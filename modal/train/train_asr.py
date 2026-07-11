"""
Fine-tune Whisper ASR for Twi/Akan on Modal — beat Round 2 baseline.

Baseline to beat (full Waxal test n=1522):
  WER 32.83%  CER 11.79%  model=teckedd/whisper-small-waxal-round2-specaug-v1

v3 lesson: continued full FT without SpecAugment overfit (test WER 33.99%).
v4 recipe: freeze encoder, SpecAugment, lower LR, early-stop on val WER,
always save full WhisperProcessor with the weights.

  modal run modal/train/train_asr.py --smoke
  modal run modal/train/train_asr.py \\
    --max-steps 1000 \\
    --freeze-encoder \\
    --push-repo teckedd/gha-whisper-small-twi-v4

Secret: huggingface-token
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-asr-train"
DEFAULT_BASE = "teckedd/whisper-small-waxal-round2-specaug-v1"
# Official bar — do not promote unless full-test WER is below this
BASELINE_WER = 0.3283

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2.post1",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
        "tensorboard==2.18.0",
        "numpy<2.3",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _normalize_text(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFC", text or "")
    t = t.lower()
    t = re.sub(r"[^\w\sɛɔáàâäéèêëíìîïóòôöúùûüńŋ]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=5 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/checkpoints": ckpt_vol,
        "/results": results_vol,
    },
    secrets=SECRETS,
    memory=32768,
)
def train(
    base_model: str = DEFAULT_BASE,
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    max_steps: int = 1000,
    learning_rate: float = 5e-6,
    batch_size: int = 8,
    grad_accum: int = 4,
    freeze_encoder: bool = True,
    weight_decay: float = 0.01,
    push_repo: Optional[str] = None,
    smoke: bool = False,
    run_name: str = "v4",
) -> dict[str, Any]:
    """
    Careful continued fine-tune from Round 2.
    Freezes encoder by default (v3 overfit with full FT).
    SpecAugment on train features. Early-stop on val WER.
    """
    import json
    import random

    import numpy as np
    import torch
    import evaluate
    from datasets import Audio, load_dataset
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if smoke:
        max_steps = min(max_steps, 20)
        batch_size = min(batch_size, 2)

    token = _hf_token()
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    cache = "/root/.cache/huggingface"
    out_dir = f"/checkpoints/gha-asr/{run_name}_{base_model.replace('/', '_')}_s{max_steps}"
    os.makedirs(out_dir, exist_ok=True)

    # Safer LR when unfreezing encoder
    if not freeze_encoder and learning_rate > 5e-6:
        learning_rate = 5e-6
    if freeze_encoder and learning_rate < 5e-6:
        learning_rate = 1e-5

    print(
        f"[train] {run_name} base={base_model} steps={max_steps} "
        f"freeze_encoder={freeze_encoder} lr={learning_rate}"
    )

    raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
    print("[train] splits", list(raw.keys()))

    split_train = "train" if "train" in raw else list(raw.keys())[0]
    # Prefer true dev/validation — never use test for model selection
    if "validation" in raw:
        split_eval = "validation"
    elif "dev" in raw:
        split_eval = "dev"
    elif "test" in raw:
        # last resort: hold out 5% of train instead of peeking at test
        split_eval = None
    else:
        split_eval = None

    train_ds = raw[split_train]
    if split_eval:
        eval_ds = raw[split_eval]
    else:
        n = len(train_ds)
        idx = list(range(n))
        random.Random(42).shuffle(idx)
        cut = max(1, int(0.05 * n))
        eval_ds = train_ds.select(idx[:cut])
        train_ds = train_ds.select(idx[cut:])
        print(f"[train] no dev split — held out {cut} from train for selection")

    if smoke:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(16, len(eval_ds))))

    audio_col = "audio" if "audio" in train_ds.column_names else train_ds.column_names[0]
    text_col = next(
        (
            c
            for c in ("text", "sentence", "transcription", "transcript", "normalized_text")
            if c in train_ds.column_names
        ),
        None,
    )
    if text_col is None:
        raise RuntimeError(f"No text column in {train_ds.column_names}")

    train_ds = train_ds.cast_column(audio_col, Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column(audio_col, Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(base_model, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, cache_dir=cache, token=token
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[train] encoder frozen · trainable {trainable}/{total}")

    def prepare_batch(batch: dict) -> dict:
        audio = batch[audio_col]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        ).input_features[0]
        labels = processor.tokenizer(_normalize_text(batch[text_col])).input_ids
        return {"input_features": feats, "labels": labels}

    train_prep = train_ds.map(
        prepare_batch, remove_columns=train_ds.column_names, desc="prep-train"
    )
    eval_prep = eval_ds.map(
        prepare_batch, remove_columns=eval_ds.column_names, desc="prep-eval"
    )

    class DataCollatorSpecAug:
        """Pad + SpecAugment (time + feature masks) on train features."""

        def __init__(self, processor, train: bool = True):
            self.processor = processor
            self.train = train
            self.mask_time_prob = 0.05
            self.mask_time_length = 10
            self.mask_feature_prob = 0.05
            self.mask_feature_length = 16

        def _spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
            # feats: (B, n_mels, T)
            if not self.train or feats.dim() != 3:
                return feats
            b, n_mels, t = feats.shape
            out = feats.clone()
            for i in range(b):
                # time masks
                num_t = max(1, int(self.mask_time_prob * t / self.mask_time_length))
                for _ in range(num_t):
                    length = random.randint(1, self.mask_time_length)
                    if t - length <= 0:
                        continue
                    start = random.randint(0, t - length)
                    out[i, :, start : start + length] = 0
                # feature masks
                num_f = max(1, int(self.mask_feature_prob * n_mels / self.mask_feature_length))
                for _ in range(num_f):
                    length = random.randint(1, self.mask_feature_length)
                    if n_mels - length <= 0:
                        continue
                    start = random.randint(0, n_mels - length)
                    out[i, start : start + length, :] = 0
            return out

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (
                self.processor.tokenizer.bos_token_id is not None
                and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
            ):
                labels = labels[:, 1:]
            batch["input_features"] = self._spec_augment(batch["input_features"])
            batch["labels"] = labels
            return batch

    train_collator = DataCollatorSpecAug(processor, train=True)
    # Eval collator without augment — reuse class with train=False
    eval_collator = DataCollatorSpecAug(processor, train=False)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [_normalize_text(s) for s in pred_str]
        label_str = [_normalize_text(s) for s in label_str]
        return {
            "wer": float(wer_metric.compute(predictions=pred_str, references=label_str)),
            "cer": float(cer_metric.compute(predictions=pred_str, references=label_str)),
        }

    eval_steps = 50 if not smoke else 10
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=min(100, max_steps // 10),
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=not smoke,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,  # explicit processor+model push after train
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_prep,
        eval_dataset=eval_prep,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=(
            []
            if smoke
            else [EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.001)]
        ),
    )
    # Use non-augment collator at eval time
    trainer.data_collator = train_collator

    # Monkey-patch evaluate to use no-augment collator
    _orig_get_eval = trainer.get_eval_dataloader

    def get_eval_dataloader(eval_dataset=None):
        old = trainer.data_collator
        trainer.data_collator = eval_collator
        try:
            return _orig_get_eval(eval_dataset)
        finally:
            trainer.data_collator = old

    trainer.get_eval_dataloader = get_eval_dataloader  # type: ignore[method-assign]

    train_result = trainer.train()
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()

    # Always persist full processor (v3 hub was missing tokenizer)
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    ckpt_vol.commit()

    hub_status = None
    if push_repo and not smoke and token:
        try:
            model.push_to_hub(push_repo, token=token, private=False)
            processor.push_to_hub(push_repo, token=token, private=False)
            hub_status = f"pushed:{push_repo}"
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"

    val_wer = float(eval_metrics.get("eval_wer", 1.0))
    summary = {
        "status": "ok",
        "run_name": run_name,
        "base_model": base_model,
        "output_dir": out_dir,
        "freeze_encoder": freeze_encoder,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "baseline_wer_to_beat": BASELINE_WER,
        "val_wer": val_wer,
        "val_cer": float(eval_metrics.get("eval_cer", 1.0)),
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()
        },
        "eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()
        },
        "hub": hub_status,
        "push_repo": push_repo,
        "note": "Promote only after full-test WER < 0.3283 (eval_asr.py n=1522)",
    }
    with open(f"/results/train_{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print("[train] done", summary)
    return summary


@app.local_entrypoint()
def main(
    base_model: str = DEFAULT_BASE,
    max_steps: int = 1000,
    push_repo: str = "teckedd/gha-whisper-small-twi-v4",
    smoke: bool = False,
    freeze_encoder: bool = True,
    learning_rate: float = 1e-5,
    run_name: str = "v4",
):
    result = train.remote(
        base_model=base_model,
        max_steps=max_steps,
        push_repo=push_repo or None,
        smoke=smoke,
        freeze_encoder=freeze_encoder,
        learning_rate=learning_rate,
        run_name=run_name,
    )
    print(result)
