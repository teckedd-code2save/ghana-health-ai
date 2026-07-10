"""
Fine-tune Whisper ASR for Twi/Akan on Modal — real training job.

Default base: Round 2 production checkpoint.
Default data: google/WaxalNLP aka_asr (same family that produced Round 2).

  # dry structure / smoke
  modal run modal/train/train_asr.py --max-steps 10 --smoke

  # serious run (needs HF token with write if --push-repo set)
  modal run modal/train/train_asr.py \\
    --base-model teckedd/whisper-small-waxal-round2-specaug-v1 \\
    --max-steps 1500 \\
    --push-repo teckedd/gha-whisper-small-twi-v3

Requires Modal secret `huggingface` with HF_TOKEN (read for datasets; write to push).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-asr-train"
DEFAULT_BASE = "teckedd/whisper-small-waxal-round2-specaug-v1"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)

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

# Workspace secret: modal secret create huggingface-token HF_TOKEN=...
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


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/checkpoints": ckpt_vol,
    },
    secrets=SECRETS,
    memory=32768,
)
def train(
    base_model: str = DEFAULT_BASE,
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    max_steps: int = 1500,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    grad_accum: int = 4,
    freeze_encoder: bool = False,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Full Whisper fine-tune. On smoke=True runs a few steps to validate the pipeline.
    """
    import torch
    from datasets import Audio, load_dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    import evaluate

    if smoke:
        max_steps = min(max_steps, 15)
        batch_size = min(batch_size, 2)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    cache = "/root/.cache/huggingface"
    out_dir = f"/checkpoints/gha-asr/{base_model.replace('/', '_')}_steps{max_steps}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[train] base={base_model} steps={max_steps} freeze_encoder={freeze_encoder}")

    # --- data ---
    # Waxal aka_asr typically has train / validation / test (names vary by revision)
    raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
    split_train = "train" if "train" in raw else list(raw.keys())[0]
    split_eval = (
        "validation"
        if "validation" in raw
        else "dev"
        if "dev" in raw
        else "test"
        if "test" in raw
        else split_train
    )

    train_ds = raw[split_train]
    eval_ds = raw[split_eval]
    if smoke:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(16, len(eval_ds))))

    # Column heuristics
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
    # Open decode for Akan fine-tunes
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("[train] encoder frozen")

    def prepare_batch(batch: dict) -> dict:
        audio = batch[audio_col]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        ).input_features[0]
        labels = processor.tokenizer(
            _normalize_text(batch[text_col]),
        ).input_ids
        return {"input_features": feats, "labels": labels}

    # Map without batched=True first for heterogeneous audio objects
    cols_to_remove = [c for c in train_ds.column_names if c not in ()]
    train_prep = train_ds.map(
        prepare_batch, remove_columns=cols_to_remove, desc="prep-train"
    )
    eval_prep = eval_ds.map(
        prepare_batch,
        remove_columns=[c for c in eval_ds.column_names],
        desc="prep-eval",
    )

    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
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
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
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
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=min(200, max_steps // 5),
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=max(50, max_steps // 10) if not smoke else 10,
        save_steps=max(50, max_steps // 10) if not smoke else 10,
        save_total_limit=3,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=not smoke,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=bool(push_repo) and not smoke,
        hub_model_id=push_repo,
        hub_token=token,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_prep,
        eval_dataset=eval_prep,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    ckpt_vol.commit()

    summary = {
        "status": "ok",
        "base_model": base_model,
        "output_dir": out_dir,
        "train_metrics": {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()},
        "eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()
        },
        "push_repo": push_repo,
        "smoke": smoke,
    }
    print("[train] done", summary)
    return summary


@app.local_entrypoint()
def main(
    base_model: str = DEFAULT_BASE,
    max_steps: int = 1500,
    push_repo: str = "",
    smoke: bool = False,
    freeze_encoder: bool = False,
):
    result = train.remote(
        base_model=base_model,
        max_steps=max_steps,
        push_repo=push_repo or None,
        smoke=smoke,
        freeze_encoder=freeze_encoder,
    )
    print(result)
