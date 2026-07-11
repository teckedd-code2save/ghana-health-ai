"""
Fine-tune Whisper ASR for Twi/Akan on Modal — beat Round 2 baseline.

Baseline to beat (full Waxal test n=1522, greedy generate):
  WER 32.83%  CER 11.79%  model=teckedd/whisper-small-waxal-round2-specaug-v1

Lessons:
  v3 — full FT, no SpecAug, same Waxal only → test WER 33.99% (overfit)
  v4 — freeze encoder + SpecAug, same Waxal only → test WER 34.96% (still overfit)
  v5 — NEW DATA + stronger regularization. Do not only replay Waxal train.

v5 recipe:
  - Mix: Waxal train (foundation) + GhanaNLP Twi multispeaker (domain diversity)
  - Filter clips 0.5–28 s; drop empty / near-empty labels
  - Speed perturbation {0.9, 1.0, 1.1} on train audio
  - Stronger SpecAugment (time + feature)
  - Freeze encoder; low decoder LR; label smoothing; early-stop on val WER
  - Full processor always saved with weights
  - Promote ONLY if full-test WER < 0.3283

  modal run modal/train/train_asr.py --smoke
  modal run modal/train/train_asr.py \\
    --run-name v5 \\
    --max-steps 800 \\
    --push-repo teckedd/gha-whisper-small-twi-v5

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
BASELINE_CER = 0.1179

# Extra Twi speech (multi-speaker) — new signal Round 2 never saw
EXTRA_DATASETS = [
    {
        "name": "ghananlpcommunity/twi-speech-text-multispeaker-16k",
        "config": None,
        "split": "train",
        "weight": 0.35,
        "note": "CC BY-NC 4.0 — research/non-commercial train only",
    },
]

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
        "tqdm",
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


def _find_text_col(columns: list[str]) -> Optional[str]:
    for c in ("text", "sentence", "transcription", "transcript", "normalized_text"):
        if c in columns:
            return c
    return None


def _find_audio_col(columns: list[str]) -> str:
    if "audio" in columns:
        return "audio"
    return columns[0]


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 60 * 60,
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
    max_steps: int = 800,
    learning_rate: float = 8e-6,
    batch_size: int = 8,
    grad_accum: int = 4,
    freeze_encoder: bool = True,
    weight_decay: float = 0.05,
    waxal_weight: float = 0.65,
    use_extra_data: bool = True,
    push_repo: Optional[str] = None,
    smoke: bool = False,
    run_name: str = "v5",
    full_test_after: bool = True,
) -> dict[str, Any]:
    """
    v5 multi-source continued fine-tune from Round 2.
    Mixes external Twi speech so we are not only replaying Waxal train.
    """
    import json
    import random

    import numpy as np
    import torch
    import evaluate
    from datasets import Audio, Dataset, interleave_datasets, load_dataset
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
        full_test_after = False

    token = _hf_token()
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    cache = "/root/.cache/huggingface"
    out_dir = f"/checkpoints/gha-asr/{run_name}_{base_model.replace('/', '_')}_s{max_steps}"
    os.makedirs(out_dir, exist_ok=True)

    # Decoder-only FT defaults: slightly higher LR when encoder frozen
    if freeze_encoder and learning_rate < 5e-6:
        learning_rate = 8e-6
    if not freeze_encoder and learning_rate > 5e-6:
        learning_rate = 3e-6

    print(
        f"[train] {run_name} base={base_model} steps={max_steps} "
        f"freeze_encoder={freeze_encoder} lr={learning_rate} "
        f"extra_data={use_extra_data} waxal_w={waxal_weight}"
    )

    # ── primary: Waxal ──────────────────────────────────────────────
    raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
    print("[train] waxal splits", list(raw.keys()))

    split_train = "train" if "train" in raw else list(raw.keys())[0]
    if "validation" in raw:
        split_eval = "validation"
    elif "dev" in raw:
        split_eval = "dev"
    else:
        split_eval = None

    waxal_train = raw[split_train]
    if split_eval:
        eval_ds = raw[split_eval]
    else:
        n = len(waxal_train)
        idx = list(range(n))
        random.Random(42).shuffle(idx)
        cut = max(1, int(0.05 * n))
        eval_ds = waxal_train.select(idx[:cut])
        waxal_train = waxal_train.select(idx[cut:])
        print(f"[train] no dev split — held out {cut} from waxal train for selection")

    # NEVER use test for training or model selection
    print(f"[train] waxal train={len(waxal_train)} eval={len(eval_ds)}")

    audio_col = _find_audio_col(waxal_train.column_names)
    text_col = _find_text_col(waxal_train.column_names)
    if text_col is None:
        raise RuntimeError(f"No text column in waxal: {waxal_train.column_names}")

    def _to_audio_text(ds, a_col: str, t_col: str, source: str, max_n: int | None = None):
        """
        Fast path: filter on non-audio columns only (input_columns avoids decode),
        rename, optional cap. Bad clips dropped later in feature prep.
        """
        # Duration filter — metadata only, never touches audio bytes
        if "duration" in ds.column_names:
            before = len(ds)
            ds = ds.filter(
                lambda d: 0.5 <= float(d) <= 28.0,
                input_columns=["duration"],
                desc=f"dur-{source}",
            )
            print(f"[train] {source} duration filter {before} → {len(ds)}")

        # Text filter without loading audio
        before = len(ds)
        ds = ds.filter(
            lambda t: len(_normalize_text(t or "")) >= 2,
            input_columns=[t_col],
            desc=f"text-{source}",
        )
        print(f"[train] {source} text filter {before} → {len(ds)}")

        if max_n and len(ds) > max_n:
            idx = list(range(len(ds)))
            random.Random(hash(source) % 10_000).shuffle(idx)
            ds = ds.select(idx[:max_n])
            print(f"[train] {source} capped to {max_n}")

        keep = [a_col, t_col]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        if a_col != "audio":
            ds = ds.rename_column(a_col, "audio")
        if t_col != "text":
            ds = ds.rename_column(t_col, "text")
        # Decode deferred until feature map
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        print(f"[train] {source} ready n={len(ds)}")
        return ds

    waxal_std = _to_audio_text(waxal_train, audio_col, text_col, "waxal")
    eval_std = _to_audio_text(eval_ds, audio_col, text_col, "waxal-val")

    train_parts: list[Dataset] = [waxal_std]
    mix_weights: list[float] = [waxal_weight]
    sources_used = ["waxal"]

    # Cap external set so prep finishes in minutes not hours
    EXTRA_CAP = 4000

    if use_extra_data and not smoke:
        for extra in EXTRA_DATASETS:
            try:
                print(f"[train] loading extra {extra['name']} …")
                load_kwargs: dict[str, Any] = {
                    "token": token,
                    "cache_dir": cache,
                    "split": extra.get("split") or "train",
                }
                if extra.get("config"):
                    eds = load_dataset(
                        extra["name"], extra["config"], **load_kwargs
                    )
                else:
                    eds = load_dataset(extra["name"], **load_kwargs)
                if hasattr(eds, "keys") and not hasattr(eds, "column_names"):
                    eds = eds[extra.get("split") or "train"]
                ea = _find_audio_col(eds.column_names)
                et = _find_text_col(eds.column_names)
                if et is None:
                    print(f"[train] skip {extra['name']}: no text col {eds.column_names}")
                    continue
                std = _to_audio_text(
                    eds, ea, et, extra["name"].split("/")[-1], max_n=EXTRA_CAP
                )
                if len(std) < 50:
                    print("[train] skip extra — too small after filter")
                    continue
                train_parts.append(std)
                mix_weights.append(float(extra.get("weight") or 0.35))
                sources_used.append(extra["name"])
            except Exception as exc:  # noqa: BLE001
                print(f"[train] extra load failed {extra['name']}: {exc}")

    wsum = sum(mix_weights)
    mix_weights = [w / wsum for w in mix_weights]

    if len(train_parts) == 1:
        train_std = train_parts[0]
        print(f"[train] single-source train n={len(train_std)}")
    else:
        train_std = interleave_datasets(
            train_parts,
            probabilities=mix_weights,
            seed=42,
            stopping_strategy="all_exhausted",
        )
        print(
            f"[train] mixed train sources={sources_used} "
            f"weights={mix_weights} parts={[len(p) for p in train_parts]}"
        )

    if smoke:
        train_std = train_std.select(range(min(32, len(train_std))))
        eval_std = eval_std.select(range(min(16, len(eval_std))))

    processor = WhisperProcessor.from_pretrained(base_model, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, cache_dir=cache, token=token
    )
    # Whisper needs these for shift_tokens_right when labels are provided
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.forced_decoder_ids = None
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.num_beams = 1
    model.generation_config.max_length = 225
    print(
        f"[train] decoder_start={model.config.decoder_start_token_id} "
        f"pad={model.config.pad_token_id}"
    )

    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[train] encoder frozen · trainable {trainable:,}/{total:,}")

    # Speed-perturb + feature extract in ONE pass (no prior audio rewrite)
    SPEEDS = (0.9, 1.0, 1.1)

    def prepare_train(batch: dict) -> dict:
        audio = batch["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        dur = len(arr) / float(sr)
        # Clip/pad absurd lengths instead of empty labels (empty → Whisper crash)
        if dur < 0.3:
            pad = max(0, int(0.5 * sr) - len(arr))
            if pad:
                arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
        elif dur > 30.0:
            arr = arr[: int(30 * sr)]
        speed = random.choice(SPEEDS)
        if speed != 1.0 and len(arr) > 160:
            n_out = max(1, int(len(arr) / speed))
            x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        feats = processor.feature_extractor(
            arr, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        text = _normalize_text(batch["text"])
        if not text:
            text = " "
        labels = processor.tokenizer(text).input_ids
        if not labels:
            labels = [processor.tokenizer.eos_token_id or 50256]
        return {"input_features": feats, "labels": labels}

    def prepare_eval(batch: dict) -> dict:
        audio = batch["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        if len(arr) < int(0.3 * sr):
            arr = np.concatenate(
                [arr, np.zeros(max(0, int(0.5 * sr) - len(arr)), dtype=np.float32)]
            )
        elif len(arr) > int(30 * sr):
            arr = arr[: int(30 * sr)]
        feats = processor.feature_extractor(
            arr, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        text = _normalize_text(batch["text"]) or " "
        labels = processor.tokenizer(text).input_ids
        if not labels:
            labels = [processor.tokenizer.eos_token_id or 50256]
        return {"input_features": feats, "labels": labels}

    # Single feature map — this is the only heavy pass
    train_prep = train_std.map(
        prepare_train,
        remove_columns=train_std.column_names,
        desc="prep-train",
        writer_batch_size=64,
    )
    eval_prep = eval_std.map(
        prepare_eval,
        remove_columns=eval_std.column_names,
        desc="prep-eval",
        writer_batch_size=64,
    )

    class DataCollatorSpecAug:
        """Pad + SpecAugment (time + feature masks) on train features."""

        def __init__(self, processor, train: bool = True, strong: bool = True):
            self.processor = processor
            self.train = train
            # v5: stronger masks than v4 (0.05 → 0.08/0.10)
            self.mask_time_prob = 0.10 if strong else 0.05
            self.mask_time_length = 12 if strong else 10
            self.mask_feature_prob = 0.08 if strong else 0.05
            self.mask_feature_length = 20 if strong else 16

        def _spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
            if not self.train or feats.dim() != 3:
                return feats
            b, n_mels, t = feats.shape
            out = feats.clone()
            for i in range(b):
                num_t = max(1, int(self.mask_time_prob * t / max(1, self.mask_time_length)))
                for _ in range(num_t):
                    length = random.randint(1, self.mask_time_length)
                    if t - length <= 0:
                        continue
                    start = random.randint(0, t - length)
                    out[i, :, start : start + length] = 0
                num_f = max(
                    1, int(self.mask_feature_prob * n_mels / max(1, self.mask_feature_length))
                )
                for _ in range(num_f):
                    length = random.randint(1, self.mask_feature_length)
                    if n_mels - length <= 0:
                        continue
                    start = random.randint(0, n_mels - length)
                    out[i, start : start + length, :] = 0
            return out

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            # Drop any rows with empty/invalid labels before padding
            clean = []
            for f in features:
                labs = f.get("labels") or []
                if isinstance(labs, torch.Tensor):
                    labs = labs.tolist()
                if not labs:
                    continue
                clean.append({"input_features": f["input_features"], "labels": labs})
            if not clean:
                # Fallback: keep first sample with a dummy eos label
                f0 = features[0]
                clean = [
                    {
                        "input_features": f0["input_features"],
                        "labels": [self.processor.tokenizer.eos_token_id or 50256],
                    }
                ]

            input_features = [{"input_features": f["input_features"]} for f in clean]
            label_features = [{"input_ids": f["labels"]} for f in clean]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # Whisper: strip leading BOS if present so shift_tokens_right works
            if (
                self.processor.tokenizer.bos_token_id is not None
                and labels.size(1) > 1
                and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
            ):
                labels = labels[:, 1:]
            # Ensure at least one non-ignored label per row (decoder needs content)
            for i in range(labels.size(0)):
                if (labels[i] != -100).sum().item() == 0:
                    labels[i, 0] = self.processor.tokenizer.eos_token_id or 50256
            batch["input_features"] = self._spec_augment(batch["input_features"])
            batch["labels"] = labels
            return batch

    train_collator = DataCollatorSpecAug(processor, train=True, strong=True)
    eval_collator = DataCollatorSpecAug(processor, train=False, strong=False)

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

    eval_steps = 40 if not smoke else 10
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=min(80, max(10, max_steps // 10)),
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        # NEVER set label_smoothing_factor with Whisper — Trainer pops `labels`
        # before forward, then decoder raises "decoder_input_ids required".
        label_smoothing_factor=0.0,
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
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        max_grad_norm=1.0,
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
            else [
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.002,
                )
            ]
        ),
    )

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
            print(f"[train] hub push ok → {push_repo}")
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"
            print(f"[train] hub push failed: {exc}")

    val_wer = float(eval_metrics.get("eval_wer", 1.0))
    val_cer = float(eval_metrics.get("eval_cer", 1.0))

    # ── optional full Waxal test (promotion gate) ───────────────────
    full_test: Optional[dict[str, Any]] = None
    if full_test_after and not smoke:
        print("[train] running FULL Waxal test (promotion gate) …")
        full_test = _run_full_test(
            model=model,
            processor=processor,
            token=token,
            cache=cache,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("[train] full test", full_test)

    beats = False
    if full_test is not None:
        beats = full_test["wer"] < BASELINE_WER

    summary = {
        "status": "ok",
        "run_name": run_name,
        "base_model": base_model,
        "output_dir": out_dir,
        "recipe": "v5-mix-specaug-speedpert-freezeenc",
        "sources": sources_used,
        "mix_weights": mix_weights,
        "freeze_encoder": freeze_encoder,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "weight_decay": weight_decay,
        "label_smoothing": 0.0,
        "baseline_wer_to_beat": BASELINE_WER,
        "baseline_cer": BASELINE_CER,
        "val_wer": val_wer,
        "val_cer": val_cer,
        "full_test": full_test,
        "beats_round2": beats,
        "promote": beats,
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()
        },
        "eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()
        },
        "hub": hub_status,
        "push_repo": push_repo,
        "note": (
            "PROMOTE" if beats else "DO NOT PROMOTE — keep Round 2 in production"
        ),
    }
    with open(f"/results/train_{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print("[train] done", summary)
    return summary


def _run_full_test(model, processor, token, cache, device) -> dict[str, Any]:
    """Immutable Waxal test n=all — same protocol as eval_asr.py."""
    import evaluate
    import torch
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    raw = load_dataset(
        "google/WaxalNLP", "aka_asr", token=token, cache_dir=cache
    )
    split = "test" if "test" in raw else list(raw.keys())[-1]
    ds = raw[split]
    audio_col = "audio" if "audio" in ds.column_names else ds.column_names[0]
    text_col = _find_text_col(ds.column_names)
    assert text_col
    ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")
    preds: list[str] = []
    refs: list[str] = []
    model.eval()

    for row in tqdm(ds, desc="full-test"):
        audio = row[audio_col]
        inputs = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            ids = model.generate(input_features, max_new_tokens=225, num_beams=1)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        preds.append(_normalize_text(hyp))
        refs.append(_normalize_text(row[text_col]))

    wer = float(wer_m.compute(predictions=preds, references=refs))
    cer = float(cer_m.compute(predictions=preds, references=refs))
    return {
        "dataset": f"google/WaxalNLP/aka_asr:{split}",
        "n": len(preds),
        "wer": wer,
        "cer": cer,
        "wer_pct": round(wer * 100, 2),
        "cer_pct": round(cer * 100, 2),
        "baseline_wer_pct": round(BASELINE_WER * 100, 2),
        "delta_wer_pp": round((wer - BASELINE_WER) * 100, 2),
    }


@app.local_entrypoint()
def main(
    base_model: str = DEFAULT_BASE,
    max_steps: int = 800,
    push_repo: str = "teckedd/gha-whisper-small-twi-v5",
    smoke: bool = False,
    freeze_encoder: bool = True,
    learning_rate: float = 8e-6,
    run_name: str = "v5",
    use_extra_data: bool = True,
    waxal_weight: float = 0.65,
    full_test_after: bool = True,
    wait: bool = True,
):
    """
    Uses spawn() so `modal run --detach` keeps the GPU job alive after client exit.
    Pass --no-wait to fire-and-forget (check Modal dashboard / results volume).
    """
    call = train.spawn(
        base_model=base_model,
        max_steps=max_steps,
        push_repo=push_repo or None,
        smoke=smoke,
        freeze_encoder=freeze_encoder,
        learning_rate=learning_rate,
        run_name=run_name,
        use_extra_data=use_extra_data,
        waxal_weight=waxal_weight,
        full_test_after=full_test_after,
    )
    print(f"[train] spawned function call: {call.object_id}")
    print("[train] follow logs in Modal dashboard; summary → akan-speech-eval-results")
    if not wait:
        print("[train] --no-wait: exiting; job continues on Modal")
        return

    result = call.get()
    print(result)
    if result.get("full_test"):
        ft = result["full_test"]
        print(
            f"\n=== PROMOTION GATE ===\n"
            f"  Round 2:  WER {BASELINE_WER*100:.2f}%  CER {BASELINE_CER*100:.2f}%\n"
            f"  {run_name}:     WER {ft['wer_pct']}%  CER {ft['cer_pct']}%  "
            f"(Δ {ft['delta_wer_pp']:+.2f} pp)\n"
            f"  → {'PROMOTE' if result.get('beats_round2') else 'KEEP ROUND 2'}\n"
        )
