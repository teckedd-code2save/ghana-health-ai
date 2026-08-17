"""
Fine-tune KhayaAI DONDO / w2v-BERT CTC for Ghana Health AI.

v1 (baseline: Waxal-only, LR 5e-6, greedy CTC):

  modal run modal/train/train_dondo_asr.py --smoke
  modal run --detach modal/train/train_dondo_asr.py \\
    --run-name dondo-waxal-twi-v1 \\
    --max-steps 800 --train-limit 1800 --eval-limit 200 \\
    --push-repo teckedd/gha-dondo-w2v-bert-twi-v1 --no-wait

v2 (full data mix + higher LR + optional KenLM beam decode; see
docs/asr-rnd-session-2026-08-15.md "Stage 2 design"):

  modal run --detach modal/train/train_dondo_asr.py \\
    --run-name dondo-twi-v2 \\
    --max-steps 2500 --learning-rate 5e-5 \\
    --train-limit 0 --cv-twi-limit 3000 --use-local-data \\
    --push-repo teckedd/gha-dondo-w2v-bert-twi-v2 --no-wait

v2 notes:
- --learning-rate default stays 5e-6 for v1 compatibility; v2 recommends 5e-5
  (DONDO paper step-1 rate; the paper anneals to 5e-6 — our cosine schedule
  approximates the anneal within one run).
- --train-limit 0 streams all available Waxal train rows (capped at 20000).
- --cv-twi-limit N mixes in up to N validated-only Common Voice 22 Twi rows.
- --use-local-data mixes in the local recorder corpus mounted at
  /root/gha_local_asr (--local-manifest-path picks the manifest file).
- --lm-path <kenlm.arpa|kenlm.bin> adds pyctcdecode beam+LM WER/CER to the
  final eval; a missing file or missing kenlm degrades to greedy-only with a
  log line (never crashes the run).
- --warmup-ratio > 0 overrides the legacy warmup_steps heuristic.

OOM-safe resume:
  modal run --detach modal/train/train_dondo_asr.py \\
    --run-name dondo-waxal-twi-v1 \\
    --max-steps 800 --train-limit 1800 --eval-limit 200 \\
    --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \\
    --resume-from-checkpoint auto \\
    --push-repo teckedd/gha-dondo-w2v-bert-twi-v1 --no-wait

Promotion bar:
- beat current v6 Whisper on our held-out Waxal / health / phone evals
- preserve English via separate evaluation
- push only with a model card
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-dondo-asr-train"
DEFAULT_MODEL = "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en"
BASELINE_WER = 0.3044  # current v6 beam=5 full Waxal test
WAXAL_FULL_CAP = 20000  # generous streaming cap when --train-limit 0 (all rows)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_ASR_DIR = os.path.join(_REPO_ROOT, "tmp", "asr-local-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # g++ compiles kenlm from sdist (no manylinux wheels). cmake must be <4:
    # cmake 4.x rejects kenlm 0.2.0's ancient CMakeLists. Installed as a
    # separate early layer so the binary is on PATH when kenlm builds.
    .apt_install("ffmpeg", "libsndfile1", "g++")
    .pip_install("cmake==3.31.6")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2.post1",
        "numpy<2.3",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
        "pyctcdecode==0.5.0",
    )
    # kenlm must build WITHOUT pip's isolated build env (which would pull the
    # latest cmake 4.x and fail on kenlm's ancient CMakeLists); the pinned
    # cmake 3.31.6 layer above is used instead.
    .run_commands("pip install --no-build-isolation kenlm==0.2.0")
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
    )
    # Local recorder corpus (manifest.jsonl / manifest.train32.jsonl + audio/)
    .add_local_dir(
        local_path=_LOCAL_ASR_DIR,
        remote_path="/root/gha_local_asr",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


LANGUAGE_MAP = {
    "Adangme": 0,
    "Akuapem Twi": 1,
    "Asante Twi": 2,
    "Dagbani": 3,
    "Dagaare": 4,
    "Ewe": 5,
    "African English": 6,
    "Fante": 7,
    "French": 8,
    "Ga": 9,
    "Gonja": 10,
    "Gurene": 11,
    "Hausa": 12,
    "Igbo": 13,
    "Kasem": 14,
    "Kikuyu": 15,
    "Konkomba (Likpakpaanl)": 16,
    "Konkomba (Likoonli)": 17,
    "Krio": 18,
    "Kusaal": 19,
    "Luo": 20,
    "Mampruli": 21,
    "Mende": 22,
    "Meru/Kimeru": 23,
    "Nzema": 24,
    "Pidgin": 25,
    "Shona": 26,
    "Swahili": 27,
    "Temne": 28,
    "Wali": 29,
    "Wolof": 30,
    "Yoruba": 31,
}


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _normalize_text(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFC", text or "").lower()
    t = re.sub(r"[^\w\sɛɔáàâäéèêëíìîïóòôöúùûüńŋ']", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_text_col(columns: list[str]) -> Optional[str]:
    for col in ("sentence", "text", "transcription", "transcript", "normalized_text"):
        if col in columns:
            return col
    return None


def _add_language_prefix(features, lang_id: int):
    import torch

    if not hasattr(features, "dim"):
        features = torch.tensor(features, dtype=torch.float32)
    if features.dim() == 3:
        features = features.squeeze(0)
    _, dim = features.shape
    lang_vec = torch.zeros(dim, dtype=features.dtype, device=features.device)
    lang_vec[lang_id % dim] = 1.0
    return torch.cat([lang_vec.unsqueeze(0), features], dim=0)


@dataclass
class CtcBatchCollator:
    processor: Any
    lang_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [
            _add_language_prefix(item["input_features"], self.lang_id) for item in features
        ]
        batch = {
            "input_features": torch.nn.utils.rnn.pad_sequence(
                input_features,
                batch_first=True,
                padding_value=0.0,
            )
        }
        labels = [item["labels"] for item in features]
        label_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch.attention_mask.ne(1),
            -100,
        )
        return batch


def _find_audio_col(columns: list[str]) -> Optional[str]:
    if "audio" in columns:
        return "audio"
    for col in columns:
        if "audio" in col.lower():
            return col
    return None


def _prepare_rows(ds, processor, limit: int, name: str) -> list[dict[str, Any]]:
    rows = []
    text_col = None
    audio_col = None
    print(f"[train-dondo] prepare {name} rows limit={limit or 'all'}", flush=True)
    for row in ds:
        if text_col is None or audio_col is None:
            columns = list(row.keys())
            text_col = _find_text_col(columns)
            audio_col = _find_audio_col(columns)
            print(
                f"[train-dondo] {name} columns text={text_col} audio={audio_col}",
                flush=True,
            )
            if text_col is None:
                raise RuntimeError(f"No transcript column found: {columns}")
            if audio_col is None:
                raise RuntimeError(f"No audio column found: {columns}")
        if limit and len(rows) >= limit:
            break
        audio = row[audio_col]
        proc = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
        features = getattr(proc, "input_features", None)
        if features is None:
            features = getattr(proc, "input_values")
        label_text = _normalize_text(str(row[text_col]))
        labels = processor.tokenizer(label_text).input_ids
        rows.append({"input_features": features.squeeze(0), "labels": labels})
        if len(rows) % 100 == 0:
            print(f"[train-dondo] prepared {name} rows={len(rows)}", flush=True)
    print(f"[train-dondo] prepared {name} total={len(rows)}", flush=True)
    return rows


def _compute_metrics(processor):
    import evaluate
    import numpy as np

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    def compute(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = [_normalize_text(x) for x in processor.batch_decode(pred_ids)]
        refs = [
            _normalize_text(x)
            for x in processor.batch_decode(label_ids, group_tokens=False)
        ]
        return {
            "wer": float(wer_m.compute(predictions=preds, references=refs)),
            "cer": float(cer_m.compute(predictions=preds, references=refs)),
        }

    return compute


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    timeout=6 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/checkpoints": ckpt_vol,
        "/results": results_vol,
    },
    secrets=SECRETS,
)
def train_dondo(
    model_id: str = DEFAULT_MODEL,
    run_name: str = "dondo-waxal-twi-v1",
    language: str = "Asante Twi",
    max_steps: int = 800,
    learning_rate: float = 5e-6,  # v1 compat; v2 recommends 5e-5 (DONDO paper step 1)
    train_limit: int = 1800,  # 0 = all Waxal train rows (streamed, capped at WAXAL_FULL_CAP)
    eval_limit: int = 200,
    cv_twi_limit: int = 0,  # >0 mixes in validated-only Common Voice 22 Twi rows
    use_local_data: bool = False,
    local_manifest_path: str = "/root/gha_local_asr/manifest.jsonl",
    lm_path: str = "",  # KenLM .arpa/.binary for beam+LM eval decode; empty = greedy only
    warmup_ratio: float = 0.0,  # >0 overrides the legacy warmup_steps heuristic
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    resume_from_checkpoint: Optional[str] = None,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    import json
    import random

    import numpy as np
    import torch
    from datasets import Audio, Dataset, concatenate_datasets, load_dataset
    from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments

    token = _hf_token()
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    lang_id = LANGUAGE_MAP[language]
    print(
        "[train-dondo] start "
        f"run={run_name} model={model_id} language={language} max_steps={max_steps} "
        f"lr={learning_rate} train_limit={train_limit or 'all'} "
        f"cv_twi_limit={cv_twi_limit} local={use_local_data} lm={lm_path or 'none'} "
        f"batch={per_device_train_batch_size} accum={gradient_accumulation_steps} "
        f"resume={resume_from_checkpoint or 'none'}",
        flush=True,
    )

    if smoke:
        max_steps = min(max_steps, 8)
        train_limit = min(train_limit, 24) if train_limit else 24
        eval_limit = min(eval_limit, 8)
        cv_twi_limit = min(cv_twi_limit, 16)
        per_device_train_batch_size = min(per_device_train_batch_size, 2)

    print("[train-dondo] loading processor", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache, token=token)
    print("[train-dondo] loading model", flush=True)
    model = AutoModelForCTC.from_pretrained(model_id, cache_dir=cache, token=token)
    print("[train-dondo] model loaded", flush=True)
    model.config.ctc_zero_infinity = True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[train-dondo] gradient checkpointing enabled", flush=True)

    def _with_retries(fn, what: str, attempts: int = 4):
        """Retry transient HF CDN failures (read timeouts, 503s). The hf-cache
        volume keeps completed shards, so each retry resumes where it died."""
        import time

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[train-dondo] {what} attempt {attempt}/{attempts} failed: {exc}",
                    flush=True,
                )
                if attempt == attempts:
                    raise
                time.sleep(15 * attempt)

    def _load_waxal_rows(split: str, limit: int, name: str) -> list[dict[str, Any]]:
        def _do() -> list[dict[str, Any]]:
            ds = load_dataset(
                "google/WaxalNLP",
                "aka_asr",
                split=split,
                streaming=True,
                token=token,
                cache_dir=cache,
            ).cast_column("audio", Audio(sampling_rate=16000))
            return _prepare_rows(ds, processor, limit, name)

        return _with_retries(_do, name)

    def _materialize_cv_twi(limit: int) -> list[dict[str, Any]]:
        """Stream Common Voice 22 Twi (validated-only) into raw audio/text rows,
        capped at `limit` — split slicing would resolve every CV shard."""
        print(f"[train-dondo] streaming common-voice-tw target={limit}", flush=True)
        ds_stream = load_dataset(
            "fsicoli/common_voice_22_0",
            "tw",
            split="train",
            streaming=True,
            token=token,
            cache_dir=cache,
            trust_remote_code=True,
        ).cast_column("audio", Audio(sampling_rate=16000))
        rows: list[dict[str, Any]] = []
        skipped = 0
        for row in ds_stream:
            try:
                if int(row.get("up_votes") or 0) < 2 or int(row.get("down_votes") or 0) != 0:
                    skipped += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
            text = str(row.get("sentence") or row.get("text") or "")
            if len(_normalize_text(text)) < 2:
                skipped += 1
                continue
            audio = row.get("audio")
            if not isinstance(audio, dict) or audio.get("array") is None:
                skipped += 1
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate") or 16000)
            dur = arr.size / max(1, sr)
            if dur < 0.5 or dur > 28.0:
                skipped += 1
                continue
            rows.append({"audio": {"array": arr, "sampling_rate": sr}, "sentence": text})
            if len(rows) >= limit:
                break
        print(
            f"[train-dondo] common-voice-tw materialized n={len(rows)} skipped={skipped}",
            flush=True,
        )
        return rows

    def _cast_audio_chunked(ds, name: str):
        """cast_column to Audio; work around the datasets 3.1.0 + pyarrow bug
        where casting a multi-chunk table of pre-decoded audio dicts crashes
        ("Cannot convert ChunkedArray to Array") — rebuild in 800-row parts
        (single-chunk casts are proven fine)."""
        try:
            return ds.cast_column("audio", Audio(sampling_rate=16000))
        except TypeError as exc:
            print(
                f"[train-dondo] {name} chunked audio-cast workaround ({exc})",
                flush=True,
            )
            rows = ds.to_list()
            part_size = 800
            parts = [
                Dataset.from_list(rows[i : i + part_size]).cast_column(
                    "audio", Audio(sampling_rate=16000)
                )
                for i in range(0, len(rows), part_size)
            ]
            return parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    def _load_local_rows(manifest_path: str) -> list[dict[str, Any]]:
        """Load the local recorder manifest (audio_path + reference per line),
        decode the WAVs via an Audio cast, and prepare feature rows."""
        if not manifest_path or not os.path.exists(manifest_path):
            print(f"[train-dondo] local ASR manifest not found: {manifest_path}", flush=True)
            return []
        raw: list[dict[str, Any]] = []
        skipped = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                audio_path = str(item.get("audio_path") or "")
                text = str(item.get("reference") or item.get("text") or "")
                if not audio_path or not os.path.exists(audio_path):
                    skipped += 1
                    continue
                if len(_normalize_text(text)) < 2:
                    skipped += 1
                    continue
                raw.append({"audio": audio_path, "text": text})
        if len(raw) < 8:
            print(
                f"[train-dondo] local ASR manifest too small n={len(raw)} skipped={skipped}",
                flush=True,
            )
            return []
        ds = _cast_audio_chunked(Dataset.from_list(raw), "local")
        print(
            f"[train-dondo] local ASR ready n={len(ds)} skipped={skipped} "
            f"manifest={manifest_path}",
            flush=True,
        )
        return _prepare_rows(ds, processor, 0, "local")

    # ── Train data: Waxal primary + optional CV-Twi + optional local ──
    waxal_limit = train_limit if train_limit > 0 else WAXAL_FULL_CAP
    print(f"[train-dondo] loading train dataset limit={waxal_limit}", flush=True)
    train_rows = _load_waxal_rows("train", waxal_limit, "train")
    sources_used = [f"google/WaxalNLP:aka_asr(n={len(train_rows)})"]

    if cv_twi_limit > 0:
        cv_raw = _with_retries(lambda: _materialize_cv_twi(cv_twi_limit), "common-voice-tw")
        if len(cv_raw) >= 30:
            cv_rows = _prepare_rows(cv_raw, processor, 0, "common-voice-tw")
            train_rows = train_rows + cv_rows
            sources_used.append(f"fsicoli/common_voice_22_0:tw(n={len(cv_rows)})")
        else:
            print(
                f"[train-dondo] common-voice-tw too small n={len(cv_raw)}; skipping",
                flush=True,
            )

    if use_local_data:
        local_rows = _load_local_rows(local_manifest_path)
        if local_rows:
            train_rows = train_rows + local_rows
            sources_used.append(f"local:ghana-health-ai-recorder(n={len(local_rows)})")

    if len(sources_used) > 1:
        random.Random(42).shuffle(train_rows)
        print(f"[train-dondo] MIX sources={sources_used} total={len(train_rows)}", flush=True)

    print("[train-dondo] loading eval dataset", flush=True)
    try:
        eval_rows = _load_waxal_rows("validation", eval_limit, "eval")
    except Exception:  # noqa: BLE001
        print("[train-dondo] validation split unavailable; falling back to test", flush=True)
        eval_rows = _load_waxal_rows("test", eval_limit, "eval")
    print("[train-dondo] eval dataset ready", flush=True)

    out_dir = f"/checkpoints/gha-dondo/{run_name}"
    resume_arg: Optional[str | bool] = None
    print(f"[train-dondo] output_dir={out_dir}", flush=True)
    if resume_from_checkpoint:
        if resume_from_checkpoint == "auto":
            from transformers.trainer_utils import get_last_checkpoint

            resume_arg = get_last_checkpoint(out_dir)
            print(f"[train-dondo] auto resume checkpoint={resume_arg}", flush=True)
        else:
            resume_arg = resume_from_checkpoint

    warmup_kwargs: dict[str, Any] = (
        {"warmup_ratio": warmup_ratio}
        if warmup_ratio and warmup_ratio > 0
        else {"warmup_steps": max(5, max_steps // 20)}
    )
    print("[train-dondo] creating training arguments", flush=True)
    args = TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(20, max_steps // 4),
        save_steps=max(50, max_steps // 4),
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        **warmup_kwargs,
    )

    print("[train-dondo] creating trainer", flush=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        data_collator=CtcBatchCollator(processor=processor, lang_id=lang_id),
        compute_metrics=_compute_metrics(processor),
    )
    print("[train-dondo] starting trainer.train", flush=True)
    train_metrics = trainer.train(resume_from_checkpoint=resume_arg).metrics
    print("[train-dondo] trainer.train complete; running final eval", flush=True)
    eval_metrics = trainer.evaluate()
    print("[train-dondo] final eval complete; saving model", flush=True)
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    ckpt_vol.commit()

    val_wer = float(eval_metrics.get("eval_wer", 999.0))
    val_cer = float(eval_metrics.get("eval_cer", 999.0))
    promote = val_wer < BASELINE_WER

    # Optional final-eval decode with a KenLM language model (beam search).
    # Greedy WER/CER above remain the promotion-gate numbers; these are the
    # v2 decode-quality measurements reported alongside them.
    lm_decode_used = False
    val_lm_wer: Optional[float] = None
    val_lm_cer: Optional[float] = None
    if lm_path:
        if not os.path.exists(lm_path):
            print(f"[train-dondo] lm file missing: {lm_path}; greedy-only eval", flush=True)
        else:
            try:
                import kenlm  # noqa: F401
                import evaluate
                from pyctcdecode import build_ctcdecoder

                vocab = processor.tokenizer.get_vocab()
                sorted_tokens = [
                    tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])
                ]
                decoder = build_ctcdecoder(sorted_tokens, kenlm_model_path=lm_path)
                print(f"[train-dondo] beam+LM decoding eval logits lm={lm_path}", flush=True)
                pred = trainer.predict(trainer.eval_dataset)
                logits = np.asarray(pred.predictions)
                label_ids = pred.label_ids
                label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                refs = [
                    _normalize_text(x)
                    for x in processor.batch_decode(label_ids, group_tokens=False)
                ]
                preds = [
                    _normalize_text(" ".join(str(decoder.decode(logits[i])).split("|")))
                    for i in range(logits.shape[0])
                ]
                wer_m = evaluate.load("wer")
                cer_m = evaluate.load("cer")
                val_lm_wer = float(wer_m.compute(predictions=preds, references=refs))
                val_lm_cer = float(cer_m.compute(predictions=preds, references=refs))
                lm_decode_used = True
                print(
                    f"[train-dondo] lm decode wer={val_lm_wer:.4f} cer={val_lm_cer:.4f} "
                    f"(greedy wer={val_wer:.4f} cer={val_cer:.4f})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[train-dondo] LM decode unavailable ({exc}); "
                    "falling back to greedy-only",
                    flush=True,
                )

    hub_status = None
    if push_repo and not smoke and token:
        try:
            model.push_to_hub(push_repo, token=token, private=False)
            processor.push_to_hub(push_repo, token=token, private=False)
            import sys

            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            card_metrics: dict[str, Any] = {"val_wer": val_wer, "val_cer": val_cer}
            if lm_decode_used:
                card_metrics["val_lm_wer"] = val_lm_wer
                card_metrics["val_lm_cer"] = val_lm_cer
            lm_section = (
                f"Beam search + KenLM (`{lm_path}`): WER `{val_lm_wer}` / CER `{val_lm_cer}`"
                if lm_decode_used
                else "Greedy CTC argmax only (no LM decode in this run)."
            )
            write_and_push_model_card(
                push_repo,
                task="automatic-speech-recognition",
                language=["tw", "ak"],
                base_model=model_id,
                datasets=[s.split("(", 1)[0] for s in sources_used],
                metrics=card_metrics,
                summary=(
                    f"DONDO w2v-BERT CTC fine-tune trial for Ghana Health AI (`{run_name}`). "
                    f"Promotion candidate: {promote}."
                ),
                extra_markdown=f"""
## DONDO language prefix

This checkpoint follows DONDO's language-conditioned CTC setup. For Asante Twi, prepend
language id `{lang_id}` to acoustic features before decoding.

## Training mix

- Sources: {", ".join(f"`{s}`" for s in sources_used)}
- Learning rate: `{learning_rate}` · max steps: `{max_steps}`

## Decode

- Greedy CTC: WER `{val_wer}` / CER `{val_cer}`
- {lm_section}

## Promotion gate

- Current v6 Whisper beam=5 WER: `{BASELINE_WER}`
- This validation WER (greedy): `{val_wer}`
- Promote: **{promote}**
""",
                tags=["dondo", "w2v-bert", "ctc", "asr", "asante-twi"],
                pipeline_tag="automatic-speech-recognition",
                token=token,
            )
            hub_status = f"pushed:{push_repo}+card"
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"

    summary = {
        "status": "ok",
        "run_name": run_name,
        "base_model": model_id,
        "architecture": "wav2vec2-bert-ctc",
        "language": language,
        "language_id": lang_id,
        "train_limit": len(train_rows),
        "eval_limit": len(eval_rows),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "resume_from_checkpoint": resume_arg,
        "data_sources": sources_used,
        "cv_twi_limit": cv_twi_limit,
        "use_local_data": use_local_data,
        "lm_path": lm_path or None,
        "lm_decode": lm_decode_used,
        "val_lm_wer": val_lm_wer,
        "val_lm_cer": val_lm_cer,
        "baseline_wer_to_beat": BASELINE_WER,
        "val_wer": val_wer,
        "val_cer": val_cer,
        "promote": promote,
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in train_metrics.items()
        },
        "eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()
        },
        "hub": hub_status,
        "push_repo": push_repo,
        "note": "PROMOTE CANDIDATE" if promote else "DO NOT PROMOTE — keep v6 serving",
    }

    with open(f"/results/train_{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print(summary)
    return summary


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    run_name: str = "dondo-waxal-twi-v1",
    language: str = "Asante Twi",
    max_steps: int = 800,
    learning_rate: float = 5e-6,
    train_limit: int = 1800,
    eval_limit: int = 200,
    cv_twi_limit: int = 0,
    use_local_data: bool = False,
    local_manifest_path: str = "/root/gha_local_asr/manifest.jsonl",
    lm_path: str = "",
    warmup_ratio: float = 0.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    resume_from_checkpoint: str = "",
    push_repo: str = "",
    smoke: bool = False,
    no_wait: bool = False,
):
    call = train_dondo.spawn(
        model_id=model_id,
        run_name=run_name,
        language=language,
        max_steps=max_steps,
        learning_rate=learning_rate,
        train_limit=train_limit,
        eval_limit=eval_limit,
        cv_twi_limit=cv_twi_limit,
        use_local_data=use_local_data,
        local_manifest_path=local_manifest_path,
        lm_path=lm_path,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint or None,
        push_repo=push_repo or None,
        smoke=smoke,
    )
    print(f"[train-dondo] spawned {call.object_id} run={run_name}")
    if not no_wait:
        print(call.get())
