"""
Fine-tune KhayaAI DONDO / w2v-BERT CTC for Ghana Health AI.

This is the credible DONDO trial path, not a serving switch:

  modal run modal/train/train_dondo_asr.py --smoke
  modal run --detach modal/train/train_dondo_asr.py \\
    --run-name dondo-waxal-twi-v1 \\
    --max-steps 800 --train-limit 1800 --eval-limit 200 \\
    --push-repo teckedd/gha-dondo-w2v-bert-twi-v1 --no-wait

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

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
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
    )
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
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
    learning_rate: float = 5e-6,
    train_limit: int = 1800,
    eval_limit: int = 200,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    resume_from_checkpoint: Optional[str] = None,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    import json
    import torch
    from datasets import Audio, Dataset, load_dataset
    from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments

    token = _hf_token()
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    lang_id = LANGUAGE_MAP[language]
    print(
        "[train-dondo] start "
        f"run={run_name} model={model_id} language={language} max_steps={max_steps} "
        f"batch={per_device_train_batch_size} accum={gradient_accumulation_steps} "
        f"resume={resume_from_checkpoint or 'none'}",
        flush=True,
    )

    if smoke:
        max_steps = min(max_steps, 8)
        train_limit = min(train_limit, 24)
        eval_limit = min(eval_limit, 8)
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

    print("[train-dondo] loading train dataset", flush=True)
    train_ds = load_dataset(
        "google/WaxalNLP",
        "aka_asr",
        split="train",
        streaming=True,
        token=token,
        cache_dir=cache,
    ).cast_column("audio", Audio(sampling_rate=16000))
    print("[train-dondo] train dataset ready", flush=True)
    print("[train-dondo] loading eval dataset", flush=True)
    try:
        eval_ds = load_dataset(
            "google/WaxalNLP",
            "aka_asr",
            split="validation",
            streaming=True,
            token=token,
            cache_dir=cache,
        ).cast_column("audio", Audio(sampling_rate=16000))
    except Exception:  # noqa: BLE001
        print("[train-dondo] validation split unavailable; falling back to test", flush=True)
        eval_ds = load_dataset(
            "google/WaxalNLP",
            "aka_asr",
            split="test",
            streaming=True,
            token=token,
            cache_dir=cache,
        ).cast_column("audio", Audio(sampling_rate=16000))
    print("[train-dondo] eval dataset ready", flush=True)

    train_rows = _prepare_rows(train_ds, processor, train_limit, "train")
    eval_rows = _prepare_rows(eval_ds, processor, eval_limit, "eval")

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

    print("[train-dondo] creating training arguments", flush=True)
    args = TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=max(5, max_steps // 20),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(20, max_steps // 4),
        save_steps=max(50, max_steps // 4),
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
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

    hub_status = None
    if push_repo and not smoke and token:
        try:
            model.push_to_hub(push_repo, token=token, private=False)
            processor.push_to_hub(push_repo, token=token, private=False)
            import sys

            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            write_and_push_model_card(
                push_repo,
                task="automatic-speech-recognition",
                language=["tw", "ak"],
                base_model=model_id,
                datasets=["google/WaxalNLP"],
                metrics={"val_wer": val_wer, "val_cer": val_cer},
                summary=(
                    f"DONDO w2v-BERT CTC fine-tune trial for Ghana Health AI (`{run_name}`). "
                    f"Promotion candidate: {promote}."
                ),
                extra_markdown=f"""
## DONDO language prefix

This checkpoint follows DONDO's language-conditioned CTC setup. For Asante Twi, prepend
language id `{lang_id}` to acoustic features before decoding.

## Promotion gate

- Current v6 Whisper beam=5 WER: `{BASELINE_WER}`
- This validation WER: `{val_wer}`
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
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "resume_from_checkpoint": resume_arg,
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
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint or None,
        push_repo=push_repo or None,
        smoke=smoke,
    )
    print(f"[train-dondo] spawned {call.object_id} run={run_name}")
    if not no_wait:
        print(call.get())
