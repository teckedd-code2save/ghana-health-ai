"""
Evaluate a Whisper ASR checkpoint on Waxal (or any HF speech dataset).

  modal run modal/train/eval_asr.py \\
    --model-id teckedd/whisper-small-waxal-round2-specaug-v1 \\
    --max-samples 200 --streaming
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

app = modal.App("ghana-health-asr-eval")
# Reuse existing HF cache from prior Akan Speech Lab runs (Waxal already there)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_ASR_DIR = os.path.join(_REPO_ROOT, "tmp", "asr-local-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2.post1",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
        "numpy<2.3",
        "tqdm",
    )
    .add_local_dir(
        local_path=_LOCAL_ASR_DIR,
        remote_path="/root/gha_local_asr",
    )
)

# Workspace secret is `huggingface-token` (HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
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
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
        "/checkpoints": ckpt_vol,
    },
    secrets=SECRETS,
)
def evaluate_checkpoint(
    model_id: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
    checkpoint_dir: str = "",
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    split: str = "test",
    audio_column: Optional[str] = None,
    text_column: Optional[str] = None,
    language: Optional[str] = None,
    max_samples: int = 500,
    num_beams: int = 1,
    streaming: bool = True,
    trust_remote_code: bool = False,
    local_manifest_path: Optional[str] = None,
) -> dict[str, Any]:
    import json
    import torch
    import evaluate
    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from tqdm import tqdm

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # Prefer keys your Modal secret may already use
    if not token:
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_API_TOKEN")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    cache = "/root/.cache/huggingface"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_from = checkpoint_dir or model_id
    processor = WhisperProcessor.from_pretrained(load_from, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        load_from, cache_dir=cache, token=token
    ).to(device)
    model.eval()
    model.config.forced_decoder_ids = None

    if local_manifest_path:
        # Local recorder corpus mode: manifest.jsonl rows with
        # audio_path / reference / bucket (mounted at /root/gha_local_asr).
        import soundfile as sf

        if not os.path.exists(local_manifest_path):
            raise RuntimeError(f"Local manifest not found: {local_manifest_path}")
        local_rows: list[dict[str, Any]] = []
        skipped = 0
        with open(local_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ap = str(item.get("audio_path") or "")
                text = str(item.get("reference") or item.get("text") or "")
                if not ap or not os.path.exists(ap) or len(_normalize_text(text)) < 2:
                    skipped += 1
                    continue
                arr, sr = sf.read(ap, dtype="float32")
                if getattr(arr, "ndim", 1) > 1:
                    arr = arr.mean(axis=1)
                if sr != 16000:
                    import librosa

                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                local_rows.append(
                    {
                        "array": arr,
                        "text": text,
                        "bucket": str(item.get("bucket") or "unknown"),
                    }
                )
        if max_samples and len(local_rows) > max_samples:
            local_rows = local_rows[:max_samples]
        print(
            f"[eval] local manifest ready n={len(local_rows)} skipped={skipped} "
            f"manifest={local_manifest_path}",
            flush=True,
        )

        def sample_rows():
            yield from local_rows

    else:
        if streaming:
            ds = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                token=token,
                cache_dir=cache,
                streaming=True,
                trust_remote_code=trust_remote_code,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            sample_iter = ds.take(max_samples) if max_samples else ds
            column_names = list(getattr(ds, "column_names", []) or [])
        else:
            raw = load_dataset(
                dataset_name,
                dataset_config,
                token=token,
                cache_dir=cache,
                trust_remote_code=trust_remote_code,
            )
            if split not in raw:
                split = list(raw.keys())[-1]
            ds = raw[split]
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            sample_iter = ds
            column_names = list(ds.column_names)

        print(
            f"[eval] dataset ready {dataset_name}/{dataset_config}:{split} streaming={streaming} cols={column_names}",
            flush=True,
        )

        audio_col = audio_column or ("audio" if "audio" in column_names else column_names[0])
        text_col = text_column or next(
            (
                c
                for c in ("text", "sentence", "transcription", "transcript", "normalized_text")
                if c in column_names
            ),
            None,
        )
        if text_col is None:
            raise RuntimeError(f"No text column: {column_names}")

        def sample_rows():
            for row in sample_iter:
                yield {
                    "array": row[audio_col]["array"],
                    "text": str(row[text_col]),
                    "bucket": f"{dataset_config or 'default'}:{split}",
                }

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    preds: list[str] = []
    refs: list[str] = []
    bucket_preds: dict[str, list[str]] = {}
    bucket_refs: dict[str, list[str]] = {}

    for row in tqdm(sample_rows(), desc=f"eval {model_id}"):
        inputs = processor(
            row["array"], sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": 225,
                "num_beams": max(1, int(num_beams)),
            }
            if language == "en":
                gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                    language="english", task="transcribe"
                )
            ids = model.generate(
                input_features,
                **gen_kwargs,
            )
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        pred = _normalize_text(hyp)
        ref = _normalize_text(row["text"])
        preds.append(pred)
        refs.append(ref)
        bucket_preds.setdefault(row["bucket"], []).append(pred)
        bucket_refs.setdefault(row["bucket"], []).append(ref)

    wer = wer_m.compute(predictions=preds, references=refs)
    cer = cer_m.compute(predictions=preds, references=refs)
    per_bucket = {
        b: {
            "n": len(bp),
            "wer_pct": round(
                float(wer_m.compute(predictions=bp, references=bucket_refs[b])) * 100, 2
            ),
            "cer_pct": round(
                float(cer_m.compute(predictions=bp, references=bucket_refs[b])) * 100, 2
            ),
        }
        for b, bp in sorted(bucket_preds.items())
    }
    result = {
        "model_id": model_id,
        "checkpoint_dir": checkpoint_dir or None,
        "dataset": (
            f"local:{local_manifest_path}"
            if local_manifest_path
            else f"{dataset_name}/{dataset_config}:{split}"
        ),
        "dataset_name": "local_recorder_corpus" if local_manifest_path else dataset_name,
        "dataset_config": None if local_manifest_path else dataset_config,
        "split": "manifest" if local_manifest_path else split,
        "local_manifest": local_manifest_path,
        "per_bucket": per_bucket,
        "audio_column": None if local_manifest_path else audio_col,
        "text_column": ("reference" if local_manifest_path else text_col),
        "language": language,
        "n": len(preds),
        "num_beams": int(num_beams),
        "streaming": bool(streaming and not local_manifest_path),
        "wer": float(wer),
        "cer": float(cer),
        "wer_pct": round(float(wer) * 100, 2),
        "cer_pct": round(float(cer) * 100, 2),
        "beat_this": {
            "goal_wer_pct": 28.0,
            "stretch_wer_pct": 22.0,
            "baseline_greedy_wer_pct": 31.49,
            "note": (
                "For Twi, promote only if new checkpoint beats v6 (31.49% greedy / "
                "30.44% beam5, full Waxal test n=1522) on same split + decode. "
                "For English, route production only if no material regression vs English baseline."
            ),
        },
    }
    model_slug = (
        "ckpt_" + checkpoint_dir.rstrip("/").split("/")[-1]
        if checkpoint_dir
        else model_id.replace("/", "_")
    )
    dataset_slug = (
        "local_recorder_corpus"
        if local_manifest_path
        else f"{dataset_name}_{dataset_config or 'default'}".replace("/", "_")
    )
    lang_slug = language or "auto"
    out_path = (
        f"/results/baseline_{model_slug}__{dataset_slug}__{result['split']}"
        f"_{lang_slug}"
        f"_n{len(preds)}_beam{int(num_beams)}{'_streaming' if streaming and not local_manifest_path else ''}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(result)
    print(f"[eval] wrote {out_path}")
    return result


@app.local_entrypoint()
def main(
    model_id: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
    checkpoint_dir: str = "",
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    max_samples: int = 200,
    split: str = "test",
    audio_column: Optional[str] = None,
    text_column: Optional[str] = None,
    language: Optional[str] = None,
    num_beams: int = 1,
    streaming: bool = True,
    trust_remote_code: bool = False,
    local_manifest_path: str = "",
    wait: bool = True,
):
    call = evaluate_checkpoint.spawn(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_samples=max_samples,
        split=split,
        audio_column=audio_column,
        text_column=text_column,
        language=language,
        num_beams=num_beams,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
        local_manifest_path=local_manifest_path or None,
    )
    print(
        f"[eval] spawned {call.object_id} model={model_id} "
        f"dataset={local_manifest_path or f'{dataset_name}/{dataset_config}:{split}'} lang={language} "
        f"beams={num_beams} streaming={streaming}"
    )
    if not wait:
        return
    print(call.get())
