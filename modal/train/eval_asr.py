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
    },
    secrets=SECRETS,
)
def evaluate_checkpoint(
    model_id: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
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

    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, cache_dir=cache, token=token
    ).to(device)
    model.eval()
    model.config.forced_decoder_ids = None

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

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    preds: list[str] = []
    refs: list[str] = []

    for row in tqdm(sample_iter, desc=f"eval {model_id}"):
        audio = row[audio_col]
        inputs = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
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
        preds.append(_normalize_text(hyp))
        refs.append(_normalize_text(row[text_col]))

    wer = wer_m.compute(predictions=preds, references=refs)
    cer = cer_m.compute(predictions=preds, references=refs)
    result = {
        "model_id": model_id,
        "dataset": f"{dataset_name}/{dataset_config}:{split}",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "audio_column": audio_col,
        "text_column": text_col,
        "language": language,
        "n": len(preds),
        "num_beams": int(num_beams),
        "streaming": bool(streaming),
        "wer": float(wer),
        "cer": float(cer),
        "wer_pct": round(float(wer) * 100, 2),
        "cer_pct": round(float(cer) * 100, 2),
        "beat_this": {
            "goal_wer_pct": 28.0,
            "stretch_wer_pct": 22.0,
            "baseline_greedy_wer_pct": 32.83,
            "note": (
                "For Twi, promote only if new checkpoint WER < Round 2 on same split + decode. "
                "For English, route production only if no material regression vs English baseline."
            ),
        },
    }
    dataset_slug = f"{dataset_name}_{dataset_config or 'default'}".replace("/", "_")
    lang_slug = language or "auto"
    out_path = (
        f"/results/baseline_{model_id.replace('/', '_')}__{dataset_slug}__{split}"
        f"_{lang_slug}"
        f"_n{len(preds)}_beam{int(num_beams)}{'_streaming' if streaming else ''}.json"
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
    wait: bool = True,
):
    call = evaluate_checkpoint.spawn(
        model_id=model_id,
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
    )
    print(
        f"[eval] spawned {call.object_id} model={model_id} "
        f"dataset={dataset_name}/{dataset_config}:{split} lang={language} "
        f"beams={num_beams} streaming={streaming}"
    )
    if not wait:
        return
    print(call.get())
