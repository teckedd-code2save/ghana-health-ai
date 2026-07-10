"""
Evaluate a Whisper ASR checkpoint on Waxal (or any HF speech dataset).

  modal run modal/train/eval_asr.py \\
    --model-id teckedd/whisper-small-waxal-round2-specaug-v1 \\
    --max-samples 200
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

app = modal.App("ghana-health-asr-eval")
vol = modal.Volume.from_name("ghana-health-asr-train", create_if_missing=True)

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

try:
    SECRETS = [modal.Secret.from_name("huggingface")]
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
    volumes={"/data": vol},
    secrets=SECRETS,
)
def evaluate_checkpoint(
    model_id: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    split: str = "test",
    max_samples: int = 500,
) -> dict[str, Any]:
    import torch
    import evaluate
    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from tqdm import tqdm

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, cache_dir=cache, token=token
    ).to(device)
    model.eval()
    model.config.forced_decoder_ids = None

    raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
    if split not in raw:
        split = list(raw.keys())[-1]
    ds = raw[split]
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    audio_col = "audio" if "audio" in ds.column_names else ds.column_names[0]
    text_col = next(
        (
            c
            for c in ("text", "sentence", "transcription", "transcript", "normalized_text")
            if c in ds.column_names
        ),
        None,
    )
    if text_col is None:
        raise RuntimeError(f"No text column: {ds.column_names}")

    ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))
    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    preds: list[str] = []
    refs: list[str] = []

    for row in tqdm(ds, desc=f"eval {model_id}"):
        audio = row[audio_col]
        inputs = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            ids = model.generate(input_features, max_new_tokens=225)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        preds.append(_normalize_text(hyp))
        refs.append(_normalize_text(row[text_col]))

    wer = wer_m.compute(predictions=preds, references=refs)
    cer = cer_m.compute(predictions=preds, references=refs)
    result = {
        "model_id": model_id,
        "dataset": f"{dataset_name}/{dataset_config}:{split}",
        "n": len(preds),
        "wer": wer,
        "cer": cer,
    }
    print(result)
    return result


@app.local_entrypoint()
def main(
    model_id: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
    max_samples: int = 200,
    split: str = "test",
):
    print(evaluate_checkpoint.remote(model_id=model_id, max_samples=max_samples, split=split))
