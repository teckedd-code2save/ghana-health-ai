"""
Fine-tune / adapt TTS for Twi health speech on Modal.

Default approach: continue from facebook/mms-tts-aka with a small
speech-text dataset (HF dataset with audio + text columns).

  modal run modal/train/train_tts.py --smoke
  modal run modal/train/train_tts.py \\
    --dataset your-org/twi-health-tts \\
    --max-steps 2000 \\
    --push-repo teckedd/gha-mms-tts-aka-health-v1

Data contract: each row has `audio` (16k+) and `text` (Twi, NFC, ɛ/ɔ preserved).
Promote only after blind A/B vs stock mms-tts-aka.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

app = modal.App("ghana-health-tts-train")
vol = modal.Volume.from_name("ghana-health-tts-train", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "huggingface_hub==0.26.2",
        "numpy<2.3",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/data": vol},
    secrets=SECRETS,
)
def train(
    base_model: str = "facebook/mms-tts-aka",
    dataset_name: str = "",
    max_steps: int = 1000,
    learning_rate: float = 5e-5,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Minimal VITS fine-tune loop. Requires a speech-text HF dataset.
    If dataset_name is empty, returns a readiness report only.
    """
    if not dataset_name:
        return {
            "status": "awaiting_data",
            "message": (
                "Publish a Twi speech-text dataset (audio+text), then re-run with --dataset. "
                "Do not fine-tune on empty data."
            ),
            "base_model": base_model,
            "recommended_hours": "2–10h clean single-speaker health reads first",
            "promote_via": "blind A/B vs facebook/mms-tts-aka",
        }

    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoTokenizer, VitsModel, Trainer, TrainingArguments

    if smoke:
        max_steps = min(max_steps, 20)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    out_dir = f"/data/tts_runs/{base_model.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(dataset_name, token=token, cache_dir=cache)
    split = "train" if "train" in ds else list(ds.keys())[0]
    train_ds = ds[split]
    if smoke:
        train_ds = train_ds.select(range(min(8, len(train_ds))))

    text_col = next(c for c in ("text", "sentence", "transcript") if c in train_ds.column_names)
    audio_col = "audio" if "audio" in train_ds.column_names else None
    if not audio_col:
        return {"status": "error", "message": f"No audio column in {train_ds.column_names}"}

    train_ds = train_ds.cast_column(audio_col, Audio(sampling_rate=16000))
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache, token=token)
    model = VitsModel.from_pretrained(base_model, cache_dir=cache, token=token)

    # VITS training via HF Trainer is non-trivial (waveform targets).
    # This entrypoint validates data + saves a "ready" tokenizer/model copy for
    # a dedicated training recipe; full waveform loss loop lands next iteration.
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    vol.commit()

    return {
        "status": "data_validated_checkpoint_copied",
        "message": (
            "Dataset load OK. Next: implement VITS waveform training or use "
            "an external recipe (e.g. coqui/XTTS) on this volume."
        ),
        "n_train": len(train_ds),
        "text_col": text_col,
        "output_dir": out_dir,
        "base_model": base_model,
        "push_repo": push_repo,
        "max_steps_requested": max_steps,
        "learning_rate": learning_rate,
    }


@app.local_entrypoint()
def main(
    dataset: str = "",
    smoke: bool = False,
    push_repo: str = "",
    max_steps: int = 1000,
):
    print(
        train.remote(
            dataset_name=dataset,
            smoke=smoke,
            push_repo=push_repo or None,
            max_steps=max_steps,
        )
    )
