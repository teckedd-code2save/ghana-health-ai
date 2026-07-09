"""
Offer 2 skeleton — Twi ASR fine-tune job on Modal (Whisper + LoRA path).

Usage (when ready):
  modal run modal/fine_tune_asr_twi.py

Point HF datasets at UGSpeechData / GhanaNLP Twi subsets.
"""

from __future__ import annotations

import modal

app = modal.App("ghana-health-asr-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)


@app.function(image=image, gpu="H100", timeout=3 * 60 * 60, secrets=[modal.Secret.from_name("huggingface-secret", required=False)])
def fine_tune(
    model_id: str = "openai/whisper-small",
    language: str = "tw",
    max_steps: int = 200,
) -> dict:
    """Stub entrypoint — replace body with PEFT LoRA training loop."""
    return {
        "status": "not_started",
        "message": "Wire dataset loaders + LoRA train loop before launch",
        "model_id": model_id,
        "language": language,
        "max_steps": max_steps,
        "target_wer": 0.25,
    }


@app.local_entrypoint()
def main():
    result = fine_tune.remote()
    print(result)
