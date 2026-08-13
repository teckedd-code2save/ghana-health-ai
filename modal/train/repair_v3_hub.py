"""Push complete Whisper checkpoint (model + processor) + model card to HF hub."""

from __future__ import annotations

import os

import modal

app = modal.App("gha-repair-v3")
hf_cache = modal.Volume.from_name("akan-speech-hf-cache")
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints")

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "huggingface_hub==0.26.2",
        "safetensors",
        "numpy<2.3",
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


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache, "/checkpoints": ckpt_vol},
    secrets=SECRETS,
)
def repair(
    local_dir: str = "/checkpoints/gha-asr/teckedd_whisper-small-waxal-round2-specaug-v1_steps1500",
    base_processor: str = "teckedd/whisper-small-waxal-round2-specaug-v1",
    repo: str = "teckedd/gha-whisper-small-twi-v3",
    base_model: str = "openai/whisper-small",
    summary: str = "Twi Whisper checkpoint repaired for Ghana Health AI (weights + processor).",
):
    import sys

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    print("local", local_dir, "exists", os.path.isdir(local_dir))
    print("files", os.listdir(local_dir) if os.path.isdir(local_dir) else None)

    model = WhisperForConditionalGeneration.from_pretrained(local_dir)
    processor = WhisperProcessor.from_pretrained(base_processor, token=token)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Also write processor into local dir for volume-based loads
    processor.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    ckpt_vol.commit()

    model.push_to_hub(repo, token=token)
    processor.push_to_hub(repo, token=token)

    sys.path.insert(0, "/root/gha_train")
    from model_card import write_and_push_model_card  # type: ignore

    write_and_push_model_card(
        repo,
        task="automatic-speech-recognition",
        language=["tw", "ak"],
        base_model=base_model,
        datasets=["google/WaxalNLP"],
        summary=summary,
        tags=["whisper", "asr", "speech-recognition"],
        pipeline_tag="automatic-speech-recognition",
        token=token,
    )
    return {"status": "ok", "repo": repo, "local_dir": local_dir, "card": True}


@app.local_entrypoint()
def main():
    print(repair.remote())
