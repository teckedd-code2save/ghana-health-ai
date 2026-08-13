"""
Backfill a proper model card on an existing HF repo (no retrain).

  modal run modal/train/push_model_card.py \\
    --repo teckedd/gha-whisper-small-twi-v6 \\
    --task automatic-speech-recognition \\
    --base-model openai/whisper-small \\
    --wer 0.3044 --cer 0.1062

Use this for historical pushes that shipped without a README.
"""

from __future__ import annotations

import os
from typing import Optional

import modal

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("ghana-health-model-card")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.26.2")
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


@app.function(image=image, timeout=120, secrets=SECRETS)
def push_card(
    repo: str,
    task: str = "automatic-speech-recognition",
    base_model: str = "openai/whisper-small",
    languages: str = "tw,ak",
    datasets: str = "google/WaxalNLP,fsicoli/common_voice_22_0",
    wer: Optional[float] = None,
    cer: Optional[float] = None,
    summary: str = "",
):
    import sys

    sys.path.insert(0, "/root/gha_train")
    from model_card import write_and_push_model_card  # type: ignore

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    metrics = {}
    if wer is not None:
        metrics["wer"] = float(wer)
    if cer is not None:
        metrics["cer"] = float(cer)

    lang = [x.strip() for x in languages.split(",") if x.strip()]
    ds = [x.strip() for x in datasets.split(",") if x.strip()]

    md = write_and_push_model_card(
        repo,
        task=task,
        language=lang or ["tw"],
        base_model=base_model,
        metrics=metrics or None,
        datasets=ds,
        summary=summary
        or f"Checkpoint for Ghana Health AI. Task={task}. Backfilled model card.",
        tags=["ghana-health-ai", "serendepify"],
        pipeline_tag=task,
        token=token,
    )
    return {"status": "ok", "repo": repo, "bytes": len(md)}


@app.local_entrypoint()
def main(
    repo: str = "teckedd/gha-whisper-small-twi-v6",
    task: str = "automatic-speech-recognition",
    base_model: str = "openai/whisper-small",
    languages: str = "tw,ak",
    datasets: str = "google/WaxalNLP,fsicoli/common_voice_22_0",
    summary: str = "",
    wer: float = 0.3044,
    cer: float = 0.1062,
):
    print(
        push_card.remote(
            repo=repo,
            task=task,
            base_model=base_model,
            languages=languages,
            datasets=datasets,
            wer=wer,
            cer=cer,
            summary=(
                summary
                or "Production Twi Whisper small (v6) for Ghana Health AI. "
                "30.44% WER beam=5 on full Waxal test."
            ),
        )
    )
