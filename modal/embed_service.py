"""
Ghana Health AI — Twi text encoder on Modal (research path).

Default model: Ghana-NLP/abena-base-asante-twi-uncased
  ABENA = A BERT Now in Akan (Azunre / Ghana NLP)
  mBERT → JW300 Twi (Akuapem-heavy) → Asante Bible continued FT

Embeddings: mean-pool last_hidden_state, L2-normalized.
Used to retrieve Twi knowledge / products from noisy ASR transcripts —
NOT a hand-written semantic bank.

  modal deploy modal/embed_service.py

Env:
  EMBED_MODEL_ID  (default Ghana-NLP/abena-base-asante-twi-uncased)
"""

from __future__ import annotations

import os
import time
from typing import Any

import modal

APP_NAME = "ghana-health-embed"
MODEL_ID = os.environ.get(
    "EMBED_MODEL_ID",
    "Ghana-NLP/abena-base-asante-twi-uncased",
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("ghana-health-embed-models", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "huggingface_hub==0.26.2",
        "numpy<2.3",
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.12")
)


@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=120,
    scaledown_window=60,
    max_containers=1,
    volumes={"/models": vol},
)
class EmbedEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        cache = "/models/hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache)
        self.model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        print(f"[embed] ABENA path loaded {MODEL_ID} dim={self.dim} device={self.device}")

    @modal.method()
    def encode(self, texts: list[str], mode: str = "query") -> dict[str, Any]:
        """mode is accepted for API compat; ABENA does not use e5 prefixes."""
        import numpy as np
        import torch

        started = time.time()
        cleaned = [(t or "").strip() or " " for t in texts]
        batch = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
            hidden = out.last_hidden_state  # [B, T, H]
            mask = batch["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            mean = summed / counts
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)
        arr = mean.detach().cpu().float().numpy().astype(np.float32)
        return {
            "embeddings": arr.tolist(),
            "dim": int(arr.shape[1]),
            "model": MODEL_ID,
            "engine": "abena-mean-pool",
            "mode": mode,
            "n": len(cleaned),
            "latency_ms": int((time.time() - started) * 1000),
        }


@app.function(image=web_image, timeout=30, scaledown_window=5, cpu=0.125, memory=256)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "ok": True,
        "service": "ghana-health-embed",
        "model": MODEL_ID,
        "engine": "abena-mean-pool",
        "research": "Ghana-NLP ABENA Asante Twi BERT",
    }


@app.function(image=web_image, timeout=120, scaledown_window=10, cpu=0.25, memory=512)
@modal.fastapi_endpoint(method="POST")
def embed(item: dict):
    """
    POST JSON:
      { "texts": ["..."], "mode": "query" | "passage" }
      or { "text": "..." }
    """
    texts = item.get("texts")
    if not texts:
        one = item.get("text")
        texts = [str(one)] if one is not None else []
    if not isinstance(texts, list) or not texts:
        return {"error": "texts or text required", "embeddings": []}
    texts = [str(t) for t in texts][:64]
    mode = str(item.get("mode") or "query")
    engine = EmbedEngine()
    return engine.encode.remote(texts, mode=mode)
