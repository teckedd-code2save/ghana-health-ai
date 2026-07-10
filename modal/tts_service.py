"""
Ghana Health AI — real Akan/Twi TTS on Modal.

Model: facebook/mms-tts-aka (Meta MMS VITS for Akan)
Same stack as akan-speech-lab.

  modal deploy modal/tts_service.py
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import modal

APP_NAME = "ghana-health-tts"
DEFAULT_MODEL = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-aka")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-tts-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "fastapi[standard]==0.115.12",
        "numpy<2.3",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=180,
    volumes={"/models": model_volume},
)
class TtsEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        model_id = os.environ.get("TTS_MODEL_ID", DEFAULT_MODEL)
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = "/models/hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
        self.model = VitsModel.from_pretrained(model_id, cache_dir=cache).to(self.device)
        self.model.eval()
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", 16000))

    @modal.method()
    def synthesize(self, text: str, language: str | None = None) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf
        import torch

        started = time.time()
        clean = (text or "").strip()
        if not clean:
            return {
                "audio_base64": "",
                "sample_rate": self.sample_rate,
                "format": "wav",
                "latency_ms": 0,
                "model": self.model_id,
                "error": "empty_text",
            }

        # MMS-aka expects Akan orthography; keep text as-is (Twi code-mix ok-ish)
        inputs = self.tokenizer(clean, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs).waveform
        waveform = out.squeeze().detach().cpu().float().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        buf = io.BytesIO()
        sf.write(buf, waveform.astype(np.float32), self.sample_rate, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "audio_base64": b64,
            "sample_rate": self.sample_rate,
            "format": "wav",
            "duration": float(len(waveform) / self.sample_rate),
            "latency_ms": int((time.time() - started) * 1000),
            "model": self.model_id,
            "text": clean,
            "language": language or "tw",
        }


@app.function(image=image, timeout=60)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    web = FastAPI(title="Ghana Health TTS", version="1.0.1")
    engine = TtsEngine()

    @web.get("/health")
    async def health():
        return {
            "ok": True,
            "service": "ghana-health-tts",
            "model": os.environ.get("TTS_MODEL_ID", DEFAULT_MODEL),
            "engine": "mms-vits-aka",
        }

    @web.post("/speak")
    async def speak(request: Request):
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        text = (data.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text required"}, status_code=400)
        language = data.get("language") or "tw"
        return_bytes = bool(data.get("return_bytes"))
        result = await engine.synthesize.remote.aio(text, language=language)
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        if return_bytes and result.get("audio_base64"):
            raw = base64.b64decode(result["audio_base64"])
            return Response(
                content=raw,
                media_type="audio/wav",
                headers={
                    "X-Model": result.get("model", ""),
                    "X-Latency-Ms": str(result.get("latency_ms", 0)),
                },
            )
        return result

    return web
