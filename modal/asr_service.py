"""
Ghana Health AI — real Twi/Akan ASR on Modal.

Model: teckedd/whisper_small-waxal_akan-asr-v1 (from akan-speech-lab)
Engine: HuggingFace WhisperForConditionalGeneration on GPU

  modal deploy modal/asr_service.py
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import modal

APP_NAME = "ghana-health-asr"
DEFAULT_MODEL = os.environ.get("MODEL_ID", "teckedd/whisper_small-waxal_akan-asr-v1")
FALLBACK_MODEL = "openai/whisper-small"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-asr-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "fastapi[standard]==0.115.12",
        "python-multipart==0.0.20",
        "numpy<2.3",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "huggingface_hub==0.26.2",
    )
)


def _write_temp_audio(data: bytes, suffix: str = ".webm") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _guess_suffix(content_type: str | None, filename: str | None) -> str:
    name = (filename or "").lower()
    ct = (content_type or "").lower()
    if "wav" in name or "wav" in ct:
        return ".wav"
    if "mp3" in name or "mpeg" in ct:
        return ".mp3"
    if "ogg" in name or "ogg" in ct:
        return ".ogg"
    if "m4a" in name or "mp4" in ct:
        return ".m4a"
    return ".webm"


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=180,
    volumes={"/models": model_volume},
)
class AsrEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        cache = "/models/hf"

        try:
            self.processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id, cache_dir=cache
            ).to(self.device)
        except Exception as exc:  # noqa: BLE001
            print(f"[asr] load {model_id} failed ({exc}); fallback {FALLBACK_MODEL}")
            self.model_id = FALLBACK_MODEL
            self.processor = WhisperProcessor.from_pretrained(FALLBACK_MODEL, cache_dir=cache)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                FALLBACK_MODEL, cache_dir=cache
            ).to(self.device)
        self.model.eval()

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        suffix: str = ".webm",
    ) -> dict[str, Any]:
        import librosa
        import torch

        started = time.time()
        if not audio_bytes:
            return {
                "text": "",
                "language": language or "tw",
                "segments": [],
                "latency_ms": 0,
                "model": self.model_id,
                "error": "empty_audio",
            }

        path = _write_temp_audio(audio_bytes, suffix=suffix)
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            # Free decode for Twi fine-tunes (forced language often hurts Akan models)
            forced = None
            if language == "en":
                forced = self.processor.get_decoder_prompt_ids(
                    language="english", task="transcribe"
                )

            with torch.no_grad():
                pred_ids = self.model.generate(
                    input_features,
                    max_new_tokens=225,
                    forced_decoder_ids=forced,
                )
            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            duration = float(len(audio) / 16000.0)
            return {
                "text": text,
                "language": "en" if language == "en" else "tw",
                "language_probability": 1.0,
                "duration": duration,
                "segments": [{"start": 0.0, "end": round(duration, 2), "text": text}],
                "latency_ms": int((time.time() - started) * 1000),
                "model": self.model_id,
                "speaker": "Speaker 1 (User)",
                "verified": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "text": "",
                "language": language or "tw",
                "segments": [],
                "latency_ms": int((time.time() - started) * 1000),
                "model": self.model_id,
                "error": str(exc),
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.function(image=image, timeout=60)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import JSONResponse

    web = FastAPI(title="Ghana Health ASR", version="1.0.0")
    engine = AsrEngine()

    @web.get("/health")
    async def health():
        return {
            "ok": True,
            "service": "ghana-health-asr",
            "model": os.environ.get("MODEL_ID", DEFAULT_MODEL),
            "engine": "transformers-whisper",
        }

    @web.post("/transcribe")
    async def transcribe(
        audio: UploadFile | None = File(None),
        language: str | None = None,
    ):
        if audio is None:
            return JSONResponse({"error": "no audio"}, status_code=400)
        raw = await audio.read()
        if not raw:
            return JSONResponse({"error": "empty audio"}, status_code=400)
        suffix = _guess_suffix(audio.content_type, audio.filename)
        return await engine.transcribe.remote.aio(raw, language=language, suffix=suffix)

    @web.post("/transcribe/bytes")
    async def transcribe_bytes(request: Request, language: str | None = None):
        raw = await request.body()
        if not raw:
            return JSONResponse({"error": "empty body"}, status_code=400)
        suffix = _guess_suffix(request.headers.get("content-type"), None)
        return await engine.transcribe.remote.aio(raw, language=language, suffix=suffix)

    return web
