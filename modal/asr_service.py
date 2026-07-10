"""
Ghana Health AI — real Twi/Akan ASR on Modal.

Model: teckedd/whisper-small-waxal-round2-specaug-v1 (Waxal Round 2)
Engine: HuggingFace WhisperForConditionalGeneration on GPU

Cost layout:
  - Slim CPU ASGI handles /health (no torch, no GPU)
  - One max GPU container runs inference only when /transcribe is called
  - Short scaledown so idle T4s die quickly

  modal deploy modal/asr_service.py
"""

# NOTE: Do not add `from __future__ import annotations` here.
# FastAPI cannot resolve postponed annotations like `UploadFile` (ForwardRef),
# which crash-loops the Modal ASGI container at import time.
import os
import tempfile
import time
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-asr"
DEFAULT_MODEL = os.environ.get(
    "MODEL_ID", "teckedd/whisper-small-waxal-round2-specaug-v1"
)
FALLBACK_MODEL = "openai/whisper-small"
# Cap utterance length so a single request can't pin the T4 forever
MAX_AUDIO_SECONDS = float(os.environ.get("ASR_MAX_AUDIO_SECONDS", "45"))

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-asr-models", create_if_missing=True)

# GPU worker only — heavy deps
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy<2.3",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "huggingface_hub==0.26.2",
    )
)

# Public API — no torch (health checks must not burn GPU)
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.12",
        "python-multipart==0.0.20",
    )
)


def _write_temp_audio(data: bytes, suffix: str = ".webm") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _guess_suffix(content_type: Optional[str], filename: Optional[str]) -> str:
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
    image=gpu_image,
    gpu="T4",
    timeout=180,
    # Die fast when idle — was 180s and kept dual containers warm
    scaledown_window=45,
    max_containers=1,
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
            self.processor = WhisperProcessor.from_pretrained(
                FALLBACK_MODEL, cache_dir=cache
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                FALLBACK_MODEL, cache_dir=cache
            ).to(self.device)
        self.model.eval()
        print(f"[asr] ready model={self.model_id} device={self.device}")

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
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

        # Hard size guard (~1.5 MB/s webm worst case × max seconds)
        max_bytes = int(MAX_AUDIO_SECONDS * 160_000)
        if len(audio_bytes) > max_bytes:
            return {
                "text": "",
                "language": language or "tw",
                "segments": [],
                "latency_ms": 0,
                "model": self.model_id,
                "error": f"audio_too_large_max_{int(MAX_AUDIO_SECONDS)}s",
            }

        path = _write_temp_audio(audio_bytes, suffix=suffix)
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            if len(audio) > int(MAX_AUDIO_SECONDS * 16000):
                audio = audio[: int(MAX_AUDIO_SECONDS * 16000)]

            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": 225,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if language == "en":
                gen_kwargs["forced_decoder_ids"] = (
                    self.processor.get_decoder_prompt_ids(
                        language="english", task="transcribe"
                    )
                )

            with torch.no_grad():
                pred_ids = self.model.generate(input_features, **gen_kwargs)
            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[
                0
            ].strip()
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


@app.function(
    image=web_image,
    timeout=200,
    scaledown_window=10,
    max_containers=2,
    cpu=0.25,
    memory=512,
)
@modal.asgi_app()
def api():
    """CPU-only gateway. GPU only spins when transcribe is called."""
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import JSONResponse

    web = FastAPI(title="Ghana Health ASR", version="1.1.0")
    engine = AsrEngine()

    @web.get("/health")
    async def health():
        # Cheap — does not touch GPU / load Whisper
        return {
            "ok": True,
            "service": "ghana-health-asr",
            "model": os.environ.get("MODEL_ID", DEFAULT_MODEL),
            "engine": "transformers-whisper",
            "gpu_scaledown_s": 45,
            "max_audio_s": MAX_AUDIO_SECONDS,
        }

    @web.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: Optional[str] = None,
    ):
        raw = await audio.read()
        if not raw:
            return JSONResponse({"error": "empty audio"}, status_code=400)
        suffix = _guess_suffix(audio.content_type, audio.filename)
        return await engine.transcribe.remote.aio(
            raw, language=language, suffix=suffix
        )

    @web.post("/transcribe/bytes")
    async def transcribe_bytes(request: Request, language: Optional[str] = None):
        raw = await request.body()
        if not raw:
            return JSONResponse({"error": "empty body"}, status_code=400)
        suffix = _guess_suffix(request.headers.get("content-type"), None)
        return await engine.transcribe.remote.aio(
            raw, language=language, suffix=suffix
        )

    return web
