"""
Ghana Health AI — stable-twi-tts candidate service on Modal.

This is intentionally separate from modal/tts_service.py so we can A/B a
Twi-native voice without disturbing the current MMS production voice.

Deploy:
  modal deploy modal/stable_twi_tts_service.py

Then set:
  TTS_TWI_PROVIDER=stable-twi
  STABLE_TWI_TTS_URL=https://...--ghana-health-tts-stable-twi-speak.modal.run
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ghana-health-tts-stable-twi"
MODEL_ID = os.environ.get("STABLE_TWI_TTS_MODEL_ID", "ghananlpcommunity/stable-twi-tts")
VOICE_TWI = os.environ.get("STABLE_TWI_TTS_VOICE", "twi-6")
VOICE_MIXED = os.environ.get("STABLE_TWI_TTS_MIXED_VOICE", "twi-1")
MAX_CHARS = int(os.environ.get("STABLE_TWI_TTS_MAX_CHARS", "500"))

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-tts-stable-twi-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1", "ffmpeg")
    .pip_install(
        "huggingface_hub==0.26.2",
        "fastapi[standard]==0.115.12",
        "stable-twi-tts[twi]==0.2.1",
        "soundfile==0.13.1",
    )
)


def _clean_text(text: str) -> str:
    clean = (text or "").replace("\n", " ").strip()
    clean = " ".join(clean.split())
    if len(clean) <= MAX_CHARS:
        return clean
    return clean[:MAX_CHARS].rsplit(" ", 1)[0] or clean[:MAX_CHARS]


def _language_mode(text: str, language: str | None) -> str:
    lang = (language or "tw").lower()
    if lang.startswith("en"):
        return "mixed"
    # Bracketed English spans are the model-card convention for code-switching.
    if "[" in text and "]" in text:
        return "mixed"
    ascii_words = [word for word in text.split() if word.isascii() and len(word) > 3]
    return "mixed" if len(ascii_words) >= 2 else "twi"


@app.cls(
    image=image,
    timeout=180,
    scaledown_window=45,
    cpu=2.0,
    memory=4096,
    volumes={"/models": model_volume},
)
class StableTwiTtsEngine:
    @modal.enter()
    def load(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir = snapshot_download(
            MODEL_ID,
            cache_dir="/models/hf",
            local_dir="/models/stable-twi-tts",
            local_dir_use_symlinks=False,
        )

    @modal.method()
    def synthesize(self, text: str, language: str | None = None, voice: str | None = None) -> dict[str, Any]:
        started = time.time()
        clean = _clean_text(text)
        if not clean:
            return {
                "audio_base64": "",
                "sample_rate": 22050,
                "format": "wav",
                "latency_ms": 0,
                "model": MODEL_ID,
                "provider": "stable-twi",
                "error": "empty_text",
            }

        mode = _language_mode(clean, language)
        picked_voice = voice or (VOICE_MIXED if mode == "mixed" else VOICE_TWI)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.wav"
            cmd = [
                "stable-twi-tts",
                "--model",
                self.model_dir,
                "--language",
                mode,
                "--voice",
                picked_voice,
                "--text",
                clean,
                "--out",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=150, check=False)
            if proc.returncode != 0:
                return {
                    "audio_base64": "",
                    "sample_rate": 22050,
                    "format": "wav",
                    "latency_ms": int((time.time() - started) * 1000),
                    "model": MODEL_ID,
                    "provider": "stable-twi",
                    "voice": picked_voice,
                    "language": mode,
                    "error": (proc.stderr or proc.stdout or "stable-twi-tts failed")[-500:],
                }

            audio = out_path.read_bytes()

        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "sample_rate": 22050,
            "format": "wav",
            "latency_ms": int((time.time() - started) * 1000),
            "model": MODEL_ID,
            "provider": "stable-twi",
            "voice": picked_voice,
            "language": mode,
            "text": clean,
        }


@app.function(image=image, timeout=30, scaledown_window=10, cpu=0.25, memory=512)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "provider": "stable-twi",
        "model": MODEL_ID,
        "voices": {"twi": VOICE_TWI, "mixed": VOICE_MIXED},
        "engine": "piper-vits-onnx",
    }


@app.function(image=image, timeout=180, scaledown_window=10, cpu=0.25, memory=512)
@modal.fastapi_endpoint(method="POST")
def speak(item: dict):
    text = str(item.get("text") or "").strip()
    language = item.get("language") or "tw"
    voice = item.get("voice")
    if not text:
        return {"error": "text required", "audio_base64": ""}
    engine = StableTwiTtsEngine()
    return engine.synthesize.remote(text, language=language, voice=voice)
