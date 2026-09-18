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
import io
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ghana-health-tts-stable-twi"
MODEL_ID = os.environ.get("STABLE_TWI_TTS_MODEL_ID", "ghananlpcommunity/stable-twi-tts")
MODEL_REVISION = os.environ.get("STABLE_TWI_TTS_MODEL_REVISION", "b6d42942c79e9a3699d2fab85c1a592dfdfd7ca5")
VOICE_TWI = os.environ.get("STABLE_TWI_TTS_VOICE", "twi-6")
VOICE_MIXED = os.environ.get("STABLE_TWI_TTS_MIXED_VOICE", "twi-1")
MAX_CHARS = int(os.environ.get("STABLE_TWI_TTS_MAX_CHARS", "2000"))

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
    raise ValueError("text_too_long")


def _chunk_text(text: str, max_chars: int = 300) -> list[str]:
    # Keep bracketed English spans intact across synthesis boundaries.
    words = re.findall(r"(?:\[[^\]]*\]|\S)+", text)
    chunks, current = [], ""
    for word in words:
        if len(word) > max_chars:
            raise ValueError("speech_span_too_long")
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {word}".strip()
        if len(current) >= max_chars // 2 and re.search(r"[.!?]$", word):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _language_mode(text: str, language: str | None) -> str:
    lang = (language or "tw").lower()
    if lang.startswith("en"):
        return "mixed"
    # Bracketed English spans are the model-card convention for code-switching.
    if "[" in text and "]" in text:
        return "mixed"
    # ASCII spelling is common in Twi too; it is not evidence of English.
    return "twi"


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
            revision=MODEL_REVISION,
            cache_dir="/models/hf",
            allow_patterns=["model.onnx", "config.json", "tokens.txt", "voices.json"],
        )

    @modal.method()
    def synthesize(self, text: str, language: str | None = None, voice: str | None = None) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

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
        chunks = _chunk_text(clean)
        waves, sample_rate = [], None

        with tempfile.TemporaryDirectory() as tmp:
            for index, chunk in enumerate(chunks):
                out_path = Path(tmp) / f"speech-{index}.wav"
                cmd = ["stable-twi-tts", "--model", self.model_dir, "--language", mode,
                       "--voice", picked_voice, "--text", chunk, "--out", str(out_path)]
                remaining = 150 - (time.time() - started)
                if remaining <= 0:
                    raise TimeoutError("speech_synthesis_timeout")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining, check=False)
                if proc.returncode != 0:
                    return {
                        "audio_base64": "", "sample_rate": sample_rate or 22050,
                        "format": "wav", "latency_ms": int((time.time() - started) * 1000),
                        "model": MODEL_ID, "provider": "stable-twi", "voice": picked_voice,
                        "language": mode, "error": "speech_synthesis_failed",
                    }
                wave, rate = sf.read(out_path, dtype="float32")
                if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
                    raise ValueError("invalid_speech_audio")
                if sample_rate is not None and rate != sample_rate:
                    raise ValueError("speech_sample_rate_mismatch")
                sample_rate = rate
                if waves:
                    waves.append(np.zeros(round(rate * 0.08), dtype=np.float32))
                waves.append(wave)
            waveform = np.concatenate(waves)
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
            audio = audio_buffer.getvalue()

        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "sample_rate": sample_rate,
            "format": "wav",
            "latency_ms": int((time.time() - started) * 1000),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "provider": "stable-twi",
            "voice": picked_voice,
            "language": mode,
            "text": clean,
            "chunks": len(chunks),
            "duration": len(waveform) / sample_rate,
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
