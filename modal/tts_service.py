"""
Ghana Health AI — TTS on Modal.

- Twi/Akan: facebook/mms-tts-aka
- English: facebook/mms-tts-eng  (do NOT read English with the Akan model)

  modal deploy modal/tts_service.py
"""

import base64
import io
import os
import re
import time
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-tts"
AKA_MODEL = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-aka")
ENG_MODEL = os.environ.get("TTS_ENG_MODEL_ID", "facebook/mms-tts-eng")
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "600"))

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-tts-models", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy<2.3",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.12")
)


def _prepare_text(text: str, language: str) -> str:
    """Expand jargon and strip symbols so VITS reads more naturally."""
    clean = (text or "").strip()
    # Expand common health jargon before speech
    expansions = [
        (r"\bCHWs?\b", "community health workers" if language == "en" else "community health worker"),
        (r"\bANC\b", "antenatal care"),
        (r"\bOTC\b", "over the counter"),
        (r"\bMoMo\b", "mobile money"),
        (r"\bGHS\b", "Ghana health service" if language == "en" else "Ghana Health Service"),
        (r"\bWHO\b", "World Health Organization"),
    ]
    for pat, repl in expansions:
        clean = re.sub(pat, repl, clean, flags=re.IGNORECASE)
    clean = re.sub(r"[*_#`>~\[\]()]", " ", clean)
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > MAX_CHARS:
        clean = clean[:MAX_CHARS].rsplit(" ", 1)[0] or clean[:MAX_CHARS]
    return clean


@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=120,
    scaledown_window=45,
    max_containers=1,
    volumes={"/models": model_volume},
)
class TtsEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = "/models/hf"
        self.voices: dict[str, dict[str, Any]] = {}

        for key, model_id in (("tw", AKA_MODEL), ("en", ENG_MODEL)):
            try:
                tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
                model = VitsModel.from_pretrained(model_id, cache_dir=cache).to(self.device)
                model.eval()
                sr = int(getattr(model.config, "sampling_rate", 16000))
                self.voices[key] = {
                    "tokenizer": tok,
                    "model": model,
                    "sample_rate": sr,
                    "model_id": model_id,
                }
                print(f"[tts] loaded {key}={model_id} device={self.device}")
            except Exception as exc:  # noqa: BLE001
                print(f"[tts] failed to load {key} ({model_id}): {exc}")

        if "tw" not in self.voices and "en" in self.voices:
            self.voices["tw"] = self.voices["en"]
        if not self.voices:
            raise RuntimeError("No TTS voices loaded")

    def _pick_voice(self, language: Optional[str]) -> dict[str, Any]:
        lang = (language or "tw").lower()
        if lang.startswith("en") and "en" in self.voices:
            return self.voices["en"]
        if "tw" in self.voices:
            return self.voices["tw"]
        return next(iter(self.voices.values()))

    @modal.method()
    def synthesize(self, text: str, language: Optional[str] = None) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf
        import torch

        started = time.time()
        lang = (language or "tw").lower()
        if lang.startswith("en"):
            lang_key = "en"
        else:
            lang_key = "tw"

        clean = _prepare_text(text, lang_key)
        voice = self._pick_voice(lang_key)

        if not clean:
            return {
                "audio_base64": "",
                "sample_rate": voice["sample_rate"],
                "format": "wav",
                "latency_ms": 0,
                "model": voice["model_id"],
                "error": "empty_text",
            }

        tokenizer = voice["tokenizer"]
        model = voice["model"]
        sample_rate = voice["sample_rate"]

        inputs = tokenizer(clean, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).waveform
        waveform = out.squeeze().detach().cpu().float().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        buf = io.BytesIO()
        sf.write(buf, waveform.astype(np.float32), sample_rate, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "audio_base64": b64,
            "sample_rate": sample_rate,
            "format": "wav",
            "duration": float(len(waveform) / sample_rate),
            "latency_ms": int((time.time() - started) * 1000),
            "model": voice["model_id"],
            "text": clean,
            "language": lang_key,
        }


@app.function(image=web_image, timeout=30, scaledown_window=5, cpu=0.125, memory=256)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "ok": True,
        "service": "ghana-health-tts",
        "models": {"tw": AKA_MODEL, "en": ENG_MODEL},
        "engine": "mms-vits",
        "gpu_scaledown_s": 45,
    }


@app.function(image=web_image, timeout=150, scaledown_window=10, cpu=0.25, memory=512)
@modal.fastapi_endpoint(method="POST")
def speak(item: dict):
    """
    POST JSON: { "text": "...", "language": "tw" | "en" }
    Picks Akan vs English MMS voice accordingly.
    """
    text = str(item.get("text") or "").strip()
    language = item.get("language") or "tw"
    if not text:
        return {"error": "text required", "audio_base64": ""}
    engine = TtsEngine()
    return engine.synthesize.remote(text, language=language)
