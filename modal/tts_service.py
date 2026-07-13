"""
Ghana Health AI — TTS on Modal.

- Twi/Akan: facebook/mms-tts-aka  (override with TTS_MODEL_ID)
- English: facebook/mms-tts-eng  (do NOT read English with the Akan model)

Focus: clearer Twi speech — number expansion, sentence chunking, jargon expansion.

  modal deploy modal/tts_service.py
"""

from __future__ import annotations

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
# VITS quality drops on very long single passes — chunk then concat
CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "160"))

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

# Cardinals for health speech (prices, weeks of pregnancy, doses)
_ONES_EN = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS_EN = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ONES_TW = [
    "ohi", "baako", "mmienu", "mmiɛnsa", "ɛnan", "enum", "nsia", "nson", "nwɔtwe", "nkron",
    "du", "dubaako", "dummienu", "dummiɛnsa", "dunuan", "dunum", "dunsia", "dunson",
    "dunwɔtwe", "dunkron",
]
_TENS_TW = [
    "", "", "aduonu", "aduasa", "aduanan", "aduonum", "aduosia", "aduoson", "aduowɔtwe", "aduokron",
]


def _num_en(n: int) -> str:
    if n < 0:
        return "minus " + _num_en(-n)
    if n < 20:
        return _ONES_EN[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS_EN[t] + (f" {_ONES_EN[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES_EN[h] + " hundred" + (f" {_num_en(r)}" if r else "")
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        return _num_en(th) + " thousand" + (f" {_num_en(r)}" if r else "")
    return str(n)


def _num_tw(n: int) -> str:
    """Approximate spoken Twi for small integers used in health/market speech."""
    if n < 0:
        return "minus " + _num_tw(-n)
    if n < 20:
        return _ONES_TW[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS_TW[t] + (f" {_ONES_TW[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        # "ɔha" / hundreds — keep simple for MMS
        head = f"{_ONES_TW[h]} ɔha" if h > 1 else "ɔha"
        return head + (f" {_num_tw(r)}" if r else "")
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        return f"{_num_tw(th)} apem" + (f" {_num_tw(r)}" if r else "")
    return str(n)


def _expand_number_token(token: str, language: str) -> str:
    """Expand 12, 3.5, 12.50, GH₵45 style tokens for clearer TTS."""
    tw = language == "tw"
    # Currency prefix
    m = re.match(r"^(?:GH₵|GHS|₵)\s*(\d+(?:\.\d{1,2})?)$", token, re.I)
    if m:
        return ("Ghana cedi " if not tw else "Ghana cedi ") + _expand_number_token(m.group(1), language)

    if re.fullmatch(r"\d+\.\d+", token):
        whole, frac = token.split(".", 1)
        w = int(whole)
        # money-style decimals → "point" or pesewas when 2 digits
        if len(frac) == 2 and frac.isdigit():
            spoken = _num_tw(w) if tw else _num_en(w)
            p = int(frac)
            if p == 0:
                return spoken
            pes = _num_tw(p) if tw else _num_en(p)
            return f"{spoken} point {pes}" if not tw else f"{spoken} point {pes}"
        digits = " ".join(
            (_num_tw(int(d)) if tw else _num_en(int(d))) for d in frac if d.isdigit()
        )
        base = _num_tw(w) if tw else _num_en(w)
        return f"{base} point {digits}"

    if re.fullmatch(r"\d+", token):
        n = int(token)
        if n > 999_999:
            return token
        return _num_tw(n) if tw else _num_en(n)

    return token


def _expand_numbers(text: str, language: str) -> str:
    # Standalone currency + numbers
    text = re.sub(
        r"(?:GH₵|GHS|₵)\s*(\d+(?:\.\d{1,2})?)",
        lambda m: _expand_number_token("GH₵" + m.group(1), language),
        text,
        flags=re.I,
    )
    parts: list[str] = []
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            parts.append(tok)
            continue
        # strip trailing punctuation for match, reattach
        core, trail = re.match(r"^(.*?)([.,;:!?]*)$", tok).groups()  # type: ignore[union-attr]
        if re.fullmatch(r"\d+(?:\.\d+)?", core):
            parts.append(_expand_number_token(core, language) + trail)
        else:
            parts.append(tok)
    return "".join(parts)


def _prepare_text(text: str, language: str) -> str:
    """Expand jargon/numbers and strip symbols so VITS reads more naturally."""
    clean = (text or "").strip()
    expansions = [
        (r"\bCHWs?\b", "community health workers" if language == "en" else "community health worker"),
        (r"\bANC\b", "antenatal care"),
        (r"\bOTC\b", "over the counter"),
        (r"\bMoMo\b", "mobile money"),
        (r"\bGHS\b", "Ghana health service" if language == "en" else "Ghana Health Service"),
        (r"\bWHO\b", "World Health Organization"),
        (r"\bmg\b", "milligrams"),
        (r"\bml\b", "milliliters"),
        (r"\bkg\b", "kilograms"),
    ]
    for pat, repl in expansions:
        clean = re.sub(pat, repl, clean, flags=re.IGNORECASE)

    clean = re.sub(r"[*_#`>~\[\]()]", " ", clean)
    clean = re.sub(r"https?://\S+", " ", clean)
    # Soft pauses: turn em/en dashes and bullets into commas (VITS-friendly)
    clean = re.sub(r"[–—•·]+", ", ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = _expand_numbers(clean, language)

    if len(clean) > MAX_CHARS:
        clean = clean[:MAX_CHARS].rsplit(" ", 1)[0] or clean[:MAX_CHARS]
    return clean


def _chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries so each VITS pass stays short and clear."""
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)

    # Hard-split any remaining oversize chunk on commas/spaces
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue
        words = c.split()
        part = ""
        for w in words:
            if not part:
                part = w
            elif len(part) + 1 + len(w) <= max_chars:
                part = f"{part} {w}"
            else:
                final.append(part)
                part = w
        if part:
            final.append(part)
    return final


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

    def _synth_one(self, clean: str, voice: dict[str, Any]):
        import numpy as np
        import torch

        tokenizer = voice["tokenizer"]
        model = voice["model"]
        inputs = tokenizer(clean, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).waveform
        waveform = out.squeeze().detach().cpu().float().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        return waveform.astype(np.float32)

    @modal.method()
    def synthesize(self, text: str, language: Optional[str] = None) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        started = time.time()
        lang = (language or "tw").lower()
        lang_key = "en" if lang.startswith("en") else "tw"

        clean = _prepare_text(text, lang_key)
        voice = self._pick_voice(lang_key)
        sample_rate = voice["sample_rate"]

        if not clean:
            return {
                "audio_base64": "",
                "sample_rate": sample_rate,
                "format": "wav",
                "latency_ms": 0,
                "model": voice["model_id"],
                "error": "empty_text",
            }

        chunks = _chunk_text(clean, CHUNK_CHARS)
        waves: list[Any] = []
        # ~120 ms silence between chunks for natural pacing
        gap = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            waves.append(self._synth_one(chunk, voice))
            if i < len(chunks) - 1:
                waves.append(gap)

        waveform = np.concatenate(waves) if waves else np.zeros(0, dtype=np.float32)
        # Soft peak normalize — avoids quiet or clipped MMS output
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1e-4:
            waveform = waveform * min(0.95 / peak, 1.5)

        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "audio_base64": b64,
            "sample_rate": sample_rate,
            "format": "wav",
            "duration": float(len(waveform) / sample_rate) if sample_rate else 0.0,
            "latency_ms": int((time.time() - started) * 1000),
            "model": voice["model_id"],
            "text": clean,
            "language": lang_key,
            "chunks": len(chunks),
        }


@app.function(image=web_image, timeout=30, scaledown_window=5, cpu=0.125, memory=256)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "ok": True,
        "service": "ghana-health-tts",
        "models": {"tw": AKA_MODEL, "en": ENG_MODEL},
        "engine": "mms-vits",
        "features": ["number_expansion", "sentence_chunking", "peak_normalize"],
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
