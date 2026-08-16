"""
Ghana Health AI — DONDO (w2v-BERT CTC) ASR on Modal.

Model: teckedd/gha-dondo-w2v-bert-twi-v1
  (KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en + Waxal fine-tune, 800 steps)

Why this service exists:
  Best product-domain WER on record (32.66% on the 40-clip local corpus vs
  v6's 54.18%), while v6 still wins the Waxal benchmark. Deployed as a
  SEPARATE app so production (v6) is untouched; point MODAL_ASR_URL at this
  endpoint locally to A/B the app against the Whisper serving path.

Evidence (docs/asr-rnd-session-2026-08-15.md):
  Waxal test n=300 CTC greedy  WER 36.47%  CER 11.36%
  CV22 English n=100           WER 43.55%  (English stays on the EN route)
  Local corpus n=40            WER 32.66%  CER 10.78%  (health_twi 28.0%)

  modal deploy modal/dondo_asr_service.py
"""

# NOTE: Do not add `from __future__ import annotations` — breaks FastAPI UploadFile.
import os
import subprocess
import tempfile
import time
from typing import Any, Optional

import modal

APP_NAME = os.environ.get("ASR_APP_NAME", "ghana-health-asr-dondo")
DEFAULT_MODEL = os.environ.get("MODEL_ID", "teckedd/gha-dondo-w2v-bert-twi-v1")
FALLBACK_MODEL = "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en"
MAX_AUDIO_SECONDS = float(os.environ.get("ASR_MAX_AUDIO_SECONDS", "45"))
MIN_RMS = float(os.environ.get("ASR_MIN_RMS", "0.008"))
MIN_SECONDS = float(os.environ.get("ASR_MIN_SECONDS", "0.35"))
# DONDO language conditioning: prepend one-hot language id row to features.
DONDO_LANGUAGE_ID = int(os.environ.get("DONDO_LANGUAGE_ID", "2"))  # Asante Twi

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-asr-models", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"MODEL_ID": DEFAULT_MODEL, "ASR_SERVICE_NAME": APP_NAME})
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

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"MODEL_ID": DEFAULT_MODEL, "ASR_SERVICE_NAME": APP_NAME})
    .pip_install(
        "fastapi[standard]==0.115.12",
        "python-multipart==0.0.20",
    )
)


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


def _ffmpeg_to_wav16k(src_path: str) -> str:
    """Decode browser webm/opus (and other formats) to mono 16 kHz WAV via ffmpeg."""
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 44:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg_decode_failed: {(proc.stderr or proc.stdout or '')[-400:]}"
        )
    return out_path


def _add_language_prefix(features, lang_id: int):
    """DONDO conditioning: prepend a one-hot language-id row to acoustic features."""
    import torch

    if features.dim() == 3:
        features = features.squeeze(0)
    _, dim = features.shape
    lang_vec = torch.zeros(dim, dtype=features.dtype, device=features.device)
    lang_vec[lang_id % dim] = 1.0
    return torch.cat([lang_vec.unsqueeze(0), features], dim=0).unsqueeze(0)


@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=180,
    scaledown_window=45,
    max_containers=1,
    volumes={"/models": model_volume},
)
class DondoEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        cache = "/models/hf"

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
            self.model = AutoModelForCTC.from_pretrained(model_id, cache_dir=cache).to(
                self.device
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[asr-dondo] load {model_id} failed ({exc}); fallback {FALLBACK_MODEL}")
            self.model_id = FALLBACK_MODEL
            self.processor = AutoProcessor.from_pretrained(FALLBACK_MODEL, cache_dir=cache)
            self.model = AutoModelForCTC.from_pretrained(
                FALLBACK_MODEL, cache_dir=cache
            ).to(self.device)
        self.model.eval()
        print(f"[asr-dondo] ready model={self.model_id} device={self.device}")

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        suffix: str = ".webm",
    ) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf
        import torch

        started = time.time()
        empty = {
            "text": "",
            "language": language or "tw",
            "segments": [],
            "latency_ms": 0,
            "model": self.model_id,
        }
        if not audio_bytes or len(audio_bytes) < 200:
            return {**empty, "error": "empty_audio", "latency_ms": 0}

        max_bytes = int(MAX_AUDIO_SECONDS * 200_000)
        if len(audio_bytes) > max_bytes:
            return {
                **empty,
                "error": f"audio_too_large_max_{int(MAX_AUDIO_SECONDS)}s",
            }

        src_fd, src_path = tempfile.mkstemp(suffix=suffix or ".webm")
        wav_path: Optional[str] = None
        try:
            with os.fdopen(src_fd, "wb") as f:
                f.write(audio_bytes)

            try:
                wav_path = _ffmpeg_to_wav16k(src_path)
                audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
                if getattr(audio, "ndim", 1) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(
                        np.asarray(audio, dtype=np.float32),
                        orig_sr=sr,
                        target_sr=16000,
                    )
                else:
                    audio = np.asarray(audio, dtype=np.float32)
            except Exception as dec_exc:  # noqa: BLE001
                import librosa

                print(f"[asr-dondo] ffmpeg path failed ({dec_exc}); librosa fallback")
                audio, _ = librosa.load(src_path, sr=16000, mono=True)
                audio = np.asarray(audio, dtype=np.float32)

            if len(audio) > int(MAX_AUDIO_SECONDS * 16000):
                audio = audio[: int(MAX_AUDIO_SECONDS * 16000)]

            duration = float(len(audio) / 16000.0)
            rms = float(np.sqrt(np.mean(np.square(audio))) if len(audio) else 0.0)

            if duration < MIN_SECONDS:
                return {
                    **empty,
                    "duration": duration,
                    "rms": rms,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": "audio_too_short",
                }
            if rms < MIN_RMS:
                return {
                    **empty,
                    "duration": duration,
                    "rms": rms,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": "audio_too_quiet_or_silent",
                }

            proc = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            feats = getattr(proc, "input_features", None)
            if feats is None:
                feats = getattr(proc, "input_values")
            feats = _add_language_prefix(feats.to(self.device), DONDO_LANGUAGE_ID)

            with torch.no_grad():
                logits = self.model(input_features=feats).logits
            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids)[0].strip()

            if not text:
                return {
                    **empty,
                    "duration": duration,
                    "rms": rms,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": "asr_empty_decode",
                }

            return {
                "text": text,
                "language": "en" if language == "en" else "tw",
                "language_probability": 1.0,
                "duration": duration,
                "rms": rms,
                "segments": [{"start": 0.0, "end": round(duration, 2), "text": text}],
                "latency_ms": int((time.time() - started) * 1000),
                "model": self.model_id,
                "speaker": "Speaker 1 (User)",
                "verified": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                **empty,
                "latency_ms": int((time.time() - started) * 1000),
                "error": str(exc),
            }
        finally:
            for p in (src_path, wav_path):
                if p:
                    try:
                        os.unlink(p)
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
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import JSONResponse

    web = FastAPI(title="Ghana Health DONDO ASR", version="0.1.0")
    engine = DondoEngine()

    @web.get("/health")
    async def health():
        return {
            "ok": True,
            "service": os.environ.get("ASR_SERVICE_NAME", APP_NAME),
            "model": os.environ.get("MODEL_ID", DEFAULT_MODEL),
            "engine": "transformers-wav2vec2-bert-ctc",
            "dondo_language_id": DONDO_LANGUAGE_ID,
            "gpu_scaledown_s": 45,
            "max_audio_s": MAX_AUDIO_SECONDS,
            "min_rms": MIN_RMS,
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
        result = await engine.transcribe.remote.aio(
            raw, language=language, suffix=suffix
        )
        if result.get("error") and not result.get("text"):
            return JSONResponse(result, status_code=422)
        return result

    @web.post("/transcribe/bytes")
    async def transcribe_bytes(request: Request, language: Optional[str] = None):
        raw = await request.body()
        if not raw:
            return JSONResponse({"error": "empty body"}, status_code=400)
        suffix = _guess_suffix(request.headers.get("content-type"), None)
        result = await engine.transcribe.remote.aio(
            raw, language=language, suffix=suffix
        )
        if result.get("error") and not result.get("text"):
            return JSONResponse(result, status_code=422)
        return result

    return web
