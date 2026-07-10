"""
Ghana Health AI — Modal ASR inference service.

Uses faster-whisper (CTranslate2) for low-latency transcription.
Default model: whisper-small (swap to fine-tuned Twi checkpoint via MODEL_ID).

Deploy:
  modal deploy modal/asr_service.py

Env / Modal secrets (optional):
  HF_TOKEN via secret "huggingface-secret"
  MODEL_ID  e.g. openai/whisper-small or teckedd/<twi-finetune>

Web endpoints:
  GET  /health
  POST /transcribe  multipart file field "audio" OR raw body (audio/*)
  WS   /ws/voice    binary audio frames → JSON transcripts
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from typing import Any

import modal

APP_NAME = "ghana-health-asr"
DEFAULT_MODEL = os.environ.get("MODEL_ID", "openai/whisper-small")

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("ghana-health-asr-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "faster-whisper==1.1.1",
        "fastapi[standard]==0.115.12",
        "python-multipart==0.0.20",
        "numpy<2.3",
        "soundfile==0.13.1",
        "websockets==14.2",
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
    if name.endswith(".wav") or "wav" in ct:
        return ".wav"
    if name.endswith(".mp3") or "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if name.endswith(".ogg") or "ogg" in ct:
        return ".ogg"
    if name.endswith(".m4a") or "mp4" in ct or "m4a" in ct:
        return ".m4a"
    return ".webm"


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=120,
    volumes={"/models": model_volume},
)
class AsrEngine:
    @modal.enter()
    def load(self) -> None:
        from faster_whisper import WhisperModel

        model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)
        # Prefer volume cache; download on first cold start
        self.model_id = model_id
        self.model = WhisperModel(
            model_id,
            device="cuda",
            compute_type="float16",
            download_root="/models",
        )
        self.ready = True

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        suffix: str = ".webm",
    ) -> dict[str, Any]:
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
            # language=None → auto-detect; "ak" not always in whisper langs → use "en" fallback with multilingual
            lang = language
            if lang in ("tw", "ak", "twi"):
                # Whisper has no dedicated Twi code; multilingual decode works better without force
                lang = None

            segments_iter, info = self.model.transcribe(
                path,
                language=lang,
                beam_size=5,
                vad_filter=True,
                word_timestamps=False,
            )
            segments = []
            texts: list[str] = []
            for seg in segments_iter:
                texts.append(seg.text.strip())
                segments.append(
                    {
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": seg.text.strip(),
                    }
                )
            text = " ".join(t for t in texts if t).strip()
            detected = getattr(info, "language", None) or language or "und"
            return {
                "text": text,
                "language": detected,
                "language_probability": float(getattr(info, "language_probability", 0) or 0),
                "duration": float(getattr(info, "duration", 0) or 0),
                "segments": segments,
                "latency_ms": int((time.time() - started) * 1000),
                "model": self.model_id,
                "speaker": "Speaker 1 (User)",
                "verified": None,
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.function(image=image, timeout=60)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    web = FastAPI(title="Ghana Health ASR", version="0.2.0")
    engine = AsrEngine()

    @web.get("/health")
    async def health():
        return {"ok": True, "service": "ghana-health-asr", "model": DEFAULT_MODEL}

    @web.post("/transcribe")
    async def transcribe(
        audio: UploadFile | None = File(None),
        language: str | None = None,
    ):
        raw = b""
        suffix = ".webm"
        if audio is not None:
            raw = await audio.read()
            suffix = _guess_suffix(audio.content_type, audio.filename)
        if not raw:
            return JSONResponse({"error": "no audio provided"}, status_code=400)

        result = await engine.transcribe.remote.aio(raw, language=language, suffix=suffix)
        return result

    @web.post("/transcribe/raw")
    async def transcribe_raw(request_body: bytes, language: str | None = None):
        # FastAPI will not bind raw body this way easily — use Request
        return JSONResponse({"error": "use multipart /transcribe"}, status_code=400)

    from fastapi import Request

    @web.post("/transcribe/bytes")
    async def transcribe_bytes(request: Request, language: str | None = None):
        raw = await request.body()
        ct = request.headers.get("content-type")
        suffix = _guess_suffix(ct, None)
        if not raw:
            return JSONResponse({"error": "empty body"}, status_code=400)
        result = await engine.transcribe.remote.aio(raw, language=language, suffix=suffix)
        return result

    @web.websocket("/ws/voice")
    async def ws_voice(websocket: WebSocket):
        await websocket.accept()
        buffer = bytearray()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    buffer.extend(message["bytes"])
                    # Transcribe when we have a reasonable chunk (~0.5s+ of compressed audio)
                    if len(buffer) >= 24_000:
                        chunk = bytes(buffer)
                        buffer.clear()
                        result = await engine.transcribe.remote.aio(chunk, language=None, suffix=".webm")
                        await websocket.send_json(result)
                elif "text" in message and message["text"]:
                    # control: {"action":"flush"} or {"action":"set_lang","language":"en"}
                    import json

                    try:
                        ctrl = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if ctrl.get("action") == "flush" and buffer:
                        chunk = bytes(buffer)
                        buffer.clear()
                        result = await engine.transcribe.remote.aio(chunk, language=ctrl.get("language"), suffix=".webm")
                        await websocket.send_json(result)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            try:
                await websocket.send_json({"error": str(exc)})
            except Exception:
                pass

    return web


@app.local_entrypoint()
def main(path: str = ""):
    """Quick local smoke: modal run modal/asr_service.py --path sample.wav"""
    if not path:
        print("Deploy with: modal deploy modal/asr_service.py")
        print("Or test: modal run modal/asr_service.py --path ./sample.wav")
        return
    data = open(path, "rb").read()
    eng = AsrEngine()
    print(eng.transcribe.remote(data, language=None, suffix=os.path.splitext(path)[1] or ".wav"))
