"""
Ghana Health AI — Modal streaming voice skeleton (Phase 0/1).

Deploy when GPU credits are ready:
  modal deploy modal/parakeet_health_voice.py

This is a production-shaped stub: wiring for FastAPI WebSocket + class lifecycle.
Swap model loads for NVIDIA Parakeet multi-talker + Sortformer when NeMo is available
on the Modal image.
"""

from __future__ import annotations

import modal

app = modal.App("ghana-health-voice")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "websockets",
        "numpy",
        "pydantic>=2",
    )
)

volume = modal.Volume.from_name("ghana-health-models", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
class GhanaVoiceTranscriber:
    @modal.enter()
    def setup(self) -> None:
        # Placeholder: load Parakeet / Sortformer from /data/asr_models
        self.ready = True
        self.model_name = "stub-parakeet-v0"

    @modal.method()
    def transcribe_stream(self, audio_chunk: bytes, speaker_id: str | None = None) -> dict:
        # Production: feature extract → ASR + diarization → optional voice verify
        n = len(audio_chunk or b"")
        return {
            "text": "[modal-stub] Me ti yɛ me ya" if n else "",
            "speaker": f"Speaker ({speaker_id[:6]})" if speaker_id else "Speaker 1",
            "verified": None,
            "lang": "tw",
            "bytes": n,
            "model": self.model_name,
        }


@app.function(image=image)
@modal.asgi_app()
def voice_ws():
    from fastapi import FastAPI, WebSocket

    web_app = FastAPI(title="Ghana Health Voice")
    transcriber = GhanaVoiceTranscriber()

    @web_app.get("/health")
    async def health():
        return {"ok": True, "service": "ghana-health-voice"}

    @web_app.websocket("/ws/voice")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                audio = await websocket.receive_bytes()
                result = await transcriber.transcribe_stream.remote.aio(audio)
                await websocket.send_json(result)
        except Exception:
            await websocket.close()

    return web_app
