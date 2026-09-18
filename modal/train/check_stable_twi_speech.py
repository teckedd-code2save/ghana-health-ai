"""Exercise the actual candidate speech service privately, without deployment."""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("ghana-twi-speech-completeness-check")
cache = modal.Volume.from_name("ghana-health-tts-stable-twi-models", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "espeak-ng", "libsndfile1", "ffmpeg",
).pip_install("huggingface_hub==0.26.2", "fastapi[standard]==0.115.12",
              "stable-twi-tts[twi]==0.2.1", "soundfile==0.13.1")
if modal.is_local():
    image = image.add_local_file(str(Path(__file__).resolve().parents[1] / "stable_twi_tts_service.py"),
                                 "/root/stable_twi_tts_service.py")


@app.function(image=image, cpu=2, memory=6144, timeout=600, retries=0,
              max_containers=1, volumes={"/models": cache})
def check():
    import base64
    import io
    import sys
    from unittest.mock import patch

    import numpy as np
    import soundfile as sf

    sys.path.insert(0, "/root")
    import stable_twi_tts_service as service

    engine = service.StableTwiTtsEngine()
    # Repetition is deliberate ONLY for this transport/chunking stress test.
    # These strings are not training data or a voice-quality evaluation set.
    paragraph = "Akwaaba. Wo ho te sɛn nnɛ? Me ho mfa me. Mepɛ sɛ mehu ayaresabea a ɛbɛn me. "
    cases = [
        ("long-twi", paragraph * 8 + "Mepɛ sɛ metɔ tomato kilogram abien. Fa brɛ me wɔ Adenta."),
        ("mixed-span", "Mepɛ [computer science] nwoma."),
    ]
    results = []
    original_run = service.subprocess.run
    for identifier, text in cases:
        with patch.object(service.subprocess, "run", wraps=original_run) as calls:
            result = engine.synthesize.local(text, language="tw")
        if result.get("error"):
            raise RuntimeError(result["error"])
        commands = [c.args[0] for c in calls.call_args_list if isinstance(c.args[0], list)
                    and c.args[0][0] == "stable-twi-tts"]
        spoken_chunks = [c[c.index("--text") + 1] for c in commands]
        assert " ".join(spoken_chunks) == service._clean_text(text)
        assert len(spoken_chunks) == result["chunks"]
        wav = base64.b64decode(result.pop("audio_base64"), validate=True)
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32")
        assert audio.ndim == 1 and len(audio) > rate and np.isfinite(audio).all()
        assert rate == result["sample_rate"]
        results.append({"id": identifier, **result, "wav": wav,
                        "text_characters": len(text), "synthesized_chunks": spoken_chunks,
                        "rms": float(np.sqrt(np.mean(audio ** 2))),
                        "peak": float(np.abs(audio).max())})
    cache.commit()
    return results


@app.local_entrypoint()
def main():
    import hashlib

    folder = Path(__file__).resolve().parents[2] / "tmp/twi-voice-comparison"
    rows = check.remote()
    for row in rows:
        wav = row.pop("wav")
        row["file"] = f"completeness-{row['id']}.wav"
        row["sha256"] = hashlib.sha256(wav).hexdigest()
        (folder / row["file"]).write_bytes(wav)
    report = {"results": rows, "scope": "Real synthesis and complete chunk transport, not native pronunciation validation.",
              "production_deployed": False}
    (folder / "completeness-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
