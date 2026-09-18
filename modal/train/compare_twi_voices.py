"""Matched-text listening samples on CPU; not a pronunciation quality score."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("ghana-twi-voice-comparison")
cache = modal.Volume.from_name("ghana-health-tts-stable-twi-models", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "espeak-ng", "libsndfile1", "ffmpeg",
).pip_install("huggingface_hub==0.26.2", "stable-twi-tts[twi]==0.2.1",
              "soundfile==0.13.1", "sherpa-onnx==1.13.7")
MODELS = {
    "nano": ("ghananlpcommunity/nano-twi", "427e8e37007544d47d5a9fd6b5376fdb2bf8094e"),
    "stable": ("ghananlpcommunity/stable-twi-tts", "b6d42942c79e9a3699d2fab85c1a592dfdfd7ca5"),
}
TEXTS = [
    ("greeting", "Akwaaba. Wo ho te sɛn nnɛ?"),
    ("question", "Me ho mfa me. Mepɛ sɛ mehu ayaresabea a ɛbɛn me."),
    ("shopping", "Mepɛ sɛ metɔ tomato kilogram abien. Fa brɛ me wɔ Adenta."),
]


@app.function(image=image, cpu=2, memory=6144, timeout=1200, max_containers=1,
              retries=0, volumes={"/models": cache})
def synthesize():
    import io
    import subprocess
    import tempfile
    import time

    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    from huggingface_hub import snapshot_download

    stable = snapshot_download(MODELS["stable"][0], revision=MODELS["stable"][1],
                               cache_dir="/models/comparison-hf", allow_patterns=["model.onnx", "config.json", "tokens.txt", "voices.json"])
    nano = Path(snapshot_download(MODELS["nano"][0], revision=MODELS["nano"][1],
                                  cache_dir="/models/comparison-hf", allow_patterns=["sherpa-onnx/*"])) / "sherpa-onnx"
    engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=str(nano / "twi_ep045_steps4.onnx"), vocoder=str(nano / "vocos-22khz-univ.onnx"),
            lexicon="", tokens=str(nano / "tokens.txt"), data_dir=str(nano / "espeak-ng-data"),
            noise_scale=0.667, length_scale=1.0,
        ), num_threads=2, provider="cpu"), max_num_sentences=1))
    results = []
    for identifier, text in TEXTS:
        for voice in ("stable-twi-1", "stable-twi-6", "nano-4step"):
            start = time.monotonic()
            if voice == "nano-4step":
                audio = engine.generate(text, sid=0, speed=1.0)
                stream = io.BytesIO()
                sf.write(stream, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
                wav = stream.getvalue()
                model, revision = MODELS["nano"]
            else:
                with tempfile.TemporaryDirectory() as folder:
                    output = Path(folder) / "speech.wav"
                    subprocess.run(["stable-twi-tts", "--model", stable, "--language", "twi",
                                    "--voice", voice.removeprefix("stable-"), "--text", text,
                                    "--out", str(output)], check=True, capture_output=True, timeout=120)
                    wav = output.read_bytes()
                model, revision = MODELS["stable"]
            samples, rate = sf.read(io.BytesIO(wav))
            if samples.size == 0 or not np.isfinite(samples).all():
                raise RuntimeError("Invalid audio")
            results.append({"id": identifier, "voice": voice, "model": model, "revision": revision,
                            "text": text, "wav": wav, "sample_rate": rate,
                            "duration": round(len(samples)/rate, 3),
                            "rms": float(np.sqrt(np.mean(samples**2))),
                            "peak": float(np.abs(samples).max()),
                            "seconds": round(time.monotonic()-start, 3)})
    cache.commit()
    return results


@app.local_entrypoint()
def main():
    import hashlib

    folder = Path(__file__).resolve().parents[2] / "tmp/twi-voice-comparison"
    folder.mkdir(parents=True, exist_ok=True)
    results = synthesize.remote()
    for row in results:
        wav = row.pop("wav")
        row["file"] = f"{row['id']}-{row['voice']}.wav"
        row["sha256"] = hashlib.sha256(wav).hexdigest()
        (folder / row["file"]).write_bytes(wav)
    report = {"created": datetime.now(timezone.utc).isoformat(), "results": results,
              "quality_status": "Unrated listening candidates. RMS and peak only validate signal integrity."}
    (folder / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
