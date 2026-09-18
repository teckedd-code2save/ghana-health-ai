"""Create loudness-matched listening copies; preserve the raw generated WAVs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

OUTPUTS = {"manifest.json": "listening.json", "cosy-manifest.json": "cosy-listening.json",
           "cosy-streaming-fp16.json": "cosy-streaming-listening.json",
           "cosy-tail-check.json": "cosy-tail-listening.json"}


def prepare(folder: Path, manifest: str = "manifest.json"):
    if manifest not in OUTPUTS:
        raise ValueError("Unknown comparison manifest")
    report = json.loads((folder / manifest).read_text())
    if manifest == "cosy-tail-check.json":
        report["results"] = [{**row, "variant": part["variant"], "runtime": part["runtime"],
                              "reference_sha256": part["reference_sha256"]}
                             for part in report["reports"] for row in part["results"]]
    for row in report["results"]:
        raw = folder / row["file"]
        if hashlib.sha256(raw.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Original audio checksum mismatch")
        samples, rate = sf.read(raw)
        rms = float(np.sqrt(np.mean(samples**2)))
        peak = float(np.abs(samples).max())
        if rms < 1e-5 or not np.isfinite(samples).all():
            raise ValueError("Silent or invalid sample")
        gain = min(0.08 / rms, 0.95 / peak)
        path = folder / f"listen-{row['file']}"
        sf.write(path, samples * gain, rate, subtype="PCM_16")
        row["listening_file"] = path.name
        row["listening_gain"] = gain
        row["listening_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest == "cosy-streaming-fp16.json":
            row["synthesis_voice"] = row["voice"]
            row["voice"] = "cosy-twi-streaming"
    report["listening_processing"] = "Constant gain only: target RMS 0.08, peak ceiling 0.95; no denoising, speed or pitch changes."
    output = OUTPUTS[manifest]
    (folder / output).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json", choices=list(OUTPUTS))
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1] / "tmp/twi-voice-comparison", args.manifest)
