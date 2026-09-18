"""Small ASR diagnostic of generated voice samples, not native speech ratings."""
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import modal


def word_errors(reference, prediction):
    def words(text):
        text = unicodedata.normalize("NFC", text).lower()
        return "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()

    ref, pred = words(reference), words(prediction)
    previous = list(range(len(pred) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(pred, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(ref)


def main():
    folder = Path(__file__).resolve().parents[1] / "tmp/twi-voice-comparison"
    rows = []
    for filename in ("manifest.json", "cosy-manifest.json"):
        rows.extend(json.loads((folder / filename).read_text())["results"])
    if len(rows) != 12:
        raise ValueError("Expected exactly twelve comparison samples")
    path = folder / "asr-roundtrip.json"
    if path.exists():
        raise FileExistsError("Preserve the completed diagnostic")
    engine = modal.Cls.from_name("ghana-health-asr-dondo", "DondoEngine")()
    report = {"created": datetime.now(timezone.utc).isoformat(),
              "limitation": "Three fixed sentences per voice; ASR errors are confounded with synthesis errors. Not native MOS, clinical accuracy, or a production gate.",
              "results": []}
    for row in rows:
        wav = (folder / row["file"]).read_bytes()
        if hashlib.sha256(wav).hexdigest() != row["sha256"]:
            raise ValueError("Generated audio integrity mismatch")
        result = engine.transcribe.remote(wav, language="tw", suffix=".wav")
        if result.get("error"):
            raise RuntimeError(f"ASR failed: {result['error']}")
        errors, count = word_errors(row["text"], result["text"])
        item = {"id": row["id"], "voice": row["voice"], "reference": row["text"],
                "audio_sha256": row["sha256"], "recognition": result,
                "word_errors": errors, "reference_words": count, "wer": errors / count}
        report["results"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        path.with_suffix(".partial.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
