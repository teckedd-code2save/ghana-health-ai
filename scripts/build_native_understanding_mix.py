"""Prepare a source-separated next candidate; this script never launches training."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/native-understanding-v5/corpus"


def read(folder, split):
    manifest = json.loads((folder / "manifest.json").read_text())
    raw = (folder / f"{split}.jsonl").read_bytes()
    record = next(r for r in manifest["artifacts"] if r["file"] == f"{split}.jsonl")
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Parent source checksum mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != record["rows"]:
        raise ValueError("Parent source count mismatch")
    return rows


def main():
    parent = ROOT / "tmp/semantic-response-v4/corpus"
    native = ROOT / "tmp/native-intent-v2"
    translation = ROOT / "tmp/general-language-corpus/adaptation-v1"
    manifest = json.loads((parent / "manifest.json").read_text())
    splits = {split: read(parent, split) for split in ("train", "validation")}
    for split in splits:
        for row in read(native, split):
            splits[split].append({**row, "group": "injongo:" + row["source_record_hash"]})
    reserved = {key: {row[key] for row in splits["validation"]} for key in ("id", "group", "source_record_hash")}
    candidates = [row for row in read(translation, "train") if row["task"] == "translation" and row["language"] == "en"
                  and not any(row[key] in reserved[key] for key in reserved)]
    chosen = sorted(candidates, key=lambda row: hashlib.sha256(("semantic-v5:" + row["id"]).encode()).hexdigest())[:1200]
    if len(chosen) != 1200:
        raise ValueError("Insufficient source-separated Twi-to-English supervision")
    splits["train"].extend({**row, "task": "auxiliary_translation"} for row in chosen)
    counts = Counter(f"{split}:{row['task']}:{row['language']}" for split, rows in splits.items() for row in rows)
    for key in reserved:
        if {row[key] for row in splits["train"]} & reserved[key]:
            raise ValueError("Cross-split source leakage: " + key)
    for split, rows in splits.items():
        if len(rows) != len({row["id"] for row in rows}):
            raise ValueError("Duplicate example IDs")
    OUT.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for split, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (OUT / f"{split}.jsonl").write_bytes(raw)
        artifacts.append({"file": f"{split}.jsonl", "rows": len(rows), "sha256": hashlib.sha256(raw).hexdigest()})
    manifest.update(experiment="native_understanding_v5", artifacts=artifacts, counts=dict(counts),
        parents=[{"folder": str(path.relative_to(ROOT)), "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()}
                 for path in (parent, native, translation)], training_started=False,
        unique_train_source_groups=len({row["group"] for row in splits["train"]}))
    manifest.pop("unique_sources", None)
    manifest["limitations"] += [
        "INJONGO adds native intent/entity annotations, not invented direct responses or completed tool actions.",
        "Twi-to-English examples use unchanged professional parallel-source targets, not newly generated meanings.",
        "Some translation sources appear in both directions; counts of task views are not independent speakers or conversations.",
        "AfriXNLI remains evaluation-only and is not included in this training mix.",
        "No claim that auxiliary task improvement guarantees natural Twi replies or clinical safety.",
    ]
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("response_core", ROOT / "modal/train/response_adaptation_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.load_inputs(OUT, "native_understanding_v5")
    print(json.dumps({"folder": str(OUT), "counts": counts, "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
