"""Summarize paired response artifacts without treating heuristics as quality scores."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_language_checkpoint import summarize


def report(folder):
    metadata = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    events = [{"type": "provenance", "comparison_scope": "Paired frozen base and adapter",
               "run_id": metadata["run_id"], "base_model": manifest["base_model"],
               "base_revision": manifest["base_revision"], "adapter_sha256": metadata["adapter_sha256"],
               "steps": metadata["steps"], "dataset_human_validated": manifest["human_validated"]}]
    hashes = {}
    for variant in ("base", "adapter"):
        data = (folder / (variant + ".json")).read_bytes()
        hashes[variant] = hashlib.sha256(data).hexdigest()
        rows = json.loads(data)
        if len(rows) != metadata["product_checks"] + metadata["heldout_source_checks"]:
            raise ValueError("Missing completed evaluation rows")
        events.extend({**row, "type": "prediction", "variant": variant} for row in rows)
    value = summarize(events)
    value["artifact_hashes"] = hashes
    value["interpretation"] = "Paired development and source-heldout diagnostic. Semantic review is required; no automatic promotion."
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = report(args.folder)
    (args.folder / "paired-diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
