"""Paired development diagnostics, with unpaired source checks kept separate."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_language_checkpoint import summarize


def report(folder):
    hashes = {}

    def read(name):
        data = (folder / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    metadata, manifest, status = read("summary.json"), read("manifest.json"), read("status.json")
    if status.get("stage") != "completed" or status.get("run_id") != metadata["run_id"]:
        raise ValueError("The run is not completed")
    events = [{"type": "provenance", "comparison_scope": "Paired frozen base and adapter",
        "run_id": metadata["run_id"], "base_model": manifest["base_model"], "base_revision": manifest["base_revision"],
        "adapter_sha256": metadata["adapter_sha256"], "steps": metadata["steps"], "dataset_human_validated": False}]
    for variant in ("base", "adapter"):
        rows = read(variant + ".json")
        if len(rows) != 30:
            raise ValueError("Expected all 30 development cases per variant")
        events.extend({**row, "type": "prediction", "variant": variant} for row in rows)
    output = summarize(events)
    heldout = read("source-heldout.json")
    if len(heldout) != 24 or len({row["id"] for row in heldout}) != 24:
        raise ValueError("Expected 24 distinct source-heldout rows")
    development_ids = {event["id"] for event in events if event["type"] == "prediction"}
    if development_ids.intersection(row["id"] for row in heldout):
        raise ValueError("Development and source-heldout IDs overlap")
    output["unpaired_source_checks"] = {"count": len(heldout), "paired_with_base": False,
        "limit_reached_ids": [row["id"] for row in heldout if row["limit_reached"]],
        "empty_ids": [row["id"] for row in heldout if not row["prediction"].strip()]}
    output["artifact_hashes"] = hashes
    output["limitations"].extend([
        "Development cases guided checkpoint selection and are not an untouched test set.",
        "Source-heldout checks have no paired base generation; do not claim a comparative gain on them.",
        "Tool syntax in offline generation is not executed-tool success.",
        "Clinical correctness and native Twi quality still require review.",
    ])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = report(args.folder)
    (args.folder / "paired-diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
