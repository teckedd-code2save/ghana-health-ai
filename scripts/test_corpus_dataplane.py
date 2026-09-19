#!/usr/bin/env python3
"""Small contract test for corpus_dataplane.py.

Run after installing scripts/requirements-ml-systems.txt.
"""
import json
import tempfile
from pathlib import Path

from corpus_dataplane import (
    build_dataset,
    load_subset_jsonl,
    load_subset_parquet,
    metadata_scan_jsonl,
    metadata_scan_parquet,
)


def row(i, split, source):
    return {
        "id": f"r{i}",
        "split": split,
        "domain": "health",
        "language": "twi",
        "source": source,
        "source_record_id": f"s{i}",
        "consent_scope": "research",
        "training_lane": "medical_research_silver",
        "quality_tier": "paired_source_silver",
        "verification_status": "source_paired_unreviewed",
        "license_policy": "noncommercial_research_only",
        "original_text": f"sample twi text {i}",
        "normalized_twi": f"sample twi text {i}",
        "natural_english": f"sample english text {i}",
        "literal_english": f"literal {i}",
        "intent": "health_symptom_report",
        "entities": {"index": i},
        "ambiguities": "",
        "requires_clarification": False,
        "label_source": "source_paired_english",
        "messages": [{"role": "user", "content": f"sample twi text {i}"}],
    }


def main():
    with tempfile.TemporaryDirectory(prefix="ghana-dataplane-test-") as folder:
        root = Path(folder)
        source = root / "source.jsonl"
        rows = [
            row(1, "train", "a"),
            row(2, "train", "a"),
            row(3, "dev", "a"),
            row(4, "train", "b"),
        ]
        source.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
        dataset = root / "dataset"
        manifest = build_dataset(source, dataset, batch_size=2)
        assert manifest["rows"] == 4
        assert manifest["splits"] == {"dev": 1, "train": 3}
        assert manifest["sources"] == {"a": 3, "b": 1}

        j = metadata_scan_jsonl(source, "train", "a")
        p = metadata_scan_parquet(dataset, "train", "a")
        assert j["rows"] == p["rows"] == 2
        assert j["result_sha256"] == p["result_sha256"]

        j = load_subset_jsonl(source, "train", "a")
        p = load_subset_parquet(dataset, "train", "a")
        assert j["rows"] == p["rows"] == 2
        assert j["result_sha256"] == p["result_sha256"]

        print(json.dumps({"ok": True, "build_id": manifest["build_id"]}, indent=2))


if __name__ == "__main__":
    main()
