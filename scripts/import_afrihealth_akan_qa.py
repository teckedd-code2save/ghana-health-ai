#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow==21.0.0",
# ]
# ///
"""Import answered Akan rows from AfriHealth-QA with immutable provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tempfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "ImhotepSystems/AfriHealth-QA"
REVISION = "61befdaa19e12afdbf9032a602067f0daa3e6c68"
HF_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE_URL = (
    "https://zindi.africa/competitions/"
    "multilingual-health-question-answering-in-low-resource-african-languages-challenge/"
    "discussions/33230"
)
SPLITS = ("train", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "tmp" / "external" / "afrihealth-qa",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "medical-response-corpus" / "afrihealth-akan-source.v1.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data" / "medical-response-corpus" / "afrihealth-akan-source.v1.summary.json",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def duplicate_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE).replace("_", " ").strip()


def source_url(split: str) -> str:
    return f"{HF_URL}/resolve/{REVISION}/{split}.parquet"


def ensure_source_file(cache_dir: Path, split: str, no_download: bool) -> Path:
    path = cache_dir / f"{split}.parquet"
    if path.exists():
        return path
    if no_download:
        raise FileNotFoundError(f"Missing {path}; remove --no-download to fetch the pinned source.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source_url(split), timeout=120) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def length_summary(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values, default=0),
        "median": statistics.median(values) if values else 0,
        "p90": percentile(values, 0.9),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "question", "answer", "language_country"])
    rows: list[dict[str, Any]] = []
    for raw in table.to_pylist():
        if clean_text(raw.get("language_country")) != "Aka_Gha":
            continue
        source_id = clean_text(raw.get("id"))
        question = clean_text(raw.get("question"))
        answer = clean_text(raw.get("answer"))
        if not source_id or not question or not answer:
            continue
        identity = f"{DATASET_ID}:{REVISION}:{split}:{source_id}"
        rows.append(
            {
                "schema_version": 1,
                "id": f"afrihealth_akan_{sha256_text(identity)[:20]}",
                "source_dataset": DATASET_ID,
                "source_revision": REVISION,
                "source_url": HF_URL,
                "source_file": f"{split}.parquet",
                "source_split": split,
                "source_record_id": source_id,
                "source_language_country": "Aka_Gha",
                "language": "tw",
                "domain": "health",
                "question_twi_source": question,
                "answer_twi_source": answer,
                "question_source_hash": sha256_text(question),
                "answer_source_hash": sha256_text(answer),
                "record_source_hash": sha256_text(
                    json.dumps(
                        {"identity": identity, "question": question, "answer": answer},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "duplicate_key": sha256_text(duplicate_key(question)),
                "license": "cc-by-sa-4.0",
                "license_evidence_url": LICENSE_URL,
                "attribution": "HASH Consortium and the Zindi Africa Multilingual Health QA Challenge",
                "source_quality_status": "unverified_source",
                "annotation_status": "not_started",
                "eligible_for_training": False,
                "eligible_for_final_evaluation": False,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(path)


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for split in SPLITS:
        path = ensure_source_file(args.cache_dir, split, args.no_download)
        payload = path.read_bytes()
        rows = read_rows(path, split)
        all_rows.extend(rows)
        source_files.append(
            {
                "split": split,
                "path": str(path.relative_to(ROOT)),
                "url": source_url(split),
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "answered_akan_rows": len(rows),
            }
        )

    all_rows.sort(key=lambda row: (row["source_split"], row["source_record_id"]))
    ids = [row["id"] for row in all_rows]
    source_ids = [f'{row["source_split"]}:{row["source_record_id"]}' for row in all_rows]
    duplicate_counts = Counter(row["duplicate_key"] for row in all_rows)
    duplicate_rows = sum(count for count in duplicate_counts.values() if count > 1)
    split_counts = Counter(row["source_split"] for row in all_rows)
    question_lengths = [len(row["question_twi_source"]) for row in all_rows]
    answer_lengths = [len(row["answer_twi_source"]) for row in all_rows]

    if len(all_rows) != 5569:
        raise RuntimeError(f"Expected 5,569 answered Akan rows at the pinned revision, found {len(all_rows)}.")
    if len(ids) != len(set(ids)) or len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Stable source identifiers are not unique.")
    if duplicate_rows:
        raise RuntimeError(f"Found {duplicate_rows} rows in normalized duplicate-question groups.")

    write_jsonl(args.out, all_rows)
    summary = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "source_revision": REVISION,
        "source_url": HF_URL,
        "license": "cc-by-sa-4.0",
        "license_evidence_url": LICENSE_URL,
        "rows": len(all_rows),
        "rows_by_source_split": dict(sorted(split_counts.items())),
        "empty_question_rows": sum(not row["question_twi_source"] for row in all_rows),
        "empty_answer_rows": sum(not row["answer_twi_source"] for row in all_rows),
        "duplicate_question_rows": duplicate_rows,
        "unique_record_ids": len(set(ids)),
        "question_characters": length_summary(question_lengths),
        "answer_characters": length_summary(answer_lengths),
        "source_files": source_files,
        "output": str(args.out.relative_to(ROOT)),
        "training_gate": (
            "Source rows are not training-eligible until separate multi-model annotations pass "
            "agreement, faithfulness, safety, and held-out leakage checks."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
