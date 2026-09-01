from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASET = "ghananlpcommunity/ghana-health-symptoms"
SOURCE_URL = "https://huggingface.co/datasets/ghananlpcommunity/ghana-health-symptoms"
PARQUET_URL = (
    "https://huggingface.co/datasets/ghananlpcommunity/ghana-health-symptoms/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)


def clean(value: object) -> str:
    return str(value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=7000)
    parser.add_argument(
        "--out",
        default="data/medical-response-corpus/ghana-health-symptoms.v0.jsonl",
    )
    args = parser.parse_args()

    df = pd.read_parquet(PARQUET_URL)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for idx, row in df.iterrows():
        symptom_twi = clean(row.get("symptom_twi"))
        tag_en = clean(row.get("tag_en"))
        if not symptom_twi or not tag_en:
            continue
        key = (symptom_twi.lower(), tag_en.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"ghana_health_symptoms_{clean(row.get('row_id')) or str(idx).zfill(5)}_{idx}",
                "row_index": int(idx),
                "text": symptom_twi,
                "symptom_twi": symptom_twi,
                "faithful_english_meaning": tag_en,
                "tag_en": tag_en,
                "intent": "health_symptom_report",
                "entities": {"body_system": clean(row.get("body_system")) or "unknown"},
                "body_system": clean(row.get("body_system")) or "unknown",
                "ipa_twi": clean(row.get("ipa_twi")),
                "source_twi": clean(row.get("source_twi")),
                "source_dataset": DATASET,
                "source_url": SOURCE_URL,
                "source_parquet": PARQUET_URL,
                "license": "cc-by-nc-4.0",
                "consent_scope": "dataset_license",
                "training_use": "noncommercial_research_only",
            }
        )
        if len(rows) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    print(
        json.dumps(
            {
                "out": str(out),
                "rows": len(rows),
                "dataset": DATASET,
                "license": "cc-by-nc-4.0",
                "dedupe": "symptom_twi+tag_en",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
