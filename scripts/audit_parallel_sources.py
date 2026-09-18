"""Download pinned public parallel sources and preserve a reproducible local audit.

No model calls, uploads, transcript modifications, or training acceptance implied.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp/general-language-corpus"
SOURCES = [
    {"name": "ghana_nlp_parallel", "repo": "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT",
     "revision": "5f59d16167c9432a6fa0dac5e7e6a7e48e161bdb",
     "file": "Public - Twi[En-Twi]_70.csv",
     "terms": "Card permits attributed noncommercial research and share-alike adaptation; license field N/A. Not commercial clearance."},
    {"name": "community_parallel", "repo": "ghananlpcommunity/twi-english-parallel-text",
     "revision": "b9ae4bc01ff2324ca6d73d1f86b38bb4f7f224cf",
     "file": "data/train-00000-of-00001.parquet",
     "terms": "cc-by-nc-4.0; Bible and dictionary/archive sources; retain source attribution."},
    {"name": "oasst_train", "repo": "OpenAssistant/oasst1",
     "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "file": "data/train-00000-of-00001-b42a775f407cee45.parquet",
     "terms": "apache-2.0; OpenAssistant contributors; preserve original conversation-tree split."},
    {"name": "oasst_validation", "repo": "OpenAssistant/oasst1",
     "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "file": "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
     "terms": "apache-2.0; evaluation only; OpenAssistant contributors."},
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    reports = []
    for source in SOURCES:
        folder = OUTPUT / source["name"]
        folder.mkdir(exist_ok=True)
        hashes = {}
        for filename in (source["file"], "README.md"):
            target = folder / Path(filename).name
            if not target.exists():
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{requests.utils.quote(filename, safe='/')}"
                response = session.get(url, timeout=120)
                response.raise_for_status()
                target.write_bytes(response.content)
            hashes[filename] = hashlib.sha256(target.read_bytes()).hexdigest()
        target = folder / Path(source["file"]).name
        if target.suffix == ".csv":
            rows = list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8-sig"))))
        else:
            rows = pq.read_table(target).to_pylist()
        rng = random.Random(42)
        report = {**source, "hashes": hashes, "rows": len(rows),
                  "fields": list(rows[0]), "sample": rng.sample(rows, min(8 if source["name"].startswith("oasst") else 24, len(rows))),
                  "source_distribution": dict(Counter(r.get("source_files", "unspecified") for r in rows)),
                  "null_fields": dict(Counter(k for row in rows for k, value in row.items() if value is None or value == ""))}
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    (OUTPUT / "public-source-audit.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
