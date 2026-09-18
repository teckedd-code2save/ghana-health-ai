"""Build a source-bound correction queue from existing candidate outputs only."""
import argparse
import json
import sqlite3
import zlib
from pathlib import Path

from corpus_release import ROOT, sha, digest, save


def build(parent, runs, output):
    if len(runs) != 2 or output.exists():
        raise ValueError("Exactly two existing candidates and a fresh output directory required")
    seal = json.loads((parent / "SEALED.json").read_text())
    if sha(parent / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Changed parent release")
    candidates, artifacts = {}, {}
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        receipt = json.loads((run / "calls/qualification.receipt.json").read_text())
        if manifest["parent_manifest_sha256"] != seal["manifest_sha256"]:
            raise ValueError("Candidates must share a sealed source release")
        if sha(run / "calls/qualification.results.json") != receipt["result_sha256"]:
            raise ValueError("Changed candidate result")
        checks = json.loads((run / "qualification-checks.json").read_text())
        if sha(run / "qualification-review.json") != checks["review_sha256"] or checks["results_sha256"] != receipt["result_sha256"]:
            raise ValueError("Changed candidate inspection")
        artifacts[str(run)] = {"model": manifest["model"], "revision": manifest["revision"],
            "results_sha256": receipt["result_sha256"], "checks_sha256": sha(run / "qualification-checks.json")}
        for row in json.loads((run / "qualification-review.json").read_text()):
            if row["kind"] != "source_validation":
                continue
            value = row.get("analysis")
            if value is None:
                try:
                    value = json.loads(row["raw_output"]["text"])
                except (ValueError, KeyError, TypeError):
                    value = None
            record = candidates.setdefault(row["source_id"], {"source_hash": row["source_hash"],
                "reference": row["reference_english"], "alternatives": []})
            if (record["source_hash"], record["reference"]) != (row["source_hash"], row["reference_english"]):
                raise ValueError("Candidates analyzed different source versions")
            if not isinstance(value, dict):
                continue
            record["alternatives"].append({"model": manifest["model"], "revision": manifest["revision"],
                "text": value.get("record_function", "Invalid annotation"),
                "fields": {"analysis": value, "automatic_checks": row["flags"] or ["Passed format and evidence checks; semantic review still required"]}})
    rows = []
    with sqlite3.connect(f"file:{parent / 'ledger.sqlite3'}?mode=ro", uri=True) as db:
        for identity, value in candidates.items():
            source = db.execute("SELECT payload,split FROM records WHERE id=?", (identity,)).fetchone()
            if not source or source[1] != "validation":
                raise ValueError("Correction sample must remain source-validation data")
            raw = json.loads(zlib.decompress(source[0]))
            if (raw["record_hash"], raw.get("english")) != (value["source_hash"], value["reference"]):
                raise ValueError("Source changed since generation")
            rows.append({"id": identity, "source_hash": raw["record_hash"], "original_tw": raw["text"],
                "reference_en": value["reference"], "source": raw["source"], "language": raw["language"],
                "alternatives": value["alternatives"], "split": "validation", "training_eligible": False,
                "annotation_status": "Candidate annotations need review",
                "reference_analysis": {"id": digest([identity, value]), "analysis": None}})
    save(output / "qualification-review.json", rows)
    save(output / "manifest.json", {"parent_manifest_sha256": seal["manifest_sha256"],
        "rows": len(rows), "unique_sources": len(candidates), "synthetic_rows": 0,
        "all_validation": True, "training_eligible": 0, "candidate_runs": artifacts,
        "review_sha256": sha(output / "qualification-review.json"), "originals_changed": False,
        "qualification_passed": False, "new_human_reviews": 0})
    print(json.dumps({"source_samples": len(rows), "training_eligible": 0, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=ROOT / "tmp/corpus-releases/twi-stage-v1-20260911")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.run, args.out)
