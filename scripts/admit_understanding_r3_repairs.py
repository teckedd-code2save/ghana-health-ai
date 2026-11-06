#!/usr/bin/env python3
"""Admit exact reviewed versions from a registered repair dataset, without editing it."""
import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_understanding_r3 import data, read_rows


def admit(dataset, review_path, out):
    data.require(not out.exists(), "Use a new immutable admission directory")
    manifest = data.read(dataset / "manifest.json")
    for name, proof in manifest["files"].items():
        data.require(data.sha(data.safe_path(dataset, name)) == proof["sha256"], "Registered artifact changed: " + name)
    review = data.read(review_path)
    data.require(review["dataset_manifest_sha256"] == data.sha(dataset / "manifest.json"), "Review refers to another dataset")
    data.require(review["reviewer"] and review["kind"] == "agent_review" and review["human_reviewed"] is False,
                 "Explicit agent-review identity required")
    candidates = read_rows(dataset / "review-candidates.jsonl")
    mappings = {r["dataset_record_id"]: r for r in read_rows(dataset / "r3-mapping.jsonl")}
    decisions = {r["id"]: r for r in review["records"]}
    data.require(len(decisions) == len(review["records"]) and set(decisions) == {r["id"] for r in candidates},
                 "Every candidate needs exactly one explicit decision")
    admitted, rejected = [], []
    for source in candidates:
        decision = decisions[source["id"]]
        checksum = data.digest(source)
        data.require(checksum == decision["record_sha256"] == mappings[source["id"]]["record_sha256"],
                     "Reviewed candidate changed")
        data.require(decision["decision"] in {"admit", "exclude"} and decision["note"].strip(), "Missing review decision/rationale")
        if decision["decision"] == "exclude":
            rejected.append({"id": source["id"], "source_id": source["source_id"], "reason": decision["note"]})
            continue
        data.require(source["review_candidate"] is True and source["review"]["status"] == "agent_checked"
                     and not source["review"]["findings"] and source["split"] == "train", "Unresolved candidate cannot be admitted")
        row = {**source, "training_eligible": True, "parent_review": decision,
            "admission": {"status": "admitted_private_research", "authority": review["reviewer"], "training_eligible": True},
            "original_ref": {**source["original_ref"], "registered_dataset": str(dataset.resolve()),
                "registered_manifest_sha256": data.sha(dataset / "manifest.json"),
                "registered_record_sha256": checksum, "parent_review_sha256": data.sha(review_path)},
            "human_reviewed": False, "production_eligible": False}
        admitted.append(row)
    data.require(admitted, "No admitted repair records")
    out.mkdir(parents=True)
    data.write_rows(out / "conversations.jsonl", admitted)
    data.write_rows(out / "excluded.jsonl", rejected)
    data.save(out / "parent-review.json", review)
    data.save(out / "report.json", {"candidate_records": len(candidates), "admitted_records": len(admitted),
        "excluded_records": len(rejected), "origin_counts": dict(Counter(r["origin"] for r in admitted)),
        "multiturn_histories": sum(len(r["messages"]) >= 4 for r in admitted),
        "native_human_reviewed": 0, "clinical_certification": False, "training_executed": False})
    result = {"dataset_id": manifest["dataset_id"], "dataset_version": str(manifest["dataset_version"]) + "-admission1",
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "registered_source_manifest_sha256": data.sha(dataset / "manifest.json"),
        "code_sha256": data.sha(Path(__file__)),
        "artifacts": {p.name: data.sha(p) for p in out.iterdir() if p.is_file()}}
    data.save(out / "manifest.json", result)
    data.save(out / "READY.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return data.read(out / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.dataset, args.review, args.out), indent=2))
