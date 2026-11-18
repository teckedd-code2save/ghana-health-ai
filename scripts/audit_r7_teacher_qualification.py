#!/usr/bin/env python3
"""Bind inspected critical teacher findings to immutable qualification output.

This is an attributed agent review, not a model judge, native-language approval,
or automatic certification of unlisted controls. Never changes the run output.
"""
import argparse
import hashlib
import json
from pathlib import Path

from r7_teacher_controls import controls


FINDINGS = {
    "gemma31": {
        "translate-experiencer": ("My child's body has become cold.", "temperature_polarity", "Hot was changed to cold."),
        "translate-half": ("The book is the one I took.", "comparison_relation", "The owner-confirmed half-size comparison was lost."),
        "translate-quantity": ("I want five kilograms of tomatoes.", "quantity", "Two kilograms was changed to five."),
    },
    "qwen35": {
        "translate-experiencer": ("I gave it to him/her as a gift.", "event_and_experiencer", "A child's temperature was replaced with a gift event."),
        "translate-eye": ("My heart is my joy.", "body_part_and_state", "Eye pain was replaced with heart/joy."),
        "translate-half": ("The book is the one I used.", "comparison_relation", "The owner-confirmed half-size comparison was lost."),
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit(report):
    profile = report["profile"]
    expected = [{"id": r["id"], "messages": r["messages"]} for r in controls()]
    if report["status"] != "complete" or profile not in FINDINGS or report["requests"] != expected:
        raise ValueError("Review is only valid for the inspected complete control run")
    results = {r["id"]: r for r in report["results"]}
    if len(results) != len(report["results"]) or set(results) != {r["id"] for r in expected}:
        raise ValueError("Missing or duplicate qualification output")
    refs = {r["id"]: r for r in controls()}
    findings = []
    for identity, (observed, category, reason) in FINDINGS[profile].items():
        result = results[identity]
        if result["output"] != observed:
            raise ValueError("Inspected output changed: " + identity)
        findings.append({"id": identity, "category": category, "severity": "meaning_critical",
            "source": refs[identity]["messages"][-1]["content"],
            "reference": refs[identity]["reference"], "observed": observed, "reason": reason,
            "result_sha256": digest(result), "reviewer_type": "agent", "human_reviewed": False})
    return {"version": "r7-teacher-qualified-scope-review-v1", "run_id": report["run_id"],
        "profile": profile, "model": report["model"], "revision": report["revision"],
        "input_sha256": digest(report["requests"]), "results_sha256": digest(report["results"]),
        "complete_outputs": len(results), "findings": findings,
        "decision": "not_qualified_for_bulk_twi_translation_or_generation", "training_eligible": False,
        "human_reviewed": False, "clinical_reviewed": False,
        "unlisted_controls": "unadjudicated_not_implicitly_passed",
        "control_caveats": [
            "The category-selection control's akutu/ankaa distinction needs lexical adjudication; do not score it as three distinct fruit types.",
            "The system prompt mentions Twi and English even on English tool controls; those do not qualify English-only tool reviewing.",
            "This development check rejects this method, not every possible use of the checkpoint.",
            "An agent's finding is attributed evidence, not native-language or clinical certification."],
        "next_action": "Continue source-preserved exports; do not scale this translation method unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in (args.gemma, args.qwen):
        data = path.read_bytes()
        reports.append({**audit(json.loads(data)), "artifact": str(path.resolve()),
                        "artifact_sha256": hashlib.sha256(data).hexdigest()})
    args.out.mkdir(parents=True, exist_ok=False)
    target = args.out / "reviews.json"
    target.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reviews": str(target), "qualified": 0, "profiles": len(reports)}))


if __name__ == "__main__":
    main()
