"""Conservative triage on top of English model votes, not gold acceptance."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/native-answer-synthesis-v1"
FLAGS = ROOT / "data/response-adaptation/source-review-findings.jsonl"


def question_issues(question, answer):
    issues = []
    if re.search(r"\b(?:he|she|him|her|his|they|their|these|this|that)\b", question, re.I):
        issues.append("context_dependent_question_needs_review")
    if re.search(r"\b(?:why|reasons?|causes?)\b", question, re.I) and re.search(r"\b(?:some|certain|various|different) reasons?\b", answer, re.I):
        issues.append("reason_not_actually_stated")
    if re.search(r"\bhow (?:many|much)\b", question, re.I) and re.search(r"\b(?:many|some|several|a number of|a lot of)\b", answer, re.I) and not re.search(r"\d", answer):
        issues.append("vague_quantity_needs_review")
    return issues


def gate(rows, flags):
    flagged = {r["source_record_hash"]: r for r in flags}
    output, reasons = [], Counter()
    for row in rows:
        candidates = []
        source_flag = flagged.get(row["source"]["source_record_hash"])
        for candidate in row["candidates"]:
            issues = question_issues(candidate["question_en"], row["answer_en"])
            if source_flag:
                issues.append("source_translation_flagged")
            if not candidate["english_match_approved"]:
                issues.append("english_reviewer_rejected")
            if candidate["duplicate_question"]:
                issues.append("duplicate_question")
            reasons.update(issues)
            candidates.append({**candidate, "triage_issues": issues, "review_candidate": not issues})
        output.append({**row, "candidates": candidates, "source_flag": source_flag,
                       "train_eligible": False, "human_reviewed": False})
    return output, {"rows": len(output), "sources_with_review_candidate": sum(any(c["review_candidate"] for c in r["candidates"]) for r in output),
        "issue_counts_per_candidate": dict(reasons), "human_reviewed": 0, "train_eligible": 0,
        "not_semantic_accuracy": True, "limitations": [
            "Rules quarantine potential problems; absence of a flag is not native-language correctness.",
            "English reviewer passed all 14 controls but approved unsupported explanations on real sources.",
            "Do not scale or train directly from the English approval count without source alignment and answerability checks."]}


def main():
    source_path = OUT / "review.json"
    rows = json.loads(source_path.read_text())
    flags = [json.loads(line) for line in FLAGS.read_text().splitlines()]
    output, summary = gate(rows, flags)
    summary.update(parent_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   flags_sha256=hashlib.sha256(FLAGS.read_bytes()).hexdigest())
    for name, value in (("gated-review.json", output), ("gated-review.summary.json", summary)):
        with (OUT / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
