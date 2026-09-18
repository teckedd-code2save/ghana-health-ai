#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow==21.0.0",
# ]
# ///
"""Build a source-faithful Twi/English Ghana health response corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from import_afrihealth_akan_qa import (
    DATASET_ID,
    HF_URL,
    LICENSE_URL,
    REVISION,
    ensure_source_file,
)


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {
    "Aka_Gha": {"language": "tw", "locale": "ak-GH"},
    "Eng_Gha": {"language": "en", "locale": "en-GH"},
}
EXPECTED = {
    ("train", "Aka_Gha"): 4455,
    ("train", "Eng_Gha"): 4443,
    ("validation", "Aka_Gha"): 1114,
    ("validation", "Eng_Gha"): 1104,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "tmp" / "external" / "afrihealth-qa",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "medical-response-corpus",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).replace("_", " ").split())


def sentence_repetition(value: str) -> float:
    sentences = [normalized(part) for part in re.split(r"(?<=[.!?])\s+|[;\n]+", value)]
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= 3]
    if len(sentences) < 3:
        return 0.0
    return (len(sentences) - len(set(sentences))) / len(sentences)


def ngram_repetition(value: str, size: int = 4) -> float:
    tokens = normalized(value).split()
    grams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    if len(grams) < 12:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def unbalanced(value: str, left: str, right: str) -> bool:
    return value.count(left) != value.count(right)


def quality_flags(question: str, answer: str, language: str) -> tuple[list[str], list[str]]:
    reject: list[str] = []
    review: list[str] = []
    question_words = normalized(question).split()
    answer_words = normalized(answer).split()
    combined = f"{question}\n{answer}"

    if len(question_words) < 2:
        reject.append("question_too_short")
    if len(answer_words) < 2:
        reject.append("answer_extremely_short")
    elif len(answer_words) < 5:
        review.append("answer_short_review")
    if len(question) > 1600:
        review.append("question_extreme_length")
    if len(answer) > 6000:
        review.append("answer_extreme_length")
    if normalized(question) == normalized(answer):
        reject.append("question_equals_answer")
    if "\ufffd" in question or "\ufffd" in answer:
        reject.append("unicode_replacement_character")
    if "<unk>" in combined.casefold():
        reject.append("placeholder_or_missing_value")
    if normalized(question) in {"nan", "null", "none"} or normalized(answer) in {"nan", "null", "none"}:
        reject.append("placeholder_or_missing_value")

    answer_sentence_repetition = sentence_repetition(answer)
    answer_ngram_repetition = ngram_repetition(answer)
    if answer_sentence_repetition >= 0.34 or answer_ngram_repetition >= 0.42:
        reject.append("severe_answer_repetition")
    elif answer_sentence_repetition > 0 or answer_ngram_repetition >= 0.22:
        review.append("possible_answer_repetition")

    if re.search(r"(?<![\d-])(?:911|999|111)(?![\d-])", combined):
        review.append("foreign_or_unverified_emergency_number")
    if re.search(r"\b(?:United States|U\.S\.|American CDC|NHS)\b", combined, re.IGNORECASE):
        review.append("non_ghana_care_context")
    if re.search(r"https?://|www\.|\b\S+@\S+\.\S+\b", combined, re.IGNORECASE):
        review.append("external_link_or_email")
    if any(unbalanced(combined, left, right) for left, right in (("(", ")"), ("[", "]"), ("{", "}"))):
        review.append("unbalanced_delimiter")
    if answer.count('"') % 2 or answer.count("“") != answer.count("”"):
        review.append("unbalanced_quotation")

    if language == "en" and re.search(r"\b(?:and|or|with|to|the|a|an|of|for|such as)\s*$", answer, re.IGNORECASE):
        review.append("possibly_truncated_answer")
    if language == "tw" and re.search(r"\b(?:ne|anaa|sɛ|na|de|fa)\s*$", answer, re.IGNORECASE):
        review.append("possibly_truncated_answer")
    return sorted(set(reject)), sorted(set(review))


def source_rows(cache_dir: Path, no_download: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed: Counter[tuple[str, str]] = Counter()
    for split in ("train", "validation"):
        source_path = ensure_source_file(cache_dir, split, no_download)
        table = pq.read_table(source_path, columns=["id", "question", "answer", "language_country"])
        for raw in table.to_pylist():
            language_country = text(raw.get("language_country"))
            if language_country not in LANGUAGES:
                continue
            source_id = text(raw.get("id"))
            question = text(raw.get("question"))
            answer = text(raw.get("answer"))
            if not source_id or not question or not answer:
                continue
            observed[(split, language_country)] += 1
            language = LANGUAGES[language_country]
            identity = f"{DATASET_ID}:{REVISION}:{split}:{source_id}"
            rows.append(
                {
                    "schema_version": 1,
                    "id": f"afrihealth_ghana_{digest(identity)[:20]}",
                    "source_dataset": DATASET_ID,
                    "source_revision": REVISION,
                    "source_url": HF_URL,
                    "source_split": split,
                    "source_record_id": source_id,
                    "source_language_country": language_country,
                    "language": language["language"],
                    "locale": language["locale"],
                    "domain": "health",
                    "task": "health_response_generation",
                    "question": question,
                    "answer": answer,
                    "question_source_hash": digest(question),
                    "answer_source_hash": digest(answer),
                    "record_source_hash": digest(
                        json.dumps(
                            {"identity": identity, "question": question, "answer": answer},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                    "license": "cc-by-sa-4.0",
                    "license_evidence_url": LICENSE_URL,
                    "attribution": "HASH Consortium and the Zindi Africa Multilingual Health QA Challenge",
                    "source_claimed_answer_quality": "clinical_consensus",
                    "project_human_review_status": "not_reviewed",
                }
            )
    if dict(observed) != EXPECTED:
        raise RuntimeError(f"Pinned source counts changed: expected {EXPECTED}, found {dict(observed)}")
    return rows


def annotate_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicate_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[(row["language"], normalized(row["question"]))].append(row)

    annotated: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["source_split"], item["language"], item["source_record_id"])):
        reject, review = quality_flags(row["question"], row["answer"], row["language"])
        group = duplicate_groups[(row["language"], normalized(row["question"]))]
        if len(group) > 1:
            canonical = min(item["source_record_id"] for item in group)
            if row["source_record_id"] != canonical:
                reject.append("duplicate_normalized_question")
            else:
                review.append("duplicate_question_group_canonical")
        reject = sorted(set(reject))
        review = sorted(set(review))
        status = "reject" if reject else "review" if review else "pass"
        research_eligible = row["source_split"] == "train" and status == "pass"
        annotated.append(
            {
                **row,
                "quality_annotation": {
                    "method": "deterministic-source-audit-v1",
                    "status": status,
                    "reject_reasons": reject,
                    "review_reasons": review,
                    "intent": None,
                    "safety_level": "unreviewed",
                },
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
                "eligible_for_research_training": research_eligible,
                "eligible_for_production_training": False,
                "eligible_for_final_evaluation": False,
            }
        )
    return annotated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = annotate_quality(source_rows(args.cache_dir, args.no_download))
    train = [row for row in rows if row["eligible_for_research_training"]]
    evaluation = [
        row
        for row in rows
        if row["source_split"] == "validation" and row["quality_annotation"]["status"] == "pass"
    ]
    review = [row for row in rows if row["quality_annotation"]["status"] != "pass"]
    outputs = {
        "train": args.out_dir / "afrihealth-ghana-response-train.v1.jsonl",
        "evaluation": args.out_dir / "afrihealth-ghana-response-eval.v1.jsonl",
        "review": args.out_dir / "afrihealth-ghana-response-review.v1.jsonl",
    }
    write_jsonl(outputs["train"], train)
    write_jsonl(outputs["evaluation"], evaluation)
    write_jsonl(outputs["review"], review)

    status_counts = Counter(row["quality_annotation"]["status"] for row in rows)
    language_counts = Counter(row["language"] for row in rows)
    train_language_counts = Counter(row["language"] for row in train)
    evaluation_language_counts = Counter(row["language"] for row in evaluation)
    reason_counts = Counter(
        reason
        for row in review
        for reason in (
            row["quality_annotation"]["reject_reasons"]
            + row["quality_annotation"]["review_reasons"]
        )
    )
    summary = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "source_revision": REVISION,
        "license": "cc-by-sa-4.0",
        "source_rows": len(rows),
        "source_rows_by_language": dict(sorted(language_counts.items())),
        "quality_status": dict(sorted(status_counts.items())),
        "research_train_rows": len(train),
        "research_train_rows_by_language": dict(sorted(train_language_counts.items())),
        "locked_evaluation_rows": len(evaluation),
        "locked_evaluation_rows_by_language": dict(sorted(evaluation_language_counts.items())),
        "review_or_reject_rows": len(review),
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "outputs": {key: str(value.relative_to(ROOT)) for key, value in outputs.items()},
        "training_policy": (
            "Only deterministic-pass source-train rows are eligible for research training. "
            "No row is production-eligible until semantic and medical review is complete."
        ),
    }
    summary_path = args.out_dir / "afrihealth-ghana-response-corpus.v1.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
