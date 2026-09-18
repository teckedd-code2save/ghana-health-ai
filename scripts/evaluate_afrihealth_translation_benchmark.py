#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sacrebleu==2.5.1",
# ]
# ///
"""Score open Twi-English proposals against the stronger 62-row reference pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--translations",
        type=Path,
        default=ROOT / "tmp" / "understanding-corpus" / "afrihealth-translation-v3-62.raw.jsonl",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "medical-response-corpus" / "afrihealth-akan-annotations.v1.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "medical-response-corpus" / "afrihealth-translation-benchmark.v3.summary.json",
    )
    parser.add_argument("--expected-rows", type=int, default=62)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pearson(rows: list[tuple[float, float]]) -> float | None:
    mean_left = sum(left for left, _ in rows) / len(rows)
    mean_right = sum(right for _, right in rows) / len(rows)
    numerator = sum((left - mean_left) * (right - mean_right) for left, right in rows)
    denominator = math.sqrt(
        sum((left - mean_left) ** 2 for left, _ in rows)
        * sum((right - mean_right) ** 2 for _, right in rows)
    )
    return numerator / denominator if denominator else None


def main() -> None:
    args = parse_args()
    translations = {row["row_id"]: row for row in read_jsonl(args.translations)}
    annotations = {row["row_id"]: row for row in read_jsonl(args.annotations)}
    row_ids = sorted(set(translations) & set(annotations))
    if len(row_ids) != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} matched rows, found {len(row_ids)}.")

    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)
    metrics: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for candidate in ("aya", "comparison"):
        metrics[candidate] = {}
        calibration[candidate] = {}
        for field in ("question", "answer"):
            text_field = f"{field}_english"
            quality_field = f"{field}_quality_probability"
            hypotheses = [translations[row_id][candidate][text_field] for row_id in row_ids]
            references = [
                [annotations[row_id]["proposals"][index][text_field] for row_id in row_ids]
                for index in (0, 1)
            ]
            sentence_scores = [
                chrf.sentence_score(
                    translations[row_id][candidate][text_field],
                    [
                        annotations[row_id]["proposals"][0][text_field],
                        annotations[row_id]["proposals"][1][text_field],
                    ],
                ).score
                for row_id in row_ids
            ]
            metrics[candidate][text_field] = {
                "bleu": round(bleu.corpus_score(hypotheses, references).score, 2),
                "chrf_plus_plus": round(chrf.corpus_score(hypotheses, references).score, 2),
                "median_sentence_chrf_plus_plus": round(sorted(sentence_scores)[len(sentence_scores) // 2], 2),
                "rows_at_least_70": sum(score >= 70 for score in sentence_scores),
                "rows_below_50": sum(score < 50 for score in sentence_scores),
            }
            calibration_rows = [
                (float(translations[row_id][candidate][quality_field]), score)
                for row_id, score in zip(row_ids, sentence_scores, strict=True)
            ]
            correlation = pearson(calibration_rows)
            calibration[candidate][text_field] = {
                "quality_score_reference_correlation": round(correlation, 3) if correlation is not None else None,
                "mean_quality_probability": round(
                    sum(score for score, _ in calibration_rows) / len(calibration_rows),
                    3,
                ),
            }

    first = translations[row_ids[0]]
    summary = {
        "schema_version": 1,
        "benchmark_version": "afrihealth-translation-benchmark-v3",
        "rows": len(row_ids),
        "reference_kind": "two stronger model proposals; not human gold",
        "models": {
            "aya": {
                "id": first["aya"]["model"],
                "license": first["aya"]["model_license"],
                "usage_scope": first["aya"]["usage_scope"],
            },
            "comparison": {
                "id": first["comparison"]["model"],
                "license": first["comparison"]["model_license"],
                "usage_scope": first["comparison"]["usage_scope"],
            },
            "quality_estimator": {
                "id": "ghananlpcommunity/twi-eng-qe-e5",
                "license": "apache-2.0",
                "usage_scope": "benchmark_only",
            },
        },
        "metrics": metrics,
        "quality_estimator_calibration": calibration,
        "decision": {
            "aya": "reject_as_automatic_teacher",
            "comparison": "research_only_candidate_not_automatic_truth",
            "quality_estimator": "reject_as_automatic_gate",
            "corpus_scale_annotation": "blocked_pending_adaptation_or_stronger_teacher",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
