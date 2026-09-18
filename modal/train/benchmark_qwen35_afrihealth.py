"""Benchmark direct Twi semantics and response generation with Qwen3.5.

The 62 reference annotations are dual-teacher research references, not human
gold. This run is a gate before Qwen3.5 is allowed to propose annotations for
the full AfriHealth corpus.

Smoke test:
  modal run modal/train/benchmark_qwen35_afrihealth.py --limit 4

Full reference run:
  modal run modal/train/benchmark_qwen35_afrihealth.py --limit 0
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import modal


app = modal.App("ghana-health-qwen35-afrihealth-benchmark")
hf_cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("ghana-health-understanding-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_SOURCE_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-akan-source.v1.jsonl",
)
_REFERENCE_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-akan-annotations.v1.jsonl",
)
_REMOTE_SOURCE_PATH = "/root/corpus/afrihealth-akan-source.v1.jsonl"
_REMOTE_REFERENCE_PATH = "/root/corpus/afrihealth-akan-annotations.v1.jsonl"
_PROMPT_VERSION = "afrihealth-qwen35-direct-v1"
_INTENTS = [
    "health_information_request",
    "health_symptom_report",
    "health_prevention_question",
    "health_testing_question",
    "health_treatment_question",
    "health_medication_question",
    "health_service_question",
    "health_personal_safety_question",
    "health_emergency_report",
    "health_followup",
    "unclear_or_out_of_scope",
]
_TOPICS = [
    "sexual_reproductive_health",
    "maternal_pregnancy",
    "child_adolescent_health",
    "infectious_disease",
    "mental_health",
    "violence_abuse_consent",
    "substance_use",
    "nutrition",
    "medication",
    "health_services",
    "general_health",
    "other",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.13",
    )
    .pip_install(
        "sacrebleu==2.5.1",
        "vllm==0.28.0",
    )
    .add_local_file(_SOURCE_PATH, _REMOTE_SOURCE_PATH)
    .add_local_file(_REFERENCE_PATH, _REMOTE_REFERENCE_PATH)
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _parse_json(value: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", value, re.IGNORECASE)
    cleaned = (fenced.group(1) if fenced else value).strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _selected_reference(annotation: dict[str, Any]) -> dict[str, Any]:
    proposals = list(annotation["proposals"])
    if annotation.get("synthesized_proposal"):
        proposals.append(annotation["synthesized_proposal"])
    return next(
        (
            proposal
            for proposal in proposals
            if proposal["proposal_id"] == annotation["recommended_proposal_id"]
        ),
        proposals[0],
    )


def _load_rows(limit: int) -> list[dict[str, Any]]:
    with open(_REMOTE_SOURCE_PATH, encoding="utf-8") as handle:
        sources = {row["id"]: row for row in map(json.loads, handle)}
    references: list[dict[str, Any]] = []
    with open(_REMOTE_REFERENCE_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            annotation = json.loads(line)
            source = sources.get(annotation["row_id"])
            if source is None:
                continue
            references.append(
                {
                    "source": source,
                    "reference": _selected_reference(annotation),
                    "reference_status": annotation["adjudication"]["status"],
                }
            )
    references.sort(key=lambda row: row["source"]["record_source_hash"])
    return references[:limit] if limit > 0 else references


def _response_shape() -> dict[str, Any]:
    return {
        "question_twi_normalized": "faithful normalized Twi question",
        "question_english": "complete natural English meaning of the question",
        "answer_english": "complete natural English meaning of the source answer",
        "intent": "one allowed intent",
        "topics": ["one or more allowed topics"],
        "entities": {
            "symptoms": [],
            "conditions": [],
            "medicines": [],
            "body_parts": [],
            "durations": [],
            "quantities": [],
            "persons": [],
            "locations": [],
            "other": [],
        },
        "safety_level": "routine|same_day|urgent|emergency|needs_review",
        "requires_clarification": False,
        "source_answer_assessment": (
            "supportable|needs_minor_edit|needs_expert_review|unsafe_or_incorrect"
        ),
        "source_answer_issues": [],
        "reply_twi": "short spoken Twi reply supported by the source answer",
        "reply_english": "exact English meaning of reply_twi",
        "evidence_spans_twi": ["one to eight exact source substrings"],
        "confidence": 0.0,
    }


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    source = row["source"]
    return [
        {
            "role": "system",
            "content": (
                "You annotate immutable Twi health question-answer evidence for a Ghanaian "
                "voice-assistant research corpus. Return exactly one valid JSON object and no "
                "commentary. Translate every meaning-bearing detail, including negation, tense, "
                "person, quantities, duration, uncertainty, and code-switching. Do not diagnose, "
                "invent claims, add medicine or dosage advice, or silently repair the source "
                "answer. Mark a questionable source answer for review. The Twi reply must be a "
                "short, natural compression of supportable source claims only. Copy evidence "
                "spans exactly from the source question or answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question_twi_source": source["question_twi_source"],
                    "answer_twi_source": source["answer_twi_source"],
                    "allowed_intents": _INTENTS,
                    "allowed_topics": _TOPICS,
                    "response_shape": _response_shape(),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _generate(
    model_id: str,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> list[str]:
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model_id,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=12288,
        max_num_seqs=max(1, batch_size),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = engine.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            _messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    requests = engine.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1800),
        use_tqdm=True,
    )
    return [request.outputs[0].text.strip() for request in requests]


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    return [_text(item) for item in value] if isinstance(value, list) else []


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(rows: list[dict[str, Any]], parsed: list[dict[str, Any] | None]) -> dict[str, Any]:
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    valid = [(row, prediction) for row, prediction in zip(rows, parsed, strict=True) if prediction]
    fields = ("question_english", "answer_english", "reply_twi", "reply_english")
    similarities: dict[str, list[float]] = {field: [] for field in fields}
    intent_matches: list[float] = []
    safety_matches: list[float] = []
    topic_jaccards: list[float] = []
    for row, prediction in valid:
        reference = row["reference"]
        for field in fields:
            similarities[field].append(
                float(chrf.sentence_score(_text(prediction.get(field)), [_text(reference[field])]).score)
            )
        intent_matches.append(float(_text(prediction.get("intent")) == reference["intent"]))
        safety_matches.append(float(_text(prediction.get("safety_level")) == reference["safety_level"]))
        predicted_topics = set(_string_list(prediction.get("topics")))
        reference_topics = set(reference["topics"])
        topic_jaccards.append(
            len(predicted_topics & reference_topics) / len(predicted_topics | reference_topics)
            if predicted_topics | reference_topics
            else 1.0
        )
    return {
        "rows": len(rows),
        "parsed": len(valid),
        "parse_rate": len(valid) / len(rows),
        "mean_chrf_pp": {field: _mean(values) for field, values in similarities.items()},
        "intent_accuracy": _mean(intent_matches),
        "safety_accuracy": _mean(safety_matches),
        "topic_jaccard": _mean(topic_jaccards),
    }


@app.function(
    image=image,
    gpu="H100:2",
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_volume},
    secrets=SECRETS,
)
def benchmark(
    limit: int = 4,
    batch_size: int = 1,
    model_id: str = "Qwen/Qwen3.5-9B",
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    rows = _load_rows(limit)
    if not rows:
        raise RuntimeError("No reference rows selected")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    info = HfApi(token=token).model_info(model_id)
    started = time.perf_counter()
    raw = _generate(model_id, rows, max(1, batch_size))
    elapsed = time.perf_counter() - started
    parsed = [_parse_json(value) for value in raw]
    metrics = _score(rows, parsed)

    predictions = []
    for row, raw_value, parsed_value in zip(rows, raw, parsed, strict=True):
        predictions.append(
            {
                "row_id": row["source"]["id"],
                "source": row["source"],
                "reference": row["reference"],
                "reference_status": row["reference_status"],
                "parsed": parsed_value,
                "raw_sha256": hashlib.sha256(raw_value.encode()).hexdigest(),
                "failure_excerpt": raw_value[:2400] if parsed_value is None else None,
            }
        )
    artifact = {
        "schema_version": 1,
        "prompt_version": _PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_status": "dual_teacher_not_human_gold",
        "model": {
            "id": model_id,
            "revision": info.sha,
            "declared_license": (info.card_data or {}).get("license", "missing"),
        },
        "metrics": {
            **metrics,
            "elapsed_seconds": round(elapsed, 3),
            "mean_latency_ms": round(elapsed * 1000 / len(rows), 2),
        },
        "predictions": predictions,
    }
    output_dir = "/results/response-annotations/qwen35-direct-v1"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/latest.json", "w", encoding="utf-8") as destination:
        json.dump(artifact, destination, ensure_ascii=False, indent=2)
    results_volume.commit()
    hf_cache.commit()
    return {
        "status": "complete",
        "output_path": f"{output_dir}/latest.json",
        "model": artifact["model"],
        "metrics": artifact["metrics"],
    }


@app.local_entrypoint()
def main(
    limit: int = 4,
    batch_size: int = 1,
    model_id: str = "Qwen/Qwen3.5-9B",
) -> None:
    print(
        json.dumps(
            benchmark.remote(limit=limit, batch_size=batch_size, model_id=model_id),
            ensure_ascii=False,
            indent=2,
        )
    )
