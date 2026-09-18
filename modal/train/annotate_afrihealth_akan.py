"""Generate open-model AfriHealth Akan annotation proposals on Modal.

Smoke test:
  modal run modal/train/annotate_afrihealth_akan.py --limit 4

Retrieve:
  modal volume get ghana-health-understanding-results \
    /response-annotations/v1/latest.raw.jsonl \
    tmp/understanding-corpus/afrihealth-modal-v1.raw.jsonl
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import modal


app = modal.App("ghana-health-afrihealth-annotation")
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
_REMOTE_SOURCE_PATH = "/root/corpus/afrihealth-akan-source.v1.jsonl"
_PROMPT_VERSION = "afrihealth-akan-response-open-v2"
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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "sentencepiece==0.2.0",
        "huggingface_hub==1.3.0",
    )
    .add_local_file(_SOURCE_PATH, _REMOTE_SOURCE_PATH)
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


def _load_rows(offset: int, limit: int) -> list[dict[str, Any]]:
    with open(_REMOTE_SOURCE_PATH, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row["record_source_hash"])
    start = max(0, int(offset))
    selected = rows[start:]
    return selected[:limit] if limit > 0 else selected


def _annotation_shape() -> dict[str, Any]:
    return {
        "question_twi_normalized": "source question with spelling/spacing normalization only",
        "question_english": "faithful natural English meaning",
        "answer_english": "faithful English meaning of the complete source answer",
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
        "reply_twi": "concise spoken Twi answer grounded only in the source answer",
        "reply_english": "faithful English meaning of reply_twi",
        "evidence_spans_twi": ["one to eight exact source substrings"],
        "confidence": 0.0,
    }


def _teacher_prompt(row: dict[str, Any], role: str) -> str:
    if role == "teacher_a":
        return (
            "You are a Twi-English corpus annotator for a Ghanaian medical voice assistant. "
            "Read the source question and source answer below, then create a real annotation. "
            "Return only one JSON object. Do not repeat these instructions or copy field "
            "descriptions as values. Preserve tense, negation, quantities, uncertainty, and "
            "code-switching. Do not add medical claims or silently repair a questionable answer. "
            "The JSON must contain these fields: question_twi_normalized, question_english, "
            "answer_english, intent, topics, entities, safety_level, requires_clarification, "
            "source_answer_assessment, source_answer_issues, reply_twi, reply_english, "
            "evidence_spans_twi, confidence. entities must contain arrays named symptoms, "
            "conditions, medicines, body_parts, durations, quantities, persons, locations, other. "
            f"intent must be one of: {', '.join(_INTENTS)}. "
            f"topics must use only: {', '.join(_TOPICS)}. "
            "safety_level must be routine, same_day, urgent, emergency, or needs_review. "
            "source_answer_assessment must be supportable, needs_minor_edit, "
            "needs_expert_review, or unsafe_or_incorrect. evidence_spans_twi must contain one "
            "to eight exact substrings copied from the source. The concise reply_twi may only "
            "compress claims already present in the source answer. confidence is from 0 to 1.\n\n"
            f"SOURCE QUESTION (TWI):\n{row['question_twi_source']}\n\n"
            f"SOURCE ANSWER (TWI):\n{row['answer_twi_source']}"
        )
    focus = (
        "Prioritize semantic structure, medical caution, and source-answer quality."
    )
    return (
        "You are an independent corpus annotator for a Ghanaian Twi medical voice assistant. "
        f"{focus} Return one valid JSON object only. Preserve exact meaning, tense, negation, "
        "quantities, uncertainty, and code-switching. Do not add medical claims and do not "
        "silently repair a questionable source answer. The short Twi reply may only compress "
        "claims already in the source answer and must never add an emergency phone number.\n\n"
        + json.dumps(
            {
                "prompt_version": _PROMPT_VERSION,
                "row_id": row["id"],
                "question_twi_source": row["question_twi_source"],
                "answer_twi_source": row["answer_twi_source"],
                "allowed_intents": _INTENTS,
                "allowed_topics": _TOPICS,
                "response_shape": _annotation_shape(),
            },
            ensure_ascii=False,
        )
    )


def _judge_prompt(
    row: dict[str, Any],
    teacher_a: dict[str, Any],
    teacher_b: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict adjudicator for a Ghanaian Twi medical training corpus. "
                "Return JSON only. Compare both proposals against the immutable source. "
                "Prefer needs_human_review over false confidence. Never add medical claims, "
                "diagnoses, doses, or emergency numbers."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt_version": _PROMPT_VERSION,
                    "row_id": row["id"],
                    "question_twi_source": row["question_twi_source"],
                    "answer_twi_source": row["answer_twi_source"],
                    "teacher_a": teacher_a,
                    "teacher_b": teacher_b,
                    "instruction": (
                        "Select teacher_a or teacher_b only when fully faithful and safe. "
                        "Use synthesized only to combine supported strengths without adding claims. "
                        "Use needs_human_review for material uncertainty."
                    ),
                    "response_shape": {
                        "selected_proposal": (
                            "teacher_a|teacher_b|synthesized|needs_human_review"
                        ),
                        "confidence": 0.0,
                        "rationale": "brief evidence-based reason",
                        "material_disagreement_fields": [],
                        "review_reasons": [],
                        "synthesized": _annotation_shape(),
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def _load_aya(model_id: str, token: str | None, cache: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def _generate_aya(model: Any, tokenizer: Any, prompts: list[str], batch_size: int) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=2200,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def _load_qwen(model_id: str, token: str | None, cache: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def _generate_qwen(
    model: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    batch_size: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(messages), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in messages[start : start + batch_size]
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=2400,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded.input_ids.shape[1] :]
        outputs.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return outputs


def _bundle(model: str, raw: str) -> dict[str, Any]:
    parsed = _parse_json(raw)
    return {
        "model": model,
        "parsed": parsed,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "failure_excerpt": raw[:2400] if parsed is None else None,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_volume},
    secrets=SECRETS,
)
def annotate(
    offset: int = 0,
    limit: int = 20,
    batch_size: int = 1,
    teacher_a_model: str = "CohereLabs/aya-101",
    teacher_b_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> dict[str, Any]:
    import torch

    rows = _load_rows(offset, limit)
    if not rows:
        raise RuntimeError("No source rows selected")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)

    aya, aya_tokenizer = _load_aya(teacher_a_model, token, cache)
    teacher_a_raw = _generate_aya(
        aya,
        aya_tokenizer,
        [_teacher_prompt(row, "teacher_a") for row in rows],
        max(1, batch_size),
    )
    teacher_a = [_bundle(teacher_a_model, value) for value in teacher_a_raw]
    del aya, aya_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    qwen, qwen_tokenizer = _load_qwen(teacher_b_model, token, cache)
    teacher_b_raw = _generate_qwen(
        qwen,
        qwen_tokenizer,
        [[{"role": "user", "content": _teacher_prompt(row, "teacher_b")}] for row in rows],
        max(1, batch_size),
    )
    teacher_b = [_bundle(teacher_b_model, value) for value in teacher_b_raw]
    judge_indexes = [
        index
        for index in range(len(rows))
        if teacher_a[index]["parsed"] is not None and teacher_b[index]["parsed"] is not None
    ]
    judge_raw = _generate_qwen(
        qwen,
        qwen_tokenizer,
        [
            _judge_prompt(rows[index], teacher_a[index]["parsed"], teacher_b[index]["parsed"])
            for index in judge_indexes
        ],
        max(1, batch_size),
    )
    judge_by_index = {
        index: _bundle(teacher_b_model, value)
        for index, value in zip(judge_indexes, judge_raw, strict=True)
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                "schema_version": 1,
                "prompt_version": _PROMPT_VERSION,
                "generated_at": generated_at,
                "row_id": row["id"],
                "source_record_hash": row["record_source_hash"],
                "teacher_a": teacher_a[index],
                "teacher_b": teacher_b[index],
                "adjudicator": judge_by_index.get(index),
            }
        )

    output_dir = "/results/response-annotations/v1"
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"{output_dir}/open-{offset}-{len(rows)}-{run_id}.raw.jsonl"
    for target in (output_path, f"{output_dir}/latest.raw.jsonl"):
        with open(target, "w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "prompt_version": _PROMPT_VERSION,
        "created_at": generated_at,
        "offset": offset,
        "selected": len(rows),
        "teacher_a_model": teacher_a_model,
        "teacher_b_model": teacher_b_model,
        "teacher_a_parsed": sum(item["parsed"] is not None for item in teacher_a),
        "teacher_b_parsed": sum(item["parsed"] is not None for item in teacher_b),
        "adjudicator_parsed": sum(
            item is not None and item["parsed"] is not None for item in judge_by_index.values()
        ),
        "output_path": output_path,
    }
    with open(f"{output_dir}/latest.manifest.json", "w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
    results_volume.commit()
    hf_cache.commit()
    return manifest


@app.local_entrypoint()
def main(
    offset: int = 0,
    limit: int = 20,
    batch_size: int = 1,
    detach: bool = False,
    teacher_a_model: str = "CohereLabs/aya-101",
    teacher_b_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> None:
    kwargs = {
        "offset": offset,
        "limit": limit,
        "batch_size": batch_size,
        "teacher_a_model": teacher_a_model,
        "teacher_b_model": teacher_b_model,
    }
    if detach:
        call = annotate.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(annotate.remote(**kwargs), ensure_ascii=False, indent=2))
