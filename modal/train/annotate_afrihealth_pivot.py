"""Generate source-linked AfriHealth Twi response annotations on Modal.

The two proposal paths are deliberately different:

1. Qwen3.5 reads the Twi source directly and proposes full semantics/reply.
2. NLLB translates the immutable source to English, Qwen3.5 labels that
   translation, and the reply remains an exact concise excerpt of the original
   Twi source answer.

Qwen3.5 then adjudicates both proposals. Output remains research silver or
human-review material; this job never creates human gold.

Reference smoke test:
  modal run modal/train/annotate_afrihealth_pivot.py --reference-only --limit 4

Training-source shard:
  modal run --detach modal/train/annotate_afrihealth_pivot.py --offset 0 --limit 128
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


app = modal.App("ghana-health-afrihealth-pivot-annotation")
hf_cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ghana-health-understanding-vllm-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("ghana-health-understanding-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_SOURCE_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-akan-source.v1.jsonl",
)
_TRAIN_CORPUS_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-ghana-response-train.v1.jsonl",
)
_REFERENCE_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-akan-annotations.v1.jsonl",
)
_REMOTE_SOURCE_PATH = "/root/corpus/afrihealth-akan-source.v1.jsonl"
_REMOTE_TRAIN_CORPUS_PATH = "/root/corpus/afrihealth-ghana-response-train.v1.jsonl"
_REMOTE_REFERENCE_PATH = "/root/corpus/afrihealth-akan-annotations.v1.jsonl"
_PROMPT_VERSION = "afrihealth-akan-response-pivot-v1"
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
_RISK_TERMS = re.compile(
    r"(?:mogya|home|nyins[ɛe]n|akokoaa|abofra|aduro|malaria|fever|h[ɔo]spital|"
    r"emergency|mmonnaato|assault|suicide|kum me ho|awuo|wu|breath|bleed|dose|"
    r"pregnan|baby|child|medicine|pill|condom|hiv|sti)",
    re.IGNORECASE,
)

_ANNOTATION_KEYS = {
    "question_twi_normalized",
    "question_english",
    "answer_english",
    "intent",
    "topics",
    "entities",
    "safety_level",
    "requires_clarification",
    "source_answer_assessment",
    "source_answer_issues",
    "reply_twi",
    "reply_english",
    "evidence_spans_twi",
    "confidence",
}
_ADJUDICATION_KEYS = {
    "selected_proposal",
    "confidence",
    "rationale",
    "material_disagreement_fields",
    "review_reasons",
    "synthesized",
}

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
    .add_local_file(_TRAIN_CORPUS_PATH, _REMOTE_TRAIN_CORPUS_PATH)
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
            nested = parsed.get("response_shape")
            known_keys = _ANNOTATION_KEYS | _ADJUDICATION_KEYS
            outer_has_payload = bool(known_keys.intersection(parsed))
            nested_has_payload = isinstance(nested, dict) and bool(known_keys.intersection(nested))
            if not outer_has_payload and nested_has_payload:
                return nested
            return parsed
    return None


def _annotation_errors(
    parsed: dict[str, Any] | None,
    *,
    include_translations: bool,
) -> list[str]:
    if parsed is None:
        return ["no valid JSON object"]
    required_strings = ["intent", "safety_level", "source_answer_assessment"]
    if include_translations:
        required_strings.extend(["question_english", "answer_english", "reply_twi", "reply_english"])
    errors = [f"missing or empty {key}" for key in required_strings if not str(parsed.get(key) or "").strip()]
    for key in ("topics", "source_answer_issues"):
        if not isinstance(parsed.get(key), list) or (key == "topics" and not parsed[key]):
            errors.append(f"{key} must be a {'non-empty ' if key == 'topics' else ''}array")
    if not isinstance(parsed.get("entities"), dict):
        errors.append("entities must be an object")
    if not isinstance(parsed.get("requires_clarification"), bool):
        errors.append("requires_clarification must be a boolean")
    if not isinstance(parsed.get("confidence"), (int, float)):
        errors.append("confidence must be a number")
    return errors


def _adjudication_errors(parsed: dict[str, Any] | None) -> list[str]:
    if parsed is None:
        return ["no valid JSON object"]
    selection = parsed.get("selected_proposal")
    allowed = {"teacher_a", "teacher_b", "synthesized", "needs_human_review"}
    errors: list[str] = []
    if selection not in allowed:
        errors.append("selected_proposal is missing or invalid")
    if not isinstance(parsed.get("confidence"), (int, float)):
        errors.append("confidence must be a number")
    if not isinstance(parsed.get("rationale"), str):
        errors.append("rationale must be a string")
    for key in ("material_disagreement_fields", "review_reasons"):
        if not isinstance(parsed.get(key), list):
            errors.append(f"{key} must be an array")
    if selection == "synthesized":
        synthesis = parsed.get("synthesized")
        errors.extend(
            f"synthesized.{error}"
            for error in _annotation_errors(
                synthesis if isinstance(synthesis, dict) else None,
                include_translations=True,
            )
        )
    return errors


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _risk_score(row: dict[str, Any]) -> tuple[int, str]:
    combined = f"{row['question_twi_source']}\n{row['answer_twi_source']}"
    score = len(_RISK_TERMS.findall(combined)) * 10
    score += min(20, len(re.findall(r"\b\d+(?:\.\d+)?\b", combined)) * 4)
    score += min(10, len(row["answer_twi_source"]) // 600)
    return score, row["record_source_hash"]


def _load_rows(
    offset: int,
    limit: int,
    reference_only: bool,
    include_existing: bool,
) -> list[dict[str, Any]]:
    sources = _read_jsonl(_REMOTE_SOURCE_PATH)
    references = _read_jsonl(_REMOTE_REFERENCE_PATH)
    reference_ids = {row["row_id"] for row in references}
    if reference_only:
        eligible = [row for row in sources if row["id"] in reference_ids]
    else:
        train_keys = {
            (row["source_split"], row["source_record_id"])
            for row in _read_jsonl(_REMOTE_TRAIN_CORPUS_PATH)
            if row["language"] == "tw" and row["eligible_for_research_training"]
        }
        eligible = [
            row
            for row in sources
            if (row["source_split"], row["source_record_id"]) in train_keys
            and (include_existing or row["id"] not in reference_ids)
        ]
    eligible.sort(key=lambda row: (-_risk_score(row)[0], _risk_score(row)[1]))
    selected = eligible[max(0, int(offset)) :]
    return selected[:limit] if limit > 0 else selected


def _chunk_text(text: str, max_characters: int = 360) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_characters:
                chunks.append(current)
                current = word
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _source_reply(text: str, max_characters: int = 700) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*selected, sentence]).strip()
        if selected and (len(candidate) > max_characters or len(selected) >= 3):
            break
        if len(sentence) > max_characters and not selected:
            return sentence[:max_characters].rsplit(" ", 1)[0].strip()
        selected.append(sentence)
    return " ".join(selected).strip()


def _translate_nllb(
    rows: list[dict[str, Any]],
    model_id: str,
    tokenizer_id: str,
    token: str | None,
    cache: str,
    batch_size: int,
) -> tuple[list[str], list[str], list[str]]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        token=token,
        cache_dir=cache,
        src_lang="twi_Latn",
        tgt_lang="eng_Latn",
    )
    tokenizer.src_lang = "twi_Latn"
    target_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()

    groups: list[list[str]] = []
    for row in rows:
        groups.extend(
            [
                _chunk_text(row["question_twi_source"]),
                _chunk_text(row["answer_twi_source"]),
                _chunk_text(_source_reply(row["answer_twi_source"])),
            ]
        )
    flat = [chunk for group in groups for chunk in group]
    translated: list[str] = []
    for start in range(0, len(flat), batch_size):
        encoded = tokenizer(
            flat[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=192,
        ).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                num_beams=4,
                max_new_tokens=320,
                no_repeat_ngram_size=3,
            )
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    regrouped: list[str] = []
    cursor = 0
    for group in groups:
        regrouped.append(" ".join(part.strip() for part in translated[cursor : cursor + len(group)]).strip())
        cursor += len(group)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return regrouped[0::3], regrouped[1::3], regrouped[2::3]


def _annotation_shape(include_translations: bool = True) -> dict[str, Any]:
    shape: dict[str, Any] = {
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
        "confidence": 0.0,
    }
    if include_translations:
        return {
            "question_twi_normalized": "faithful normalized Twi question",
            "question_english": "complete natural English meaning of the question",
            "answer_english": "complete natural English meaning of the answer",
            **shape,
            "reply_twi": "short spoken Twi reply supported only by the source answer",
            "reply_english": "exact English meaning of reply_twi",
            "evidence_spans_twi": ["one to eight exact source substrings"],
        }
    return shape


def _direct_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the direct-Twi proposal agent for a Ghanaian health corpus. Return one "
                "valid JSON object only. Preserve every meaning-bearing detail, including person, "
                "negation, tense, duration, quantity, uncertainty, and code-switching. Do not "
                "invent a diagnosis, medicine, dose, fact, or emergency number. Treat the source "
                "answer as evidence, not guaranteed medical truth, and flag questionable claims. "
                "Write entity values as concise canonical English terms and include only entities "
                "stated in the source. The reply must be short natural Twi grounded only in "
                "supportable source claims. Return the fields shown inside response_shape directly; "
                "do not include a response_shape wrapper."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question_twi_source": row["question_twi_source"],
                    "answer_twi_source": row["answer_twi_source"],
                    "allowed_intents": _INTENTS,
                    "allowed_topics": _TOPICS,
                    "response_shape": _annotation_shape(),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _pivot_messages(
    row: dict[str, Any],
    question_english: str,
    answer_english: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the English semantic and safety proposal agent for a Ghanaian health "
                "corpus. The supplied English is an independent machine translation and may be "
                "wrong. Return one valid JSON object only. Label only what that translation "
                "supports; flag ambiguity or a questionable source answer. Do not diagnose, add "
                "medical claims, prescribe medicines, or repair the translation silently. Write "
                "entity values as concise canonical English terms. Return the fields shown inside "
                "response_shape directly; do not include a response_shape wrapper."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question_english_machine_translation": question_english,
                    "answer_english_machine_translation": answer_english,
                    "allowed_intents": _INTENTS,
                    "allowed_topics": _TOPICS,
                    "response_shape": _annotation_shape(include_translations=False),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _judge_messages(
    row: dict[str, Any],
    direct: dict[str, Any],
    pivot: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the conservative adjudicator for a Ghanaian Twi medical research "
                "corpus. Return one valid JSON object only. Compare both proposals with the "
                "immutable Twi source. Select one only when meaning, intent, entities, safety, and "
                "reply are supported. Use needs_human_review for conflicts involving negation, "
                "person, symptom, body part, medicine, duration, quantity, consent, pregnancy, "
                "children, urgency, or source-answer correctness. A synthesis may combine only "
                "supported content and must not add diagnoses, medicines, doses, or phone numbers. "
                "Return the fields shown inside response_shape directly; do not include a "
                "response_shape wrapper."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question_twi_source": row["question_twi_source"],
                    "answer_twi_source": row["answer_twi_source"],
                    "teacher_a_direct_twi": direct,
                    "teacher_b_translation_pivot": pivot,
                    "response_shape": {
                        "selected_proposal": (
                            "teacher_a|teacher_b|synthesized|needs_human_review"
                        ),
                        "confidence": 0.0,
                        "rationale": "brief source-based reason",
                        "material_disagreement_fields": [],
                        "review_reasons": [],
                        "synthesized": _annotation_shape(),
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def _generate(
    engine: Any,
    message_sets: list[list[dict[str, str]]],
    max_tokens: int,
) -> list[str]:
    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in message_sets
    ]
    requests = engine.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        use_tqdm=True,
    )
    return [request.outputs[0].text.strip() for request in requests]


def _repair_messages(
    original: list[dict[str, str]],
    raw: str,
    errors: list[str],
) -> list[dict[str, str]]:
    return [
        *original,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "Your previous output failed validation: "
                + "; ".join(errors)
                + ". Return one corrected, complete JSON object only. Preserve the source meaning "
                "and do not invent missing medical facts."
            ),
        },
    ]


def _bundle(
    model: str,
    raw: str,
    parsed: dict[str, Any] | None,
    repair_raw: str = "",
) -> dict[str, Any]:
    final_raw = repair_raw or raw
    return {
        "model": model,
        "parsed": parsed,
        "raw_text": raw,
        "repair_raw_text": repair_raw or None,
        "generation_attempts": 2 if repair_raw else 1,
        "raw_sha256": hashlib.sha256(final_raw.encode()).hexdigest(),
        "failure_excerpt": final_raw[:2400] if parsed is None else None,
    }


def _complete_direct(row: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if _annotation_errors(parsed, include_translations=True):
        return None
    assert parsed is not None
    parsed = dict(parsed)
    parsed["question_twi_normalized"] = (
        str(parsed.get("question_twi_normalized") or "").strip() or row["question_twi_source"]
    )
    spans = parsed.get("evidence_spans_twi")
    evidence = f"{row['question_twi_source']}\n{row['answer_twi_source']}".lower()
    if not isinstance(spans, list) or not all(str(span).strip().lower() in evidence for span in spans):
        parsed["evidence_spans_twi"] = [row["question_twi_source"]]
    return parsed


def _complete_pivot(
    row: dict[str, Any],
    parsed: dict[str, Any] | None,
    question_english: str,
    answer_english: str,
    reply_english: str,
) -> dict[str, Any] | None:
    if _annotation_errors(parsed, include_translations=False):
        return None
    if not all(value.strip() for value in (question_english, answer_english, reply_english)):
        return None
    assert parsed is not None
    return {
        **parsed,
        "question_twi_normalized": row["question_twi_source"],
        "question_english": question_english,
        "answer_english": answer_english,
        "reply_twi": _source_reply(row["answer_twi_source"]),
        "reply_english": reply_english,
        "evidence_spans_twi": [row["question_twi_source"]],
    }


def _complete_adjudication(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    return parsed if not _adjudication_errors(parsed) else None


@app.function(
    image=image,
    gpu="H100:2",
    timeout=24 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/results": results_volume,
    },
    secrets=SECRETS,
)
def annotate(
    offset: int = 0,
    limit: int = 16,
    batch_size: int = 8,
    reference_only: bool = False,
    include_existing: bool = False,
    model_id: str = "Qwen/Qwen3.5-35B-A3B",
    translation_model: str = "ninte/twi-en-nllb-v2",
    translation_tokenizer: str = "facebook/nllb-200-distilled-600M",
) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from vllm import LLM

    rows = _load_rows(offset, limit, reference_only, include_existing)
    if not rows:
        raise RuntimeError("No eligible source rows selected")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    api = HfApi(token=token)
    model_info = api.model_info(model_id)
    translation_info = api.model_info(translation_model)
    model_ref = f"{model_id}@{model_info.sha}"
    translation_ref = f"{translation_model}@{translation_info.sha}"

    question_english, answer_english, reply_english = _translate_nllb(
        rows,
        translation_model,
        translation_tokenizer,
        token,
        cache,
        max(4, batch_size * 2),
    )
    engine = LLM(
        model=model_id,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=12288,
        max_num_seqs=max(1, batch_size),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    direct_message_sets = [_direct_messages(row) for row in rows]
    pivot_message_sets = [
        _pivot_messages(row, question, answer)
        for row, question, answer in zip(rows, question_english, answer_english, strict=True)
    ]
    direct_raw = _generate(engine, direct_message_sets, 1800)
    pivot_raw = _generate(
        engine,
        pivot_message_sets,
        900,
    )
    direct_parsed = [
        _complete_direct(row, _parse_json(raw))
        for row, raw in zip(rows, direct_raw, strict=True)
    ]
    pivot_parsed = [
        _complete_pivot(row, _parse_json(raw), question, answer, reply)
        for row, raw, question, answer, reply in zip(
            rows,
            pivot_raw,
            question_english,
            answer_english,
            reply_english,
            strict=True,
        )
    ]
    direct_repair_raw = [""] * len(rows)
    direct_repair_indexes = [index for index, parsed in enumerate(direct_parsed) if parsed is None]
    if direct_repair_indexes:
        repaired = _generate(
            engine,
            [
                _repair_messages(
                    direct_message_sets[index],
                    direct_raw[index],
                    _annotation_errors(_parse_json(direct_raw[index]), include_translations=True),
                )
                for index in direct_repair_indexes
            ],
            1800,
        )
        for index, raw in zip(direct_repair_indexes, repaired, strict=True):
            direct_repair_raw[index] = raw
            direct_parsed[index] = _complete_direct(rows[index], _parse_json(raw))

    pivot_repair_raw = [""] * len(rows)
    pivot_repair_indexes = [index for index, parsed in enumerate(pivot_parsed) if parsed is None]
    if pivot_repair_indexes:
        repaired = _generate(
            engine,
            [
                _repair_messages(
                    pivot_message_sets[index],
                    pivot_raw[index],
                    _annotation_errors(_parse_json(pivot_raw[index]), include_translations=False),
                )
                for index in pivot_repair_indexes
            ],
            900,
        )
        for index, raw in zip(pivot_repair_indexes, repaired, strict=True):
            pivot_repair_raw[index] = raw
            pivot_parsed[index] = _complete_pivot(
                rows[index],
                _parse_json(raw),
                question_english[index],
                answer_english[index],
                reply_english[index],
            )

    judge_indexes = [
        index
        for index in range(len(rows))
        if direct_parsed[index] is not None and pivot_parsed[index] is not None
    ]
    judge_message_sets = [
        _judge_messages(rows[index], direct_parsed[index], pivot_parsed[index])
        for index in judge_indexes
    ]
    judge_raw = _generate(
        engine,
        judge_message_sets,
        2200,
    )
    judge_by_index = {
        index: value for index, value in zip(judge_indexes, judge_raw, strict=True)
    }
    judge_parsed_by_index = {
        index: _complete_adjudication(_parse_json(raw))
        for index, raw in judge_by_index.items()
    }
    judge_repair_by_index: dict[int, str] = {}
    judge_repair_positions = [
        position
        for position, index in enumerate(judge_indexes)
        if judge_parsed_by_index[index] is None
    ]
    if judge_repair_positions:
        repaired = _generate(
            engine,
            [
                _repair_messages(
                    judge_message_sets[position],
                    judge_raw[position],
                    _adjudication_errors(_parse_json(judge_raw[position])),
                )
                for position in judge_repair_positions
            ],
            2200,
        )
        for position, raw in zip(judge_repair_positions, repaired, strict=True):
            index = judge_indexes[position]
            judge_repair_by_index[index] = raw
            judge_parsed_by_index[index] = _complete_adjudication(_parse_json(raw))

    generated_at = datetime.now(timezone.utc).isoformat()
    records = []
    for index, row in enumerate(rows):
        judge_value = judge_by_index.get(index, "")
        records.append(
            {
                "schema_version": 1,
                "prompt_version": _PROMPT_VERSION,
                "generated_at": generated_at,
                "row_id": row["id"],
                "source_record_hash": row["record_source_hash"],
                "source_risk_score": _risk_score(row)[0],
                "teacher_a": _bundle(
                    model_ref,
                    direct_raw[index],
                    direct_parsed[index],
                    direct_repair_raw[index],
                ),
                "teacher_b": _bundle(
                    f"{translation_ref} + {model_ref} + source-answer-excerpt-v1",
                    pivot_raw[index],
                    pivot_parsed[index],
                    pivot_repair_raw[index],
                ),
                "adjudicator": _bundle(
                    model_ref,
                    judge_value,
                    judge_parsed_by_index.get(index),
                    judge_repair_by_index.get(index, ""),
                )
                if judge_value
                else None,
            }
        )

    output_dir = "/results/response-annotations/pivot-v1"
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "reference" if reference_only else "train"
    output_path = f"{output_dir}/{mode}-{offset}-{len(rows)}-{run_id}.raw.jsonl"
    for target in (output_path, f"{output_dir}/latest.raw.jsonl"):
        with open(target, "w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "prompt_version": _PROMPT_VERSION,
        "created_at": generated_at,
        "mode": mode,
        "offset": offset,
        "selected": len(rows),
        "model": {
            "id": model_id,
            "revision": model_info.sha,
            "license": (model_info.card_data or {}).get("license", "missing"),
        },
        "translation_model": {
            "id": translation_model,
            "revision": translation_info.sha,
            "license": (translation_info.card_data or {}).get("license", "missing"),
        },
        "teacher_a_parsed": sum(value is not None for value in direct_parsed),
        "teacher_b_parsed": sum(value is not None for value in pivot_parsed),
        "teacher_a_retried": sum(bool(value) for value in direct_repair_raw),
        "teacher_b_retried": sum(bool(value) for value in pivot_repair_raw),
        "adjudicator_retried": len(judge_repair_by_index),
        "adjudicator_parsed": sum(value is not None for value in judge_parsed_by_index.values()),
        "output_path": output_path,
        "promotion_policy": (
            "Model outputs are research proposals. Deterministic validation and human review "
            "remain required; no row is human gold from this job."
        ),
    }
    with open(f"{output_dir}/latest.manifest.json", "w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
    results_volume.commit()
    hf_cache.commit()
    vllm_cache.commit()
    return manifest


@app.local_entrypoint()
def main(
    offset: int = 0,
    limit: int = 16,
    batch_size: int = 8,
    reference_only: bool = False,
    include_existing: bool = False,
    detach: bool = False,
    model_id: str = "Qwen/Qwen3.5-35B-A3B",
) -> None:
    kwargs = {
        "offset": offset,
        "limit": limit,
        "batch_size": batch_size,
        "reference_only": reference_only,
        "include_existing": include_existing,
        "model_id": model_id,
    }
    if detach:
        call = annotate.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(annotate.remote(**kwargs), ensure_ascii=False, indent=2))
