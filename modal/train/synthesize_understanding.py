"""Generate multi-agent Twi corpus annotations on Modal.

The job writes raw, source-linked model outputs to a private Modal volume. A
local importer applies deterministic scoring and promotion gates before the
records enter the review UI.

Smoke test:
  modal run modal/train/synthesize_understanding.py --limit 2

Corpus run:
  modal run --detach modal/train/synthesize_understanding.py --limit 7000 --batch-size 8

Retrieve:
  modal volume get ghana-health-understanding-results \
    /synthesis/v9/latest.raw.jsonl tmp/understanding-corpus/modal-synthesis-v9.raw.jsonl
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

app = modal.App("ghana-health-understanding-synthesis")
hf_cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("ghana-health-understanding-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_CANDIDATE_PATH = os.path.join(_REPO_ROOT, "data", "understanding-corpus", "candidates.v0.jsonl")
_REMOTE_CANDIDATE_PATH = "/root/corpus/candidates.v0.jsonl"
_PROMPT_VERSION = "understanding-synthesis-v9"
_INTENTS = [
    "health_symptom_report",
    "health_followup",
    "health_facility_search",
    "health_medication_question",
    "health_general_question",
    "health_emergency_report",
    "commerce_product_search",
    "commerce_purchase_request",
    "commerce_order_followup",
    "commerce_general_question",
    "general_statement",
    "general_question",
    "unclear_fragment",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.3.0",
        "accelerate==1.1.1",
        "sentencepiece==0.2.0",
        "sacremoses==0.1.1",
        "huggingface_hub==1.3.0",
    )
    .add_local_file(_CANDIDATE_PATH, _REMOTE_CANDIDATE_PATH)
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _parse_json(value: str, required_fields: tuple[str, ...] = ()) -> dict[str, Any] | None:
    placeholder_values = {
        "original wording with punctuation/spacing corrections only",
        "faithful natural english meaning",
        "literal gloss when useful",
        "one value from allowed_intents",
        "remaining uncertainty or empty string",
        "source-backed reply only or empty",
        "source-backed level only or empty",
    }

    def acceptable(candidate: Any) -> bool:
        if not isinstance(candidate, dict):
            return False
        for field in required_fields:
            field_value = candidate.get(field)
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                return False
            if isinstance(field_value, str) and field_value.strip().lower() in placeholder_values:
                return False
        return True

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
        if acceptable(parsed):
            return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    raw = cleaned[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        repaired = (
            raw.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace(":=", ":")
            .replace("=:", ":")
        )
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[=:])", r'\1"\2":', repaired)
        repaired = re.sub(r'"\s*=', '":', repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return parsed if acceptable(parsed) else None


def _load_rows(source: str, offset: int, limit: int, stride: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(_REMOTE_CANDIDATE_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            proposal = row.get("model_proposal") or {}
            if source and row.get("source") != source:
                continue
            if proposal.get("status") != "draft" or not str(proposal.get("natural_english") or "").strip():
                continue
            rows.append(row)
    start = max(0, int(offset))
    step = max(1, int(stride))
    selected = rows[start::step]
    return selected[:limit] if limit > 0 else selected


def _schema() -> dict[str, Any]:
    return {
        "normalized_twi": "original wording with punctuation/spacing corrections only",
        "natural_english": "faithful natural English meaning",
        "literal_english": "literal gloss when useful",
        "intent": "one value from allowed_intents",
        "entities": {
            "symptom": [],
            "body_part": [],
            "product": [],
            "quantity": [],
            "location": [],
            "time": [],
            "negated": [],
        },
        "ambiguities": "remaining uncertainty or empty string",
        "reply_twi": "source-backed reply only or empty",
        "safety_level": "source-backed level only or empty",
        "requires_clarification": False,
    }


def _agent_messages(row: dict[str, Any], role: str) -> list[dict[str, str]]:
    proposal = row.get("model_proposal") or {}
    focus = (
        "Act as a Ghanaian Twi translation specialist. Preserve exact meaning, tense, negation, quantities, code-switching, uncertainty, and discourse references."
        if role == "translator"
        else "Act as a Ghanaian health and commerce semantic annotator. Focus on intent, entities, negation, time, quantity, location, symptoms, and body-part references."
    )
    system = (
        f"{focus} Return one valid JSON object only. The utterance is immutable evidence: normalized_twi may correct punctuation, spacing, and obvious orthography but must not add, remove, or duplicate meaning-bearing words. "
        "Do not diagnose or infer absent facts. Never add reply_twi or safety_level unless the source supplies them."
    )
    payload = {
        "utterance": row["text"],
        "source": row["source"],
        "domain": row["domain"],
        "source_reply_twi": proposal.get("reply_twi", ""),
        "source_safety_level": proposal.get("safety_level", ""),
        "allowed_intents": _INTENTS,
        "response_shape": _schema(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _semantic_messages(row: dict[str, Any], translated_english: str) -> list[dict[str, str]]:
    """Structure an independent MT result without letting the model retranslate it."""
    system = (
        "You structure an existing Twi-to-English machine translation for corpus comparison. "
        "Return one valid JSON object only. Copy source_twi exactly into normalized_twi and "
        "copy translated_english exactly into natural_english. Do not translate, rewrite, "
        "answer, diagnose, or add facts. Extract only intent, entities, ambiguity, and whether "
        "clarification is needed. Intent must be one allowed value."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "source_twi": row["text"],
                    "translated_english": translated_english,
                    "domain": row.get("domain", "general"),
                    "allowed_intents": _INTENTS,
                    "response_shape": _schema(),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _judge_messages(
    row: dict[str, Any],
    translator: dict[str, Any],
    semantic: dict[str, Any],
) -> list[dict[str, str]]:
    source = row.get("model_proposal") or {}
    try:
        source_entities = json.loads(source.get("entities") or "{}")
    except (TypeError, json.JSONDecodeError):
        source_entities = {}
    source_ambiguities = "\n".join(
        line.strip()
        for line in str(source.get("ambiguities") or "").splitlines()
        if line.strip()
        and not re.match(
            r"^(source|sources|source_twi|training_use|body_system|license|notes)=",
            line.strip(),
            re.IGNORECASE,
        )
    )
    source_bundle = {
        "proposal_id": "source",
        "normalized_twi": source.get("normalized_twi") or row["text"],
        "natural_english": source.get("natural_english", ""),
        "literal_english": source.get("literal_english", ""),
        "intent": source.get("intent", ""),
        "entities": source_entities,
        "ambiguities": source_ambiguities,
        "reply_twi": source.get("reply_twi", ""),
        "safety_level": source.get("safety_level", ""),
        "requires_clarification": bool(source.get("requires_clarification")),
    }
    system = (
        "Adjudicate paired Twi corpus evidence and return one compact valid JSON object only. "
        "The paired source English may be a concise clinical summary. A candidate that adds details explicitly present in the Twi is compatible, not contradictory. "
        "Flag a material disagreement only for changed or missing symptoms, negation, quantities, time, body parts, products, locations, intent, or uncertainty. Specific modifiers are material: herbal medicine is not equivalent to unspecified medicine, and vomiting is not equivalent to poor appetite. Equivalent wording and compatible detail are not material. Never treat body_system taxonomy metadata as a disagreement. "
        "Require human review when two independent candidates support the same contradiction to the paired reference, or when the evidence remains genuinely ambiguous. Otherwise select source when it preserves the core meaning; select another proposal only when it clearly corrects a material error. "
        "Preserve the original utterance; normalization may only correct punctuation, spacing, and obvious orthography. Do not invent a diagnosis, medical response, or safety level. "
        "review_reasons may contain only these short codes: paired_reference_conflict, candidate_conflict, negation_conflict, quantity_conflict, time_conflict, entity_conflict, ambiguous_twi, low_confidence. "
        "Use confidence at least 0.78 only when no material disagreement remains, and below 0.78 when review is needed."
    )
    payload = {
        "utterance": row["text"],
        "paired_source_english_reference": source.get("natural_english", ""),
        "source_reply_twi": source.get("reply_twi", ""),
        "allowed_intents": _INTENTS,
        "proposals": [source_bundle, {"proposal_id": "translator", **translator}, {"proposal_id": "semantic", **semantic}],
        "response_shape": {
            "selected_proposal_id": "source|translator|semantic|synthesized",
            "confidence": 0.0,
            "material_disagreement_fields": [],
            "review_reasons": [],
            "rationale": "brief evidence-based reason",
            **_schema(),
        },
    }
    conflict_example = {
        "utterance": "Nnansa ni meyɛ mmerɛ. Menom aduro bi nso ɛnyɛɛ adwuma.",
        "paired_source_english_reference": "Weakness for three days not relieved by herbal medicine.",
        "proposals": [
            {"proposal_id": "source", "natural_english": "Weakness for three days not relieved by herbal medicine."},
            {"proposal_id": "translator", "natural_english": "I have been weak for three days. Some medicine did not help."},
            {"proposal_id": "semantic", "natural_english": "Three days of weakness; medication did not work."},
        ],
    }
    conflict_answer = {
        "selected_proposal_id": "source",
        "confidence": 0.7,
        "material_disagreement_fields": ["product"],
        "review_reasons": ["paired_reference_conflict"],
        "rationale": "Both independent candidates support unspecified medicine; the paired reference adds herbal.",
        "normalized_twi": conflict_example["utterance"],
        "natural_english": conflict_example["paired_source_english_reference"],
        "literal_english": "",
        "intent": "health_symptom_report",
        "entities": {"symptom": ["weakness"], "product": ["herbal medicine"], "time": ["three days"]},
        "ambiguities": "Medicine type conflicts across the evidence.",
        "reply_twi": "",
        "safety_level": "",
        "requires_clarification": True,
    }
    paraphrase_example = {
        "utterance": "Nnansa ni m'akyi yɛ me yaw na akyere me.",
        "paired_source_english_reference": "Three-day lower-back pain with stiffness.",
        "proposals": [
            {"proposal_id": "source", "natural_english": "Three-day lower-back pain with stiffness."},
            {"proposal_id": "translator", "natural_english": "My lower back has hurt and felt stiff for three days."},
            {"proposal_id": "semantic", "natural_english": "I have had back pain and stiffness for three days."},
        ],
    }
    paraphrase_answer = {
        "selected_proposal_id": "source",
        "confidence": 0.94,
        "material_disagreement_fields": [],
        "review_reasons": [],
        "rationale": "All candidates express the same symptoms and duration using compatible wording.",
        "normalized_twi": paraphrase_example["utterance"],
        "natural_english": paraphrase_example["paired_source_english_reference"],
        "literal_english": "",
        "intent": "health_symptom_report",
        "entities": {"symptom": ["pain", "stiffness"], "body_part": ["lower back"], "time": ["three days"]},
        "ambiguities": "",
        "reply_twi": "",
        "safety_level": "",
        "requires_clarification": False,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(conflict_example, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(conflict_answer, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(paraphrase_example, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(paraphrase_answer, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(messages), batch_size):
        chunk = messages[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in chunk
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded.input_ids.shape[1] :]
        outputs.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return outputs


def _load_model(model_id: str, token: str | None, cache: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _language_id(tokenizer: Any, candidates: tuple[str, ...]) -> int:
    language_ids = getattr(tokenizer, "lang_code_to_id", {}) or {}
    for code in candidates:
        if code in language_ids:
            return int(language_ids[code])
        token_id = tokenizer.convert_tokens_to_ids(code)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    raise RuntimeError(f"Tokenizer has none of the required language codes: {candidates}")


def _load_translation_model(
    model_id: str,
    tokenizer_id: str,
    token: str | None,
    cache: str,
) -> tuple[Any, Any, int]:
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
    target_id = _language_id(tokenizer, ("eng_Latn",))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return model, tokenizer, target_id


def _translate(
    model: Any,
    tokenizer: Any,
    target_id: int,
    texts: list[str],
    batch_size: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=192,
                num_beams=4,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [value.strip() for value in outputs]


@app.function(
    image=image,
    gpu="L40S",
    timeout=24 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_volume,
    },
    secrets=SECRETS,
)
def synthesize(
    source: str = "ghana_health_symptoms",
    offset: int = 0,
    limit: int = 20,
    stride: int = 1,
    batch_size: int = 4,
    translator_model: str = "McGill-NLP/AfriqueQwen3.5-9B-50Langs",
    structure_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adjudicator_model: str = "Qwen/Qwen2.5-7B-Instruct",
    alternate_translation_model: str = "ninte/twi-en-nllb-v2",
    alternate_tokenizer: str = "facebook/nllb-200-distilled-600M",
) -> dict[str, Any]:
    import torch

    rows = _load_rows(source, offset, limit, stride)
    if not rows:
        raise RuntimeError("No eligible corpus rows selected")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)

    translator, translator_tokenizer = _load_model(translator_model, token, cache)
    translator_raw = _generate(
        translator,
        translator_tokenizer,
        [_agent_messages(row, "translator") for row in rows],
        max(1, batch_size),
        640,
    )
    del translator, translator_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    structure, structure_tokenizer = _load_model(structure_model, token, cache)
    alternate, alternate_tokenizer_instance, alternate_target_id = _load_translation_model(
        alternate_translation_model,
        alternate_tokenizer,
        token,
        cache,
    )
    alternate_translations = _translate(
        alternate,
        alternate_tokenizer_instance,
        alternate_target_id,
        [str(row["text"]) for row in rows],
        max(1, batch_size),
    )
    semantic_raw = _generate(
        structure,
        structure_tokenizer,
        [
            _semantic_messages(row, translated)
            for row, translated in zip(rows, alternate_translations, strict=True)
        ],
        max(1, batch_size),
        420,
    )
    del alternate, alternate_tokenizer_instance, structure, structure_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    annotation_fields = ("normalized_twi", "natural_english", "intent", "entities")
    judgment_fields = ("selected_proposal_id", "confidence", "natural_english", "intent", "entities")
    translator_parsed = [_parse_json(value, annotation_fields) for value in translator_raw]
    semantic_parsed = [_parse_json(value, annotation_fields) for value in semantic_raw]
    for index, parsed in enumerate(semantic_parsed):
        if parsed is None:
            continue
        parsed["normalized_twi"] = rows[index]["text"]
        parsed["natural_english"] = alternate_translations[index]
        parsed["literal_english"] = parsed.get("literal_english") or alternate_translations[index]
        parsed["reply_twi"] = ""
        parsed["safety_level"] = ""

    judge_messages: list[list[dict[str, str]]] = []
    judge_indexes: list[int] = []
    for index, row in enumerate(rows):
        if translator_parsed[index] is None or semantic_parsed[index] is None:
            continue
        judge_indexes.append(index)
        judge_messages.append(_judge_messages(row, translator_parsed[index], semantic_parsed[index]))
    adjudicator, adjudicator_tokenizer = _load_model(adjudicator_model, token, cache)
    judge_raw = _generate(
        adjudicator,
        adjudicator_tokenizer,
        judge_messages,
        max(1, batch_size),
        720,
    )
    judge_by_index = {index: value for index, value in zip(judge_indexes, judge_raw, strict=True)}
    judge_parsed_by_index = {
        index: _parse_json(value, judgment_fields)
        for index, value in judge_by_index.items()
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        judge_value = judge_by_index.get(index, "")
        records.append(
            {
                "schema_version": 1,
                "prompt_version": _PROMPT_VERSION,
                "generated_at": generated_at,
                "row_id": row["id"],
                "source_hash": row["source_hash"],
                "translator": {
                    "model": translator_model,
                    "parsed": translator_parsed[index],
                    "raw_sha256": hashlib.sha256(translator_raw[index].encode()).hexdigest(),
                    "failure_excerpt": translator_raw[index][:2000]
                    if translator_parsed[index] is None
                    else None,
                },
                "semantic": {
                    "model": f"{alternate_translation_model} + {structure_model}",
                    "translation_model": alternate_translation_model,
                    "tokenizer": alternate_tokenizer,
                    "structure_model": structure_model,
                    "translation": alternate_translations[index],
                    "parsed": semantic_parsed[index],
                    "raw_sha256": hashlib.sha256(semantic_raw[index].encode()).hexdigest(),
                    "failure_excerpt": semantic_raw[index][:2000] if semantic_parsed[index] is None else None,
                },
                "adjudicator": {
                    "model": adjudicator_model,
                    "parsed": judge_parsed_by_index.get(index),
                    "raw_sha256": hashlib.sha256(judge_value.encode()).hexdigest() if judge_value else None,
                    "failure_excerpt": judge_value[:2000]
                    if judge_value and judge_parsed_by_index.get(index) is None
                    else None,
                },
            }
        )

    output_dir = "/results/synthesis/v9"
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"{output_dir}/{source}-{offset}-{len(rows)}-{run_id}.raw.jsonl"
    for target in (output_path, f"{output_dir}/latest.raw.jsonl"):
        with open(target, "w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "prompt_version": _PROMPT_VERSION,
        "created_at": generated_at,
        "source": source,
        "offset": offset,
        "stride": stride,
        "selected": len(rows),
        "translator_model": translator_model,
        "structure_model": structure_model,
        "adjudicator_model": adjudicator_model,
        "alternate_translation_model": alternate_translation_model,
        "alternate_tokenizer": alternate_tokenizer,
        "translator_parsed": sum(record["translator"]["parsed"] is not None for record in records),
        "semantic_parsed": sum(record["semantic"]["parsed"] is not None for record in records),
        "adjudicator_parsed": sum(record["adjudicator"]["parsed"] is not None for record in records),
        "output_path": output_path,
    }
    with open(f"{output_dir}/latest.manifest.json", "w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
    results_volume.commit()

    hf_cache.commit()
    del adjudicator, adjudicator_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


@app.local_entrypoint()
def main(
    source: str = "ghana_health_symptoms",
    offset: int = 0,
    limit: int = 20,
    stride: int = 1,
    batch_size: int = 4,
    detach: bool = False,
    translator_model: str = "McGill-NLP/AfriqueQwen3.5-9B-50Langs",
    structure_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adjudicator_model: str = "Qwen/Qwen2.5-7B-Instruct",
    alternate_translation_model: str = "ninte/twi-en-nllb-v2",
    alternate_tokenizer: str = "facebook/nllb-200-distilled-600M",
) -> None:
    kwargs = {
        "source": source,
        "offset": offset,
        "limit": limit,
        "stride": stride,
        "batch_size": batch_size,
        "translator_model": translator_model,
        "structure_model": structure_model,
        "adjudicator_model": adjudicator_model,
        "alternate_translation_model": alternate_translation_model,
        "alternate_tokenizer": alternate_tokenizer,
    }
    if detach:
        call = synthesize.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(synthesize.remote(**kwargs), ensure_ascii=False, indent=2))
