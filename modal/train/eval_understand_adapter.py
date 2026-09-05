"""Evaluate the Twi semantic-recovery LoRA or its base model.

This is a product-facing smoke eval, not a benchmark scorecard. It proves the
adapter loads, emits parseable JSON, and preserves key health/commerce meanings.

  modal run modal/train/eval_understand_adapter.py
  modal run modal/train/eval_understand_adapter.py --adapter-id base
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import modal

app = modal.App("ghana-health-understand-adapter-eval")
hf_cache = modal.Volume.from_name("ghana-health-understand-models", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_FIXTURE_PATH = os.path.join(_REPO_ROOT, "scripts", "understanding-fixtures.json")
_REMOTE_FIXTURE_PATH = "/root/eval/understanding-fixtures.json"
_TEST_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "understanding-corpus",
    "silver-medical-paired-v2",
    "test.jsonl",
)
_REMOTE_TEST_PATH = "/root/eval/silver-medical-paired-v2-test.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "peft==0.17.1",
        "huggingface_hub==0.26.2",
    )
    .add_local_file(_FIXTURE_PATH, _REMOTE_FIXTURE_PATH)
    .add_local_file(_TEST_PATH, _REMOTE_TEST_PATH)
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []

SYSTEM = (
    "You are Ghana Health AI's semantic recovery model. Given a Twi/Akan, "
    "English, or code-switched user utterance, output faithful structured "
    "understanding. Do not diagnose. Do not invent missing symptoms. Preserve "
    "uncertainty. Return JSON only with keys normalized_twi, natural_english, "
    "literal_english, intent, entities, ambiguities, requires_clarification. "
    "Use double quotes, lowercase true/false, and no markdown."
)


def _json_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object")
    raw = text[start : end + 1]
    try:
        value = json.loads(raw)
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
        value = json.loads(repaired)
    if not isinstance(value, dict):
        raise ValueError("json_not_object")
    return value


def _contains_any(value: str, terms: list[str]) -> bool:
    lower = value.lower()
    return any(term.lower() in lower for term in terms)


def _covers_concepts(value: str, concept_groups: list[list[str]]) -> bool:
    return all(_contains_any(value, alternatives) for alternatives in concept_groups)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    remaining = list(expected)
    overlap = 0
    for token in predicted:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _body_system(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    return str(value.get("body_system", "")).strip() if isinstance(value, dict) else ""


def _load_adapter(base_model: str, adapter_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/models/hf"
    use_adapter = bool(adapter_id) and adapter_id.lower() != "base"
    tokenizer_source = adapter_id if use_adapter else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache,
        torch_dtype="auto",
        device_map="auto",
        token=token,
    )
    if use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_id, cache_dir=cache, token=token)
    else:
        model = base
    model.eval()
    return tokenizer, model


def _generate_json(tokenizer, model, messages: list[dict[str, str]]) -> tuple[dict[str, Any], str, str]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        output[0][encoded["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    try:
        return _json_from_text(decoded), decoded, ""
    except Exception as exc:  # noqa: BLE001
        return {}, decoded, str(exc)


def _generate_json_batch(
    tokenizer,
    model,
    message_batches: list[list[dict[str, str]]],
) -> list[tuple[dict[str, Any], str, str]]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in message_batches
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_width = encoded["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results: list[tuple[dict[str, Any], str, str]] = []
    for output in outputs:
        decoded = tokenizer.decode(output[input_width:], skip_special_tokens=True).strip()
        try:
            results.append((_json_from_text(decoded), decoded, ""))
        except Exception as exc:  # noqa: BLE001
            results.append(({}, decoded, str(exc)))
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=45 * 60,
    volumes={"/models": hf_cache},
    secrets=SECRETS,
)
def evaluate(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_id: str = "teckedd/gha-understand-twi-medical-v4",
    limit: int = 0,
) -> dict[str, Any]:
    tokenizer, model = _load_adapter(base_model, adapter_id)

    with open(_REMOTE_FIXTURE_PATH, encoding="utf-8") as source:
        fixtures = json.load(source)
    if limit > 0:
        fixtures = fixtures[:limit]

    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        messages = [{"role": "system", "content": SYSTEM}]
        messages.extend(fixture.get("history") or [])
        messages.append({"role": "user", "content": fixture["text"]})
        payload, decoded, error = _generate_json(tokenizer, model, messages)
        parse_ok = not error

        natural = str(payload.get("natural_english", ""))
        intent = str(payload.get("intent", ""))
        expected_intent = str(fixture.get("expectedIntent", "")).lower()
        concept_groups = fixture.get("expectedEnglishConcepts") or []
        concept_ok = _covers_concepts(natural, concept_groups)
        forbidden_terms = fixture.get("forbiddenUnderstandingTerms") or []
        forbidden_ok = not _contains_any(natural, forbidden_terms) if forbidden_terms else True
        schema_ok = bool(intent) and isinstance(payload.get("entities"), dict) and isinstance(
            payload.get("requires_clarification"), bool
        )
        clarification_ok = (
            payload.get("requires_clarification") is True
            if fixture.get("expectedClarifying") is True
            else True
        )
        intent_lower = intent.lower()
        if expected_intent == "ecommerce":
            intent_ok = intent_lower.startswith("commerce_")
        elif expected_intent == "health":
            intent_ok = intent_lower.startswith("health_")
        else:
            intent_ok = not expected_intent or expected_intent in intent_lower
        ok = parse_ok and schema_ok and concept_ok and forbidden_ok and clarification_ok and intent_ok
        results.append(
            {
                "id": fixture["id"],
                "focus": fixture.get("focus", "unknown"),
                "ok": ok,
                "parse_ok": parse_ok,
                "schema_ok": schema_ok,
                "intent_ok": intent_ok,
                "concept_ok": concept_ok,
                "forbidden_ok": forbidden_ok,
                "clarification_ok": clarification_ok,
                "expected_intent": expected_intent,
                "expected_english_concepts": concept_groups,
                "prediction": payload,
                "raw": decoded[:500],
                "error": error,
            }
        )

    passed = sum(1 for row in results if row["ok"])
    failed = len(results) - passed
    by_focus: dict[str, dict[str, int]] = {}
    for row in results:
        focus = str(row["focus"])
        by_focus.setdefault(focus, {"passed": 0, "failed": 0, "total": 0})
        by_focus[focus]["total"] += 1
        by_focus[focus]["passed" if row["ok"] else "failed"] += 1
    return {
        "status": "complete" if failed == 0 else "needs_review",
        "base_model": base_model,
        "adapter_id": adapter_id,
        "case_count": len(results),
        "passed": passed,
        "failed": failed,
        "by_focus": by_focus,
        "results": results,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=90 * 60,
    volumes={"/models": hf_cache},
    secrets=SECRETS,
)
def evaluate_test_set(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_id: str = "teckedd/gha-understand-twi-medical-v4",
    limit: int = 0,
) -> dict[str, Any]:
    tokenizer, model = _load_adapter(base_model, adapter_id)
    with open(_REMOTE_TEST_PATH, encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if limit > 0:
        rows = rows[:limit]

    parse_passed = 0
    intent_passed = 0
    body_system_passed = 0
    strict_passed = 0
    meaning_f1_sum = 0.0
    by_body_system: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []

    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
        message_batches = []
        for row in batch_rows:
            messages = row.get("messages") or [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": str(row.get("original_text", ""))},
            ]
            message_batches.append(messages[:2])
        generated = _generate_json_batch(tokenizer, model, message_batches)
        for row, (payload, decoded, error) in zip(batch_rows, generated):
            parse_ok = not error
            intent_ok = str(payload.get("intent", "")).strip() == str(row.get("intent", "")).strip()
            expected_body = _body_system(row.get("entities"))
            body_ok = _body_system(payload.get("entities")) == expected_body
            meaning_f1 = _token_f1(
                str(payload.get("natural_english", "")),
                str(row.get("natural_english", "")),
            )
            strict_ok = parse_ok and intent_ok and body_ok and meaning_f1 >= 0.5
            parse_passed += int(parse_ok)
            intent_passed += int(intent_ok)
            body_system_passed += int(body_ok)
            strict_passed += int(strict_ok)
            meaning_f1_sum += meaning_f1
            bucket = expected_body or "unknown"
            by_body_system.setdefault(bucket, {"passed": 0, "total": 0})
            by_body_system[bucket]["total"] += 1
            by_body_system[bucket]["passed"] += int(strict_ok)
            if not strict_ok and len(failures) < 20:
                failures.append({
                    "id": row.get("id"),
                    "expected_english": row.get("natural_english"),
                    "predicted_english": payload.get("natural_english", ""),
                    "expected_body_system": expected_body,
                    "predicted_body_system": _body_system(payload.get("entities")),
                    "expected_intent": row.get("intent"),
                    "predicted_intent": payload.get("intent", ""),
                    "meaning_token_f1": round(meaning_f1, 4),
                    "parse_error": error,
                    "raw": decoded[:500],
                })
        if (start + len(batch_rows)) % 40 == 0 or start + len(batch_rows) == len(rows):
            print(f"[held-out-eval] {start + len(batch_rows)}/{len(rows)} rows", flush=True)

    total = len(rows)
    return {
        "status": "complete",
        "base_model": base_model,
        "adapter_id": adapter_id,
        "test_set": "silver-medical-paired-v2/test.jsonl",
        "case_count": total,
        "parse_passed": parse_passed,
        "parse_rate": parse_passed / total if total else 0.0,
        "intent_passed": intent_passed,
        "intent_accuracy": intent_passed / total if total else 0.0,
        "body_system_passed": body_system_passed,
        "body_system_accuracy": body_system_passed / total if total else 0.0,
        "strict_passed": strict_passed,
        "strict_pass_rate": strict_passed / total if total else 0.0,
        "mean_natural_english_token_f1": meaning_f1_sum / total if total else 0.0,
        "by_body_system": by_body_system,
        "failure_examples": failures,
    }


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_id: str = "teckedd/gha-understand-twi-medical-v4",
    limit: int = 0,
    test_set: bool = False,
) -> None:
    if test_set:
        print(
            json.dumps(
                evaluate_test_set.remote(base_model=base_model, adapter_id=adapter_id, limit=limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            evaluate.remote(base_model=base_model, adapter_id=adapter_id, limit=limit),
            ensure_ascii=False,
            indent=2,
        )
    )
