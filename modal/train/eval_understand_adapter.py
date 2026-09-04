"""Evaluate the Twi semantic-recovery LoRA on product fixtures.

This is a product-facing smoke eval, not a benchmark scorecard. It proves the
adapter loads, emits parseable JSON, and preserves key health/commerce meanings.

  modal run modal/train/eval_understand_adapter.py
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


@app.function(
    image=image,
    gpu="T4",
    timeout=45 * 60,
    volumes={"/models": hf_cache},
    secrets=SECRETS,
)
def evaluate(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_id: str = "teckedd/gha-understand-twi-medical-plus-language-v3",
    limit: int = 0,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/models/hf"

    tokenizer = AutoTokenizer.from_pretrained(adapter_id, cache_dir=cache, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache,
        torch_dtype="auto",
        device_map="auto",
        token=token,
    )
    model = PeftModel.from_pretrained(base, adapter_id, cache_dir=cache, token=token)
    model.eval()

    with open(_REMOTE_FIXTURE_PATH, encoding="utf-8") as source:
        fixtures = json.load(source)
    if limit > 0:
        fixtures = fixtures[:limit]

    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "language=tw\nfocus=health\nrecent_history=[]\n"
                    "utterance=me ba no ho yɛ hyew"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '{"normalized_twi":"me ba no ho yɛ hyew",'
                    '"natural_english":"My child has a fever or feels hot.",'
                    '"literal_english":"My child body is hot.",'
                    '"intent":"health_symptom_report",'
                    '"entities":{"person":"child","symptom":"fever"},'
                    '"ambiguities":"temperature not measured; child age unknown",'
                    '"requires_clarification":true}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"language={fixture.get('language', 'tw')}\n"
                    f"focus={fixture.get('focus', 'health')}\n"
                    f"recent_history={json.dumps(fixture.get('history', []), ensure_ascii=False)}\n"
                    f"utterance={fixture['text']}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            output[0][encoded["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        try:
            payload = _json_from_text(decoded)
            parse_ok = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            payload = {}
            parse_ok = False
            error = str(exc)

        natural = str(payload.get("natural_english", ""))
        normalized = str(payload.get("normalized_twi", ""))
        intent = str(payload.get("intent", ""))
        expected_intent = str(fixture.get("expectedIntent", "")).lower()
        expected_terms = fixture.get("expectedReplyIncludesAny") or []
        shopping_term = fixture.get("expectedShoppingIntent")
        term_ok = True
        if expected_terms:
            term_ok = _contains_any(natural + " " + normalized, expected_terms)
        if shopping_term:
            term_ok = _contains_any(natural + " " + normalized, [shopping_term])
        intent_ok = not expected_intent or expected_intent in intent.lower()
        ok = parse_ok and term_ok and (intent_ok or fixture.get("focus") == "commerce")
        results.append(
            {
                "id": fixture["id"],
                "ok": ok,
                "parse_ok": parse_ok,
                "intent_ok": intent_ok,
                "term_ok": term_ok,
                "expected_intent": expected_intent,
                "prediction": payload,
                "raw": decoded[:500],
                "error": error,
            }
        )

    passed = sum(1 for row in results if row["ok"])
    failed = len(results) - passed
    return {
        "status": "complete" if failed == 0 else "needs_review",
        "base_model": base_model,
        "adapter_id": adapter_id,
        "case_count": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_id: str = "teckedd/gha-understand-twi-medical-plus-language-v3",
    limit: int = 0,
) -> None:
    print(
        json.dumps(
            evaluate.remote(base_model=base_model, adapter_id=adapter_id, limit=limit),
            ensure_ascii=False,
            indent=2,
        )
    )
