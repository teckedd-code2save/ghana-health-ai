"""Compare the AfriHealth bilingual response adapter with its untouched base.

The evaluator is independent of a proprietary LLM judge. It uses the locked
upstream validation split, deterministic product fixtures, response-language
checks, and repetition/safety-artifact checks.

  modal run modal/train/eval_afrihealth_response.py
  modal run modal/train/eval_afrihealth_response.py --heldout-limit 32
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

import modal


app = modal.App("ghana-health-afrihealth-response-eval")
volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_DATA_DIR = os.path.join(_REPO_ROOT, "data", "medical-response-corpus")
_EVAL_FILE = "afrihealth-ghana-response-eval.v1.jsonl"
_PRODUCT_FILE = "response-product-eval.v1.jsonl"
_REMOTE_DATA_DIR = "/root/afrihealth-response-eval"
_DEFAULT_BASE = "ghananlpcommunity/MiniCPM5-1B-Twi"
_DEFAULT_ADAPTER = (
    "/data/sft/ghananlpcommunity_MiniCPM5-1B-Twi_"
    "afrihealth_bilingual_response_v1_full"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "peft==0.18.0",
        "sacrebleu==2.5.1",
        "safetensors==0.6.2",
    )
    .add_local_file(
        os.path.join(_LOCAL_DATA_DIR, _EVAL_FILE),
        os.path.join(_REMOTE_DATA_DIR, _EVAL_FILE),
    )
    .add_local_file(
        os.path.join(_LOCAL_DATA_DIR, _PRODUCT_FILE),
        os.path.join(_REMOTE_DATA_DIR, _PRODUCT_FILE),
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    per_language = max(1, limit // 2)
    selected: list[dict[str, Any]] = []
    for language in ("tw", "en"):
        language_rows = sorted(
            (row for row in rows if row["language"] == language),
            key=lambda row: row["record_source_hash"],
        )
        selected.extend(language_rows[:per_language])
    return selected[:limit]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wɛɔƐƆ]+", text.casefold(), flags=re.UNICODE)


def _rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, target in enumerate(expected, start=1):
            if token == target:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    overlap = previous[-1]
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _repeated_ngram_ratio(text: str, size: int = 3) -> float:
    tokens = _tokens(text)
    if len(tokens) < size:
        return 0.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


_TWI_MARKERS = {
    "abofra", "aduro", "anaa", "ani", "asram", "ayaresabea", "dɛn", "ɛhe",
    "fa", "home", "ho", "hyew", "kɔ", "ma", "mfe", "na", "ne", "nni",
    "no", "nti", "nyinsɛn", "paa", "sɛ", "ti", "wɔ", "yɛ", "ɔsram",
}
_ENGLISH_MARKERS = {
    "and", "are", "baby", "because", "child", "doctor", "for", "have", "health",
    "hospital", "if", "immediately", "is", "of", "should", "the", "this", "to",
    "urgent", "with", "you", "your",
}


def _language_matches(text: str, expected: str) -> bool:
    tokens = _tokens(text)
    twi = sum(token in _TWI_MARKERS for token in tokens)
    english = sum(token in _ENGLISH_MARKERS for token in tokens)
    if expected == "tw":
        return twi >= max(2, english)
    return english >= max(2, twi)


def _contains_any(text: str, alternatives: list[str]) -> bool:
    folded = text.casefold()
    return any(term.casefold() in folded for term in alternatives)


def _product_result(fixture: dict[str, Any], prediction: str) -> dict[str, Any]:
    concept_results = [
        _contains_any(prediction, alternatives)
        for alternatives in fixture["expected_concept_groups"]
    ]
    forbidden_hits = [
        term for term in fixture.get("forbidden_terms", []) if term.casefold() in prediction.casefold()
    ]
    language_ok = _language_matches(prediction, fixture["language"])
    repetition_ratio = _repeated_ngram_ratio(prediction)
    ok = (
        len(prediction.strip()) >= 12
        and all(concept_results)
        and not forbidden_hits
        and language_ok
        and repetition_ratio <= 0.12
    )
    return {
        "id": fixture["id"],
        "language": fixture["language"],
        "critical": fixture["critical"],
        "ok": ok,
        "concept_groups_passed": sum(concept_results),
        "concept_groups_total": len(concept_results),
        "language_ok": language_ok,
        "forbidden_hits": forbidden_hits,
        "repeated_trigram_ratio": round(repetition_ratio, 4),
        "prediction": prediction,
    }


def _generate_batches(
    tokenizer: Any,
    model: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
    adapter_enabled: bool,
) -> tuple[list[str], float]:
    import torch

    predictions: list[str] = []
    started = time.perf_counter()
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            input_width = encoded["input_ids"].shape[-1]
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            predictions.extend(
                tokenizer.decode(item[input_width:], skip_special_tokens=True).strip()
                for item in output
            )
            print(
                f"[{'adapter' if adapter_enabled else 'base'}] "
                f"{min(offset + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )
    return predictions, time.perf_counter() - started


def _score_reference_rows(
    rows: list[dict[str, Any]], predictions: list[str], elapsed: float
) -> dict[str, Any]:
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    references = [row["answer"] for row in rows]
    details: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        details.append(
            {
                "id": row["id"],
                "language": row["language"],
                "rouge_l_f1": _rouge_l_f1(prediction, row["answer"]),
                "language_ok": _language_matches(prediction, row["language"]),
                "repeated_trigram_ratio": _repeated_ngram_ratio(prediction),
                "prediction": prediction,
                "reference": row["answer"],
            }
        )

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_predictions = [row["prediction"] for row in selected]
        selected_references = [row["reference"] for row in selected]
        count = len(selected)
        return {
            "count": count,
            "chrf_pp": float(chrf.corpus_score(selected_predictions, [selected_references]).score),
            "mean_rouge_l_f1": sum(row["rouge_l_f1"] for row in selected) / count,
            "language_match_rate": sum(row["language_ok"] for row in selected) / count,
            "mean_repeated_trigram_ratio": sum(
                row["repeated_trigram_ratio"] for row in selected
            ) / count,
        }

    return {
        "overall": summarize(details),
        "by_language": {
            language: summarize([row for row in details if row["language"] == language])
            for language in ("tw", "en")
        },
        "elapsed_seconds": round(elapsed, 3),
        "mean_latency_ms": round(elapsed * 1000 / len(rows), 2),
        "worst_reference_examples": sorted(details, key=lambda row: row["rouge_l_f1"])[:12],
    }


def _score_product_rows(
    fixtures: list[dict[str, Any]], predictions: list[str], elapsed: float
) -> dict[str, Any]:
    results = [
        _product_result(fixture, prediction)
        for fixture, prediction in zip(fixtures, predictions, strict=True)
    ]
    passed = sum(row["ok"] for row in results)
    critical = [row for row in results if row["critical"]]
    critical_passed = sum(row["ok"] for row in critical)
    return {
        "case_count": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "critical_count": len(critical),
        "critical_passed": critical_passed,
        "critical_pass_rate": critical_passed / len(critical),
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={"/data": volume},
    secrets=SECRETS,
)
def evaluate(
    base_model: str = _DEFAULT_BASE,
    adapter_path: str = _DEFAULT_ADAPTER,
    heldout_limit: int = 128,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise RuntimeError(f"Adapter training output is not available: {adapter_path}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    load_kwargs: dict[str, Any] = {"cache_dir": cache}
    if token:
        load_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **load_kwargs,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    heldout = _balanced_rows(
        _read_jsonl(os.path.join(_REMOTE_DATA_DIR, _EVAL_FILE)), heldout_limit
    )
    fixtures = _read_jsonl(os.path.join(_REMOTE_DATA_DIR, _PRODUCT_FILE))
    combined = heldout + fixtures
    comparisons: dict[str, Any] = {}
    for label, enabled in (("base", False), ("adapter", True)):
        predictions, elapsed = _generate_batches(
            tokenizer,
            model,
            combined,
            batch_size=max(1, batch_size),
            max_new_tokens=max_new_tokens,
            adapter_enabled=enabled,
        )
        heldout_predictions = predictions[: len(heldout)]
        product_predictions = predictions[len(heldout) :]
        heldout_elapsed = elapsed * len(heldout) / len(combined)
        product_elapsed = elapsed - heldout_elapsed
        comparisons[label] = {
            "reference": _score_reference_rows(heldout, heldout_predictions, heldout_elapsed),
            "product": _score_product_rows(fixtures, product_predictions, product_elapsed),
        }

    base_scores = comparisons["base"]
    adapter_scores = comparisons["adapter"]
    base_ref = base_scores["reference"]
    adapter_ref = adapter_scores["reference"]
    checks = {
        "overall_reference_improved": (
            adapter_ref["overall"]["chrf_pp"] >= base_ref["overall"]["chrf_pp"] + 2.0
        ),
        "twi_reference_improved": (
            adapter_ref["by_language"]["tw"]["chrf_pp"]
            >= base_ref["by_language"]["tw"]["chrf_pp"] + 2.0
        ),
        "english_reference_not_regressed": (
            adapter_ref["by_language"]["en"]["chrf_pp"]
            >= base_ref["by_language"]["en"]["chrf_pp"] - 1.0
        ),
        "response_language_preserved": (
            adapter_ref["overall"]["language_match_rate"] >= 0.9
        ),
        "repetition_controlled": (
            adapter_ref["overall"]["mean_repeated_trigram_ratio"] <= 0.12
        ),
        "product_cases_improved": (
            adapter_scores["product"]["pass_rate"] > base_scores["product"]["pass_rate"]
        ),
        "critical_product_cases_pass": (
            adapter_scores["product"]["critical_pass_rate"] == 1.0
        ),
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter_path": adapter_path,
        "heldout_rows": len(heldout),
        "heldout_languages": {
            language: sum(row["language"] == language for row in heldout)
            for language in ("tw", "en")
        },
        "product_cases": len(fixtures),
        "comparisons": comparisons,
        "promotion_checks": checks,
        "passes_research_gate": all(checks.values()),
    }
    output_dir = os.path.join(adapter_path, "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "response-eval.v1.json")
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
    volume.commit()
    return {**result, "output_path": output_path}


@app.local_entrypoint()
def main(
    base_model: str = _DEFAULT_BASE,
    adapter_path: str = _DEFAULT_ADAPTER,
    heldout_limit: int = 128,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> None:
    print(
        json.dumps(
            evaluate.remote(
                base_model=base_model,
                adapter_path=adapter_path,
                heldout_limit=heldout_limit,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
