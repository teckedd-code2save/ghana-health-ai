"""Benchmark independent Twi-to-English translation proposals on Modal.

Run a small gate before using either model for corpus annotation:
  modal run modal/train/benchmark_afrihealth_translation.py --limit 4

Retrieve the latest result:
  modal volume get ghana-health-understanding-results \
    /response-annotations/translation-v1/latest.raw.jsonl \
    tmp/understanding-corpus/afrihealth-translation-v1.raw.jsonl
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import modal


app = modal.App("ghana-health-afrihealth-translation-benchmark")
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
_PROMPT_VERSION = "afrihealth-translation-benchmark-v3"

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


def _load_rows(offset: int, limit: int) -> list[dict[str, Any]]:
    with open(_REMOTE_SOURCE_PATH, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row["record_source_hash"])
    selected = rows[max(0, int(offset)) :]
    return selected[:limit] if limit > 0 else selected


def _load_model(model_id: str, token: str | None, cache: str) -> tuple[Any, Any]:
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


def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [output.strip() for output in outputs]


def _aya_prompt(text: str) -> str:
    return f"Translate from Twi to English. Return only the English translation.\nTwi: {text}\nEnglish:"


def _chunk_text(text: str, max_characters: int = 360) -> list[str]:
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        while words:
            room = max_characters - len(current) - (1 if current else 0)
            candidate_words: list[str] = []
            candidate_length = 0
            while words and candidate_length + len(words[0]) + (1 if candidate_words else 0) <= max(1, room):
                word = words.pop(0)
                candidate_words.append(word)
                candidate_length += len(word) + (1 if candidate_words[:-1] else 0)
            if candidate_words:
                part = " ".join(candidate_words)
                current = f"{current} {part}".strip()
            if words or len(current) >= max_characters:
                if current:
                    chunks.append(current)
                current = ""
            if not candidate_words and words:
                chunks.append(words.pop(0))
    if current:
        chunks.append(current)
    return chunks or [text]


def _generate_nllb(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
) -> list[str]:
    import torch

    tokenizer.src_lang = "twi_Latn"
    target_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    segmented = [_chunk_text(text) for text in texts]
    flat = [chunk for chunks in segmented for chunk in chunks]
    translated: list[str] = []
    for start in range(0, len(flat), batch_size):
        encoded = tokenizer(
            flat[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_token_id,
                num_beams=4,
                max_length=256,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    outputs: list[str] = []
    cursor = 0
    for chunks in segmented:
        count = len(chunks)
        outputs.append(" ".join(part.strip() for part in translated[cursor : cursor + count]).strip())
        cursor += count
    return outputs


def _score_quality(
    model: Any,
    tokenizer: Any,
    twi_texts: list[str],
    english_texts: list[str],
    batch_size: int,
) -> list[float]:
    import torch

    scores: list[float] = []
    pairs = [
        f"query: {twi} passage: {english}"
        for twi, english in zip(twi_texts, english_texts, strict=True)
    ]
    for start in range(0, len(pairs), batch_size):
        encoded = tokenizer(
            pairs[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to("cuda")
        with torch.inference_mode():
            probabilities = model(**encoded).logits.softmax(dim=-1)[:, 1]
        scores.extend(float(value) for value in probabilities.cpu())
    return scores


def _proposal(
    model: str,
    model_license: str,
    usage_scope: str,
    question: str,
    answer: str,
    question_quality_probability: float,
    answer_quality_probability: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "model_license": model_license,
        "usage_scope": usage_scope,
        "question_english": question,
        "answer_english": answer,
        "question_quality_probability": question_quality_probability,
        "answer_quality_probability": answer_quality_probability,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_volume},
    secrets=SECRETS,
)
def benchmark(
    offset: int = 0,
    limit: int = 4,
    batch_size: int = 1,
    aya_model: str = "CohereLabs/aya-101",
    comparison_model: str = "ninte/twi-en-nllb-v2",
    quality_model: str = "ghananlpcommunity/twi-eng-qe-e5",
) -> dict[str, Any]:
    import torch

    rows = _load_rows(offset, limit)
    if not rows:
        raise RuntimeError("No source rows selected")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)

    aya, aya_tokenizer = _load_model(aya_model, token, cache)
    aya_questions = _generate(
        aya,
        aya_tokenizer,
        [_aya_prompt(row["question_twi_source"]) for row in rows],
        max_input_tokens=1024,
        max_new_tokens=320,
        batch_size=max(1, batch_size),
    )
    aya_answers = _generate(
        aya,
        aya_tokenizer,
        [_aya_prompt(row["answer_twi_source"]) for row in rows],
        max_input_tokens=4096,
        max_new_tokens=1600,
        batch_size=max(1, batch_size),
    )
    del aya, aya_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    comparison, comparison_tokenizer = _load_model(comparison_model, token, cache)
    comparison_questions = _generate_nllb(
        comparison,
        comparison_tokenizer,
        [row["question_twi_source"] for row in rows],
        max(1, batch_size),
    )
    comparison_answers = _generate_nllb(
        comparison,
        comparison_tokenizer,
        [row["answer_twi_source"] for row in rows],
        max(1, batch_size),
    )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    quality_tokenizer = AutoTokenizer.from_pretrained(quality_model, token=token, cache_dir=cache)
    quality = AutoModelForSequenceClassification.from_pretrained(
        quality_model,
        token=token,
        cache_dir=cache,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    quality.eval()
    source_questions = [row["question_twi_source"] for row in rows]
    source_answers = [row["answer_twi_source"] for row in rows]
    aya_question_quality = _score_quality(
        quality,
        quality_tokenizer,
        source_questions,
        aya_questions,
        max(1, batch_size * 4),
    )
    aya_answer_quality = _score_quality(
        quality,
        quality_tokenizer,
        source_answers,
        aya_answers,
        max(1, batch_size * 4),
    )
    comparison_question_quality = _score_quality(
        quality,
        quality_tokenizer,
        source_questions,
        comparison_questions,
        max(1, batch_size * 4),
    )
    comparison_answer_quality = _score_quality(
        quality,
        quality_tokenizer,
        source_answers,
        comparison_answers,
        max(1, batch_size * 4),
    )

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
                "aya": _proposal(
                    aya_model,
                    "apache-2.0",
                    "commercial_candidate",
                    aya_questions[index],
                    aya_answers[index],
                    aya_question_quality[index],
                    aya_answer_quality[index],
                ),
                "comparison": _proposal(
                    comparison_model,
                    "cc-by-nc-4.0",
                    "research_only",
                    comparison_questions[index],
                    comparison_answers[index],
                    comparison_question_quality[index],
                    comparison_answer_quality[index],
                ),
            }
        )

    output_dir = "/results/response-annotations/translation-v1"
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"{output_dir}/translations-{offset}-{len(rows)}-{run_id}.raw.jsonl"
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
        "aya_model": aya_model,
        "comparison_model": comparison_model,
        "comparison_usage_scope": "research_only",
        "quality_model": quality_model,
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
    limit: int = 4,
    batch_size: int = 1,
    detach: bool = False,
) -> None:
    kwargs = {"offset": offset, "limit": limit, "batch_size": batch_size}
    if detach:
        call = benchmark.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(benchmark.remote(**kwargs), ensure_ascii=False, indent=2))
