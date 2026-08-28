"""Run the synthetic Twi meaning benchmark on Modal.

This is an internal research job. It reads only the repository's synthetic seed
set, caches public model weights in Modal, and persists small JSON results.

Usage:
  modal run modal/train/benchmark_understanding.py
  modal run modal/train/benchmark_understanding.py --limit 5
  modal volume get ghana-health-understanding-results / tmp/understanding-results
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import modal

app = modal.App("ghana-health-understanding-benchmark")
hf_cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "ghana-health-understanding-results", create_if_missing=True
)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_SEED_PATH = os.path.join(
    _REPO_ROOT, "data", "understanding-benchmark", "seed.v0.jsonl"
)
_REMOTE_SEED_PATH = "/root/benchmark/seed.v0.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "sacremoses==0.1.1",
        "huggingface_hub==0.26.2",
    )
    .add_local_file(_SEED_PATH, _REMOTE_SEED_PATH)
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _load_rows(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _language_id(tokenizer: Any, candidates: tuple[str, ...]) -> tuple[str, int]:
    language_ids = getattr(tokenizer, "lang_code_to_id", {}) or {}
    for code in candidates:
        if code in language_ids:
            return code, int(language_ids[code])

        token_id = tokenizer.convert_tokens_to_ids(code)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return code, int(token_id)
    raise RuntimeError(f"Tokenizer has none of the required language codes: {candidates}")


@app.function(
    image=image,
    gpu="T4",
    timeout=30 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_volume,
    },
    secrets=SECRETS,
)
def benchmark(
    model_id: str = "ninte/twi-en-nllb-v2",
    tokenizer_id: str = "facebook/nllb-200-distilled-600M",
    revision: str = "main",
    tokenizer_revision: str = "main",
    limit: int = 0,
    batch_size: int = 8,
) -> dict[str, Any]:
    import time

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = _load_rows(_REMOTE_SEED_PATH, limit)
    if not rows:
        raise RuntimeError("Benchmark seed is empty")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    api = HfApi(token=token)
    resolved_revision = api.model_info(model_id, revision=revision).sha
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        cache_dir=cache,
        token=token,
        src_lang="twi_Latn",
        tgt_lang="eng_Latn",
    )
    source_code, _ = _language_id(tokenizer, ("twi_Latn", "aka_Latn"))
    target_code, target_id = _language_id(tokenizer, ("eng_Latn",))
    tokenizer.src_lang = source_code

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=resolved_revision,
        cache_dir=cache,
        token=token,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_size = max(1, min(int(batch_size), 32))
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        texts = [str(row["text"]) for row in batch]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        batch_started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        elapsed_ms = round((time.perf_counter() - batch_started) * 1000)
        per_row_ms = round(elapsed_ms / len(batch))
        for row, prediction in zip(batch, decoded, strict=True):
            predictions.append(
                {
                    **row,
                    "prediction": prediction,
                    "latency_ms_approx": per_row_ms,
                }
            )

    with open(_REMOTE_SEED_PATH, "rb") as source:
        seed_sha256 = hashlib.sha256(source.read()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "requested_revision": revision,
        "requested_tokenizer_revision": tokenizer_revision,
        "resolved_revision": resolved_revision,
        "source_language": source_code,
        "target_language": target_code,
        "seed_sha256": seed_sha256,
        "case_count": len(predictions),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": device,
        "review_status": "unverified",
        "predictions": predictions,
    }
    safe_model_id = model_id.replace("/", "--")
    output_dir = f"/results/understanding/{safe_model_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
    results_volume.commit()

    print(json.dumps(result, ensure_ascii=False), flush=True)
    return {
        "status": "complete",
        "output_path": output_path,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "resolved_revision": resolved_revision,
        "case_count": len(predictions),
        "elapsed_seconds": result["elapsed_seconds"],
    }


@app.local_entrypoint()
def main(
    model_id: str = "ninte/twi-en-nllb-v2",
    tokenizer_id: str = "facebook/nllb-200-distilled-600M",
    revision: str = "main",
    tokenizer_revision: str = "main",
    limit: int = 0,
    batch_size: int = 8,
) -> None:
    result = benchmark.remote(
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        limit=limit,
        batch_size=batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
