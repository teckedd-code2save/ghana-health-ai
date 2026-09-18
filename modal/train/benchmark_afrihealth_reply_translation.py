"""Benchmark open English-to-Twi reply translators on stronger references.

The 62 reference rows are dual-teacher annotations rather than human gold. This
benchmark is a candidate-selection gate for corpus proposal generation, not a
claim of production translation quality.

  modal run modal/train/benchmark_afrihealth_reply_translation.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import modal


app = modal.App("ghana-health-afrihealth-reply-translation-benchmark")
hf_cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("ghana-health-understanding-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_ANNOTATION_PATH = os.path.join(
    _REPO_ROOT,
    "data",
    "medical-response-corpus",
    "afrihealth-akan-annotations.v1.jsonl",
)
_REMOTE_ANNOTATION_PATH = "/root/benchmark/afrihealth-akan-annotations.v1.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "sentencepiece==0.2.0",
        "sacrebleu==2.5.1",
        "huggingface_hub==1.3.0",
    )
    .add_local_file(_ANNOTATION_PATH, _REMOTE_ANNOTATION_PATH)
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _load_rows(limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(_REMOTE_ANNOTATION_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            annotation = json.loads(line)
            proposals = list(annotation["proposals"])
            if annotation.get("synthesized_proposal"):
                proposals.append(annotation["synthesized_proposal"])
            selected = next(
                (
                    proposal
                    for proposal in proposals
                    if proposal["proposal_id"] == annotation["recommended_proposal_id"]
                ),
                proposals[0],
            )
            rows.append(
                {
                    "row_id": annotation["row_id"],
                    "english": selected["reply_english"].strip(),
                    "twi": selected["reply_twi"].strip(),
                    "reference_model": selected["model"],
                    "reference_status": annotation["adjudication"]["status"],
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("Reference annotation set is empty")
    return rows


def _target_id(tokenizer: Any) -> int:
    for code in ("twi_Latn", "aka_Latn"):
        language_ids = getattr(tokenizer, "lang_code_to_id", {}) or {}
        if code in language_ids:
            return int(language_ids[code])
        token_id = tokenizer.convert_tokens_to_ids(code)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    raise RuntimeError("Tokenizer does not expose a Twi/Akan language token")


def _translate_nllb(
    rows: list[dict[str, str]], model_id: str, revision: str, cache: str, token: str | None
) -> tuple[list[str], float]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache,
        token=token,
        src_lang="eng_Latn",
        tgt_lang="twi_Latn",
    )
    target_id = _target_id(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    predictions: list[str] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), 8):
        batch = rows[offset : offset + 8]
        encoded = tokenizer(
            [row["english"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=512,
                num_beams=4,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    elapsed = time.perf_counter() - started
    del model, tokenizer
    torch.cuda.empty_cache()
    return predictions, elapsed


def _translate_opani(
    rows: list[dict[str, str]], model_id: str, revision: str, cache: str, token: str | None
) -> tuple[list[str], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache, token=token
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    predictions: list[str] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), 8):
        batch = rows[offset : offset + 8]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Translate this text faithfully to natural Ghanaian Twi. "
                            "Return only the translation, with no explanation:\n" + row["english"]
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")
        input_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        predictions.extend(
            tokenizer.decode(output[input_width:], skip_special_tokens=True).strip()
            for output in generated
        )
    elapsed = time.perf_counter() - started
    del model, tokenizer
    torch.cuda.empty_cache()
    return predictions, elapsed


def _score(
    rows: list[dict[str, str]], predictions: list[str], elapsed: float
) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    references = [row["twi"] for row in rows]
    chrf = CHRF(word_order=2)
    bleu = BLEU(effective_order=True)
    details = []
    for row, prediction in zip(rows, predictions, strict=True):
        details.append(
            {
                **row,
                "prediction": prediction,
                "sentence_chrf_pp": float(chrf.sentence_score(prediction, [row["twi"]]).score),
            }
        )
    return {
        "case_count": len(rows),
        "bleu": float(bleu.corpus_score(predictions, [references]).score),
        "chrf_pp": float(chrf.corpus_score(predictions, [references]).score),
        "elapsed_seconds": round(elapsed, 3),
        "mean_latency_ms": round(elapsed * 1000 / len(rows), 2),
        "predictions": details,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_volume,
    },
    secrets=SECRETS,
)
def benchmark(limit: int = 0) -> dict[str, Any]:
    from huggingface_hub import HfApi

    rows = _load_rows(limit)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    api = HfApi(token=token)
    candidates = [
        {
            "id": "ghananlpcommunity/nllb-600m-eng-twi-merged",
            "kind": "nllb",
            "declared_license": "missing",
        },
        {
            "id": "ghananlpcommunity/opani-translate_1b-merged-16bit",
            "kind": "opani",
            "declared_license": "apache-2.0",
        },
    ]
    results: dict[str, Any] = {}
    for candidate in candidates:
        model_id = candidate["id"]
        revision = api.model_info(model_id).sha
        if candidate["kind"] == "nllb":
            predictions, elapsed = _translate_nllb(rows, model_id, revision, cache, token)
        else:
            predictions, elapsed = _translate_opani(rows, model_id, revision, cache, token)
        results[model_id] = {
            "revision": revision,
            "declared_license": candidate["declared_license"],
            **_score(rows, predictions, elapsed),
        }

    with open(_REMOTE_ANNOTATION_PATH, "rb") as source:
        reference_sha256 = hashlib.sha256(source.read()).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_artifact": "afrihealth-akan-annotations.v1.jsonl",
        "reference_sha256": reference_sha256,
        "reference_status": "dual_teacher_not_human_gold",
        "results": results,
    }
    output_dir = "/results/response-annotations/reply-translation-v1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/latest.json"
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
    results_volume.commit()
    hf_cache.commit()
    return {
        "status": "complete",
        "output_path": output_path,
        "reference_rows": len(rows),
        "candidates": {
            model_id: {
                key: value
                for key, value in result.items()
                if key not in {"predictions"}
            }
            for model_id, result in results.items()
        },
    }


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    print(json.dumps(benchmark.remote(limit=limit), ensure_ascii=False, indent=2))
