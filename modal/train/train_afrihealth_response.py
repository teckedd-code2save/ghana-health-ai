"""Train a balanced Twi/English AfriHealth response adapter on Modal.

Smoke test:
  modal run modal/train/train_afrihealth_response.py --smoke

Full research run (publishing is intentionally separate from training):
  modal run --detach modal/train/train_afrihealth_response.py --max-steps 600
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import modal


app = modal.App("ghana-health-afrihealth-response-train")
volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_DATA_DIR = os.path.join(_REPO_ROOT, "data", "medical-response-corpus")
_REMOTE_DATA_DIR = "/root/afrihealth-response"
_TRAIN_FILE = "afrihealth-ghana-response-train.v1.jsonl"
_EVAL_FILE = "afrihealth-ghana-response-eval.v1.jsonl"
_OUTPUT_SUFFIX = "afrihealth_bilingual_response_v1"
_DATASET_REVISION = "61befdaa19e12afdbf9032a602067f0daa3e6c68"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "peft==0.18.0",
        "huggingface_hub==1.3.0",
        "safetensors==0.6.2",
    )
    .add_local_file(
        os.path.join(_LOCAL_DATA_DIR, _TRAIN_FILE),
        os.path.join(_REMOTE_DATA_DIR, _TRAIN_FILE),
    )
    .add_local_file(
        os.path.join(_LOCAL_DATA_DIR, _EVAL_FILE),
        os.path.join(_REMOTE_DATA_DIR, _EVAL_FILE),
    )
    .add_local_file(
        os.path.join(_TRAIN_DIR, "model_card.py"),
        "/root/gha_train/model_card.py",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_eval(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    per_language = max(1, limit // 2)
    selected: list[dict[str, Any]] = []
    for language in ("tw", "en"):
        candidates = sorted(
            (row for row in rows if row["language"] == language),
            key=lambda row: row["record_source_hash"],
        )
        selected.extend(candidates[:per_language])
    return selected[:limit]


@dataclass
class ResponseCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    question = row["messages"][0]["content"].strip()
    answer = row["messages"][1]["content"].strip()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    available = max(1, max_length - len(prompt_ids) - len(eos))
    answer_ids = answer_ids[:available]
    input_ids = list(prompt_ids) + list(answer_ids) + eos
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + list(answer_ids) + eos,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=12 * 60 * 60,
    volumes={"/data": volume},
    secrets=SECRETS,
)
def train(
    base_model: str = "ghananlpcommunity/MiniCPM5-1B-Twi",
    max_steps: int = 600,
    learning_rate: float = 8e-5,
    max_length: int = 2048,
    eval_limit: int = 256,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    run_kind = "smoke" if smoke else "full"
    output_dir = f"/data/sft/{base_model.replace('/', '_')}_{_OUTPUT_SUFFIX}_{run_kind}"
    train_rows = _read_jsonl(os.path.join(_REMOTE_DATA_DIR, _TRAIN_FILE))
    eval_rows = _balanced_eval(
        _read_jsonl(os.path.join(_REMOTE_DATA_DIR, _EVAL_FILE)),
        eval_limit,
    )
    if smoke:
        max_steps = min(max_steps, 12)
        max_length = min(max_length, 1024)
        train_rows = _balanced_eval(train_rows, 32)
        eval_rows = _balanced_eval(eval_rows, 8)

    language_counts = {
        language: sum(row["language"] == language for row in train_rows)
        for language in ("tw", "en")
    }
    if not smoke and abs(language_counts["tw"] - language_counts["en"]) > 16:
        raise RuntimeError(f"Training mix is not balanced: {language_counts}")
    if any(not row["eligible_for_research_training"] for row in train_rows):
        raise RuntimeError("Training artifact contains an ineligible row")
    if any(row["source_split"] != "validation" for row in eval_rows):
        raise RuntimeError("Evaluation artifact contains a non-validation row")

    load_kwargs: dict[str, Any] = {"cache_dir": cache}
    if token:
        load_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(base_model, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        **load_kwargs,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    class EncodedRows(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return _encode_row(tokenizer, self.rows[index], max_length)

    random.seed(42)
    random.shuffle(train_rows)
    train_dataset = EncodedRows(train_rows)
    eval_dataset = EncodedRows(eval_rows)
    arguments = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(max_steps * 0.05)),
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=max(50, max_steps // 4),
        eval_strategy="steps",
        eval_steps=max(50, max_steps // 4),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ResponseCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "run_kind": run_kind,
                "base_model": base_model,
                "source_dataset": "ImhotepSystems/AfriHealth-QA",
                "source_revision": _DATASET_REVISION,
                "source_license": "cc-by-sa-4.0",
                "train_rows": len(train_rows),
                "train_languages": language_counts,
                "training_eval_rows": len(eval_rows),
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "metrics": metrics,
                "promotion_status": "research_only_pending_semantic_and_safety_evaluation",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    volume.commit()

    hub_status = "not_requested"
    if push_repo and not smoke:
        if not token:
            hub_status = "missing_hf_token"
        else:
            model.push_to_hub(push_repo, token=token)
            tokenizer.push_to_hub(push_repo, token=token)
            import sys

            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            write_and_push_model_card(
                push_repo,
                task="text-generation",
                language=["tw", "ak", "en"],
                base_model=base_model,
                datasets=["ImhotepSystems/AfriHealth-QA"],
                metrics={
                    "source_train_rows": 8809,
                    "twi_train_rows": 4407,
                    "english_train_rows": 4402,
                    "locked_eval_rows": 2198,
                    "training_eval_rows": len(eval_rows),
                    "max_steps": max_steps,
                    **{
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and key in {"train_loss", "eval_loss"}
                    },
                },
                summary=(
                    "Balanced Twi and Ghanaian-English medical response LoRA for research. "
                    "It learns direct question-to-answer behavior without machine-translated labels."
                ),
                extra_markdown=(
                    "## Dataset and verification status\n\n"
                    "- Source revision: `" + _DATASET_REVISION + "`\n"
                    "- Training rows: 8,809 (4,407 Twi; 4,402 Ghanaian English).\n"
                    "- Locked source validation: 2,198 rows.\n"
                    "- 109 structurally suspicious rows were isolated before training.\n"
                    "- Source answers are described upstream as clinical consensus, but this "
                    "project has not medically reviewed every row.\n"
                    "- This checkpoint is not approved for production medical advice.\n\n"
                    "The balanced English replay is intentional because previous Twi-only "
                    "fine-tunes regressed English behavior. Semantic, safety, repetition, and "
                    "product-fixture evaluations are required before any deployment.\n"
                ),
                license_id="cc-by-sa-4.0",
                tags=["lora", "sft", "twi", "akan", "ghana", "health", "research"],
                pipeline_tag="text-generation",
                intended_use=[
                    "Research on direct Twi and Ghanaian-English health response generation.",
                    "Evaluation of bilingual retention and Ghanaian health language behavior.",
                ],
                out_of_scope=[
                    "Clinical diagnosis, autonomous triage, or unsupervised patient care.",
                    "Production routing before the published semantic and safety gates pass.",
                ],
                token=token,
            )
            hub_status = f"pushed:{push_repo}+card"

    return {
        "status": "ok",
        "base_model": base_model,
        "output_dir": output_dir,
        "source_revision": _DATASET_REVISION,
        "n_train": len(train_rows),
        "train_languages": language_counts,
        "n_eval": len(eval_rows),
        "max_steps": max_steps,
        "metrics": metrics,
        "hub": hub_status,
    }


@app.local_entrypoint()
def main(
    base_model: str = "ghananlpcommunity/MiniCPM5-1B-Twi",
    max_steps: int = 600,
    learning_rate: float = 8e-5,
    max_length: int = 2048,
    eval_limit: int = 256,
    push_repo: str = "",
    smoke: bool = False,
    detach: bool = False,
) -> None:
    kwargs = {
        "base_model": base_model,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "eval_limit": eval_limit,
        "push_repo": push_repo or None,
        "smoke": smoke,
    }
    if detach:
        call = train.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(train.remote(**kwargs), ensure_ascii=False, indent=2))
