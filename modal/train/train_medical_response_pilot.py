"""One-GPU, one-hour maximum pilot; no hosted teacher, deployment, or Hub push.

modal run --detach modal/train/train_medical_response_pilot.py --acknowledge-uncalibrated
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("ghana-health-annotated-response-pilot")
volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==5.3.0", "accelerate==1.10.1", "peft==0.18.0", "huggingface_hub==1.3.0", "safetensors==0.6.2", "sacrebleu==2.5.1")
)
if modal.is_local():
    image = (
        image.add_local_dir(str(ROOT / "tmp/medical-response-pilot/v1"), "/root/pilot")
        .add_local_file(str(Path(__file__).with_name("medical_pilot_core.py")), "/root/medical_pilot_core.py")
        .add_local_file(str(Path(__file__).with_name("model_card.py")), "/root/model_card.py")
        .add_local_file(str(ROOT / "data/medical-response-corpus/afrihealth-ghana-response-eval.v1.jsonl"), "/root/locked.jsonl")
        .add_local_file(str(ROOT / "data/medical-response-corpus/response-product-eval.v1.jsonl"), "/root/product.jsonl")
    )


def read_rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@app.function(image=image, gpu="A100-40GB", cpu=2, memory=16384, timeout=3600, retries=0, max_containers=1, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-token")])
def run_pilot(acknowledge_uncalibrated: bool = False, epochs: int = 3) -> dict[str, Any]:
    if not acknowledge_uncalibrated or epochs not in (1, 2, 3):
        raise ValueError("Explicit experimental acknowledgement and 1-3 epochs required")
    import re
    import sys
    from collections import Counter
    from contextlib import nullcontext
    from datetime import datetime, timezone

    import torch
    from peft import LoraConfig, get_peft_model
    from sacrebleu.metrics import CHRF
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    sys.path.insert(0, "/root")
    from medical_pilot_core import PilotCollator, check_pilot_files, encode_row, parsed_interpretation
    from model_card import build_model_card_md

    started = time.monotonic()
    manifest = json.loads(Path("/root/pilot/manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        actual = hashlib.sha256(Path("/root/pilot", artifact["file"]).read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact['file']}")
    train = read_rows("/root/pilot/train.jsonl")
    holdout = read_rows("/root/pilot/holdout.jsonl")
    check_pilot_files(manifest, train, holdout)
    run_id = "annotated_response_pilot_v1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("/data/sft", run_id)
    output.mkdir(parents=True, exist_ok=False)

    def save(name: str, value: Any) -> None:
        Path(output, name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        volume.commit()

    save("corpus_manifest.json", manifest)
    save("run_status.json", {"stage": "loading", "run_id": run_id})
    print(json.dumps({"run_id": run_id, "output_dir": str(output), "train_sources": manifest["training_sources"], "train_examples": len(train)}), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {"cache_dir": "/data/hf", "revision": manifest["base_revision"], "token": token}
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_row(tokenizer, row, 2048) for row in train]
    lengths = sorted(len(row["input_ids"]) for row in encoded)
    save("tokenization.json", {"count": len(lengths), "max_tokens": max(lengths), "median_tokens": lengths[len(lengths) // 2], "truncated_targets": 0})
    base = AutoModelForCausalLM.from_pretrained(manifest["base_model"], dtype=torch.bfloat16, **kwargs).to("cuda")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.peft_config["default"].revision = manifest["base_revision"]
    trainable, total = model.get_nb_trainable_parameters()
    config = {"seed": 42, "epochs": epochs, "learning_rate": 5e-5, "lora_rank": 16, "lora_alpha": 32, "max_length": 2048, "trainable_parameters": trainable, "total_parameters": total, "effective_batch_size": 8, "timeout_seconds": 3600, "gpu": "A100-40GB"}
    save("training_config.json", config)
    model.config.use_cache = False
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=5e-5, warmup_ratio=0.1, lr_scheduler_type="cosine", weight_decay=0.01,
        bf16=True, gradient_checkpointing=True, logging_steps=5, save_strategy="epoch",
        save_total_limit=1, report_to="none", remove_unused_columns=False,
        seed=42, data_seed=42,
    ), train_dataset=encoded, data_collator=PilotCollator(tokenizer.pad_token_id))
    save("run_status.json", {"stage": "training", "run_id": run_id})
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    save("training_metrics.json", {**result.metrics, "log_history": trainer.state.log_history})
    save("run_status.json", {"stage": "evaluating", "run_id": run_id})
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"
    chrf = CHRF(word_order=2)

    def normalized(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold()))

    def repeated(value: str) -> float:
        words = normalized(value).split()
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        return sum(v-1 for v in Counter(grams).values()) / max(1, len(grams))

    def generate(rows: list[dict[str, Any]], enabled: bool, structured: bool) -> list[dict[str, Any]]:
        outputs = []
        context = nullcontext() if enabled else model.disable_adapter()
        with context:
            for index in range(0, len(rows), 4):
                batch = rows[index:index+4]
                prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                set_seed(42 + index)
                before = time.monotonic()
                # N-gram blocking corrupts repeated JSON keys/arrays, so it is free-text only.
                decoding = {"do_sample": False, "repetition_penalty": 1.05, "no_repeat_ngram_size": 0} if structured else {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=1024 if structured else 256, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **decoding)
                elapsed = time.monotonic() - before
                for row, ids in zip(batch, generated[:, inputs["input_ids"].shape[1]:], strict=True):
                    prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    outputs.append({**row, "prediction": prediction, "batch_seconds": elapsed, "batch_size": len(batch), "token_limit_reached": len(ids) >= (1024 if structured else 256) and tokenizer.eos_token_id not in ids.tolist()})
                print(f"eval {'adapter' if enabled else 'base'} {'structured' if structured else 'reply'} {len(outputs)}/{len(rows)}", flush=True)
        return outputs

    structured_rows = [{"id": r["id"], "prompt": r["messages"][:-1], "reference": r["reference"]} for r in holdout if r["task"] == "interpret_and_reply"]
    reply_rows = [{"id": r["id"], "set": "pilot_holdout", "language": r["language"], "prompt": r["messages"][:-1], "reference": r["messages"][-1]["content"]} for r in holdout if r["task"] == "direct_reply"]
    locked = read_rows("/root/locked.jsonl")
    for language in ("tw", "en"):
        for row in sorted((r for r in locked if r["language"] == language), key=lambda r: r["record_source_hash"])[:16]:
            reply_rows.append({"id": row["id"], "set": "locked_source", "language": language, "prompt": [{"role": "user", "content": row["question"]}], "reference": row["answer"]})
    fixtures = read_rows("/root/product.jsonl")
    reply_rows += [{"id": row["id"], "set": "product", "language": row["language"], "prompt": row["messages"], "fixture": row} for row in fixtures]
    comparisons = {}
    for label, enabled in (("base", False), ("adapter", True)):
        raw_structured = generate(structured_rows, enabled, True)
        save(f"{label}_structured_predictions.json", raw_structured)
        parsed = [parsed_interpretation(row["prediction"]) for row in raw_structured]
        count = len(parsed)
        valid = sum(row is not None for row in parsed)
        grounded = sum(p is not None and all(f" {normalized(v)} " in f" {normalized(raw['prompt'][-1]['content'])} " for values in p["entities"].values() for v in values) for raw, p in zip(raw_structured, parsed))
        structured_scores = {
            "count": count, "valid_json_schema": valid, "schema_rate": valid / count,
            "intent_agreement": sum(p is not None and p["intent"] == raw["reference"]["intent"] for raw, p in zip(raw_structured, parsed)) / count,
            "grounded_entities_rate": grounded / count,
            "meaning_chrf_pp": chrf.corpus_score([p["natural_english"] if p else "" for p in parsed], [[r["reference"]["natural_english"] for r in raw_structured]]).score,
            "reply_chrf_pp": chrf.corpus_score([p["reply"] if p else "" for p in parsed], [[r["reference"]["reply"] for r in raw_structured]]).score,
            "interpretation": "Agreement with uncalibrated model labels; NOT semantic accuracy",
        }
        raw_replies = generate(reply_rows, enabled, False)
        save(f"{label}_reply_predictions.json", raw_replies)
        reply_scores = {}
        for subset in ("pilot_holdout", "locked_source"):
            for language in ("tw", "en"):
                rows = [r for r in raw_replies if r["set"] == subset and r["language"] == language]
                reply_scores[f"{subset}_{language}"] = {"count": len(rows), "chrf_pp": chrf.corpus_score([r["prediction"] for r in rows], [[r["reference"] for r in rows]]).score, "mean_repeated_trigram_ratio": sum(repeated(r["prediction"]) for r in rows) / len(rows)}
        product_results = []
        for row in (r for r in raw_replies if r["set"] == "product"):
            fixture, text = row["fixture"], row["prediction"].casefold()
            hits = [any(term.casefold() in text for term in terms) for terms in fixture["expected_concept_groups"]]
            forbidden = [term for term in fixture.get("forbidden_terms", []) if term.casefold() in text]
            product_results.append({"id": row["id"], "critical": fixture["critical"], "lexical_check_pass": all(hits) and not forbidden, "concept_groups": hits, "forbidden_hits": forbidden})
        comparisons[label] = {"structured": structured_scores, "replies": reply_scores, "product_lexical_proxies": product_results}
        save("comparison.partial.json", comparisons)

    elapsed = time.monotonic() - started
    report = {"run_id": run_id, "base_model": manifest["base_model"], "base_revision": manifest["base_revision"], "adapter_path": str(output), "comparisons": comparisons, "elapsed_seconds": elapsed, "gpu_cost_estimate_usd": elapsed * 0.000583, "cost_note": "GPU-only list-price estimate, not billing; CPU, memory, startup and storage excluded", "production_promoted": False, "human_validated": False, "research_gate": "not_eligible_uncalibrated_pilot", "limitations": manifest["limitations"], "decoding": {"structured": "greedy, repetition_penalty=1.05, no_repeat_ngram_size=0", "free_reply": "seed=42+batch_offset, temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3"}}
    save("evaluation.json", report)
    card = build_model_card_md(repo_id=run_id, task="text-generation", language=["tw", "en"], base_model=manifest["base_model"], datasets=["ImhotepSystems/AfriHealth-QA"], license_id="cc-by-sa-4.0", library_name="peft", summary="Uncalibrated model-consensus pilot for Twi interpretation and direct bilingual replies. NOT a medical assistant or a production release.", intended_use=["Private supervised research inspection of saved model annotations"], out_of_scope=["Medical advice, triage, diagnosis, autonomous commerce, or production deployment", "Claiming model-label agreement is human-verified accuracy"], extra_markdown="## Corpus and provenance\n\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n\n## Recipe\n\n```json\n" + json.dumps(config, indent=2) + "\n```\n\n## Evaluation\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n\nBase weights declare Apache-2.0; source derivatives retain the AfriHealth challenge CC-BY-SA-4.0 attribution. No public redistribution clearance is inferred. Teacher outputs are not human gold. No proprietary model is called at inference time.\n")
    Path(output, "README.md").write_text(card)
    save("run_status.json", {"stage": "completed", "run_id": run_id, "output_dir": str(output)})
    return report


@app.local_entrypoint()
def main(acknowledge_uncalibrated: bool = False, epochs: int = 3) -> None:
    report = run_pilot.remote(acknowledge_uncalibrated=acknowledge_uncalibrated, epochs=epochs)
    destination = ROOT / "tmp/medical-response-pilot/evaluation.v1.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
