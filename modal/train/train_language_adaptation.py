"""Bounded private Gemma language adaptation; no production or Hub promotion.

modal run modal/train/train_language_adaptation.py --preflight-only
modal run --detach modal/train/train_language_adaptation.py --max-steps 960
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("ghana-general-language-adaptation")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0", "sacrebleu==2.5.1")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
    .run_commands("python -c 'from transformers import Gemma4ForCausalLM, Trainer; from peft import get_peft_model'")
)
if modal.is_local():
    for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
        image = image.add_local_file(str(ROOT / "tmp/general-language-corpus/adaptation-v1" / name), "/inputs/" + name)
    image = image.add_local_file(str(Path(__file__).with_name("language_adaptation_core.py")), "/root/language_adaptation_core.py")
    image = image.add_local_file(str(Path(__file__).with_name("medical_pilot_core.py")), "/root/medical_pilot_core.py")
    image = image.add_local_file(str(ROOT / "tmp/general-foundation-comparison/cases-81a7aad880d9.json"), "/inputs/product.json")


def prepare():
    import sys
    sys.path.insert(0, "/root")
    from language_adaptation_core import load_inputs, tokenize_inputs
    from transformers import AutoTokenizer
    manifest, rows = load_inputs(Path("/inputs"))
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"], revision=manifest["base_revision"])
    encoded, accepted, report = tokenize_inputs(tokenizer, rows)
    if len(encoded["train"]) < 12000 or report["examples"].get("train:translation:tw", 0) < 5000:
        raise ValueError("Unexpected corpus collapse; do not spend GPU credits")
    return manifest, tokenizer, encoded, accepted, report


def training_arguments(output_dir: str, steps: int, cpu: bool = False):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir, max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1 if cpu else 16,
        learning_rate=2e-5, warmup_steps=max(1, round(steps * 0.05)),
        lr_scheduler_type="cosine", weight_decay=0.01,
        bf16=not cpu, use_cpu=cpu, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_steps=50, save_total_limit=2, save_only_model=True,
        report_to="none", remove_unused_columns=False, seed=42, data_seed=42,
        per_device_eval_batch_size=1, dataloader_num_workers=0,
    )


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/cache": cache})
def preflight():
    import tempfile
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, Trainer
    manifest, tokenizer, encoded, rows, report = prepare()
    from medical_pilot_core import PilotCollator
    # Check this exact runtime's optimizer API and backward pass without a GPU.
    config = Gemma4TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_global_key_value_heads=1, head_dim=16, global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0, use_cache=False)
    model = get_peft_model(Gemma4ForCausalLM(config), LoraConfig(r=4, lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    with tempfile.TemporaryDirectory() as folder:
        trainer = Trainer(model=model, args=training_arguments(folder, 1, cpu=True),
            train_dataset=[{"input_ids": [11, 12, 13, 21, 22, 106], "attention_mask": [1] * 6,
                            "labels": [-100, -100, -100, 21, 22, 106]}],
            data_collator=PilotCollator(0))
        report["toy_runtime_smoke"] = trainer.train().metrics
    report["sample_serialized"] = tokenizer.decode(encoded["train"][0]["input_ids"], skip_special_tokens=False)
    report["sample_target"] = tokenizer.decode([t for t in encoded["train"][0]["labels"] if t != -100], skip_special_tokens=False)
    report["base_model"] = manifest["base_model"]
    cache.commit()
    return report


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def train(max_steps: int = 960):
    import hashlib
    import time
    from collections import Counter
    from datetime import datetime, timezone
    import torch
    from peft import LoraConfig, get_peft_model
    from sacrebleu.metrics import CHRF
    from transformers import Gemma4ForCausalLM, Trainer, TrainerCallback, set_seed

    started = time.monotonic()
    if not 1 <= max_steps <= 960:
        raise ValueError("At most 960 optimizer steps are authorized by this runner")
    manifest, tokenizer, encoded, accepted, token_report = prepare()
    from medical_pilot_core import PilotCollator
    run_id = "gemma31_language_v1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path("/artifacts/language-adaptation", run_id)
    destination.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        artifacts.commit()

    save("manifest.json", manifest)
    save("tokenization.json", token_report)
    save("status.json", {"stage": "loading", "run_id": run_id})
    print(json.dumps({"run_id": run_id, "stage": "loading", "examples": token_report["examples"], "supervised_tokens": token_report["supervised_tokens"]}), flush=True)
    set_seed(42)
    base, info = Gemma4ForCausalLM.from_pretrained(manifest["base_model"], revision=manifest["base_revision"],
                                                 dtype=torch.bfloat16, device_map={"": "cuda"},
                                                 attn_implementation="sdpa", output_loading_info=True,
                                                 key_mapping={r"^model\.language_model\.": "model."})
    save("loading_info.json", {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()})
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Base weights were not loaded completely; refusing to train random tensors")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.peft_config["default"].revision = manifest["base_revision"]
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    max_steps = min(max_steps, (len(encoded["train"]) + 15) // 16)
    config = {"seed": 42, "max_steps": max_steps, "max_length": 896, "lora_rank": 16,
              "lora_alpha": 32, "learning_rate": 2e-5, "effective_batch_size": 16,
              "trainable_parameters": trainable, "total_parameters": total,
              "gpu": torch.cuda.get_device_name(), "gradient_checkpointing": True,
              "wall_time_limit_seconds": 7200, "training_stop_elapsed_seconds": 5700,
              "loss_mask": "final assistant and end-of-turn only", "production_promoted": False}
    save("training_config.json", config)

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            print(json.dumps({"step": state.global_step, **(logs or {}), "elapsed_s": round(time.monotonic() - started)}), flush=True)
        def on_save(self, args, state, control, **kwargs):
            save("status.json", {"stage": "training", "step": state.global_step, "run_id": run_id})
        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - started > 5700:
                control.should_training_stop = True
                control.should_save = True
            return control

    trainer = Trainer(model=model, args=training_arguments(str(destination / "checkpoints"), max_steps),
                      train_dataset=encoded["train"], data_collator=PilotCollator(tokenizer.pad_token_id), callbacks=[Progress()])

    subsets = {}
    for task in (("translation", "tw"), ("translation", "en"), ("general_conversation_replay", "en")):
        indexes = [i for i, r in enumerate(accepted["validation"]) if (r["task"], r["language"]) == task]
        indexes.sort(key=lambda i: accepted["validation"][i]["source_record_hash"])
        subsets[":".join(task)] = indexes[:32]

    def losses():
        return {name: trainer.evaluate(eval_dataset=[encoded["validation"][i] for i in indexes])["eval_loss"] for name, indexes in subsets.items()}

    def generations(label):
        cases = []
        for name, indexes in subsets.items():
            if not name.startswith("translation"):
                continue
            for i in indexes[:16]:
                r = accepted["validation"][i]
                cases.append({"id": r["id"], "task": name, "messages": r["messages"][:-1], "reference": r["messages"][-1]["content"]})
        cases += [{**r, "task": "product"} for r in json.loads(Path("/inputs/product.json").read_text())]
        tokenizer.padding_side = "left"
        model.eval()
        model.config.use_cache = True
        results = []
        for start in range(0, len(cases), 2):
            batch = cases[start:start + 2]
            prompts = []
            for row in batch:
                messages = [dict(m) for m in row["messages"]]
                if row["task"] == "product" and messages[0]["role"] == "system":
                    messages[0]["content"] += " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
                prompts.append(tokenizer.apply_chat_template(messages, tools=row.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=False))
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=384, do_sample=False, eos_token_id=[1, 106, 50], pad_token_id=tokenizer.pad_token_id)
            for row, ids in zip(batch, output[:, inputs["input_ids"].shape[1]:], strict=True):
                raw = ids.tolist()
                results.append({**row, "prediction": tokenizer.decode(raw, skip_special_tokens=True).strip(),
                                "raw_output": tokenizer.decode(raw, skip_special_tokens=False),
                                "token_ids": raw, "limit_reached": not any(i in (1, 106, 50) for i in raw)})
            print(f"Generation checks {len(results)}/{len(cases)}", flush=True)
            if len(results) % 8 == 0 or len(results) == len(cases):
                save(f"{label}_generations.partial.json", results)
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        return results

    save("status.json", {"stage": "baseline_evaluation", "run_id": run_id})
    baseline_loss = losses()
    save("baseline_loss.json", baseline_loss)
    save("status.json", {"stage": "training", "run_id": run_id})
    try:
        result = trainer.train()
    except Exception as exc:
        save("status.json", {"stage": "failed_training", "run_id": run_id,
                              "error_type": type(exc).__name__, "step": trainer.state.global_step})
        raise
    model.save_pretrained(destination / "adapter")
    tokenizer.save_pretrained(destination / "adapter")
    save("training_metrics.json", {**result.metrics, "log_history": trainer.state.log_history})
    save("status.json", {"stage": "adapter_evaluation", "run_id": run_id})
    adapted_loss = losses()
    model.gradient_checkpointing_disable()
    # Base weights are frozen: disabling LoRA gives the unchanged baseline.
    # Generate it after checkpointing training so a lost evaluation cannot erase
    # all useful work from a short-lived/preempted GPU session.
    with model.disable_adapter():
        baseline = generations("baseline")
    save("baseline_generations.json", baseline)
    adapted = generations("adapter")
    save("adapter_generations.json", adapted)
    chrf = CHRF(word_order=2)
    metrics = {}
    for label, rows in (("base", baseline), ("adapter", adapted)):
        metrics[label] = {}
        for direction in ("translation:en", "translation:tw"):
            subset = [r for r in rows if r["task"] == direction]
            metrics[label][direction] = {"count": len(subset), "chrf_pp": chrf.corpus_score([r["prediction"] for r in subset], [[r["reference"] for r in subset]]).score,
                                          "limit_reached": sum(r["limit_reached"] for r in subset)}
    adapter_sha = hashlib.sha256((destination / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    report = {"run_id": run_id, "path": str(destination), "adapter_sha256": adapter_sha,
              "steps_completed": trainer.state.global_step, "baseline_loss": baseline_loss, "adapter_loss": adapted_loss,
              "translation_similarity": metrics, "elapsed_seconds": round(time.monotonic() - started, 2),
              "production_promoted": False, "human_validated": False,
              "interpretation": "Held-out loss and reference similarity are diagnostics, not native or clinical accuracy. Inspect raw product outputs before serving."}
    save("evaluation.json", report)
    card = "---\nlibrary_name: peft\nbase_model: google/gemma-4-31B-it\nlanguage:\n- tw\n- en\npipeline_tag: text-generation\n---\n\n# Private Twi-English Language Adaptation\n\nExperimental stage-one LoRA. Not a medical assistant or production release.\n\n"
    card += "## Data and limitations\n\n" + "\n".join("- " + item for item in manifest["limitations"])
    card += "\n\n## Source attribution\n\nGhana-NLP/ENGLISH_TWI_PARALLEL_TEXT, Ghana NLP Community parallel text, and OpenAssistant contributors. Source research restrictions remain; no commercial clearance or public redistribution license is asserted.\n\n"
    card += "## Recipe\n\n```json\n" + json.dumps(config, indent=2) + "\n```\n\n## Measured results\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n\nNo proprietary fallback; no synthetic answers added to source transcripts. Full provenance hashes, token counts, pre/post generations, and exclusions are retained in the private run artifacts. No clinical, tool-execution, or speech quality validation is implied.\n"
    (destination / "adapter/README.md").write_text(card)
    save("status.json", {"stage": "completed", "run_id": run_id})
    return report


@app.local_entrypoint()
def main(preflight_only: bool = False, max_steps: int = 960):
    report = preflight.remote() if preflight_only else train.remote(max_steps=max_steps)
    output = ROOT / "tmp/general-language-corpus" / ("preflight.json" if preflight_only else report["run_id"] + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rejected"}, ensure_ascii=False, indent=2))
