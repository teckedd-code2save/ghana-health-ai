"""Durable, bounded private response SFT. Deploy, then submit through the runner."""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

RECIPE = os.environ.get("GHA_RESPONSE_RECIPE", "response-v2")
if RECIPE not in ("response-v2", "semantic-v4", "native-v5"):
    raise ValueError("Unknown bounded response recipe")
V4 = RECIPE == "semantic-v4"
V5 = RECIPE == "native-v5"
APP = "ghana-native-understanding-v5" if V5 else "ghana-semantic-response-v4" if V4 else "ghana-response-adaptation-v2"
INPUT_FOLDER = "native-understanding-v5" if V5 else "semantic-response-v4" if V4 else "response-adaptation-v2"
EXPERIMENT = "native_understanding_v5" if V5 else "grounded_response_v4" if V4 else "balanced_direct_response_v2"
PREFIX = "response_v5_" if V5 else "response_v4_" if V4 else "response_v2_"
LEARNING_RATE = 3e-6 if V4 else 1e-5
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"] if V5 else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
app = modal.App(APP)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0",
).env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "GHA_RESPONSE_RECIPE": RECIPE,
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
        image = image.add_local_file(str(root / "tmp" / INPUT_FOLDER / "corpus" / name), "/inputs/" + name)
    for name in ("response_adaptation_core.py", "medical_pilot_core.py", "adapter_checkpoint.py"):
        image = image.add_local_file(str(Path(__file__).with_name(name)), "/root/" + name)
    image = image.add_local_file(str(root / "tmp/general-foundation-comparison/cases-81a7aad880d9.json"), "/inputs/product.json")


def prepare():
    import sys
    sys.path.insert(0, "/root")
    from response_adaptation_core import load_inputs, tokenize_inputs
    from transformers import AutoTokenizer
    manifest, rows = load_inputs(Path("/inputs"), EXPERIMENT)
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"], revision=manifest["base_revision"], local_files_only=True)
    encoded, accepted, report = tokenize_inputs(tokenizer, rows)
    if len(encoded["train"]) < (5000 if V4 else 6000) or report["examples"].get("train:tool_call:en", 0) < 500:
        raise ValueError("Training mixture collapsed during native tokenization")
    if V5 and (report["examples"].get("train:native_intent_entities:tw", 0) != 2240
               or report["examples"].get("train:auxiliary_translation:en", 0) < 1000):
        raise ValueError("Native understanding supervision was lost")
    return manifest, tokenizer, encoded, accepted, report


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0, volumes={"/cache": cache.read_only()})
def preflight():
    import tempfile
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM, Trainer, TrainingArguments
    manifest, tokenizer, encoded, accepted, report = prepare()
    from adapter_checkpoint import adapter_trainer_class
    Trainer = adapter_trainer_class()
    from medical_pilot_core import PilotCollator
    from response_adaptation_core import encode
    tool = {"type": "function", "function": {"name": "list_items", "description": "List matching items.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    messages = [{"role": "user", "content": "Find onions."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "list_items", "arguments": {"query": "onions"}}}]}]
    call = encode(tokenizer, {"messages": messages, "tools": [tool]})
    messages += [{"role": "tool", "tool_call_id": "call_1", "name": "list_items", "content": "No matching items."},
                 {"role": "assistant", "content": "No matching items were found."}]
    reply = encode(tokenizer, {"messages": messages, "tools": [tool]})
    report["native_call_target"] = tokenizer.decode([i for i in call["labels"] if i != -100])
    report["native_result_reply_target"] = tokenizer.decode([i for i in reply["labels"] if i != -100])
    config = Gemma4TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1, head_dim=16, global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"], hidden_size_per_layer_input=0, use_cache=False)
    model = get_peft_model(Gemma4ForCausalLM(config), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.peft_config["default"].base_model_name_or_path = manifest["base_model"]
    model.peft_config["default"].revision = manifest["base_revision"]
    model.enable_input_require_grads()
    with tempfile.TemporaryDirectory() as folder:
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=folder, use_cpu=True, max_steps=1,
            per_device_train_batch_size=1, report_to="none", save_steps=1, save_only_model=True),
            train_dataset=[{"input_ids": [11, 12, 21, 22, 106], "attention_mask": [1] * 5, "labels": [-100, -100, 21, 22, 106]}],
            data_collator=PilotCollator(0))
        report["toy_backward_pass"] = trainer.train().metrics
        checkpoint = Path(folder) / "checkpoint-1"
        restored = PeftModel.from_pretrained(Gemma4ForCausalLM(config), checkpoint)
        report["toy_checkpoint_roundtrip"] = {"adapter_bytes": (checkpoint / "adapter_model.safetensors").stat().st_size,
            "revision": restored.peft_config["default"].revision, "embeddings_saved": False}
    report["manifest_sha256"] = __import__("hashlib").sha256(Path("/inputs/manifest.json").read_bytes()).hexdigest()
    return report


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=5400, retries=0,
              max_containers=1, volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def train(run_id: str, preflight_manifest_sha: str, max_steps: int = 320):
    import hashlib
    import re
    import time
    from collections import defaultdict
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4ForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    if not re.fullmatch(PREFIX + r"\d{8}T\d{6}Z", run_id) or not 1 <= max_steps <= 320:
        raise ValueError("Invalid bounded response run")
    actual_sha = hashlib.sha256(Path("/inputs/manifest.json").read_bytes()).hexdigest()
    if actual_sha != preflight_manifest_sha:
        raise ValueError("CPU preflight does not match this deployed training input")
    started = time.monotonic()
    destination = Path("/artifacts/response-adaptation", run_id)
    destination.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        artifacts.commit()

    try:
        manifest, tokenizer, encoded, accepted, report = prepare()
        from adapter_checkpoint import adapter_trainer_class, save_adapter
        Trainer = adapter_trainer_class()
        from medical_pilot_core import PilotCollator
        save("manifest.json", manifest)
        save("tokenization.json", report)
        save("status.json", {"stage": "loading", "run_id": run_id})
        set_seed(42)
        base, info = Gemma4ForCausalLM.from_pretrained(manifest["base_model"], revision=manifest["base_revision"],
            dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa", output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."}, local_files_only=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("Base weights did not load completely")
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES))
        model.peft_config["default"].revision = manifest["base_revision"]
        model.enable_input_require_grads()
        model.config.use_cache = False
        save_adapter(model, destination / "initial-adapter")
        artifacts.commit()
        config = {"max_steps": min(max_steps, (len(encoded["train"]) + 15) // 16), "effective_batch": 16,
                  "learning_rate": LEARNING_RATE, "seed": 42, "lora_rank": 16, "lora_alpha": 32, "max_length": 1024,
                  "recipe": RECIPE, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "lora_target_modules": TARGET_MODULES,
                  "initialization": "Fresh LoRA on pinned instruction base; no failed translation adapter or external merge",
                  "time_limit_seconds": 5400, "training_stop_elapsed_seconds": 3900, "production_promoted": False}
        save("training_config.json", config)
        cases = json.loads(Path("/inputs/product.json").read_text())

        @torch.random.fork_rng()
        def generate(label, rows):
            results = []
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            tokenizer.padding_side = "left"
            for offset in range(0, len(rows), 2):
                batch = rows[offset:offset + 2]
                prompts = []
                for row in batch:
                    messages = [dict(m) for m in row["messages"]]
                    if row.get("category") != "heldout_source":
                        messages[0]["content"] += " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
                    prompts.append(tokenizer.apply_chat_template(messages, tools=row.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=False))
                tokens = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output = model.generate(**tokens, max_new_tokens=384, do_sample=False, eos_token_id=[1, 106, 50], pad_token_id=tokenizer.pad_token_id)
                for row, value in zip(batch, output[:, tokens["input_ids"].shape[1]:], strict=True):
                    ids = value.tolist()
                    results.append({**row, "prediction": tokenizer.decode(ids, skip_special_tokens=True),
                                    "raw_output": tokenizer.decode(ids, skip_special_tokens=False),
                                    "limit_reached": not any(i in (1, 106, 50) for i in ids)})
                save(label + ".partial.json", results)
            save(label + ".json", results)
            tokenizer.padding_side = "right"
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            return results

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                print(json.dumps({"step": state.global_step, **(logs or {}), "elapsed": round(time.monotonic() - started)}), flush=True)
            def on_save(self, args, state, control, **kwargs):
                save("status.json", {"stage": "training", "run_id": run_id, "step": state.global_step})
                if V4 or V5:
                    generate(f"checkpoint-{state.global_step}-probe", [row for row in cases if row["id"] in (
                        "budget-tw", "negation-tw", "language-switch", "tw-breathing-difficulty", "tw-hospital-choice", "tool-none-en")])
            def on_step_end(self, args, state, control, **kwargs):
                if time.monotonic() - started > 3900:
                    control.should_training_stop = True
                    control.should_save = True
                return control

        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(destination / "checkpoints"),
            max_steps=config["max_steps"], per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=LEARNING_RATE, warmup_steps=max(1, round(config["max_steps"] * 0.05)), lr_scheduler_type="cosine", weight_decay=0.01,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10, save_steps=50, save_total_limit=3, save_only_model=True,
            report_to="none", remove_unused_columns=False, seed=42, data_seed=42,
            per_device_eval_batch_size=1, dataloader_num_workers=0), train_dataset=encoded["train"],
            data_collator=PilotCollator(tokenizer.pad_token_id), callbacks=[Progress()])
        save("status.json", {"stage": "training", "run_id": run_id, "step": 0})
        result = trainer.train()
        save_adapter(model, destination / "adapter")
        tokenizer.save_pretrained(destination / "adapter")
        save("training_metrics.json", {**result.metrics, "log_history": trainer.state.log_history})
        save("status.json", {"stage": "evaluating", "run_id": run_id, "step": trainer.state.global_step})
        groups = defaultdict(list)
        for row in accepted["validation"]:
            groups[row["task"]].append(row)
        holdout = [{"id": row["id"], "category": "heldout_source", "task": row["task"], "language": row["language"],
                    "messages": row["messages"][:-1], "tools": row.get("tools"), "reference": row["messages"][-1]}
                   for task in sorted(groups) for row in sorted(groups[task], key=lambda r: r["id"])[:6]]
        with model.disable_adapter():
            generate("base", cases + holdout)
        generate("adapter", cases + holdout)
        adapter_sha = hashlib.sha256((destination / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
        summary = {"run_id": run_id, "steps": trainer.state.global_step, "adapter_sha256": adapter_sha,
                   "path": str(destination), "elapsed_seconds": time.monotonic() - started,
                   "product_checks": len(cases), "heldout_source_checks": len(holdout), "production_promoted": False,
                   "decision": "Inspect actual paired outputs; completed training does not establish model quality"}
        card = "---\nlibrary_name: peft\nbase_model: google/gemma-4-31B-it\nlanguage: [tw, en]\npipeline_tag: text-generation\n---\n\n# Private Balanced Response V2\n\nExperimental direct-answer and native-tool LoRA. Not clinically validated or approved for production.\n\n"
        card += "## Data\n\n" + "\n".join("- " + item for item in manifest["limitations"])
        card += "\n\nAttribution: HASH Consortium / AfriHealth-QA; Mich-Seth Owusu and Ghana NLP Community; Hugging Face; OpenAssistant contributors; Argilla and Salesforce xLAM / Synth-APIGen contributors. Source restrictions remain, including noncommercial and share-alike terms. No public redistribution clearance asserted.\n\n"
        card += "## Recipe\n\n```json\n" + json.dumps(config, indent=2) + "\n```\n\n## Run results\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\nRaw base/adapter responses and input provenance accompany this model. Training loss is not semantic accuracy. Native tool generation is not proof of successful real-world execution. No proprietary fallback.\n"
        if V4:
            card = card.replace("Private Balanced Response V2", "Private Grounded Response V4")
            card += "\nAdditional attribution: AfriQA authors / Masakhane and original Wikipedia contributors. Pinned sources, exact answer spans, unchanged questions/answers and split exclusions accompany the corpus. Human-translation is an upstream label, not project clinical certification.\n"
        if V5:
            card = card.replace("Private Balanced Response V2", "Private Native Understanding V5")
            card += "\nAdditional attribution: INJONGO authors / Masakhane / Lacuna Fund; AfriQA authors and Wikipedia contributors; GhanaNLP professional parallel text contributors. Native intent/entity supervision is an auxiliary task, not full conversation or clinical certification. AfriXNLI is excluded from training.\n"
        (destination / "adapter/README.md").write_text(card)
        save("summary.json", summary)
        save("status.json", {"stage": "completed", **summary})
        return summary
    except Exception as exc:
        save("status.json", {"stage": "failed", "run_id": run_id, "error_type": type(exc).__name__})
        raise
