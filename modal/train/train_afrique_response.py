"""Durable instruction SFT on the unchanged Twi-including CPT foundation."""
import json
import os
from pathlib import Path

import modal

RECIPE = os.environ.get("GHA_AFRIQUE_RECIPE", "afrique-v3")
if RECIPE not in ("afrique-v3", "balanced-v6"):
    raise ValueError("Unknown Afrique recipe")
V6 = RECIPE == "balanced-v6"
APP = "ghana-balanced-afrique-v6" if V6 else "ghana-afrique-response-v3"
FOLDER = "balanced-afrique-v6" if V6 else "afrique-response-v3"
EXPERIMENT = "balanced_afrique_response_v6" if V6 else "afrique_response_sft_v3"
PREFIX = "afrique_v6_" if V6 else "afrique_v3_"
LR = 3e-5 if V6 else 5e-5
BASE_CACHE = "/cache/hf/hub" if V6 else "/cache/hf"
app = modal.App(APP)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0",
).env({"HF_HOME": "/cache/hf", "HF_HUB_CACHE": "/cache/hf", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "GHA_AFRIQUE_RECIPE": RECIPE})
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
        image = image.add_local_file(str(root / "tmp" / FOLDER / "corpus" / name), "/inputs/" + name)
    for name in ("afrique_response_core.py", "response_adaptation_core.py", "medical_pilot_core.py", "adapter_checkpoint.py"):
        image = image.add_local_file(str(Path(__file__).with_name(name)), "/root/" + name)
    image = image.add_local_file(str(root / "tmp/general-foundation-comparison/cases-81a7aad880d9.json"), "/inputs/product.json")
    if V6:
        image = image.add_local_file(str(root / "data/response-adaptation/conversation-checks.v6.json"), "/inputs/conversations.json")


def prepare():
    import sys
    sys.path.insert(0, "/root")
    from afrique_response_core import tokenize
    from response_adaptation_core import load_inputs
    from transformers import AutoTokenizer
    manifest, rows = load_inputs(Path("/inputs"), EXPERIMENT)
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_model"], revision=manifest["tokenizer_revision"], cache_dir="/cache/hf", local_files_only=True)
    original = AutoTokenizer.from_pretrained(manifest["base_model"], revision=manifest["base_revision"], cache_dir=BASE_CACHE, local_files_only=True)
    if tokenizer.get_vocab() != original.get_vocab():
        raise ValueError("Instruction tokenizer IDs differ from language foundation")
    encoded, accepted, report = tokenize(tokenizer, rows)
    if len(encoded["train"]) < 5000:
        raise ValueError("Corpus collapsed during native tokenization")
    if V6:
        tw = {k: n for k, n in report["supervised_tokens"].items() if k.startswith("train:") and k.endswith(":tw")}
        report["twi_health_target_token_fraction"] = tw.get("train:health_response:tw", 0) / sum(tw.values())
        if report["twi_health_target_token_fraction"] > 0.35:
            raise ValueError("Twi targets remain health-dominated")
        report["native_vocabulary_equal"] = True
    return manifest, tokenizer, encoded, accepted, report


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0, volumes={"/cache": cache.read_only()})
def preflight():
    import hashlib
    import tempfile
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Trainer, TrainingArguments
    manifest, tokenizer, encoded, accepted, report = prepare()
    from adapter_checkpoint import adapter_trainer_class
    Trainer = adapter_trainer_class()
    from medical_pilot_core import PilotCollator
    from afrique_response_core import encode
    sample = next(row for row in accepted["train"] if row["task"] == "tool_call")
    tool = encode(tokenizer, sample)
    report["native_tool_target"] = tokenizer.decode([i for i in tool["labels"] if i != -100])
    report["native_reply_target"] = tokenizer.decode([i for i in encoded["train"][0]["labels"] if i != -100])
    config = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, layer_types=["linear_attention", "full_attention"],
        use_cache=False, pad_token_id=0, eos_token_id=2)
    model = get_peft_model(Qwen3_5ForCausalLM(config), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.peft_config["default"].base_model_name_or_path = manifest["base_model"]
    model.peft_config["default"].revision = manifest["base_revision"]
    model.enable_input_require_grads()
    with tempfile.TemporaryDirectory() as folder:
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=folder, use_cpu=True, max_steps=1,
            per_device_train_batch_size=1, report_to="none", save_steps=1, save_only_model=True),
            train_dataset=[{"input_ids": [11, 12, 21, 22, 2], "attention_mask": [1] * 5, "labels": [-100, -100, 21, 22, 2]}],
            data_collator=PilotCollator(0))
        report["toy_backward_pass"] = trainer.train().metrics
        checkpoint = Path(folder) / "checkpoint-1"
        restored = PeftModel.from_pretrained(Qwen3_5ForCausalLM(config), checkpoint)
        report["toy_checkpoint_roundtrip"] = {"adapter_bytes": (checkpoint / "adapter_model.safetensors").stat().st_size,
            "revision": restored.peft_config["default"].revision, "embeddings_saved": False}
    report["manifest_sha256"] = hashlib.sha256(Path("/inputs/manifest.json").read_bytes()).hexdigest()
    return report


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=5400, retries=0, max_containers=1,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def train(run_id: str, preflight_sha: str):
    import hashlib
    import re
    import time
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    if not re.fullmatch(PREFIX + r"\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid run ID")
    if hashlib.sha256(Path("/inputs/manifest.json").read_bytes()).hexdigest() != preflight_sha:
        raise ValueError("Preflight and training inputs differ")
    started = time.monotonic()
    destination = Path("/artifacts/afrique-response", run_id)
    destination.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        artifacts.commit()

    try:
        manifest, tokenizer, encoded, accepted, report = prepare()
        from adapter_checkpoint import adapter_trainer_class, save_adapter
        Trainer = adapter_trainer_class()
        from medical_pilot_core import PilotCollator
        set_seed(42)
        save("manifest.json", manifest)
        save("tokenization.json", report)
        save("status.json", {"stage": "loading", "run_id": run_id})
        base, info = Qwen3_5ForCausalLM.from_pretrained(manifest["base_model"], revision=manifest["base_revision"],
            dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa", output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."}, cache_dir=BASE_CACHE, local_files_only=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("Foundation weights did not load completely")
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
        model.peft_config["default"].revision = manifest["base_revision"]
        model.enable_input_require_grads()
        model.config.use_cache = False
        # Fail before spending GPU time if the actual adapter cannot be saved.
        save_adapter(model, destination / "initial-adapter")
        artifacts.commit()
        batch = 16 if V6 else 8
        config = {"max_steps": min(400 if V6 else 700, (len(encoded["train"]) + batch - 1) // batch), "effective_batch": batch, "learning_rate": LR,
                  "seed": 42, "lora_rank": 16, "lora_alpha": 32, "target_modules": "all-linear", "max_length": 1536,
                  "initialization": "Unchanged Twi-including CPT foundation; no task-vector merge or prior adapter",
                  "training_stop_seconds": 3900, "function_timeout_seconds": 5400, "production_promoted": False,
                  "decoding": "greedy" if V6 else "temperature .7, top_p .8, top_k 20"}
        save("training_config.json", config)
        cases = json.loads(Path("/inputs/product.json").read_text())

        @torch.random.fork_rng()
        def generate(label, rows):
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            tokenizer.padding_side = "left"
            output = []
            for offset in range(0, len(rows), 4):
                batch = rows[offset:offset + 4]
                prompts = [tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"), tokenize=False,
                           add_generation_prompt=True, enable_thinking=False) for row in batch]
                inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
                torch.manual_seed(42 + offset)
                with torch.inference_mode():
                    decoding = {"do_sample": False} if V6 else {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
                    generated = model.generate(**inputs, max_new_tokens=384, **decoding,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                for row, ids in zip(batch, generated[:, inputs["input_ids"].shape[1]:], strict=True):
                    output.append({**row, "prediction": tokenizer.decode(ids, skip_special_tokens=True),
                                   "raw_output": tokenizer.decode(ids, skip_special_tokens=False),
                                   "limit_reached": tokenizer.eos_token_id not in ids.tolist()})
                save(label + ".partial.json", output)
            save(label + ".json", output)
            tokenizer.padding_side = "right"
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            return output

        def conversations(label):
            if not V6:
                return
            scenarios = json.loads(Path("/inputs/conversations.json").read_text())
            system = cases[0]["messages"][0]
            histories = {r["id"]: [system] for r in scenarios}
            collected = []
            for turn in range(2):
                batch_rows = []
                for row in scenarios:
                    histories[row["id"]].append({"role": "user", "content": row["turns"][turn]})
                    batch_rows.append({"id": row["id"], "turn": turn, "messages": list(histories[row["id"]])})
                outputs = generate(f"{label}-conversation-turn-{turn}", batch_rows)
                for result in outputs:
                    histories[result["id"]].append({"role": "assistant", "content": result["prediction"]})
                collected.extend(outputs)
            save(label + "-conversations.json", collected)

        with model.disable_adapter():
            generate("base", cases)
            conversations("base")

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                print(json.dumps({"step": state.global_step, **(logs or {}), "elapsed": round(time.monotonic() - started)}), flush=True)
            def on_save(self, args, state, control, **kwargs):
                save("status.json", {"stage": "training", "run_id": run_id, "step": state.global_step})
                generate(f"checkpoint-{state.global_step}-probe", [row for row in cases if row["id"] in (
                    "budget-tw", "language-switch", "tw-breathing-difficulty", "tw-hospital-choice")])
            def on_step_end(self, args, state, control, **kwargs):
                if time.monotonic() - started > 3900:
                    control.should_training_stop = True
                    control.should_save = True
                return control

        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(destination / "checkpoints"), max_steps=config["max_steps"],
            per_device_train_batch_size=2, gradient_accumulation_steps=batch // 2, learning_rate=LR, warmup_steps=35,
            lr_scheduler_type="cosine", weight_decay=0.01, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10, save_steps=100, save_total_limit=3,
            save_only_model=True, report_to="none", remove_unused_columns=False, seed=42, data_seed=42),
            train_dataset=encoded["train"], data_collator=PilotCollator(tokenizer.pad_token_id), callbacks=[Progress()])
        save("status.json", {"stage": "training", "run_id": run_id, "step": 0})
        result = trainer.train()
        save_adapter(model, destination / "adapter")
        tokenizer.save_pretrained(destination / "adapter")
        save("training_metrics.json", {**result.metrics, "log_history": trainer.state.log_history})
        save("status.json", {"stage": "evaluating", "run_id": run_id, "step": trainer.state.global_step})
        generate("adapter", cases)
        conversations("adapter")
        validation = [{"id": row["id"], "task": row["task"], "language": row["language"], "messages": row["messages"][:-1],
                       "tools": row.get("tools"), "reference": row["messages"][-1]} for task in sorted({r["task"] for r in accepted["validation"]})
                      for row in [r for r in accepted["validation"] if r["task"] == task][:8]]
        generate("source-heldout", validation)
        if V6:
            with model.disable_adapter():
                generate("base-source-heldout", validation)
        with (destination / "adapter/adapter_model.safetensors").open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        summary = {"run_id": run_id, "steps": trainer.state.global_step, "adapter_sha256": sha,
                   "path": str(destination), "elapsed_seconds": time.monotonic() - started, "production_promoted": False,
                   "decision": "Inspect raw replies; no automatic semantic or clinical acceptance"}
        version = "V6" if V6 else "V3"
        card = f"---\nlibrary_name: peft\nbase_model: {manifest['base_model']}\nlanguage: [tw, en]\npipeline_tag: text-generation\n---\n\n# Private Afrique Response {version}\n\nResearch instruction adaptation, not a medical product or verified general assistant. No proprietary response fallback.\n\n## Sources and limitations\n\n"
        card += "\n".join("- " + item for item in manifest["limitations"])
        card += "\n\nAttribution: McGill NLP / AfriqueLLM; Qwen team; HASH Consortium; Hugging Face; OpenAssistant; Argilla, Salesforce and Synth-APIGen; original parallel-text contributors. Full row-level source/revision/license attribution is retained in the corpus. No public redistribution clearance asserted.\n\n"
        card += "## Training\n\n```json\n" + json.dumps(config, indent=2) + "\n```\n\n## Results\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\nActual base, adapter, checkpoint-probe and source-heldout outputs accompany this model. Loss is not semantic accuracy. Tool generation is not executed-tool success.\n"
        (destination / "adapter/README.md").write_text(card)
        save("summary.json", summary)
        save("status.json", {"stage": "completed", **summary})
        return summary
    except Exception as exc:
        save("status.json", {"stage": "failed", "run_id": run_id, "error_type": type(exc).__name__})
        raise
