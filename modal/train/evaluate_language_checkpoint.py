"""Read-only early checkpoint diagnostic; does not stop or modify training."""
from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path

import modal

app = modal.App("ghana-language-checkpoint-diagnostic")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
        image = image.add_local_file(str(root / "tmp/general-language-corpus/adaptation-v1" / name), "/inputs/" + name)
    image = image.add_local_file(str(Path(__file__).with_name("language_adaptation_core.py")), "/root/language_adaptation_core.py")
    image = image.add_local_file(str(root / "tmp/general-foundation-comparison/cases-81a7aad880d9.json"), "/inputs/product.json")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, timeout=1800,
              max_containers=1, is_generator=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts.read_only()})
def evaluate(run_id: str, step: int, paired: bool = False, adapter_scale: float = 1.0, translation_only: bool = False):
    import hashlib
    import sys
    import time

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, Gemma4ForCausalLM

    if not re.fullmatch(r"gemma31_language_v1_\d{8}T\d{6}Z", run_id) or not 1 <= step <= 813:
        raise ValueError("Invalid checkpoint")
    if adapter_scale not in (1.0, 0.25):
        raise ValueError("Only full and quarter-strength diagnostics are supported")
    sys.path.insert(0, "/root")
    from language_adaptation_core import load_inputs, tokenize_inputs, translation_diagnostics
    directory = Path("/artifacts/language-adaptation", run_id)
    checkpoint = directory / "checkpoints" / f"checkpoint-{step}"
    manifest, inputs = load_inputs(Path("/inputs"))
    if json.loads((directory / "manifest.json").read_text()) != manifest:
        raise ValueError("Checkpoint/input provenance mismatch")
    if not (checkpoint / "adapter_model.safetensors").exists():
        raise ValueError("Requested checkpoint is not saved")
    adapter_sha = hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"], revision=manifest["base_revision"], local_files_only=True)
    encoded, accepted, _ = tokenize_inputs(tokenizer, inputs)
    base, info = Gemma4ForCausalLM.from_pretrained(manifest["base_model"], revision=manifest["base_revision"],
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa", output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."}, local_files_only=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete base weights")
    configuration = LoraConfig.from_pretrained(str(checkpoint))
    if configuration.use_rslora or configuration.use_dora or configuration.alpha_pattern:
        raise ValueError("Unexpected adapter scaling convention")
    original_alpha = configuration.lora_alpha
    configuration.lora_alpha = original_alpha * adapter_scale
    model = PeftModel.from_pretrained(base, str(checkpoint), config=configuration, is_trainable=False).eval()
    model.config.use_cache = False
    yield {"type": "provenance", "run_id": run_id, "step": step, "adapter_sha256": adapter_sha,
           "base_model": manifest["base_model"], "revision": manifest["base_revision"],
           "adapter_scale": adapter_scale, "original_lora_alpha": original_alpha,
           "effective_lora_alpha": configuration.lora_alpha,
           "weight_status": "Original checkpoint tensors, inference-only scaling; no further training",
           "task_scope": "Held-out source translation" if translation_only else "Product development checks",
           "comparison_scope": "Paired frozen base and adapter" if paired else "Adapter-only early diagnostic",
           "generation": {"max_new_tokens": 384, "do_sample": False, "batch_size": 2,
                          "runtime": "transformers", "dtype": "bfloat16"}}
    losses = {}
    for task in (() if translation_only else (("translation", "tw"), ("translation", "en"), ("general_conversation_replay", "en"))):
        indexes = [i for i, row in enumerate(accepted["validation"]) if (row["task"], row["language"]) == task]
        indexes.sort(key=lambda i: accepted["validation"][i]["source_record_hash"])
        values = []
        for i in indexes[:32]:
            batch = {k: torch.tensor([v], device=model.device) for k, v in encoded["validation"][i].items()}
            with torch.inference_mode():
                values.append(model(**batch).loss.float().item())
        losses[":".join(task)] = sum(values) / len(values)
    if not translation_only:
        yield {"type": "losses", "adapter": losses, "base": json.loads((directory / "baseline_loss.json").read_text())}
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    cases = translation_diagnostics(accepted["validation"]) if translation_only else json.loads(Path("/inputs/product.json").read_text())
    for variant in (("base", "adapter") if paired else ("adapter",)):
        with model.disable_adapter() if variant == "base" else nullcontext():
            for offset in range(0, len(cases), 2):
                batch = cases[offset:offset + 2]
                prompts = []
                for row in batch:
                    messages = [dict(m) for m in row["messages"]]
                    if not translation_only:
                        messages[0]["content"] += " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
                    prompts.append(tokenizer.apply_chat_template(messages, tools=row.get("tools"), tokenize=False,
                                                                 add_generation_prompt=True, enable_thinking=False))
                tokens = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
                started = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**tokens, max_new_tokens=384, do_sample=False,
                                            eos_token_id=[1, 106, 50], pad_token_id=tokenizer.pad_token_id)
                for row, result in zip(batch, output[:, tokens["input_ids"].shape[1]:], strict=True):
                    ids = result.tolist()
                    yield {"type": "prediction", "variant": variant, **row,
                           "prediction": tokenizer.decode(ids, skip_special_tokens=True),
                           "raw_output": tokenizer.decode(ids, skip_special_tokens=False), "token_ids": ids,
                           "limit_reached": not any(i in (1, 106, 50) for i in ids), "batch_seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(run_id: str, step: int = 100, paired: bool = False, adapter_scale: float = 1.0, translation_only: bool = False):
    folder = Path(__file__).resolve().parents[2] / "tmp/general-language-corpus"
    suffix = "paired" if paired else "diagnostic"
    if adapter_scale != 1.0:
        suffix += "-scale025"
    if translation_only:
        suffix += "-source-translation"
    path = folder / f"{run_id}-checkpoint-{step}-{suffix}.jsonl"
    if path.exists():
        raise ValueError("Diagnostic already exists; do not overwrite results")
    with path.open("x") as output:
        for event in evaluate.remote_gen(run_id, step, paired, adapter_scale, translation_only):
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({k: v for k, v in event.items() if k not in ("token_ids", "raw_output", "messages")}, ensure_ascii=False), flush=True)
