"""Compare an external direct-answer adapter, without training or deployment."""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import modal

BASE = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
REPO = "DariusTheGeek/mhqa-itu-adapters"
REVISION = "16247b4983ed169c5cfc18f2099701cf31425500"
WEIGHT_SHA = "9f3a8e6d3982e6c673c9bd5f533a93102bb32e6da588466537e55a2b86894245"
PREFIX = "base_model.model.model."
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

app = modal.App("ghana-public-response-adapter-comparison")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})


def mapped_name(name):
    source = PREFIX + "language_model."
    if name.startswith(source):
        name = PREFIX + name[len(source):]
    if not re.fullmatch(re.escape(PREFIX) + r"layers\.\d+\.(self_attn|mlp)\.\w+\.lora_[AB]\.weight", name):
        raise ValueError(f"Unexpected external adapter tensor: {name}")
    return name


def validate_shapes(actual, expected):
    mapped = {}
    for name, shape in actual.items():
        key = mapped_name(name)
        if key in mapped:
            raise ValueError("Adapter tensor mapping collision")
        mapped[key] = tuple(shape)
    if mapped != {k: tuple(v) for k, v in expected.items()}:
        raise ValueError("External adapter tensor names or shapes do not match the pinned base")
    return mapped


def text_weights(weights):
    # The text-only base never invokes the image encoder. Retain every other key
    # for strict validation, rather than silently ignoring unknown tensors.
    return {key: value for key, value in weights.items() if not key.startswith(PREFIX + "vision_tower.")}


def configuration(folder):
    from peft import LoraConfig
    source = json.loads((folder / "adapter_config.json").read_text())
    if (source["r"], source["lora_alpha"], source["bias"], set(source["target_modules"])) != (64, 128, "none", PROJECTIONS):
        raise ValueError("Published adapter configuration changed")
    if any(source.get(k) for k in ("use_rslora", "use_dora", "modules_to_save", "alpha_pattern", "rank_pattern", "lora_bias")):
        raise ValueError("Unexpected adapter convention")
    return LoraConfig(r=64, lora_alpha=128, lora_dropout=0, bias="none",
                      target_modules=sorted(PROJECTIONS), task_type="CAUSAL_LM", inference_mode=True)


def check_hash(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != WEIGHT_SHA:
        raise ValueError("External adapter SHA256 mismatch")


@app.function(image=image, cpu=4, memory=12288, timeout=900, retries=0,
              volumes={"/cache": cache})
def prepare():
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4ForCausalLM

    folder = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=[
        "README.md", "gen7454/adapter_config.json", "gen7454/adapter_model.safetensors", "gen7454/chat_template.jinja",
    ])) / "gen7454"
    check_hash(folder / "adapter_model.safetensors")
    config = AutoConfig.from_pretrained(BASE, revision=BASE_REVISION, local_files_only=True)
    with init_empty_weights():
        model = get_peft_model(Gemma4ForCausalLM(config.text_config), configuration(folder))
    expected = {key: tuple(value.shape) for key, value in get_peft_model_state_dict(model).items()}
    with safe_open(folder / "adapter_model.safetensors", framework="pt") as tensors:
        actual = {key: tuple(tensors.get_slice(key).get_shape()) for key in tensors.keys()}
    selected = text_weights(actual)
    validate_shapes(selected, expected)
    cache.commit()
    return {"folder": str(folder), "adapter_sha256": WEIGHT_SHA, "tensor_count": len(actual),
            "shape_check": "Exact mapped key and shape match against pinned base on meta device",
            "text_tensor_count": len(selected), "unused_image_encoder_tensors": len(actual) - len(selected),
            "sample_original_key": next(iter(selected)), "sample_mapped_key": mapped_name(next(iter(selected)))}


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, timeout=1800,
              max_containers=1, is_generator=True,
              volumes={"/cache": cache.read_only()})
def compare(rows: list[dict], prepared: dict):
    import time
    import torch
    from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Gemma4ForCausalLM

    if len(rows) != 30 or any(set(row) - {"id", "messages", "tools"} for row in rows):
        raise ValueError("Only the 30 model-visible development inputs are allowed")
    folder = Path(prepared["folder"])
    check_hash(folder / "adapter_model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, local_files_only=True)
    base, info = Gemma4ForCausalLM.from_pretrained(BASE, revision=BASE_REVISION,
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
        key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True, local_files_only=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete base weights")
    model = get_peft_model(base, configuration(folder))
    weights = text_weights(load_file(folder / "adapter_model.safetensors"))
    validate_shapes({k: v.shape for k, v in weights.items()},
                    {k: v.shape for k, v in get_peft_model_state_dict(model).items()})
    loaded = set_peft_model_state_dict(model, {mapped_name(k): v for k, v in weights.items()})
    if loaded.unexpected_keys or any("lora_" in key for key in loaded.missing_keys):
        raise ValueError("External adapter failed to load completely")
    del weights
    model.eval()
    tokenizer.padding_side = "left"
    effective = []
    for row in rows:
        messages = [dict(message) for message in row["messages"]]
        messages[0]["content"] += " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
        effective.append({**row, "messages": messages})
    yield {"type": "provenance", "comparison_scope": "Paired frozen base and adapter",
           "base_model": BASE, "revision": BASE_REVISION, "external_adapter": REPO,
           "external_revision": REVISION, "subfolder": "gen7454", "adapter_sha256": WEIGHT_SHA,
           "authorship": "External DariusTheGeek research reference, not project-trained",
           "limitations": ["Publisher does not pin the original base training revision.",
                           "Text-only inference; unused image-encoder adapter tensors are excluded.",
                           "Research only. No clinical or product acceptance.",
                           "No retrieval, answer bank, judge, ensemble or proprietary fallback.",
                           "These 30 cases are development checks, not independent gold."],
           "preflight": prepared, "gpu": torch.cuda.get_device_name(),
           "generation": {"max_new_tokens": 384, "do_sample": False, "batch_size": 2,
                          "runtime": "transformers==5.17.0", "dtype": "bfloat16"}}
    for variant in ("base", "adapter"):
        with model.disable_adapter() if variant == "base" else nullcontext():
            for offset in range(0, len(effective), 2):
                batch = effective[offset:offset + 2]
                prompts = [tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
                tokens = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
                if tokens["input_ids"].shape[1] > 4096:
                    raise ValueError("Unexpected input size; do not truncate")
                started = time.monotonic()
                with torch.inference_mode():
                    outputs = model.generate(**tokens, do_sample=False, max_new_tokens=384,
                        eos_token_id=[1, 106, 50], pad_token_id=tokenizer.pad_token_id)
                for row, result in zip(batch, outputs[:, tokens["input_ids"].shape[1]:], strict=True):
                    ids = result.tolist()
                    yield {"type": "prediction", "variant": variant, **row,
                           "prediction": tokenizer.decode(ids, skip_special_tokens=True),
                           "raw_output": tokenizer.decode(ids, skip_special_tokens=False), "token_ids": ids,
                           "limit_reached": not any(i in (1, 106, 50) for i in ids),
                           "batch_seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(preflight_only: bool = False):
    prepared = prepare.remote()
    print(json.dumps(prepared), flush=True)
    if preflight_only:
        return
    root = Path(__file__).resolve().parents[2]
    raw = (root / "tmp/general-foundation-comparison/cases-81a7aad880d9.json").read_bytes()
    cases = json.loads(raw)
    if hashlib.sha256(raw).hexdigest()[:12] != "81a7aad880d9":
        raise ValueError("Development fixture digest changed")
    inputs = [{k: row[k] for k in ("id", "messages", "tools") if k in row} for row in cases]
    path = root / "tmp/general-foundation-comparison" / ("external-gen7454-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    with path.open("x") as output:
        for event in compare.remote_gen(inputs, prepared):
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({k: v for k, v in event.items() if k not in ("token_ids", "raw_output", "messages")}, ensure_ascii=False), flush=True)
    print(f"Saved {path}", flush=True)
