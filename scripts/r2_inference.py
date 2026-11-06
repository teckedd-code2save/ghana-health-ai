"""Shared, pinned R2 research inference; no training, fallback or external APIs."""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time

RUNTIME = {"torch": "2.13.0", "transformers": "5.17.0", "accelerate": "1.15.0",
           "flash-linear-attention": "0.5.2", "triton": "3.7.1", "peft": "0.20.0", "numpy": "2.4.2"}
VARIANTS = ("base", "r1_meaning", "r1_response", "r2_language", "r2_meaning", "r2_response")
DECODING = {"do_sample": False, "enable_thinking": False, "max_new_tokens": 256,
            "num_beams": 1, "repetition_penalty": 1.0, "seed": 17, "truncate": False}
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
BASE_SHA = "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
           "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, "Nonfinite JSON"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def safe(root, name):
    root = Path(root).resolve()
    path = root / name
    require(not Path(name).is_absolute() and not path.is_symlink(), "Unsafe artifact path")
    path = path.resolve()
    require(path.is_relative_to(root) and path != root, "Escaping artifact path")
    return path


def adapter_config(config):
    require(config["base_model_name_or_path"] == MODEL and config["revision"] == REVISION,
            "Wrong adapter foundation")
    require(config["peft_type"] == "LORA" and config["task_type"] == "CAUSAL_LM"
            and config["r"] == 16 and config["lora_alpha"] == 32 and config["lora_dropout"] == .05
            and config["bias"] == "none" and set(config["target_modules"]) == TARGETS,
            "Wrong adapter configuration")
    require(not any(config.get(k) for k in ("modules_to_save", "use_dora", "use_rslora",
            "rank_pattern", "alpha_pattern", "layer_replication", "trainable_token_indices")),
            "Unexpected additional adapter components")


def load_bundle(folder, expected_sha=None):
    folder = Path(folder)
    if expected_sha is not None:
        require(sha(folder / "manifest.json") == expected_sha, "Bundle identity changed")
    manifest = read(folder / "manifest.json")
    require(manifest["version"] == "r2-paired-inference-v1" and manifest["model"] == MODEL
            and manifest["revision"] == REVISION and manifest["base_sha256"] == BASE_SHA,
            "Wrong inference foundation")
    require(manifest["runtime"] == RUNTIME and manifest["decoding"] == DECODING,
            "Inference protocol changed")
    require(set(manifest["variants"]) == set(VARIANTS) and manifest["variants"]["base"] is None,
            "Wrong checkpoint set")
    for name, checksum in manifest["files"].items():
        require(sha(safe(folder, name)) == checksum, "Bundle artifact changed: " + name)
    for name, spec in manifest["variants"].items():
        if name == "base":
            continue
        require(re.fullmatch(r"/artifacts/understanding-adaptation-r[12]/understanding_\d{8}T\d{6}Z/(language|meaning|responses)-adapter", spec["directory"]),
                "Invalid adapter directory")
        require(set(spec["files"]) == {"adapter_config.json", "adapter_model.safetensors"}, "Incomplete adapter binding")
    require(manifest["private"] is True and manifest["training"] is False, "Research-only bundle required")
    return manifest


def validate_messages(messages):
    require(isinstance(messages, list) and 1 <= len(messages) <= 129, "Provide a bounded conversation")
    start = int(messages[0].get("role") == "system")
    require((len(messages) - start) % 2 == 1, "Conversation must end in user")
    for i, m in enumerate(messages):
        role = "system" if start and i == 0 else "user" if (i - start) % 2 == 0 else "assistant"
        require(set(m) == {"role", "content"} and m["role"] == role and isinstance(m["content"], str)
                and bool(m["content"].strip()), "Invalid complete conversation history")
    require(sum(len(m["content"]) for m in messages) <= 65536, "Conversation too long; no automatic truncation")


def shifted_nll(logits, inputs):
    from torch.nn import functional as F
    require(logits.shape[:2] == inputs.shape and inputs.shape[1] > 1, "Invalid causal-loss shapes")
    total = 0.0
    for first in range(0, inputs.shape[1] - 1, 128):
        end = min(first + 128, inputs.shape[1] - 1)
        total += F.cross_entropy(logits[0, first:end].float(), inputs[0, first+1:end+1], reduction="sum").item()
    require(math.isfinite(total) and total >= 0, "Nonfinite held-out loss")
    return {"predicted_tokens": inputs.shape[1] - 1, "summed_nll": total}


def saved_prompt_hash(record):
    hashes = [record[k] for k in ("prompt_token_ids_sha256", "prompt_ids_sha256") if k in record]
    require(bool(hashes) and all(h == hashes[0] for h in hashes), "Missing/conflicting saved prompt hash")
    return hashes[0]


def reference_record(ref, case, tokenizer):
    from afrique_instruction_baseline_core import prompt_ids
    from check_afrique_training_contract import ASSET_HASHES
    ids = prompt_ids(tokenizer, case["messages"], {"context_tokens": 4096, "max_new_tokens": 256})
    record = ref["record"]
    require(ref["case_id"] == case["id"] and ref["decoding"] == DECODING
            and record["messages_sha256"] == digest(case["messages"])
            and saved_prompt_hash(record) == digest(ids)
            and ref["runtime"] == {"environment_sha256": digest(RUNTIME),
                "tokenizer_sha256": ASSET_HASHES["tokenizer.json"],
                "chat_template_sha256": ASSET_HASHES["chat_template.jinja"]}, "Saved reply protocol mismatch")
    return {**record, "case_id": case["id"], "prompt_token_ids_sha256": digest(ids),
            "seconds": record.get("seconds", record["diagnostics"]["seconds"]),
            "reference_identity": record.get("identity"), "origin": "saved_exact_reference",
            "reference_source_sha256": ref["source_sha256"]}


class Runtime:
    def __init__(self, bundle_dir, expected_sha, artifacts_root=Path("/artifacts"), *, bundle_loader=load_bundle):
        self.bundle = Path(bundle_dir)
        self.manifest = bundle_loader(self.bundle, expected_sha)
        self.manifest_sha = expected_sha
        require({n: importlib.metadata.version(n) for n in RUNTIME} == RUNTIME, "Runtime versions differ")
        self.paths = {}
        for name, checksum in self.manifest["remote_files"].items():
            require(name.startswith("/artifacts/"), "Invalid evidence root")
            require(sha(safe(artifacts_root, name.removeprefix("/artifacts/"))) == checksum,
                    "Remote training evidence changed: " + name)
        for name, spec in self.manifest["variants"].items():
            if spec is None:
                continue
            path = safe(artifacts_root, spec["directory"].removeprefix("/artifacts/"))
            for file, checksum in spec["files"].items():
                require(sha(safe(path, file)) == checksum, "Actual adapter bytes differ: " + name)
            adapter_config(read(path / "adapter_config.json"))
            self.paths[name] = path
        from check_afrique_training_contract import load_tokenizer, ASSET_HASHES
        from afrique_instruction_baseline_core import verify_cache_receipt, verify_weight, verify_loading, WEIGHT
        from native_adaptation_run_core import require_kernel_binding
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native_model
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        from safetensors import safe_open
        from transformers import Qwen3_5ForCausalLM
        import torch
        self.tokenizer = load_tokenizer(self.bundle / "tokenizer")
        snapshot = Path("/cache/hf/hub", "models--" + MODEL.replace("/", "--"), "snapshots", REVISION)
        verify_cache_receipt(read(snapshot / "AFRIQUE_CACHE_READY.json"))
        for name, checksum in ASSET_HASHES.items():
            require(sha(snapshot / name) == checksum, "Cached foundation asset changed")
        verify_weight(snapshot / "model.safetensors", WEIGHT)
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
            shapes = {n: list(handle.get_slice(n).get_shape()) for n in handle.keys()}
        require_kernel_binding(native_model.torch_chunk_gated_delta_rule, chunk_gated_delta_rule)
        require_kernel_binding(native_model.torch_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule)
        self.model, info = Qwen3_5ForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
            key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        self.loading = verify_loading(info, shapes, {k: list(v.shape) for k, v in self.model.state_dict().items()})
        require(not any(p.is_meta for p in self.model.parameters()), "Unmaterialized weights")
        self.model.requires_grad_(False); self.model.eval(); self.model.config.use_cache = True
        torch.backends.cuda.matmul.allow_tf32 = False
        self.loaded = set()

    def identity(self, variant):
        require(variant in self.manifest["variants"], "Unknown model variant")
        spec = self.manifest["variants"][variant]
        return {"manifest_sha256": self.manifest_sha, "variant": variant, "model": MODEL,
                "revision": REVISION, "base_weights_sha256": BASE_SHA,
                "adapter_weights_sha256": None if spec is None else spec["files"]["adapter_model.safetensors"]}

    def select(self, variant):
        self.identity(variant)
        if variant == "base":
            return self.model.disable_adapter() if self.loaded else nullcontext()
        if variant not in self.loaded:
            import torch
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            if not self.loaded:
                self.model = PeftModel.from_pretrained(self.model, str(self.paths[variant]), adapter_name=variant,
                    is_trainable=False, local_files_only=True, autocast_adapter_dtype=True)
            else:
                self.model.load_adapter(str(self.paths[variant]), adapter_name=variant,
                    is_trainable=False, local_files_only=True, autocast_adapter_dtype=True)
            expected = load_file(str(self.paths[variant] / "adapter_model.safetensors"), device="cpu")
            actual = get_peft_model_state_dict(self.model, adapter_name=variant, save_embedding_layers=False)
            require(expected.keys() == actual.keys() and sum(t.numel() for t in expected.values()) == 43278336
                    and all("lora_" in n and torch.equal(t, actual[n].detach().cpu()) for n, t in expected.items()),
                    "Loaded adapter tensors differ")
            self.loaded.add(variant)
        self.model.set_adapter(variant)
        self.model.requires_grad_(False); self.model.eval(); self.model.config.use_cache = True
        return nullcontext()

    def reply(self, messages, variant, on_text=None, cancelled=None):
        import random
        import numpy as np
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextStreamer
        from check_afrique_training_contract import EOS, PAD
        from afrique_instruction_baseline_core import prompt_ids, completion
        validate_messages(messages)
        ids = prompt_ids(self.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})
        with self.select(variant):
            tick = time.monotonic()
            class Stop(StoppingCriteria):
                timed_out = False
                was_cancelled = False
                def __call__(self, input_ids, scores, **kwargs):
                    self.timed_out = time.monotonic() - tick >= 60
                    self.was_cancelled = bool(cancelled and cancelled())
                    return torch.full((input_ids.shape[0],), self.timed_out or self.was_cancelled,
                                      dtype=torch.bool, device=input_ids.device)
            class Stream(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    if text and on_text:
                        on_text(text)
            timer = Stop()
            random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
            config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                max_new_tokens=256, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=None,
                use_cache=True, return_dict_in_generate=True)
            with torch.inference_mode():
                output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    generation_config=config, stopping_criteria=StoppingCriteriaList([timer]),
                    streamer=Stream(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if on_text else None)
            torch.cuda.synchronize()
            generated = output.sequences[0, len(ids):].tolist()
            value = completion(self.tokenizer, generated, 256, timer.timed_out, time.monotonic() - tick)
        return {"identity": self.identity(variant), "messages_sha256": digest(messages),
            "prompt_ids_sha256": digest(ids), "decoding": DECODING, "output_ids": generated,
            "output": value["prediction"], "seconds": value["seconds"], "diagnostics": value,
            "finish_reason": "cancelled" if timer.was_cancelled else value["finish_reason"],
            "complete": value["complete"] and not timer.was_cancelled}

    def language_loss(self, text, variant):
        import torch
        from check_afrique_training_contract import EOS
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [EOS]
        require(1 < len(ids) <= 4096, "Invalid held-out length; no truncation")
        with self.select(variant), torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
            logits = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
            return shifted_nll(logits, inputs)
