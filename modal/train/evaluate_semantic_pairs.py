"""Bounded paired development evaluation. Labels/references remain on the client."""
from pathlib import Path
import modal

app = modal.App("ghana-semantic-pair-evaluation")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0", "jsonschema==4.26.0",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
if modal.is_local():
    image = image.add_local_file(str(Path(__file__).resolve().parents[1] / "response_runtime.py"), "/root/response_runtime.py")


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=1200,
              max_containers=1, retries=0, volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def evaluate(candidate: dict, cases: list[dict], attempt: int = 1, task: str = "nli"):
    import hashlib
    import json
    import sys
    import time
    from contextlib import nullcontext
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Gemma4ForCausalLM
    sys.path.insert(0, "/root")
    from response_runtime import model_spec, validate_input

    spec = model_spec(candidate["run_id"])
    if task not in ("nli", "native-intent", "native-intent-schema-v2") or attempt not in (1, 2, 3) or spec["family"] != "gemma" or not 1 <= len(cases) <= 900 or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Invalid semantic comparison inputs")
    for row in cases:
        if set(row) != {"id", "messages"}:
            raise ValueError("Evaluation labels must not cross to the model runner")
        messages = row["messages"]
        if task.startswith("native-intent"):
            if (len(messages) != 2 or set(messages[0]) != {"role", "content"} or messages[0]["role"] != "system"
                    or not isinstance(messages[0]["content"], str) or not 1 <= len(messages[0]["content"]) <= 4000):
                raise ValueError("Invalid annotation task instruction")
            messages = messages[1:]
        validate_input(messages, spec["variants"][0], candidate["run_id"], candidate["checkpoint"], candidate["adapter_sha256"])
    # Reused workers otherwise retain a snapshot from before new checkpoints.
    artifacts.reload()
    path = Path("/artifacts", spec["folder"], candidate["run_id"], candidate["checkpoint"])
    with (path / "adapter_model.safetensors").open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != candidate["adapter_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
    output = Path("/artifacts/semantic-evaluation", candidate["run_id"], candidate["checkpoint"].replace("/", "-"),
                  (f"attempt-{attempt}" if task == "nli" else f"{task}-attempt-{attempt}"))
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    tokenizer.padding_side = "left"
    prompts = [tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) for row in cases]
    if any(len(tokenizer.encode(prompt, add_special_tokens=False)) > 1024 for prompt in prompts):
        raise ValueError("Semantic prompt exceeds context budget; do not truncate")
    started = time.monotonic()
    base, info = Gemma4ForCausalLM.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa", output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."}, local_files_only=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete base weights")
    model = PeftModel.from_pretrained(base, path, is_trainable=False).eval()
    max_tokens = 8 if task == "nli" else 256
    report = {"candidate": candidate, "model": spec["model"], "revision": spec["revision"], "task": task,
              "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "decoding": {"do_sample": False, "max_new_tokens": max_tokens, "batch_size": 8, "thinking": False},
              "results": [], "proprietary_fallback": False}
    for variant in ("base", "adapter"):
        with model.disable_adapter() if variant == "base" else nullcontext():
            for offset in range(0, len(cases), 8):
                tokens = tokenizer(prompts[offset:offset+8], padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**tokens, do_sample=False, max_new_tokens=max_tokens,
                        eos_token_id=[1, 106, 50], pad_token_id=tokenizer.pad_token_id)
                for row, ids in zip(cases[offset:offset+8], generated[:, tokens.input_ids.shape[1]:].tolist(), strict=True):
                    report["results"].append({"variant": variant, "id": row["id"], "token_ids": ids,
                        "prediction": tokenizer.decode(ids, skip_special_tokens=True),
                        "raw_output": tokenizer.decode(ids, skip_special_tokens=False),
                        "limit_reached": not any(t in [1, 106, 50] for t in ids)})
                if offset % 64 == 0:
                    (output / "partial.json").write_text(json.dumps(report, ensure_ascii=False))
                    artifacts.commit()
                    print(json.dumps({"variant": variant, "completed": min(offset+8, len(cases)), "seconds": time.monotonic()-started}), flush=True)
    report["total_seconds"] = time.monotonic()-started
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False))
    artifacts.commit()
    return report
