"""One bounded open-weight foundation comparison; no training or public service."""
from pathlib import Path
import modal

MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
REVISION = "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
app = modal.App("ghana-stronger-response-comparison")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cpu_image = modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub==1.5.0").env({
    "HF_HOME": "/cache/hf", "HF_XET_CACHE": "/tmp/hf-xet", "DO_NOT_TRACK": "1"})
image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.13").pip_install(
    "vllm==0.28.0", "sacrebleu==2.5.1").pip_install("transformers==5.17.0").run_commands(
    "python -c 'from transformers import PreTrainedTokenizerBase; assert hasattr(PreTrainedTokenizerBase, \"get_response_parser\")'").env({
    "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "TRITON_CACHE_DIR": "/tmp/triton", "VLLM_CACHE_ROOT": "/tmp/vllm", "OMP_NUM_THREADS": "1"})
if modal.is_local():
    image = image.add_local_file(str(Path(__file__).resolve().parents[1] / "response_runtime.py"), "/root/response_runtime.py")


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 30:
        raise ValueError("Expected at most 30 distinct diagnostic cases")
    for row in cases:
        if not isinstance(row, dict) or not {"id", "messages"} <= set(row) or set(row) - {"id", "messages", "tools"} or not isinstance(row["id"], str):
            raise ValueError("Only model-visible inputs, no answers or rubrics")
        messages = row["messages"]
        if not messages or messages[0]["role"] != "system" or len(messages) % 2 != 0 or len(messages) > 32:
            raise ValueError("Expected system and alternating conversation")
        for i, message in enumerate(messages):
            role = "system" if i == 0 else "user" if i % 2 else "assistant"
            if set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str) or not 1 <= len(message["content"]) <= 8000:
                raise ValueError("Invalid diagnostic message")
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate diagnostic ID")


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1800, retries=0, max_containers=1,
              volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "qwen122fp8", True)


def prepare_weights(model, revision, folder, fp8, reload_cache=True):
    import json
    import time
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    if reload_cache:
        cache.reload()
    path = Path(snapshot_download(model, revision=revision,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md", "LICENSE*"],
        ignore_patterns=["original/*", "metal/*", "onnx/*"], max_workers=4))
    config = json.loads((path / "config.json").read_text())
    if (config.get("quantization_config", {}).get("quant_method") == "fp8") != fp8:
        raise ValueError("Unexpected source quantization")
    index_path = path / "model.safetensors.index.json"
    files = (sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
             if index_path.exists() else ["model.safetensors"])
    if not files or any(not (path / name).is_file() for name in files):
        raise ValueError("Incomplete public weights")
    report = {"model": model, "revision": revision, "weights_bytes": sum((path / name).stat().st_size for name in files),
              "weight_files": files, "quantization": config.get("quantization_config", {}).get("quant_method"),
              "seconds": time.monotonic()-started, "gpu_used": False}
    cache.commit()
    output = Path("/artifacts/stronger-response", folder)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache-manifest.json").write_text(json.dumps(report, indent=2))
    artifacts.commit()
    return report


@app.function(image=image, gpu="H100:2", cpu=8, memory=98304, timeout=1800, retries=0, max_containers=1,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def compare(run_id: str, cases: list[dict]):
    return compare_model(run_id, cases, MODEL, REVISION, "qwen122", 2, "fp8")


def compare_model(run_id, cases, model, revision, prefix, parallel, quantization, reasoning_effort=None):
    import hashlib
    import json
    import re
    import sys
    import time
    import torch
    from vllm import LLM, SamplingParams
    sys.path.insert(0, "/root")
    from response_runtime import QWEN_RESPONSE_TEMPLATE
    validate_cases(cases)
    if not re.fullmatch(re.escape(prefix) + r"_\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid comparison ID")
    artifacts.reload()
    cache.reload()
    output = Path("/artifacts/stronger-response", run_id)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"model": model, "revision": revision, "run_id": run_id, "runtime": "vllm==0.28.0",
              "gpu_actual": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "quantization": "publisher FP8" if quantization == "fp8" else "none; BF16 weights", "tensor_parallel_size": parallel,
              "reasoning_effort": reasoning_effort,
              "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "inputs": cases, "results": [], "training": False, "tools_executed": False,
              "production_promoted": False, "proprietary_fallback": False}

    def save(name):
        report["elapsed_seconds"] = time.monotonic()-started
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        artifacts.commit()

    save("loading.json")
    engine = LLM(model=model, revision=revision, tokenizer_revision=revision,
        language_model_only=True, dtype="bfloat16", quantization=quantization, tensor_parallel_size=parallel,
        max_model_len=8192, max_num_seqs=8, gpu_memory_utilization=0.9, enforce_eager=True,
        trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    tokenizer.response_template = QWEN_RESPONSE_TEMPLATE
    report["load_seconds"] = time.monotonic()-started
    for thinking in (False, True):
        decoding = {"temperature": 1.0 if thinking else 0.7, "top_p": 0.95 if thinking else 0.8,
                    "top_k": 20, "presence_penalty": 0.0 if thinking and reasoning_effort else 1.5,
                    "max_tokens": 4096 if thinking else 640, "seed": 42}
        for offset in range(0, len(cases), 8):
            batch = cases[offset:offset+8]
            template_kwargs = {"reasoning_effort": reasoning_effort, "preserve_thinking": True} if reasoning_effort else {}
            prompts = [tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"), tokenize=False,
                       add_generation_prompt=True, enable_thinking=thinking, **template_kwargs) for row in batch]
            if any(len(tokenizer.encode(p, add_special_tokens=False)) > 4000 for p in prompts):
                raise ValueError("Input exceeds budget; no truncation")
            batch_started = time.monotonic()
            predictions = engine.generate(prompts, SamplingParams(**decoding))
            elapsed = time.monotonic()-batch_started
            for row, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                value = prediction.outputs[0]
                raw = tokenizer.decode(value.token_ids, skip_special_tokens=False)
                parser = tokenizer.get_response_parser(prefix=prompt, tools=row.get("tools"))
                parser.feed(raw)
                parsed, _ = parser.finalize()
                report["results"].append({"id": row["id"], "thinking": thinking, "decoding": decoding,
                    "raw_output": raw, "parsed": parsed, "answer": parsed.get("content", ""),
                    "output_tokens": len(value.token_ids), "finish_reason": value.finish_reason,
                    "batch_seconds": elapsed, "batch_size": len(batch)})
            save("partial.json")
            print(json.dumps({"thinking": thinking, "completed": offset+len(batch), "seconds": time.monotonic()-started}), flush=True)
    save("results.json")
    return report
