"""One open-weight reasoning baseline on private Modal, not a hosted GPT fallback."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights, validate_cases

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
app = modal.App("ghana-oss-response-comparison")
if modal.is_local():
    for name in ("benchmark_stronger_response.py", "oss_response_core.py"):
        source = str(Path(__file__).with_name(name))
        cpu_image = cpu_image.add_local_file(source, "/root/" + name)
        image = image.add_local_file(source, "/root/" + name)


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1800, retries=0, max_containers=1,
              volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    report = prepare_weights(MODEL, REVISION, "oss120", False)
    if report["quantization"] != "mxfp4":
        raise ValueError("Expected publisher MXFP4 weights")
    return report


@app.function(image=image, gpu="H100", cpu=4, memory=98304, timeout=1800, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def compare(run_id: str, cases: list[dict]):
    import hashlib
    import json
    import re
    import time
    import torch
    from vllm import LLM, SamplingParams
    from oss_response_core import parse
    validate_cases(cases)
    if not re.fullmatch(r"oss120_\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid comparison")
    cache.reload()
    artifacts.reload()
    destination = Path("/artifacts/stronger-response", run_id)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"run_id": run_id, "model": MODEL, "revision": REVISION, "inputs": cases,
        "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "gpu_actual": torch.cuda.get_device_name(0), "quantization": "publisher MXFP4",
        "reasoning_effort": "medium", "decoding": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2048, "seed": 42},
        "results": [], "training": False, "proprietary_fallback": False,
        "tools_executed": False, "production_promoted": False}

    def save(name):
        report["elapsed_seconds"] = time.monotonic()-started
        (destination / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        artifacts.commit()

    save("loading.json")
    engine = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
        quantization="mxfp4", dtype="bfloat16", max_model_len=8192, max_num_seqs=4,
        gpu_memory_utilization=0.9, enforce_eager=True, trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    for token in ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>"):
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            raise ValueError("Native Harmony tokens changed")
    report["load_seconds"] = time.monotonic()-started
    for offset in range(0, len(cases), 4):
        batch = cases[offset:offset+4]
        prompts = [tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"), tokenize=False,
                   add_generation_prompt=True, reasoning_effort="medium") for row in batch]
        if any(len(tokenizer.encode(p, add_special_tokens=False)) > 5800 for p in prompts):
            raise ValueError("No silent input truncation")
        outputs = engine.generate(prompts, SamplingParams(**report["decoding"]))
        for row, prompt, prediction in zip(batch, prompts, outputs, strict=True):
            value = prediction.outputs[0]
            result = {"id": row["id"], "finish_reason": value.finish_reason, "output_tokens": len(value.token_ids),
                      "raw_output": tokenizer.decode(value.token_ids, skip_special_tokens=False)}
            try:
                parsed = parse(tokenizer, prompt, value.token_ids, row.get("tools"))
                result.update(parsed=parsed, answer=parsed.get("content", ""))
            except (ValueError, TypeError, KeyError) as exc:
                result.update(parse_error=type(exc).__name__, answer="")
            report["results"].append(result)
        save("partial.json")
        print(json.dumps({"completed": offset+len(batch), "total": len(cases)}), flush=True)
    report["diagnostic"] = {
        "empty_answer_and_no_tool_ids": [r["id"] for r in report["results"] if not r["answer"].strip() and not r.get("parsed", {}).get("tool_calls")],
        "length_limit_ids": [r["id"] for r in report["results"] if r["finish_reason"] == "length"],
        "semantic_accuracy": None, "clinical_accuracy": None,
    }
    save("results.json")
    return report
