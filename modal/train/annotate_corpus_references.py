"""Bounded, private English-reference annotation. No translation or training."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
app = modal.App("ghana-corpus-reference-annotation")
_engine = None
if modal.is_local():
    for name in ("benchmark_stronger_response.py", "corpus_checkpoint.py"):
        source = str(Path(__file__).with_name(name))
        image = image.add_local_file(source, "/root/" + name)
        cpu_image = cpu_image.add_local_file(source, "/root/" + name)


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=900, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "reference-qwen9", False)


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=65536, timeout=900,
              retries=0, max_containers=1, scaledown_window=60,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def annotate(run_id: str, requests: list[dict]):
    return annotate_model(run_id, requests, MODEL, REVISION)


def annotate_model(run_id, requests, model, revision, parallel=1, quantization=None, harmony=False):
    global _engine
    import hashlib
    import json
    import re
    import time
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from corpus_checkpoint import stage_checkpoint

    if not re.fullmatch(r"reference_[a-f0-9]{24}", run_id):
        raise ValueError("Invalid immutable run ID")
    if not 1 <= len(requests) <= 512 or len({r["id"] for r in requests}) != len(requests):
        raise ValueError("At most 512 unique reference requests")
    for row in requests:
        if set(row) != {"id", "messages"} or [m["role"] for m in row["messages"]] != ["system", "user"]:
            raise ValueError("Only model-visible messages, no rubric")
        if any(set(m) != {"role", "content"} or not isinstance(m["content"], str) or len(m["content"]) > 12000 for m in row["messages"]):
            raise ValueError("Invalid message")
    request_hash = hashlib.sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    artifacts.reload(); cache.reload()
    folder = Path("/artifacts/corpus-reference", run_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "partial.json"
    started = time.monotonic()
    report = {"model": model, "revision": revision, "run_id": run_id, "request_hash": request_hash,
              "results": [], "training": False, "translation": False, "provider": "modal",
              "proprietary_fallback": False, "gpu": "H200:2" if parallel == 2 else "H100" if harmony else "A100-40GB", "max_seconds": 900,
              "reasoning_effort": "medium" if harmony else None,
              "decoding": {"temperature": 1 if harmony else 0, "max_tokens": 2048 if harmony else 1024, "seed": 42}}
    if path.exists():
        report = json.loads(path.read_text())
        if report["request_hash"] != request_hash or report["revision"] != revision:
            raise ValueError("Changed resumable request")
    done = {r["id"] for r in report["results"]}
    if len(done) == len(requests):
        return report
    report["attempts"] = report.get("attempts", 0) + 1
    if report["attempts"] > 2:
        raise ValueError("Inspect interrupted run before further recovery")

    def save():
        report["last_attempt_seconds"] = time.monotonic() - started
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False))
        temp.replace(path); artifacts.commit()

    save()
    report["reused_engine"] = _engine is not None
    if _engine is None:
        source = Path(snapshot_download(model, revision=revision, local_files_only=True,
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md", "LICENSE*"],
            ignore_patterns=["original/*", "metal/*", "onnx/*"]))
        staged = Path("/tmp") / run_id
        report["staging"] = stage_checkpoint(source, staged, revision)
        save()
        _engine = LLM(model=str(staged), tokenizer=str(staged), language_model_only=True,
                      tensor_parallel_size=parallel, quantization=quantization,
                      dtype="bfloat16", max_model_len=8192 if harmony else 4096, max_num_seqs=16,
                      safetensors_load_strategy="lazy", gpu_memory_utilization=.9,
                      enforce_eager=True, trust_remote_code=False, seed=42)
    tokenizer = _engine.get_tokenizer()
    report["load_seconds"] = time.monotonic() - started
    pending = [r for r in requests if r["id"] not in done]
    for offset in range(0, len(pending), 16):
        if time.monotonic() - started > 820:
            save()
            return report
        batch = pending[offset:offset + 16]
        template = {"reasoning_effort": "medium"} if harmony else {"enable_thinking": False}
        prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                   add_generation_prompt=True, **template) for r in batch]
        if any(len(tokenizer.encode(p)) > 3000 for p in prompts):
            raise ValueError("Input exceeds budget; no truncation")
        tick = time.monotonic()
        outputs = _engine.generate(prompts, SamplingParams(**report["decoding"]))
        for row, p, output in zip(batch, prompts, outputs, strict=True):
            result = output.outputs[0]
            text = result.text
            diagnostics = {}
            if harmony:
                from oss_response_core import parse
                parsed = parse(tokenizer, p, result.token_ids)
                text = parsed.get("content", "")
                diagnostics = {"raw_harmony_output": tokenizer.decode(result.token_ids, skip_special_tokens=False),
                    "stop_reason": result.stop_reason,
                    "last_token_id": result.token_ids[-1] if result.token_ids else None,
                    "parsed_fields": sorted(parsed), "final_channel_missing": not bool(text.strip())}
                if parsed.get("tool_calls"):
                    raise ValueError("Annotation does not permit tool calls")
            report["results"].append({"id": row["id"], "text": text,
                "finish_reason": result.finish_reason, "output_tokens": len(result.token_ids),
                "prompt_tokens": len(output.prompt_token_ids), "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(),
                "batch_seconds": time.monotonic() - tick, **diagnostics})
        save()
        print(json.dumps({"completed": len(report["results"]), "total": len(requests)}), flush=True)
    return report
