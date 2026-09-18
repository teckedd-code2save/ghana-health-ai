"""Bounded private Modal-only qualification of the proposed corpus teacher."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
REVISION = "e156cb4efae43fbee1a1ab073f946a1377e6b969"
app = modal.App("ghana-corpus-teacher-qualification")
_engine = None
_staging = None
if modal.is_local():
    image = image.add_local_file(str(Path(__file__).with_name("benchmark_stronger_response.py")), "/root/benchmark_stronger_response.py")
    cpu_image = cpu_image.add_local_file(str(Path(__file__).with_name("benchmark_stronger_response.py")), "/root/benchmark_stronger_response.py")
    image = image.add_local_file(str(Path(__file__).with_name("corpus_checkpoint.py")), "/root/corpus_checkpoint.py")


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1800, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "qwen235-corpus", True)


@app.function(image=image, gpu="H200:2", cpu=8, memory=196608, timeout=1200,
              retries=0, max_containers=1, scaledown_window=60,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def qualify(run_id: str, requests: list[dict]):
    global _engine, _staging
    import hashlib
    import json
    import re
    import time
    from vllm import LLM, SamplingParams
    if not re.fullmatch(r"corpus_teacher_\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid immutable run ID")
    if not 1 <= len(requests) <= 64 or len({r["id"] for r in requests}) != len(requests):
        raise ValueError("At most 64 unique qualification requests")
    for row in requests:
        if set(row) != {"id", "messages"} or not 1 <= len(row["messages"]) <= 2:
            raise ValueError("Only public model-visible messages")
        if any(set(m) != {"role", "content"} or m["role"] not in ("system", "user") or
               not isinstance(m["content"], str) or len(m["content"]) > 16000 for m in row["messages"]):
            raise ValueError("Invalid message")
    request_hash = hashlib.sha256(json.dumps(requests, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    artifacts.reload(); cache.reload()
    output = Path("/artifacts/corpus-teacher", run_id)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "partial.json"
    report = {"run_id": run_id, "model": MODEL, "revision": REVISION, "request_hash": request_hash,
              "started_at_unix": time.time(), "results": [], "runtime": "vllm 0.28.0; transformers 5.17.0",
              "gpu": "H200:2", "max_seconds": 1200, "qualified": False, "training": False,
              "weight_load_strategy": "verified_ephemeral_staging_then_local_lazy",
              "human_reviewed": False, "attempts": 0, "decoding": {"temperature": 0, "max_tokens": 1024, "seed": 42}}
    if partial.exists():
        report = json.loads(partial.read_text())
        if report["request_hash"] != request_hash or report["revision"] != REVISION:
            raise ValueError("Changed resumable request")
    report["attempts"] += 1
    if report["attempts"] > 2 or time.time() - report["started_at_unix"] > 1200:
        raise ValueError("Recovery budget exhausted; inspect before retry")

    def save():
        report["elapsed_seconds"] = time.time() - report["started_at_unix"]
        temporary = partial.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(partial)
        artifacts.commit()

    done = {r["id"] for r in report["results"]}
    if len(done) == len(requests):
        return report
    save()
    report["reused_loaded_engine"] = _engine is not None
    if _engine is None:
        from huggingface_hub import snapshot_download
        from corpus_checkpoint import stage_checkpoint
        source = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
        staged = Path("/tmp") / ("corpus-qwen235-" + run_id)
        _staging = stage_checkpoint(source, staged, REVISION)
        report["staging"] = _staging
        save()
        print(json.dumps({"phase": "staged", "seconds": _staging["seconds"], "bytes": _staging["bytes"]}), flush=True)
        _engine = LLM(model=str(staged), tokenizer=str(staged),
                     dtype="bfloat16", quantization="fp8", tensor_parallel_size=2,
                     safetensors_load_strategy="lazy",
                     max_model_len=8192, max_num_seqs=4, gpu_memory_utilization=.94,
                     enforce_eager=True, trust_remote_code=False, seed=42)
    engine = _engine
    report["staging"] = _staging
    tokenizer = engine.get_tokenizer()
    report["load_seconds"] = time.time() - report["started_at_unix"]
    pending = [r for r in requests if r["id"] not in done]
    for offset in range(0, len(pending), 4):
        if time.time() - report["started_at_unix"] > 1150:
            raise TimeoutError("Qualification budget reached")
        batch = pending[offset:offset + 4]
        prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
        if any(len(tokenizer.encode(p)) > 7000 for p in prompts):
            raise ValueError("No silent source truncation")
        start = time.monotonic()
        predictions = engine.generate(prompts, SamplingParams(**report["decoding"]))
        elapsed = time.monotonic() - start
        for r, p, prediction in zip(batch, prompts, predictions, strict=True):
            result = prediction.outputs[0]
            report["results"].append({"id": r["id"], "text": result.text, "finish_reason": result.finish_reason,
                "prompt_hash": hashlib.sha256(p.encode()).hexdigest(), "output_tokens": len(result.token_ids),
                "prompt_tokens": len(prediction.prompt_token_ids), "batch_seconds": elapsed})
        save()
        print(json.dumps({"completed": len(report["results"]), "total": len(requests)}), flush=True)
    return report
