"""Private, checkpointed Modal teacher qualification. No training or API relay."""
from pathlib import Path
import modal

PROFILES = {
    "gemma31": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "qwen35": ("Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
}
app = modal.App("ghana-r7-teacher-private")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cpu_image = modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub==1.5.0").env({
    "HF_HOME": "/cache/hf", "HF_XET_CACHE": "/tmp/hf-xet", "DO_NOT_TRACK": "1"})
image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.13").pip_install(
    "vllm==0.28.0", "sacrebleu==2.5.1").pip_install("transformers==5.17.0").env({
    "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1", "TRITON_CACHE_DIR": "/tmp/triton", "VLLM_CACHE_ROOT": "/tmp/vllm",
    "OMP_NUM_THREADS": "1"})


def digest(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(run_id, profile, requests):
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,100}", run_id) or profile not in PROFILES:
        raise ValueError("Invalid immutable run/profile")
    if not isinstance(requests, list) or not 1 <= len(requests) <= 64:
        raise ValueError("Qualification only: 1-64 requests")
    identities = set()
    for row in requests:
        if set(row) != {"id", "messages"} or not isinstance(row["id"], str) or row["id"] in identities:
            raise ValueError("Only distinct model-visible IDs/messages")
        identities.add(row["id"])
        messages = row["messages"]
        if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
            raise ValueError("Invalid message count")
        start = int(messages[0].get("role") == "system")
        if (len(messages) - start) % 2 != 1:
            raise ValueError("Must end in user")
        for i, message in enumerate(messages):
            role = "system" if i == 0 and start else "user" if (i - start) % 2 == 0 else "assistant"
            if (set(message) != {"role", "content"} or message["role"] != role
                    or not isinstance(message["content"], str) or not 1 <= len(message["content"]) <= 24000):
                raise ValueError("Invalid complete text history")
    return digest({"profile": profile, "checkpoint": PROFILES[profile], "requests": requests})


def save(path, value):
    import json
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1800, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def prepare(profile):
    import json
    import time
    from huggingface_hub import snapshot_download
    model, revision = PROFILES[profile]
    cache.reload()
    started = time.monotonic()
    source = Path(snapshot_download(model, revision=revision, local_files_only=profile == "gemma31",
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "README.md", "LICENSE*"], max_workers=4))
    names = sorted(set(json.loads((source / "model.safetensors.index.json").read_text())["weight_map"].values()))
    if not names or any(Path(n).name != n or not (source / n).is_file() for n in names):
        raise ValueError("Incomplete checkpoint")
    result = {"profile": profile, "model": model, "revision": revision, "shards": len(names),
              "bytes": sum((source / n).stat().st_size for n in names),
              "seconds": time.monotonic() - started, "gpu_used": False}
    cache.commit()
    save(Path("/artifacts/understanding-adaptation-r7/teacher-cache") / (profile + ".json"), result)
    artifacts.commit()
    return result


def stage(source, destination):
    import hashlib
    import json
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    names = sorted(set(json.loads((source / "model.safetensors.index.json").read_text())["weight_map"].values()))
    if destination.exists() or not names or any(Path(n).name != n for n in names):
        raise ValueError("Invalid fresh staging location/shards")
    size = sum((source / n).stat().st_size for n in names)
    if shutil.disk_usage(destination.parent).free < size * 1.1:
        raise ValueError("Insufficient ephemeral space")
    destination.mkdir()

    def copy(name):
        path = source / name
        sha = hashlib.sha256()
        with path.open("rb") as src, (destination / name).open("xb") as out:
            while chunk := src.read(16 * 1024 * 1024):
                sha.update(chunk)
                out.write(chunk)
        if sha.hexdigest() != path.resolve().name:
            raise ValueError("HF weight content hash differs")
        return {"name": name, "sha256": sha.hexdigest(), "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        shards = list(pool.map(copy, names))
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".json", ".jinja", ".txt", ".model"}:
            shutil.copyfile(path, destination / path.name)
    return shards


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, retries=0,
              max_containers=1, scaledown_window=60,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def qualify(run_id, profile, requests):
    import json
    import time
    import uuid
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    request_sha = validate(run_id, profile, requests)
    artifacts.reload()
    output = Path("/artifacts/understanding-adaptation-r7/teacher") / run_id
    path = output / "progress.json"
    model, revision = PROFILES[profile]
    report = {"run_id": run_id, "profile": profile, "model": model, "revision": revision,
              "requests_sha256": request_sha, "requests": requests, "results": [],
              "gpu": "H200", "runtime": {"vllm": "0.28.0", "transformers": "5.17.0"},
              "decoding": {"temperature": 0, "max_tokens": 768, "seed": 42},
              "status": "loading", "qualified": False, "human_reviewed": False,
              "started_at_unix": time.time(), "attempts": 0}
    if path.exists():
        report = json.loads(path.read_text())
        if report["requests_sha256"] != request_sha:
            raise ValueError("Immutable run identity changed")
    if report["status"] == "complete":
        return report
    if report["attempts"] >= 2:
        raise ValueError("Inspect failed attempts before further recovery")
    report["attempts"] += 1

    def checkpoint():
        report["elapsed_seconds"] = time.time() - report["started_at_unix"]
        save(path, report)
        artifacts.commit()

    checkpoint()
    try:
        source = Path(snapshot_download(model, revision=revision, local_files_only=True,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"]))
        staged = Path("/tmp") / ("r7-teacher-" + uuid.uuid4().hex)
        report["shards"] = stage(source, staged)
        checkpoint()
        print("Verified checkpoint staged; loading teacher", flush=True)
        engine = LLM(model=str(staged), tokenizer=str(staged), language_model_only=True,
            dtype="bfloat16", max_model_len=8192, max_num_seqs=4, gpu_memory_utilization=.90,
            enforce_eager=True, safetensors_load_strategy="lazy", trust_remote_code=False, seed=42)
        tokenizer = engine.get_tokenizer()
        report["load_seconds"] = time.time() - report["started_at_unix"]
        report["status"] = "running"
        done = {r["id"] for r in report["results"]}
        pending = [r for r in requests if r["id"] not in done]
        for offset in range(0, len(pending), 4):
            batch = pending[offset:offset + 4]
            prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for r in batch]
            if any(len(tokenizer.encode(p, add_special_tokens=False)) + 768 > 8192 for p in prompts):
                raise ValueError("Input too long; no silent truncation")
            tick = time.monotonic()
            predictions = engine.generate(prompts, SamplingParams(**report["decoding"]))
            for row, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                result = prediction.outputs[0]
                report["results"].append({"id": row["id"], "output": result.text,
                    "prompt_sha256": digest(prompt), "finish_reason": result.finish_reason,
                    "input_tokens": len(prediction.prompt_token_ids), "output_tokens": len(result.token_ids),
                    "batch_seconds": time.monotonic() - tick})
            checkpoint()
            print(json.dumps({"completed": len(report["results"]), "total": len(requests)}), flush=True)
        report["status"] = "complete"
        checkpoint()
        return report
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)[:2000]
        checkpoint()
        raise
