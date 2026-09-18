"""Private bounded calibration; no training, public endpoints or hidden fallback."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, image
from dialogue_translation_core import MODELS, digest, requests, restore_progress

app = modal.App("ghana-dialogue-translation-calibration")
if modal.is_local():
    for filename in ("benchmark_stronger_response.py", "dialogue_translation_core.py"):
        image = image.add_local_file(str(Path(__file__).with_name(filename)), "/root/" + filename)


def run_translation(run_id: str, variant: str, rows: list[dict], demonstrations: list[dict]):
    import json
    import re
    import time
    import torch
    from vllm import LLM, SamplingParams
    prepared = requests(rows, variant, demonstrations)
    if not re.fullmatch(r"dialogue_\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid immutable run ID")
    cache.reload()
    artifacts.reload()
    output = Path("/artifacts/dialogue-translation", run_id, variant)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, revision = MODELS[variant]
    decoding = {"temperature": 0, "max_tokens": 768, "seed": 42}
    if variant == "afrique":
        # Stop at a new demonstration, not at a legitimate translated paragraph.
        decoding["stop"] = ["\nEnglish:", "\n\nEnglish:"]
    report = {"run_id": run_id, "variant": variant, "model": model, "revision": revision,
              "inputs_sha256": digest(rows), "demonstrations_sha256": digest(demonstrations),
              "requests_sha256": digest(prepared), "inputs": rows, "demonstrations": demonstrations,
              "requests": prepared, "decoding": decoding, "results": [],
              "runtime": "vllm==0.28.0; transformers==5.17.0", "thinking": False,
              "started_at_unix": time.time(), "resume_count": 0,
              "gpu_actual": torch.cuda.get_device_name(), "training": False,
              "human_verified": False, "production_promoted": False}
    report = restore_progress(output, report)
    if report["resume_count"] > 2 or time.time() - report["started_at_unix"] > 1200:
        raise RuntimeError("Calibration recovery budget exhausted; inspect before another attempt")
    done = {r["id"] for r in report["results"]}
    if len(done) == len(prepared):
        return report
    pending = [r for r in prepared if r["id"] not in done]

    def save(name):
        report["elapsed_seconds"] = round(time.time() - report["started_at_unix"], 3)
        temporary = output / (name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output / name)
        artifacts.commit()

    # Persist recovery accounting before model loading can be preempted again.
    save("partial.json")
    engine = LLM(model=model, revision=revision, tokenizer_revision=revision,
                 language_model_only=True, dtype="bfloat16", max_model_len=8192,
                 max_num_seqs=8, gpu_memory_utilization=0.88, enforce_eager=True,
                 trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    report["latest_attempt_load_seconds"] = round(time.monotonic() - started, 3)
    for offset in range(0, len(pending), 8):
        if time.time() - report["started_at_unix"] > 1200:
            raise RuntimeError("Calibration wall-time budget exhausted")
        batch = pending[offset:offset+8]
        prompts = [r["prompt"] if "prompt" in r else tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        if any(len(tokenizer.encode(p, add_special_tokens=False)) > 7000 for p in prompts):
            raise ValueError("Input exceeds budget; no silent truncation")
        batch_started = time.monotonic()
        predictions = engine.generate(prompts, SamplingParams(**decoding))
        seconds = round(time.monotonic() - batch_started, 3)
        for row, prompt, result in zip(batch, prompts, predictions, strict=True):
            value = result.outputs[0]
            report["results"].append({k: row[k] for k in ("id", "source_id", "turn", "method")} | {
                "translation": value.text, "raw_output": tokenizer.decode(value.token_ids, skip_special_tokens=False),
                "prompt_sha256": digest(prompt), "prompt_tokens": len(result.prompt_token_ids),
                "output_tokens": len(value.token_ids), "finish_reason": value.finish_reason,
                "stop_reason": value.stop_reason, "batch_seconds": seconds})
        save("partial.json")
        print(json.dumps({"variant": variant, "completed": len(report["results"]), "total": len(prepared)}), flush=True)
    save("results.json")
    return report


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=49152, timeout=1200,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def translate_afrique(run_id: str, rows: list[dict], demonstrations: list[dict]):
    return run_translation(run_id, "afrique", rows, demonstrations)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, timeout=1200,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def translate_gemma(run_id: str, rows: list[dict], demonstrations: list[dict]):
    return run_translation(run_id, "gemma", rows, demonstrations)
