"""Paired source-language evaluation for two base models, not a chat deployment."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights

MODELS = {
    "base": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
    "afrique": ("McGill-NLP/AfriqueQwen3.5-9B-50Langs", "4358dcbc062421751174279da1efe9f88f85e1d5"),
}
app = modal.App("ghana-afrique-native-comparison")
if modal.is_local():
    source = str(Path(__file__).with_name("benchmark_stronger_response.py"))
    cpu_image = cpu_image.add_local_file(source, "/root/benchmark_stronger_response.py")
    image = image.add_local_file(source, "/root/benchmark_stronger_response.py")


def validate_inputs(inputs):
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 1000:
        raise ValueError("Expected bounded source inputs")
    for row in inputs:
        if not isinstance(row, dict) or set(row) != {"id", "task", "prompt"} or row["task"] not in ("nli", "translation"):
            raise ValueError("Only model inputs; no held-out answers")
        if not isinstance(row["id"], str) or not isinstance(row["prompt"], str) or not 1 <= len(row["prompt"]) <= 16000:
            raise ValueError("Invalid source input")
    if len({r["id"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate case")


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1200, retries=0, max_containers=1,
              volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    cache.reload()
    return {key: prepare_weights(model, revision, "afrique9-" + key, False, reload_cache=False)
            for key, (model, revision) in MODELS.items()}


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=49152, timeout=1200, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def evaluate(run_id: str, variant: str, inputs: list[dict]):
    import hashlib
    import json
    import re
    import time
    import torch
    from vllm import LLM, SamplingParams
    validate_inputs(inputs)
    if variant not in MODELS or not re.fullmatch(r"afrique9_\d{8}T\d{6}Z", run_id):
        raise ValueError("Unknown comparison")
    artifacts.reload()
    cache.reload()
    model, revision = MODELS[variant]
    output = Path("/artifacts/native-foundation", run_id, variant)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"run_id": run_id, "variant": variant, "model": model, "revision": revision,
        "gpu_actual": torch.cuda.get_device_name(0), "results": [],
        "inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "training": False, "chat_quality_evaluated": False, "tools_executed": False,
        "nli_method": "Three English demonstrations, counterbalanced choices, greedy one-token restricted A/B/C classification",
        "translation_method": "Three unchanged training-source demonstrations per direction, greedy continuation, newline stop, 256-token ceiling"}

    def save(name):
        report["elapsed_seconds"] = time.monotonic()-started
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        artifacts.commit()

    save("loading.json")
    engine = LLM(model=model, revision=revision, tokenizer_revision=revision, language_model_only=True,
        dtype="bfloat16", max_model_len=4096, max_num_seqs=32, gpu_memory_utilization=0.85,
        enforce_eager=True, trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(ids) != 1 for ids in labels) or len({ids[0] for ids in labels}) != 3:
        raise ValueError("Classifier labels must have distinct single tokens")
    report["load_seconds"] = time.monotonic()-started
    for task in ("nli", "translation"):
        rows = [row for row in inputs if row["task"] == task]
        params = (SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=[ids[0] for ids in labels], seed=42)
                  if task == "nli" else SamplingParams(temperature=0, max_tokens=256, stop=["\n"], seed=42))
        for offset in range(0, len(rows), 32):
            batch = rows[offset:offset+32]
            if any(len(tokenizer.encode(row["prompt"])) > 3700 for row in batch):
                raise ValueError("Source prompt exceeds context; no truncation")
            predictions = engine.generate([row["prompt"] for row in batch], params)
            for row, prediction in zip(batch, predictions, strict=True):
                value = prediction.outputs[0]
                report["results"].append({"id": row["id"], "task": task, "prediction": value.text,
                    "output_tokens": len(value.token_ids), "finish_reason": value.finish_reason})
            save("partial.json")
            print(json.dumps({"variant": variant, "task": task, "completed": offset+len(batch)}), flush=True)
    save("results.json")
    return report
