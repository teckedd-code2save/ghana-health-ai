"""Two pinned English-only stages, fixed source Twi answers retained locally."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, image
from native_answer_core import validate_inputs

app = modal.App("ghana-native-answer-synthesis")
MODELS = {
    "generate": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "review": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "ground": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}
if modal.is_local():
    for name in ("benchmark_stronger_response.py", "native_answer_core.py"):
        image = image.add_local_file(str(Path(__file__).with_name(name)), "/root/" + name)


def cached_model(stage):
    model, revision = MODELS[stage]
    for root in (Path("/cache/hf/hub"), Path("/cache/hf")):
        snapshot = root / ("models--" + model.replace("/", "--")) / "snapshots" / revision
        if (snapshot / "config.json").is_file() and any(snapshot.glob("*.safetensors")):
            return str(snapshot)
    raise ValueError("Pinned model is not cached; no GPU download permitted")


@app.function(image=image, cpu=2, memory=8192, timeout=300, retries=0,
              volumes={"/cache": cache.read_only()})
def preflight():
    from transformers import AutoTokenizer
    cache.reload()
    return {stage: {"model": model, "revision": revision,
        "tokenizer_size": len(AutoTokenizer.from_pretrained(cached_model(stage), local_files_only=True,
                                                             trust_remote_code=False))}
        for stage, (model, revision) in MODELS.items()}


@app.function(image=image, gpu="H100", cpu=4, memory=73728, timeout=1200,
              max_containers=1, single_use_containers=True, retries=0,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def synthesize(run_id: str, stage: str, rows: list[dict]):
    import hashlib
    import json
    import re
    import time
    from jsonschema.exceptions import ValidationError
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from native_answer_core import GENERATION_SCHEMA, REVIEW_SCHEMA, GROUND_SCHEMA, messages, parse_output
    validate_inputs(rows, stage)
    if not re.fullmatch(r"native_answers_\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid research run")
    cache.reload()
    artifacts.reload()
    model, revision = MODELS[stage]
    destination = Path("/artifacts/native-answers", run_id, stage)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"run_id": run_id, "stage": stage, "model": model, "revision": revision,
        "input_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "results": [], "training": False, "twi_targets_generated": 0, "human_reviewed": False,
        "limitations": ["Only English question/answer compatibility is assessed.",
            "The independent model is not a native-language or clinical certifier."]}
    report["prompt_instruction"] = messages(rows[0], stage)[0]["content"]
    report["schema"] = {"generate": GENERATION_SCHEMA, "review": REVIEW_SCHEMA, "ground": GROUND_SCHEMA}[stage]
    report["decoding"] = {"temperature": 0.3 if stage == "generate" else 0,
                          "max_tokens": 768 if stage == "ground" else 384, "seed": 42}

    def save(name):
        report["elapsed_seconds"] = time.monotonic()-started
        (destination / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        artifacts.commit()

    save("loading.json")
    engine = LLM(model=cached_model(stage), dtype="bfloat16",
        language_model_only=True, max_model_len=4096, max_num_seqs=8,
        gpu_memory_utilization=0.9, enforce_eager=True, trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    report["load_seconds"] = time.monotonic()-started
    schema = report["schema"]
    params = SamplingParams(**report["decoding"], structured_outputs=StructuredOutputsParams(json=schema))
    for offset in range(0, len(rows), 8):
        batch = rows[offset:offset+8]
        prompts = [tokenizer.apply_chat_template(messages(r, stage), tokenize=False,
            enable_thinking=False, add_generation_prompt=True) for r in batch]
        if any(len(tokenizer.encode(p, add_special_tokens=False)) > 3500 for p in prompts):
            raise ValueError("No silent source truncation")
        predictions = engine.generate(prompts, params)
        for row, prediction in zip(batch, predictions, strict=True):
            output = prediction.outputs[0]
            result = {"id": row["id"], "raw_output": output.text, "finish_reason": output.finish_reason,
                      "output_tokens": len(output.token_ids)}
            try:
                if output.finish_reason == "length":
                    raise ValueError("Output exhausted token budget")
                result["parsed"] = parse_output(output.text, stage, len(row.get("questions", [])), row["answer_en"])
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                result["error"] = type(exc).__name__
            report["results"].append(result)
        save("partial.json")
        print(json.dumps({"stage": stage, "completed": offset+len(batch), "total": len(rows)}), flush=True)
    save("results.json")
    return report
