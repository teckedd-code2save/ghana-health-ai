"""Bounded language-specific comparator, not a trained project model or teacher."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights, validate_cases

MODEL = "ghananlpcommunity/MiniCPM5-1B-Twi"
REVISION = "d807ca1a3323972afafabff8f9affe2639e37b5c"
app = modal.App("ghana-native-fluency-comparison")
if modal.is_local():
    for name in ("benchmark_stronger_response.py",):
        cpu_image = cpu_image.add_local_file(str(Path(__file__).with_name(name)), "/root/" + name)
        image = image.add_local_file(str(Path(__file__).with_name(name)), "/root/" + name)


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=600, retries=0, max_containers=1,
              volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "minicpm_twi", False)


@app.function(image=image, cpu=2, memory=8192, timeout=300, retries=0,
              volumes={"/cache": cache.read_only()})
def preflight():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache.reload()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")):
        raise ValueError("Incomplete source weights")
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0] or len(tokenizer) != 136560:
        raise ValueError("Extended tokenizer and embeddings disagree")
    inputs = tokenizer("Me pa wo kyɛw.", return_tensors="pt")
    with torch.inference_mode():
        logits = model.eval()(**inputs).logits[:, -1]
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite source forward pass")
    return {"model": MODEL, "revision": REVISION, "vocabulary": len(tokenizer),
            "weight_loading_complete": True, "finite_forward": True, "gpu_used": False}


@app.function(image=image, gpu="T4", cpu=2, memory=16384, timeout=900, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def compare(run_id: str, cases: list[dict]):
    import hashlib
    import json
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from response_runtime import QWEN_RESPONSE_TEMPLATE
    validate_cases(cases)
    if not re.fullmatch(r"minicpm_twi_\d{8}T\d{6}Z", run_id) or any(r.get("tools") for r in cases):
        raise ValueError("Only non-tool language comparison cases are supported")
    cache.reload()
    artifacts.reload()
    output = Path("/artifacts/stronger-response", run_id)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")):
        raise ValueError("Incomplete language-model checkpoint")
    model = model.to("cuda").eval()
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0] or len(tokenizer) != 136560:
        raise ValueError("Extended Twi tokenizer and model disagree")
    report = {"run_id": run_id, "model": MODEL, "revision": REVISION,
        "inputs": cases, "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "gpu_actual": torch.cuda.get_device_name(), "runtime": "transformers==5.17.0",
        "quantization": "none; publisher FP16", "reasoning_effort": "disabled",
        "decoding": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50,
                     "repetition_penalty": 1.3, "no_repeat_ngram_size": 3, "max_new_tokens": 384},
        "seed_per_case": 42, "load_seconds": time.monotonic()-started,
        "results": [], "training": False, "proprietary_fallback": False,
        "tools_executed": False, "production_promoted": False}

    def save(name):
        report["elapsed_seconds"] = time.monotonic()-started
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        artifacts.commit()

    save("loading.json")
    for row in cases:
        set_seed(42)
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        if tokens.input_ids.shape[-1] > 3000:
            raise ValueError("No silent source truncation")
        case_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**tokens, **report["decoding"], eos_token_id=[1, 130073], pad_token_id=1)
        ids = generated[0, tokens.input_ids.shape[-1]:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        parser = tokenizer.get_response_parser(response_template=QWEN_RESPONSE_TEMPLATE, prefix=prompt)
        parser.feed(raw)
        parsed, _ = parser.finalize()
        report["results"].append({"id": row["id"], "answer": parsed.get("content", ""), "parsed": parsed,
            "raw_output": raw, "output_tokens": len(ids), "finish_reason": "stop" if ids[-1] in (1, 130073) else "length",
            "generation_seconds": time.monotonic()-case_started})
        save("partial.json")
        print(json.dumps({"completed": len(report["results"]), "total": len(cases)}), flush=True)
    save("results.json")
    return report
