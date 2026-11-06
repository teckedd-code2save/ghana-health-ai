"""Bounded six-checkpoint evaluation with per-example durable output journals."""
import os
from pathlib import Path
import sys
import modal

BUNDLE = Path(os.environ.get("GHA_R2_EVAL_BUNDLE", "/bundle")) if modal.is_local() else Path("/bundle")
sys.path.insert(0, str(BUNDLE / "code"))
from r2_inference import RUNTIME, load_bundle

app = modal.App("ghana-understanding-paired-evaluation-r2")
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
claims = modal.Dict.from_name("ghana-understanding-evaluation-r2-invocations", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in RUNTIME.items()]
).env({"PYTHONPATH": "/bundle/code", "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1",
       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
if modal.is_local():
    load_bundle(BUNDLE)
    image = image.add_local_dir(str(BUNDLE), "/bundle")


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200,
              retries=0, max_containers=1, min_containers=0, single_use_containers=True,
              volumes={"/artifacts": artifacts, "/cache": cache.read_only()}, include_source=False)
def evaluate(identifier: str, expected_sha: str):
    from collections import Counter
    import math
    import re
    import time
    from r2_inference import Runtime, read, save, sha, digest, require, VARIANTS, DECODING, reference_record
    from build_understanding_evaluation import objective_score
    started = time.monotonic()
    require(re.fullmatch(r"r2_eval_\d{8}T\d{6}Z", identifier), "Invalid evaluation ID")
    require(claims.put(identifier, {"call_id": modal.current_function_call_id(), "manifest_sha256": expected_sha}, skip_if_exists=True), "Duplicate run")
    artifacts.reload(); cache.reload()
    folder = Path("/artifacts/understanding-evaluation-r2", identifier)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = load_bundle(BUNDLE, expected_sha)
    cases = read(BUNDLE / "evaluation/cases.json")
    heldout = read(BUNDLE / "evaluation/language-heldout.json")
    references = {(r["checkpoint_id"], r["case_id"]): r for r in read(BUNDLE / "evaluation/reference-records.json")}
    results, losses = {}, {}
    def status(state, **extra):
        value = {"state": state, "run_id": identifier, "manifest_sha256": expected_sha,
            "elapsed_seconds": time.monotonic() - started, "chat_slots": sum(len(v) for v in results.values()),
            "language_rows": sum(len(v) for v in losses.values()), "training": False,
            "semantic_quality_established": False, **extra}
        save(folder / "status.json", value); artifacts.commit()
        return value
    def persist(path, value, commit=True):
        require(not path.exists(), "Record already exists; never overwrite a completed request")
        save(path, value)
        if commit:
            artifacts.commit()
    status("loading")
    try:
        runtime = Runtime(BUNDLE, expected_sha)
        for variant in ("r2_response", "r2_meaning", "r2_language", "base", "r1_response", "r1_meaning"):
            results[variant], losses[variant] = [], []
            status("evaluating", variant=variant)
            for case in cases:
                if time.monotonic() - started >= manifest["limits"]["soft_seconds"]:
                    return status("paused_budget", variant=variant)
                key = (variant, case["id"])
                ref = references.get(key)
                if ref:
                    record = {**reference_record(ref, case, runtime.tokenizer), "identity": runtime.identity(variant)}
                else:
                    value = runtime.reply(case["messages"], variant)
                    record = {**value, "case_id": case["id"], "origin": "generated_r2_evaluation",
                              "prompt_token_ids_sha256": value["prompt_ids_sha256"], "generated": True}
                record["objective"] = objective_score(case, record)
                persist(folder / variant / "chat" / (case["id"] + ".json"), record)
                results[variant].append(record)
                status("evaluating_chat", variant=variant)
            for index, row in enumerate(heldout):
                if time.monotonic() - started >= manifest["limits"]["soft_seconds"]:
                    return status("paused_budget", variant=variant)
                value = {"id": row["id"], "source_id": row["source_id"],
                         **runtime.language_loss(row["text"], variant)}
                # Do not copy held-out text into demo or report artifacts.
                persist(folder / variant / "loss" / (str(index).zfill(4) + ".json"), value, commit=False)
                losses[variant].append(value)
                if (index + 1) % 25 == 0 or index + 1 == len(heldout):
                    status("evaluating_language_loss", variant=variant)
        summary = {}
        for variant in VARIANTS:
            records, terms = results[variant], losses[variant]
            tokens = sum(t["predicted_tokens"] for t in terms)
            nll = sum(t["summed_nll"] for t in terms)
            objectives = [r["objective"] for r in records if r["objective"] is not None]
            summary[variant] = {"chat_cases": len(records), "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
                "strict_objective_passed": sum(r["passed"] for r in objectives), "strict_objective_cases": len(objectives),
                "replies_with_repeated_4grams": sum(r["diagnostics"]["repetition"]["4_word_ngrams"]["repeated_positions"] > 0 for r in records),
                "language_loss": {"rows": len(terms), "predicted_tokens": tokens, "summed_nll": nll,
                                  "token_weighted_mean_nll": nll / tokens, "exp_mean_nll": math.exp(nll / tokens)},
                "native_review": "pending"}
        save(folder / "summary.json", {"checkpoints": summary, "manifest_sha256": expected_sha,
            "claims": "Development diagnostics only. Strict JSON is task-specific; loss/repetition do not certify meaning or safety.",
            "primary_pairs": read(BUNDLE / "evaluation/comparison-plan.json")["primary_pairs"]})
        save(folder / "results-manifest.json", {"execution_sha256": expected_sha,
            "files": {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob("*.json") if p.name != "status.json"}})
        return status("completed", recorded_slots=216, language_records=2058)
    except Exception as error:
        status("failed", error_type=type(error).__name__, error=str(error)[:2000])
        raise
