#!/usr/bin/env python3
"""Freeze, launch and collect a private R1/R2 paired evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from r2_inference import (RUNTIME, VARIANTS, DECODING, MODEL, REVISION, BASE_SHA,
                          adapter_config, load_bundle, read, require, safe, save, sha, reference_record)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai")
VOLUME = "ghana-health-understand-train"
PREFIX = "/understanding-evaluation-r2"
HELPERS = ("scripts/check_afrique_training_contract.py", "scripts/training_data_inputs.py",
           "modal/train/afrique_instruction_baseline_core.py", "modal/train/afrique_response_core.py",
           "modal/train/native_adaptation_run_core.py", "modal/train/medical_pilot_core.py")


def remote_bytes(volume, path):
    data = bytearray()
    for chunk in volume.read_file(path.removeprefix("/artifacts")):
        data.extend(chunk)
        require(len(data) <= 12_000_000, "Metadata exceeds bound")
    return bytes(data)


def check_completion(status, execution, stages):
    require(status.get("state") == "completed" and status.get("training_completed") is True,
            "Completed training required")
    require(status["mode"] == execution["mode"] == "train" and execution["runtime"] == RUNTIME,
            "Wrong training/runtime")
    require(status["data_sha256"] == execution["data_sha256"], "Training data identity differs")
    require([s["stage"] for s in status["stages"]] == [s["name"] for s in stages], "Wrong stage sequence")
    for actual, plan in zip(status["stages"], stages, strict=True):
        pos = actual["position"]
        require(actual["status"] == "completed" and actual["completion_verified"] is True
                and pos["completed_updates"] == plan["optimizer_updates"]
                and pos["completed_microbatches"] == plan["scheduled_sequences"]
                and pos["microbatches_in_update"] == 0
                and pos["exposure"]["effective_target_tokens"] == plan["effective_target_tokens"]
                and pos["exposure"]["input_tokens"] == plan["sequence_tokens"], "Incomplete training exposure")


def prepare(receipt, evaluation, out):
    import modal
    from prepare_understanding_r2_evaluation import verify_pack
    value = read(receipt)
    os.environ["MODAL_PROFILE"] = value["profile"]
    require(value["state"] == "submitted" and not out.exists(), "Use a fresh immutable bundle directory")
    package = Path(value["package"])
    verify_pack(evaluation, package / "inputs")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    terminal = modal.FunctionCall.from_id(value["call_id"]).get(timeout=0)
    root = "/artifacts" + value["remote_directory"]
    evidence = {}
    def fetch(path):
        raw = remote_bytes(volume, path)
        evidence[path] = raw
        return json.loads(raw)
    status = fetch(root + "/status.json")
    execution = fetch(root + "/execution.json")
    require(status == terminal and status["run_id"] == value["run_id"], "Terminal outcome differs")
    require(hashlib.sha256(evidence[root + "/execution.json"]).hexdigest() == value["execution_sha256"], "Wrong training code")
    stages = read(package / "inputs/stages.json")
    check_completion(status, execution, stages)
    eval_manifest = read(evaluation / "manifest.json")
    require(status["evaluation_manifest_sha256"] == sha(evaluation / "manifest.json")
            and execution["data_sha256"] == eval_manifest["bindings"]["training_manifest_sha256"], "Evaluation not bound to training")
    fetch(root + "/inputs/manifest.json")
    require(hashlib.sha256(evidence[root + "/inputs/manifest.json"]).hexdigest() == value["data_sha256"], "Data manifest changed")
    exports = fetch(root + "/adapters.json")
    require(exports == status["adapters"] and len(exports) == 3
            and all(e["checkpoint_adapter_tensors_verified"] is True for e in exports), "Unverified adapter exports")
    if value.get("resume_run_id"):
        resume, restore = fetch(root + "/resume.json"), fetch(root + "/restore-proof.json")
        require(resume["previous_run_id"] == value["resume_run_id"] and resume["checkpoint_checksums_verified"] is True
                and restore["equivalent"] is True and restore["completed_updates"] == resume["completed_updates"],
                "Missing exact restoration evidence")
        require(set(restore["checked_sections"]) == {"identity", "model", "model_modes", "optimizer", "position", "rng", "runtime", "scheduler", "version"}, "Partial restoration proof")
    plan = read(evaluation / "comparison-plan.json")
    r1root = read(evaluation / "r1-execution.json")["training_directory"]
    r1status = fetch(r1root + "/status.json")
    require(r1status["state"] == "completed" and r1status["training_completed"] is True, "R1 baseline not completed")
    variants = {"base": None}
    for item in plan["checkpoints"]:
        name = item["id"]
        if name == "base":
            continue
        if item["origin"] == "r1":
            stage = "meaning" if name == "r1_meaning" else "responses"
            directory = r1root + "/" + stage + "-adapter"
            weights = item["adapter_weights_sha256"]
        else:
            stage = item["training_stage"]
            export = next(e for e in exports if e["stage"] == stage)
            directory = root + "/" + stage + "-adapter"
            require(export["directory"] == status["physical_directory"] + "/" + stage + "-adapter", "Export outside run")
            weights = export["files"]["adapter_model.safetensors"]
        config = fetch(directory + "/adapter_config.json")
        adapter_config(config)
        config_sha = hashlib.sha256(evidence[directory + "/adapter_config.json"]).hexdigest()
        if item["origin"] == "r2":
            require(config_sha == export["files"]["adapter_config.json"], "Export config changed")
        variants[name] = {"directory": directory, "files": {
            "adapter_config.json": config_sha, "adapter_model.safetensors": weights}}
    out.mkdir(parents=True)
    shutil.copytree(evaluation, out / "evaluation")
    shutil.copytree(package / "inputs/tokenizer", out / "tokenizer")
    (out / "code").mkdir()
    for name in HELPERS:
        shutil.copyfile(package / "frozen" / name, out / "code" / Path(name).name)
    shutil.copyfile(ORIGINAL / "scripts/build_understanding_evaluation.py", out / "code/build_understanding_evaluation.py")
    for name in ("scripts/r2_inference.py", "scripts/run_understanding_r2_evaluation.py", "modal/train/evaluate_understanding_r2.py"):
        shutil.copyfile(ROOT / name, out / "code" / Path(name).name)
    sys.path.insert(0, str(out / "code"))
    from check_afrique_training_contract import load_tokenizer
    tokenizer = load_tokenizer(out / "tokenizer")
    cases = {c["id"]: c for c in read(evaluation / "cases.json")}
    references = read(evaluation / "reference-records.json")
    for ref in references:
        reference_record(ref, cases[ref["case_id"]], tokenizer)
    save(out / "reference-preflight.json", {"checked": len(references), "native_prompts_verified": True,
        "frozen_references_modified": False})
    save(out / "training-proof.json", {p: json.loads(b) for p, b in evidence.items()})
    manifest = {"version": "r2-paired-inference-v1", "model": MODEL, "revision": REVISION,
        "base_sha256": BASE_SHA, "runtime": RUNTIME, "decoding": DECODING, "private": True, "training": False,
        "r2_run_id": value["run_id"], "r2_call_id": value["call_id"], "evaluation_sha256": sha(evaluation / "manifest.json"),
        "variants": variants, "remote_files": {p: hashlib.sha256(b).hexdigest() for p, b in evidence.items()},
        "files": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limits": {"hard_seconds": 7200, "soft_seconds": 6900, "reply_seconds": 60},
        "saved_reference_policy": "Reuse only exact messages/native prompt/protocol/runtime/checkpoint; preserve original provenance.",
        "created_at": datetime.now(timezone.utc).isoformat()}
    save(out / "manifest.json", manifest)
    load_bundle(out, sha(out / "manifest.json"))
    return {"bundle": str(out), "manifest_sha256": sha(out / "manifest.json"), "training_completed": True,
            "variants": list(variants), "chat_cases": 36, "loss_rows_per_variant": 343}


def launch(bundle, receipt, authorize):
    require(authorize and not receipt.exists(), "Explicit GPU authorization and new receipt required")
    manifest = load_bundle(bundle)
    os.environ.update(GHA_R2_EVAL_BUNDLE=str(bundle.resolve()), MODAL_PROFILE="createdliving1000")
    import modal
    identifier = "r2_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"state": "submitting", "run_id": identifier, "bundle": str(bundle.resolve()),
              "manifest_sha256": sha(bundle / "manifest.json"), "remote_directory": PREFIX + "/" + identifier}
    save(receipt, record)
    spec = importlib.util.spec_from_file_location("evaluate_understanding_r2", bundle / "code/evaluate_understanding_r2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with modal.enable_output(), module.app.run(detach=True):
        call = module.evaluate.spawn(identifier, record["manifest_sha256"])
        record.update(state="submitted", call_id=call.object_id)
        save(receipt, record)
    return record


def status(receipt):
    import modal
    value = read(receipt)
    require(value.get("call_id"), "Ambiguous submission; inspect rather than retry")
    try:
        result = modal.FunctionCall.from_id(value["call_id"]).get(timeout=0)
    except TimeoutError:
        return {"state": "pending", "call_id": value["call_id"]}
    save(receipt.with_suffix(".result.json"), result)
    return result


def collect(receipt, out):
    import modal
    value = read(receipt)
    require(not out.exists(), "Use a new snapshot directory")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    prefix = value["remote_directory"]
    listing = list(volume.iterdir(prefix, recursive=True))
    out.mkdir(parents=True)
    hashes = {}
    for entry in listing:
        if entry.type.name != "FILE":
            continue
        remote = "/" + entry.path.lstrip("/")
        require(remote.startswith(prefix + "/"), "Unexpected remote prefix")
        rel = remote[len(prefix) + 1:]
        raw = remote_bytes(volume, remote)
        path = safe(out, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[rel] = hashlib.sha256(raw).hexdigest()
    save(out / "collection.json", {"files": hashes, "collected_at": datetime.now(timezone.utc).isoformat()})
    return {"directory": str(out), "files": len(hashes), "status": read(out / "status.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "launch", "status", "collect"))
    p.add_argument("--receipt", type=Path, required=True); p.add_argument("--bundle", type=Path)
    p.add_argument("--evaluation", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--authorize-gpu", action="store_true")
    a = p.parse_args()
    result = prepare(a.receipt, a.evaluation, a.out) if a.action == "prepare" else launch(a.bundle, a.receipt, a.authorize_gpu) if a.action == "launch" else collect(a.receipt, a.out) if a.action == "collect" else status(a.receipt)
    print(json.dumps(result, indent=2))
