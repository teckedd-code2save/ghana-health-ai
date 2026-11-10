#!/usr/bin/env python3
"""Separate full IFEval execution packages, bounded resumable jobs, no implicit launch."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import run_understanding_r5_evaluation as common
import r5_ifeval_source as source
import r5_ifeval_runtime as generation

require, read, save, sha, digest, files, safe = common.require, common.read, common.save, common.sha, common.digest, common.files, common.safe
VERSION = "r5-ifeval-execution-v1"
KIND = "public_english_ifeval_not_twi_ability"
VARIANTS = ("base", "r4_response", "r5_response")
LIMITS = {"soft_seconds": 6900, "hard_seconds": 7200, "default_chunk_slots": 24, "maximum_chunk_slots": 96}
VOLUME = common.VOLUME
PREFIX = "/understanding-ifeval-r5"
ROOT = Path(__file__).resolve().parents[1]
CODE = ("scripts/run_understanding_r5_ifeval.py", "scripts/r5_ifeval_source.py", "scripts/r5_ifeval_runtime.py",
        "scripts/score_understanding_r5_ifeval.py", "modal/train/evaluate_understanding_r5_ifeval.py",
        "scripts/run_understanding_r5_evaluation.py", "scripts/r5_retention.py")
LEDGER = "ghana-r5-ifeval-global-gpu-ledger-v1"


def save_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "Linked private receipt")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def preflight(bundle, source_dir, source_sha):
    _, rows = source.load(source_dir, source_sha)
    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    for name, checksum in manifest["files"].items():
        if name.startswith(("code/", "tokenizer/")):
            require(sha(safe(bundle, name)) == checksum, "Native preflight artifact changed")
    require(manifest["base_sha256"] == "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6", "Wrong native foundation")
    context = read(bundle / "tokenizer/config.json")["text_config"]["max_position_embeddings"]
    require(generation.PROTOCOL["context_tokens"] <= context, "Protocol exceeds native model context")
    core, _ = common.activate_code(bundle / "code")
    from check_afrique_training_contract import load_tokenizer
    tokenizer = load_tokenizer(bundle / "tokenizer")
    checks = []
    for row in rows:
        try:
            ids = generation.prompt_ids(core, tokenizer, row["prompt"])
            checks.append({"key": row["key"], "ready": True, "prompt_tokens": len(ids), "prompt_ids_sha256": digest(ids),
                           "messages_sha256": digest(generation.messages(row["prompt"]))})
        except Exception as error:
            checks.append({"key": row["key"], "ready": False, "error_type": type(error).__name__, "error": "Native prompt/context check failed"})
    return {"source_sha256": source_sha, "protocol": generation.PROTOCOL, "model_loaded": False, "gpu_used": False,
            "expected_prompts": len(rows), "ready": all(c["ready"] for c in checks), "cases": checks,
            "maximum_prompt_tokens": max((c["prompt_tokens"] for c in checks if c["ready"]), default=0),
            "budget_basis": {"maximum_required_minimum_words": 1200, "output_tokens": 8192,
                "rationale": "6.82 output tokens per word at the largest explicit 1200-word minimum; headroom is not a completion guarantee",
                "cap_hits_remain_incomplete": True, "unchanged_prompts": True}}


def prepare(bundle, bundle_sha, source_dir, source_sha, inputs, consumption, audit, out, policy="raw"):
    bundle, out = Path(bundle), Path(out).absolute()
    require(policy in generation.POLICIES and not out.exists() and not out.is_symlink(), "Use a new execution and explicit policy")
    require(not any(out.resolve().is_relative_to(Path(p).resolve()) for p in (bundle, source_dir, inputs)), "Output inside immutable source")
    _, manifest = common.inference(bundle, bundle_sha)
    require(set(VARIANTS) <= set(manifest["variants"]), "Missing paired base/R4/R5 models")
    _, rows = source.load(source_dir, source_sha)
    evidence, copies = common.provenance(bundle, manifest, inputs, consumption, {"kind": KIND}, evaluation_sha=source_sha)
    evidence["source_audit"], audit_copies = common.source_audit(audit, manifest["r5_training"]["data_sha256"], source_sha, source.UPSTREAM[source.DATA])
    context = preflight(bundle, source_dir, source_sha)
    require(context["ready"], "Full native preflight failed; no truncation, omission or budget reduction")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".ifeval-execution-") as temporary:
        stage = Path(temporary) / "execution"
        shutil.copytree(bundle, stage / "inference")
        stage.chmod(0o700)
        shutil.copytree(source_dir, stage / "source")
        for name, path in copies.items():
            target = safe(stage / "provenance/inputs", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name, path in audit_copies.items():
            target = safe(stage / "provenance", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(consumption, stage / "provenance/consumption.json")
        save(stage / "provenance/summary.json", evidence)
        save(stage / "context-preflight.json", context)
        (stage / "code").mkdir()
        for name in CODE:
            shutil.copyfile(ROOT / name, stage / "code" / Path(name).name)
        contract = {"version": VERSION, "kind": KIND, "private": True, "training": False,
            "inference_sha256": bundle_sha, "source_sha256": source_sha, "variants": list(VARIANTS),
            "runtime": manifest["runtime"], "protocol": generation.PROTOCOL, "generation_policy": policy,
            "guard_policy": generation.GUARD if policy == "product_guarded" else None,
            "limits": LIMITS, "expected_prompts_per_model": 541, "expected_instructions_per_model": 834,
            "expected_slots": len(rows) * len(VARIANTS), "files": files(stage)}
        save(stage / "execution.json", contract)
        checksum = sha(stage / "execution.json")
        save(stage / "SEALED.json", {"execution_sha256": checksum})
        verify(stage, checksum)
        source.private(stage)
        stage.rename(out)
    return {"execution": str(out), "execution_sha256": checksum, "expected_slots": 1623, "gpu_launched": False}


def verify(folder, checksum):
    folder = Path(folder)
    require(sha(folder / "execution.json") == checksum and read(folder / "SEALED.json") == {"execution_sha256": checksum}, "Execution seal differs")
    execution = read(folder / "execution.json")
    require(execution["version"] == VERSION and execution["kind"] == KIND and execution["private"] is True
            and execution["training"] is False and execution["protocol"] == generation.PROTOCOL
            and execution["variants"] == list(VARIANTS) and execution["limits"] == LIMITS
            and execution["generation_policy"] in generation.POLICIES
            and execution["guard_policy"] == (generation.GUARD if execution["generation_policy"] == "product_guarded" else None)
            and execution["expected_slots"] == 1623 and execution["expected_prompts_per_model"] == 541
            and execution["expected_instructions_per_model"] == 834, "IFEval execution protocol differs")
    require(files(folder) == {**execution["files"], "execution.json": checksum, "SEALED.json": sha(folder / "SEALED.json")}, "Execution files differ")
    _, rows = source.load(folder / "source", execution["source_sha256"])
    module, manifest = common.inference(folder / "inference", execution["inference_sha256"])
    require(execution["runtime"] == manifest["runtime"], "Inference environment differs")
    context = read(folder / "context-preflight.json")
    require(context["ready"] is True and context["protocol"] == generation.PROTOCOL
            and context["source_sha256"] == execution["source_sha256"] and context["expected_prompts"] == 541
            and [c["key"] for c in context["cases"]] == [r["key"] for r in rows]
            and all(c["ready"] is True and 0 < c["prompt_tokens"] <= 8192
                    and c["messages_sha256"] == digest(generation.messages(r["prompt"]))
                    and re.fullmatch(r"[a-f0-9]{64}", c["prompt_ids_sha256"])
                    for c, r in zip(context["cases"], rows, strict=True)), "Native context coverage differs")
    evidence, _ = common.provenance(folder / "inference", manifest, folder / "provenance/inputs",
        folder / "provenance/consumption.json", {"kind": KIND}, evaluation_sha=execution["source_sha256"])
    evidence["source_audit"], _ = common.source_audit(folder / "provenance/source-audit/audit.json",
        manifest["r5_training"]["data_sha256"], execution["source_sha256"], source.UPSTREAM[source.DATA])
    require(evidence == read(folder / "provenance/summary.json"), "Training/heldout source proof differs")
    return execution, manifest, module, rows


def binding(execution, checksum, run_id):
    return {"version": VERSION, "kind": KIND, "run_id": run_id, "execution_sha256": checksum,
            **{k: execution[k] for k in ("source_sha256", "inference_sha256", "generation_policy", "guard_policy", "protocol", "variants")}}


def slots(rows):
    return [{"key": str(row["key"]) + ":" + variant, "case": row, "case_key": row["key"], "variant": variant}
            for row in rows for variant in VARIANTS]


def request(slot):
    return {"case_sha256": digest(slot["case"]), "messages_sha256": digest(generation.messages(slot["case"]["prompt"]))}


def evaluate(runtime, core, rows, journal, attempt, proof_sha, policy, deadline, chunk_slots=24, clock=time.monotonic):
    require(policy in generation.POLICIES and type(chunk_slots) is int and 1 <= chunk_slots <= LIMITS["maximum_chunk_slots"], "Invalid bounded chunk")
    plan, newly_finalized = slots(rows), 0
    for slot in plan:
        key, variant, prompt = slot["key"], slot["variant"], slot["case"]["prompt"]
        inputs_sha = digest(request(slot))
        record = journal.get(key)
        if record is not None:
            require(record["inputs_sha256"] == inputs_sha and record["case_key"] == slot["case_key"]
                    and record["variant"] == variant and record["generation_policy"] == policy, "Saved IFEval request differs")
            if record.get("verified"):
                ids = generation.prompt_ids(core, runtime.tokenizer, prompt)
                generation.verify_reply(runtime, core, prompt, variant, ids, record["result"], policy, loaded=False)
                require(record["state"] == ("complete" if record["result"]["complete"] else "incomplete"), "Saved generation state differs")
            continue
        if newly_finalized >= chunk_slots or clock() + generation.PROTOCOL["case_seconds"] + 15 >= deadline:
            return {"state": "paused_chunk" if newly_finalized >= chunk_slots else "paused_budget", "newly_finalized": newly_finalized}
        started = list((journal.folder / "attempts").glob("*/" + digest(key) + ".json"))
        value = {"case_key": slot["case_key"], "variant": variant, "generation_policy": policy, "attempt": attempt,
                 "inputs_sha256": inputs_sha, "runtime_proof_sha256": proof_sha, "verified": False, "state": "error"}
        if started:
            require(all(read(p)["id"] == key and read(p)["inputs_sha256"] == inputs_sha for p in started), "Interrupted IFEval request differs")
            value.update(state="indeterminate", reason="Started generation has no durable result; no automatic retry")
        else:
            journal.begin(key, attempt, inputs_sha)
            try:
                ids = generation.prompt_ids(core, runtime.tokenizer, prompt)
                result = runtime.reply(prompt, variant, policy)
                value["result"] = result
                generation.verify_reply(runtime, core, prompt, variant, ids, result, policy)
                value.update(verified=True, state="complete" if result["complete"] else "incomplete")
            except Exception as error:
                value.update(error_type=type(error).__name__, error=str(error)[:2000])
        journal.finish(key, value)
        newly_finalized += 1
    return {"state": "completed", "newly_finalized": newly_finalized}


def report(folder, rows, journal):
    plan = slots(rows)
    cases = []
    for slot in plan:
        record = journal.get(slot["key"]) or {}
        result = record.get("result", {})
        cases.append({"slot_id": slot["key"], "case_key": slot["case_key"], "variant": slot["variant"],
            "expected_instructions": len(slot["case"]["instruction_id_list"]), "state": record.get("state", "pending"),
            "verified": record.get("verified", False), "finish_reason": result.get("finish_reason"),
            "cap_hit": result.get("cap_hit", False), "timed_out": result.get("timed_out", False),
            "error_type": record.get("error_type"), "error": record.get("error"), "reason": record.get("reason")})
    metrics = {}
    for variant in VARIANTS:
        chosen = [c for c in cases if c["variant"] == variant]
        counts = Counter(c["state"] for c in chosen)
        metrics[variant] = {"expected_prompts": len(chosen), "expected_instructions": sum(c["expected_instructions"] for c in chosen),
            "states": {state: counts[state] for state in ("complete", "incomplete", "error", "indeterminate", "pending")},
            "cap_hits": sum(c["cap_hit"] for c in chosen), "time_limit_hits": sum(c["timed_out"] for c in chosen),
            "verified_outputs": sum(c["verified"] for c in chosen), "accuracy": None}
    summary = {"binding": journal.binding, "expected_slots": len(plan), "finalized_slots": sum(c["state"] != "pending" for c in cases),
               "generation": metrics, "official_scoring": "pending_separate_offline_scorer", "completion_is_accuracy": False,
               "twi_ability_established": False}
    save(Path(folder) / "generation-summary.json", summary)
    save(Path(folder) / "generation-cases.json", {"binding": journal.binding, "cases": cases})
    snapshot = files(folder)
    save(Path(folder) / "results-manifest.json", {"binding": journal.binding,
        "files": {n: h for n, h in snapshot.items() if n not in {"status.json", "results-manifest.json", "collection.json"}}})
    journal.commit()
    return summary


def audit_results(package, checksum, folder):
    execution, manifest, _, rows = verify(package, checksum)
    folder = Path(folder)
    result = read(folder / "results-manifest.json")
    bound = result["binding"]
    require(bound == binding(execution, checksum, bound["run_id"]), "Collected run binding differs")
    require(read(folder / "journal/binding.json") == bound, "Missing collected journal binding")
    require(files(folder) == {**result["files"], **{n: sha(folder / n) for n in ("status.json", "results-manifest.json", "collection.json") if (folder / n).exists()}},
            "Result snapshot files differ")
    core, _ = common.activate(package)
    journal = core.Journal(folder / "journal", bound)
    expected_paths = set()
    preflight_rows = {c["key"]: c for c in read(Path(package) / "context-preflight.json")["cases"]}
    for slot in slots(rows):
        record = journal.get(slot["key"])
        if record is None:
            continue
        expected_paths.add(journal.path(slot["key"]).name)
        require(record["inputs_sha256"] == digest(request(slot)) and record["case_key"] == slot["case_key"]
                and record["variant"] == slot["variant"] and record["generation_policy"] == execution["generation_policy"]
                and type(record["verified"]) is bool and record["state"] in {"complete", "incomplete", "error", "indeterminate"}, "Saved case identity differs")
        proof_path = folder / "runtime" / ("attempt-" + str(record["attempt"]) + ".json")
        proof = read(proof_path)
        require(sha(proof_path) == record["runtime_proof_sha256"] and proof["binding"] == bound
                and proof["runtime"] == execution["runtime"], "Saved runtime proof differs")
        if record.get("verified"):
            raw = record["result"]
            expected_identity = {"manifest_sha256": execution["inference_sha256"], "variant": slot["variant"],
                "model": manifest["model"], "revision": manifest["revision"], "base_weights_sha256": manifest["base_sha256"],
                "adapter_weights_sha256": None if slot["variant"] == "base" else manifest["variants"][slot["variant"]]["files"]["adapter_model.safetensors"]}
            require(raw["identity"] == proof["identities"][slot["variant"]] == expected_identity
                    and raw["protocol"] == generation.PROTOCOL and raw["generation_policy"] == execution["generation_policy"]
                    and raw["messages_sha256"] == preflight_rows[slot["case_key"]]["messages_sha256"]
                    and raw["prompt_ids_sha256"] == preflight_rows[slot["case_key"]]["prompt_ids_sha256"]
                    and raw["output"] == raw["diagnostics"]["prediction"]
                    and raw["output_ids"] == raw["diagnostics"]["output_ids"]
                    and len(raw["output_ids"]) <= generation.PROTOCOL["max_new_tokens"]
                    and record["state"] == ("complete" if raw["complete"] else "incomplete"), "Saved native output/identity differs")
    actual_paths = {p.name for p in (folder / "journal/records").glob("*.json")}
    require(actual_paths == expected_paths, "Unexpected/duplicate generation records")
    return execution, rows, journal


def reserve_gpu(store, record, terminal):
    """Append-only global leases: no racy delete/reclaim and no ambiguous retry."""
    for index in range(10000):
        prior = store.get("lease:" + str(index))
        if prior is None:
            require(store.put("lease:" + str(index), record, skip_if_exists=True), "Another launch reserved the GPU; no spawn")
            return index
        call = store.get("call:" + str(index))
        require(call is not None and call["owner"] == prior["owner"],
                "Global GPU lease is active or ambiguous; do not retry submission")
        acknowledged = store.get("terminal:" + str(index))
        if acknowledged is None:
            require(terminal(call["call_id"]), "Global GPU lease is active or ambiguous; do not retry submission")
            store.put("terminal:" + str(index), call, skip_if_exists=True)
        require(store.get("terminal:" + str(index)) == call, "Terminal lease acknowledgement differs")
    raise ValueError("Finite lease ledger exhausted")


def bind_call(store, index, owner, call_id):
    require(store.get("lease:" + str(index))["owner"] == owner, "GPU lease owner differs")
    value = {"owner": owner, "call_id": call_id}
    store.put("call:" + str(index), value, skip_if_exists=True)
    require(store.get("call:" + str(index)) == value, "GPU lease call differs")


def launch(package, checksum, receipt, profile, authorize, previous=None, chunk_slots=24):
    require(authorize and not Path(receipt).exists() and not Path(receipt).is_symlink(), "Explicit GPU authorization and a new receipt required")
    require(type(chunk_slots) is int and 1 <= chunk_slots <= LIMITS["maximum_chunk_slots"], "Invalid chunk limit")
    package = Path(package).resolve()
    require(not Path(receipt).resolve().is_relative_to(package), "Receipt must be outside the sealed execution")
    verify(package, checksum)
    os.environ.update(GHA_R5_IFEVAL_PACKAGE=str(package), MODAL_PROFILE=profile)
    import modal
    run_id = "r5_ifeval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt = 1
    if previous:
        old = read(previous)
        require(old["state"] == "submitted" and old["execution_sha256"] == checksum and old["profile"] == profile
                and common.terminal(old["call_id"]), "Previous call is active, ambiguous or differently bound")
        run_id, attempt = old["run_id"], old["attempt"] + 1
    value = {"state": "reserving", "run_id": run_id, "attempt": attempt, "owner": uuid.uuid4().hex,
             "execution_sha256": checksum, "package": str(package), "profile": profile,
             "volume": VOLUME, "remote_directory": PREFIX + "/" + run_id, "chunk_slots": chunk_slots}
    save_private(receipt, value)
    store = modal.Dict.from_name(LEDGER, create_if_missing=True)
    lease = reserve_gpu(store, value, common.terminal)
    value.update(state="submitting", lease=lease)
    save_private(receipt, value)
    worker = common.module_at(package / "code/evaluate_understanding_r5_ifeval.py", "evaluate_understanding_r5_ifeval", register=True)
    with modal.enable_output(), worker.app.run(detach=True):
        call = worker.evaluate.spawn(run_id, checksum, attempt, chunk_slots, lease, value["owner"])
        value.update(state="submitted", call_id=call.object_id)
        save_private(receipt, value)
        bind_call(store, lease, value["owner"], call.object_id)
    return value


def remote_status(receipt):
    value = read(receipt)
    require(value.get("state") == "submitted" and value.get("call_id"), "Ambiguous submission; do not retry")
    os.environ["MODAL_PROFILE"] = value["profile"]
    import modal
    try:
        result = modal.FunctionCall.from_id(value["call_id"]).get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError) as error:
        if isinstance(error, (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError)):
            raise
        return {"state": "pending", "call_id": value["call_id"]}
    save_private(Path(receipt).with_suffix(".result.json"), result)
    return result


def collect(receipt, out):
    value, out = read(receipt), Path(out)
    require(value["state"] == "submitted" and not out.exists(), "Submitted receipt and new result directory required")
    require(not out.resolve().is_relative_to(Path(value["package"]).resolve()), "Results must be outside the sealed execution")
    os.environ["MODAL_PROFILE"] = value["profile"]
    import modal
    require(common.terminal(value["call_id"]), "Collect only a terminal call")
    require(re.fullmatch(r"r5_ifeval_\d{8}T\d{6}Z_[a-f0-9]{8}", value["run_id"])
            and value["remote_directory"] == PREFIX + "/" + value["run_id"], "Unexpected result prefix")
    out.mkdir(parents=True, mode=0o700)
    common.download_snapshot(modal.Volume.from_name(VOLUME, create_if_missing=False), value["remote_directory"], out)
    require(read(out / "journal/binding.json")["execution_sha256"] == value["execution_sha256"], "Collected execution differs")
    if (out / "results-manifest.json").exists():
        audit_results(Path(value["package"]), value["execution_sha256"], out)
    save(out / "collection.json", {"receipt": value, "files": files(out)})
    source.private(out)
    return {"directory": str(out), "status": read(out / "status.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--tokenizer-bundle", type=Path, required=True)
    pre.add_argument("--source", type=Path, required=True)
    pre.add_argument("--source-sha", required=True)
    prep = sub.add_parser("prepare")
    for name in ("bundle", "source", "training-inputs", "consumption", "source-audit", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--bundle-sha", required=True)
    prep.add_argument("--source-sha", required=True)
    prep.add_argument("--policy", choices=generation.POLICIES, default="raw")
    for name in ("verify", "launch"):
        command = sub.add_parser(name)
        command.add_argument("--execution", type=Path, required=True)
        command.add_argument("--execution-sha", required=True)
        if name == "launch":
            command.add_argument("--receipt", type=Path, required=True)
            command.add_argument("--profile", required=True)
            command.add_argument("--authorize-gpu", action="store_true")
            command.add_argument("--previous-receipt", type=Path)
            command.add_argument("--chunk-slots", type=int, default=24)
    for name in ("status", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "collect":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight(args.tokenizer_bundle, args.source, args.source_sha)
    elif args.action == "prepare":
        result = prepare(args.bundle, args.bundle_sha, args.source, args.source_sha, args.training_inputs,
                         args.consumption, args.source_audit, args.out, args.policy)
    elif args.action == "verify":
        result = verify(args.execution, args.execution_sha)[0]
    elif args.action == "launch":
        result = launch(args.execution, args.execution_sha, args.receipt, args.profile, args.authorize_gpu,
                        args.previous_receipt, args.chunk_slots)
    elif args.action == "collect":
        result = collect(args.receipt, args.out)
    else:
        result = remote_status(args.receipt)
    print(json.dumps(result, indent=2))
    if args.action == "preflight" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
