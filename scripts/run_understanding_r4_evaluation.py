#!/usr/bin/env python3
"""Seal and run a receipt-bound R4 development evaluation. Preparation is offline."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

VERSION = "r4-development-execution-v1"
EVALUATION_SHA = "413c6e3b7e5737ffa3360219202bfb4e8e7a86437eb0d84dc9e2c72385612a06"
R2_BUNDLE_SHA = "6168a3ff4e69756a748f271896a133b67887a6ca8565d08029a7ab184ee1d8f3"
VARIANTS = ("base", "r3_response", "r4_response")
FROZEN_VARIANTS = ("base", "r2_meaning", "r3_response")
ALL_VARIANTS = VARIANTS + ("r4_pass1",)
POLICIES = ("raw", "guarded")
GUARD_POLICY = "repeat-block-v1"
KIND = "development_not_blind_final"
LIMITS = {"soft_seconds": 6900, "hard_seconds": 7200, "reply_seconds": 60}
VOLUME = "ghana-health-understand-train"
PREFIX = "/understanding-evaluation-r4"
ROOT = Path(__file__).resolve().parents[1]
EVAL_FILES = ("manifest.json", "FROZEN.json", "cases.json", "rollouts.json", "language-loss.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def safe(root, name):
    root, relative = Path(root).resolve(), Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe relative path")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.resolve() != root,
            "Escaping artifact path")
    return path


def files(folder):
    require(not any(p.is_symlink() for p in Path(folder).rglob("*")), "Linked artifact")
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def module_at(path, name, *, register=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def inference(folder, expected_sha):
    folder = Path(folder)
    require(sha(folder / "manifest.json") == expected_sha, "Inference bundle hash differs")
    manifest = read(folder / "manifest.json")
    require(files(folder) == {**manifest["files"], "manifest.json": expected_sha,
                             "SEALED.json": sha(folder / "SEALED.json")}, "Inference bundle files differ")
    module = module_at(folder / "code/r4_inference.py", "_evaluation_pinned_r4")
    return module, module.load_bundle(folder, expected_sha)


def evaluation(folder, *, full=True, loss=True):
    folder = Path(folder)
    require(sha(folder / "manifest.json") == EVALUATION_SHA
            and read(folder / "FROZEN.json") == {"manifest_sha256": EVALUATION_SHA}, "Use immutable evaluation-v2")
    manifest = read(folder / "manifest.json")
    require(manifest["kind"] == KIND and manifest["training_eligible"] is False
            and manifest["variants"] == list(FROZEN_VARIANTS), "Wrong evaluation scope")
    selected = set(manifest["artifacts"]) if full else set(EVAL_FILES[2:]) | ({"language-heldout.json"} if loss else set())
    for name in selected:
        require(sha(safe(folder, name)) == manifest["artifacts"][name], "Evaluation artifact changed: " + name)
    if not full:
        require(set(files(folder)) == selected | {"manifest.json", "FROZEN.json"}, "Unexpected runtime evaluation data")
    cases, rollouts = read(folder / "cases.json"), read(folder / "rollouts.json")
    require(len(cases) == 46 and len({c["id"] for c in cases}) == 46
            and len(rollouts) == len({r["id"] for r in rollouts}) == 10
            and sum(len(r["user_turns"]) for r in rollouts) == 29, "Evaluation case counts differ")
    held = read(folder / "language-heldout.json") if full or loss else []
    require(not (full or loss) or len(held) == len({r["id"] for r in held}) == 343, "Language holdout differs")
    return manifest, cases, rollouts, held


def provenance(bundle, manifest, inputs, consumption, evaluation_manifest):
    """Verify trainer consumption, not inventory, against the completed checkpoint."""
    inputs, consumption = Path(inputs), Path(consumption)
    training = manifest["r4_training"]
    terminal = read(Path(bundle) / "evidence/terminal.json")
    data = read(inputs / "manifest.json")
    require(sha(inputs / "manifest.json") == training["data_sha256"], "Training input manifest differs")
    wrapper = inputs / "evaluation-manifest.json"
    require(sha(wrapper) == data["artifacts"]["evaluation-manifest.json"]
            and read(wrapper) == {"sha256": EVALUATION_SHA, "manifest": evaluation_manifest},
            "Training was not bound to this frozen evaluation")
    receipt = read(consumption)
    require(sha(consumption) == terminal.get("consumption_sha256"), "Missing or changed trainer consumption receipt")
    require(receipt["version"] == "r4-consumption-v1" and receipt["mode"] == "train"
            and receipt["scope"] == "cumulative_committed_schedule_prefix"
            and receipt["training_complete"] is True and receipt["schedule_complete"] is True
            and receipt["run_id"] == training["run_id"]
            and receipt["execution_sha256"] == training["execution_sha256"]
            and receipt["data_sha256"] == training["data_sha256"]
            and receipt["checkpoint"] == training["checkpoint"]
            and receipt["completed_updates"] == training["completed_updates"], "Consumption checkpoint differs")
    position = terminal["stages"][0]["position"]
    counts = receipt["consumed"]
    require(counts["row_exposures"] == position["completed_microbatches"]
            and all(counts[k] == position["exposure"][k] for k in ("input_tokens", "effective_target_tokens"))
            and receipt["inventory"]["archive_counts_included"] is False, "Consumption exposure differs")
    references = data["dataset_snapshots"]
    datasets, copies = [], {"manifest.json": inputs / "manifest.json", "evaluation-manifest.json": wrapper}
    for ref in references:
        source = safe(inputs, ref["manifest_path"])
        require(sha(source) == ref["manifest_sha256"] == data["artifacts"][ref["manifest_path"]], "Dataset manifest differs")
        value = read(source)
        require(all(value[k] == ref[k] for k in ("dataset_id", "dataset_version"))
                and value["records_sha256"] == data["artifacts"][value["records_path"]]
                and value["admission_status"] == "admitted" and value["private_research_only"] is True
                and value["production_eligible"] is False, "Unbound dataset version/records")
        datasets.append({**ref, **value})
        copies[ref["manifest_path"]] = source
    expected_snapshots = [{**r, "admitted_record_count": d["record_count"]} for r, d in zip(references, datasets, strict=True)]
    require(receipt["inventory"]["admitted_snapshots"] == expected_snapshots, "Consumption dataset snapshots differ")
    rows, by_class, by_dataset = receipt["rows"], {}, {}
    require(len({r["row_index"] for r in rows}) == len(rows) == counts["unique_rows"], "Duplicate consumed rows")
    keys = ("unique_rows", "row_exposures", "input_tokens", "effective_target_tokens")
    known = {(d["dataset_id"], d["dataset_version"]) for d in datasets}
    def record_key(row):
        return tuple(row["dataset_ref"][k] for k in ("dataset_id", "dataset_version", "record_id"))
    for row in rows:
        key = record_key(row)[:2]
        require(key in known and row["unique_rows"] == 1
                and all(type(row[k]) is int and row[k] > 0 for k in keys), "Invalid consumed row")
        for table, group in ((by_class, row["class"]), (by_dataset, key)):
            table.setdefault(group, Counter()).update({k: row[k] for k in keys})
    require(all(sum(r[k] for r in rows) == counts[k] for k in keys)
            and len({record_key(r) for r in rows}) == counts["unique_dataset_records"], "Consumption totals differ")
    for name, counter in by_class.items():
        counter["unique_dataset_records"] = len({record_key(r) for r in rows if r["class"] == name})
    for key, counter in by_dataset.items():
        counter["unique_dataset_records"] = len({record_key(r) for r in rows if record_key(r)[:2] == key})
    require(counts["by_class"] == by_class and counts["by_dataset"] == [
        {"dataset_id": k[0], "dataset_version": k[1], **v} for k, v in sorted(by_dataset.items())], "Consumption groups differ")
    return {"kind": KIND, "training": training, "datasets": datasets,
            "consumption_sha256": sha(consumption), "consumed": counts, "inventory": receipt["inventory"],
            "clinical_certification": False, "native_human_certification": False}, copies


def selection(variants, policy, loss, available):
    require(isinstance(variants, (list, tuple)) and len(variants) in (3, 4)
            and list(variants[:2]) == ["base", "r3_response"]
            and list(variants) == [v for v in ALL_VARIANTS if v in variants]
            and set(variants) <= set(available), "Invalid or unavailable paired variant selection")
    require(policy in POLICIES and type(loss) is bool and not (policy == "guarded" and loss),
            "Choose raw loss or a separate guarded chat execution")
    return list(variants)


def prepare(bundle, bundle_sha, evaluation_dir, inputs, consumption, out, *, loss=False,
            variants=VARIANTS, policy="raw"):
    bundle, out = Path(bundle), Path(out)
    require(not out.exists(), "Use a new immutable execution directory")
    _, manifest = inference(bundle, bundle_sha)
    variants = selection(variants, policy, loss, manifest["variants"])
    ev, cases, rollouts, held = evaluation(evaluation_dir)
    require(ev["decoding"] == manifest["decoding"], "Inference/evaluation decoding differs")
    evidence, copies = provenance(bundle, manifest, inputs, consumption, ev)
    context = context_preflight(bundle, cases, rollouts, held)
    require(context["ready"], "Native context preflight failed; no truncation or case replacement")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r4-evaluation-") as temporary:
        staged = Path(temporary) / "execution"
        shutil.copytree(bundle, staged / "inference")
        for name in EVAL_FILES + (("language-heldout.json",) if loss else ()):
            destination = staged / "evaluation" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(evaluation_dir) / name, destination)
        for name, source in copies.items():
            destination = safe(staged / "provenance/inputs", name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(consumption, staged / "provenance/consumption.json")
        save(staged / "provenance/summary.json", evidence)
        save(staged / "context-preflight.json", context)
        (staged / "code").mkdir()
        for name in ("scripts/run_understanding_r4_evaluation.py", "modal/train/evaluate_understanding_r4.py"):
            shutil.copyfile(ROOT / name, staged / "code" / Path(name).name)
        contract = {"version": VERSION, "kind": KIND, "training": False, "private": True,
                    "inference_sha256": bundle_sha, "evaluation_sha256": EVALUATION_SHA,
                    "runtime": manifest["runtime"], "decoding": manifest["decoding"], "variants": variants,
                    "generation_policy": policy, "guard_policy": GUARD_POLICY if policy == "guarded" else None,
                    "limits": LIMITS, "language_loss": loss, "expected_slots": len(variants) * (75 + (343 if loss else 0)),
                    "data_minimization": "Selected evaluation artifacts only; no feedback proofs or review stores",
                    "files": files(staged)}
        save(staged / "execution.json", contract)
        checksum = sha(staged / "execution.json")
        save(staged / "SEALED.json", {"execution_sha256": checksum})
        verify(staged, checksum)
        staged.rename(out)
    return {"execution": str(out.resolve()), "execution_sha256": checksum, "expected_slots": contract["expected_slots"],
            "kind": KIND, "gpu_launched": False}


def verify(folder, expected_sha):
    folder = Path(folder)
    require(sha(folder / "execution.json") == expected_sha
            and read(folder / "SEALED.json") == {"execution_sha256": expected_sha}, "Execution identity differs")
    contract = read(folder / "execution.json")
    require(contract["version"] == VERSION and contract["kind"] == KIND and contract["private"] is True
            and contract["training"] is False
            and contract["limits"] == LIMITS and type(contract["language_loss"]) is bool
            and contract["expected_slots"] == len(contract["variants"]) * (75 + (343 if contract["language_loss"] else 0))
            and contract["evaluation_sha256"] == EVALUATION_SHA, "Execution protocol differs")
    require(files(folder) == {**contract["files"], "execution.json": expected_sha,
                             "SEALED.json": sha(folder / "SEALED.json")}, "Execution files changed")
    context = read(folder / "context-preflight.json")
    require(context["ready"] is True and context["evaluation_sha256"] == EVALUATION_SHA
            and context["context_tokens"] == 4096 and context["reserved_generation_tokens"] == 256
            and context["truncate"] is False, "Missing native context preflight")
    module, manifest = inference(folder / "inference", contract["inference_sha256"])
    selection(contract["variants"], contract["generation_policy"], contract["language_loss"], manifest["variants"])
    require(contract["guard_policy"] == (GUARD_POLICY if contract["generation_policy"] == "guarded" else None),
            "Generation guard policy differs")
    ev, cases, rollouts, held = evaluation(folder / "evaluation", full=False, loss=contract["language_loss"])
    require(contract["runtime"] == manifest["runtime"] and contract["decoding"] == manifest["decoding"] == ev["decoding"],
            "Runtime/decoding changed")
    proof, _ = provenance(folder / "inference", manifest, folder / "provenance/inputs", folder / "provenance/consumption.json", ev)
    require(proof == read(folder / "provenance/summary.json"), "Provenance summary differs")
    return contract, manifest, module, cases, rollouts, held


def activate_code(code):
    """Reject cross-bundle imports; a worker process serves one immutable package."""
    code = Path(code).resolve()
    for path in code.glob("*.py"):
        existing = sys.modules.get(path.stem)
        require(existing is None or Path(existing.__file__).resolve() == path, "Different bundle module already imported: " + path.stem)
    sys.path.insert(0, str(code))
    prior = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        import afrique_instruction_baseline_core as core
        import build_understanding_evaluation as scoring
    finally:
        sys.dont_write_bytecode = prior
    return core, scoring


def activate(folder):
    return activate_code(Path(folder) / "inference/code")


def context_preflight(bundle, cases, rollouts, held):
    core, _ = activate_code(Path(bundle) / "code")
    from check_afrique_training_contract import load_tokenizer
    tokenizer = load_tokenizer(Path(bundle) / "tokenizer")
    checks = []
    for kind, name, messages in ([('fixed', c["id"], c["messages"]) for c in cases]
            + [('rollout_first_turn', c["id"], [{"role": "user", "content": c["user_turns"][0]}]) for c in rollouts]):
        try:
            ids = core.prompt_ids(tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})
            checks.append({"kind": kind, "case_id": name, "ready": True,
                           "prompt_tokens": len(ids), "prompt_ids_sha256": digest(ids)})
        except Exception as error:
            checks.append({"kind": kind, "case_id": name, "ready": False,
                           "error_type": type(error).__name__, "error": str(error)[:1000]})
    lengths = [len(tokenizer.encode(row["text"], add_special_tokens=False)) + 1 for row in held]
    loss_ok = all(1 < n <= 4096 for n in lengths)
    return {"evaluation_sha256": EVALUATION_SHA, "kind": KIND, "model_loaded": False, "gpu_used": False,
            "context_tokens": 4096, "reserved_generation_tokens": 256, "truncate": False,
            "ready": all(c["ready"] for c in checks) and loss_ok, "checks": checks,
            "maximum_verified_prompt_tokens": max((c["prompt_tokens"] for c in checks if c["ready"]), default=0),
            "language_loss": {"rows": len(lengths), "ready": loss_ok, "maximum_sequence_tokens": max(lengths, default=0),
                              "predicted_tokens": sum(n - 1 for n in lengths)},
            "later_rollout_turns": {"count": sum(len(r["user_turns"]) - 1 for r in rollouts),
                "status": "Checked against actual own-model history immediately before each generation; future replies not invented"}}


def preflight(tokenizer_bundle, evaluation_dir):
    bundle = Path(tokenizer_bundle)
    require(sha(bundle / "manifest.json") == R2_BUNDLE_SHA, "Preflight requires the original pinned R2 tokenizer bundle")
    manifest = read(bundle / "manifest.json")
    for name, checksum in manifest["files"].items():
        require(sha(safe(bundle, name)) == checksum, "Pinned preflight artifact changed")
    _, cases, rollouts, held = evaluation(evaluation_dir)
    return context_preflight(bundle, cases, rollouts, held)


def slots(cases, rollouts, held, variants=VARIANTS, policy="raw"):
    for case in cases:
        for variant in variants:
            yield {"key": "fixed:" + case["id"] + ":" + variant, "kind": "fixed", "case_id": case["id"],
                   "variant": variant, "policy": policy, "turn": 0, "case": case}
    for case in rollouts:
        for variant in variants:
            for turn, user in enumerate(case["user_turns"]):
                yield {"key": "rollout:" + case["id"] + ":" + variant + ":" + str(turn), "kind": "rollout",
                       "case_id": case["id"], "variant": variant, "policy": policy, "turn": turn, "user": user, "case": case}
    for row in held:
        for variant in variants:
            yield {"key": "loss:" + row["id"] + ":" + variant, "kind": "loss", "case_id": row["id"],
                   "variant": variant, "policy": policy, "turn": 0, "row": row}


def continuable(record):
    raw = record.get("result") or {}
    guarded_pause = (record.get("policy") == "guarded" and record["state"] == "paused"
                     and raw.get("finish_reason") == "cancelled" and raw.get("loop_guard", {}).get("trigger"))
    return (record.get("verified") is True and bool(raw.get("output", "").strip())
            and (guarded_pause or (record["state"] in {"complete", "incomplete"}
                                  and raw["finish_reason"] in {"eos", "max_new_tokens", "time_limit"})))


def reply_state(result):
    return "complete" if result["complete"] else "paused" if result.get("loop_guard", {}).get("trigger") else "incomplete"


def guarded_reply(runtime, messages, variant):
    from guarded_r3_runtime import reply
    from response_loop_guard import VERSION as guard_version, repeated_passage
    require(guard_version == GUARD_POLICY, "Guard policy differs")
    emitted, stopped = [], False

    def emit(text):
        nonlocal stopped
        require(isinstance(text, str), "Invalid streamed guard text")
        emitted.append(text)
        stopped = stopped or bool(repeated_passage("".join(emitted)))

    result = reply(runtime, messages, variant, on_text=emit, cancelled=lambda: stopped)
    require(result["output"] == "".join(emitted), "Guard stream differs from native output")
    result["loop_guard"] = {"version": GUARD_POLICY, "trigger":
        repeated_passage(result["output"]) if stopped and result["finish_reason"] == "cancelled" else None}
    return result


def verify_reply(runtime, core, messages, variant, ids, result, *, loaded=True, policy="raw"):
    require(result["identity"] == runtime.identity(variant) and result["messages_sha256"] == digest(messages)
            and result["prompt_ids_sha256"] == digest(ids) and result["decoding"] == runtime.manifest["decoding"],
            "Returned prompt/checkpoint/protocol identity differs")
    expected = core.completion(runtime.tokenizer, result["output_ids"], 256,
                               result["finish_reason"] == "time_limit", result["seconds"])
    require(result["diagnostics"] == expected and result["output"] == expected["prediction"],
            "Native output/completion proof differs")
    require(policy in POLICIES, "Unknown generation policy")
    if policy == "guarded":
        from response_loop_guard import validate_guard
        validate_guard(result.get("loop_guard"), result["output"], result["finish_reason"], result["complete"], True)
    else:
        require("loop_guard" not in result, "Raw result contains a guarded policy")
    paused = policy == "guarded" and bool(result["loop_guard"]["trigger"])
    require((paused and result["finish_reason"] == "cancelled" and result["complete"] is False)
            or (not paused and result["complete"] == expected["complete"]
                and result["finish_reason"] == expected["finish_reason"]), "Native output/completion proof differs")
    require(not loaded or variant == "base" or variant in runtime.loaded, "Adapter tensors were not verified")


def evaluate(runtime, core, scoring, plan, journal, attempt, proof_sha, deadline, clock=time.monotonic):
    require(plan and len({s["policy"] for s in plan}) == 1 and plan[0]["policy"] in POLICIES,
            "Raw and guarded slots require separate executions")
    histories, previous, prompts, records = {}, {}, {}, []
    for slot in plan:
        key, kind, variant = slot["key"], slot["kind"], slot["variant"]
        group = (slot["case_id"], variant, slot["policy"])
        parent = previous.get(group) if kind == "rollout" else None
        blocked = parent is not None and not continuable(parent)
        messages = (slot["case"]["messages"] if kind == "fixed" else
                    None if kind == "loss" or blocked else histories.get(group, []) + [{"role": "user", "content": slot["user"]}])
        request = ({"row_id": slot["row"]["id"], "text_sha256": digest(slot["row"]["text"])} if kind == "loss" else
                   {"scripted_user": slot["user"], "blocked": True} if blocked else {"messages": messages})
        inputs_sha, parent_sha = digest(request), digest(parent) if parent else None
        record = journal.get(key)
        if record is not None:
            require(record["inputs_sha256"] == inputs_sha and record["parent_sha256"] == parent_sha
                    and all(record[k] == slot[k] for k in ("kind", "variant", "case_id", "turn", "policy")), "Saved slot/history differs")
            if record.get("verified") and kind != "loss":
                ids = core.prompt_ids(runtime.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})
                require(ids == record["prompt_ids"], "Saved native prompt differs")
                # Prior adapter tensor checks belong to the recorded, hash-bound attempt.
                verify_reply(runtime, core, messages, variant, ids, record["result"], loaded=False, policy=slot["policy"])
                require(record["state"] == reply_state(record["result"]), "Saved completion state differs")
            elif record.get("verified"):
                ids = runtime.tokenizer.encode(slot["row"]["text"], add_special_tokens=False) + [core.EOS]
                require(record["identity"] == runtime.identity(variant) and record["state"] == "measured"
                        and record["token_ids_sha256"] == digest(ids)
                        and record["loss"]["predicted_tokens"] == record["predicted_tokens"] == len(ids) - 1
                        and math.isfinite(record["loss"]["summed_nll"]) and record["loss"]["summed_nll"] >= 0,
                        "Saved language-loss proof differs")
        else:
            if clock() >= deadline:
                return {"state": "paused_budget", "finalized_slots": len(records), "expected_slots": len(plan)}
            started = list((journal.folder / "attempts").glob("*/" + digest(key) + ".json"))
            meta = {k: slot[k] for k in ("kind", "variant", "case_id", "turn", "policy")}
            value = {**meta, "attempt": attempt, "runtime_proof_sha256": proof_sha,
                     "inputs_sha256": inputs_sha, "parent_sha256": parent_sha,
                     "request": request, "verified": False, "state": "blocked" if blocked else "error"}
            if started:
                require(all(read(p)["id"] == key and read(p)["inputs_sha256"] == inputs_sha for p in started), "Interrupted request differs")
                value.update(state="indeterminate", reason="Started request has no durable result; not retried")
            else:
                journal.begin(key, attempt, inputs_sha)
                if blocked:
                    value["reason"] = "Previous own-model reply cannot continue"
                else:
                    try:
                        if kind == "loss":
                            ids = runtime.tokenizer.encode(slot["row"]["text"], add_special_tokens=False) + [core.EOS]
                            require(1 < len(ids) <= 4096, "Invalid held-out length")
                            raw = runtime.language_loss(slot["row"]["text"], variant)
                            value["loss"] = raw
                            require(raw["predicted_tokens"] == len(ids) - 1 and math.isfinite(raw["summed_nll"])
                                    and raw["summed_nll"] >= 0, "Invalid shifted loss")
                            require(variant == "base" or variant in runtime.loaded, "Adapter tensor proof missing")
                            value.update(state="measured", verified=True, identity=runtime.identity(variant),
                                         token_ids_sha256=digest(ids), predicted_tokens=len(ids) - 1)
                        else:
                            ids = core.prompt_ids(runtime.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})
                            value["prompt_ids"] = ids
                            raw = (guarded_reply(runtime, messages, variant) if slot["policy"] == "guarded"
                                   else runtime.reply(messages, variant))
                            value["result"] = raw
                            verify_reply(runtime, core, messages, variant, ids, raw, policy=slot["policy"])
                            value.update(state=reply_state(raw), verified=True,
                                         adapter_tensors_verified=variant != "base",
                                         objective=scoring.objective_score({**slot["case"], "objective": slot["case"].get("objective")}, raw)
                                         if kind == "fixed" else None)
                    except Exception as error:
                        value.update(state="error", verified=False, error_type=type(error).__name__,
                                     error="Protected language-loss operation failed" if kind == "loss" else str(error)[:2000])
            record = journal.finish(key, value)
        if kind == "fixed" and record.get("verified"):
            require(prompts.setdefault(slot["case_id"], record["prompt_ids"]) == record["prompt_ids"], "Fixed native prompts differ across variants")
        if kind == "rollout":
            previous[group] = record
            if continuable(record):
                histories[group] = messages + [{"role": "assistant", "content": record["result"]["output"]}]
        records.append(record)
    return {"state": "completed", "finalized_slots": len(records), "expected_slots": len(plan),
            "semantic_quality_established": False}


def report(folder, plan, journal, evidence, identities):
    """Mechanical diagnostics and a separate, unscored human review queue."""
    folder = Path(folder)
    records = {slot["key"]: journal.get(slot["key"]) for slot in plan}
    reviews, metrics = [], {}
    for variant in dict.fromkeys(s["variant"] for s in plan):
        chosen = [r for r in records.values() if r and r["variant"] == variant]
        expected_chat = sum(s["variant"] == variant and s["kind"] != "loss" for s in plan)
        expected_loss = sum(s["variant"] == variant and s["kind"] == "loss" for s in plan)
        expected_objectives = sum(s["variant"] == variant and s["kind"] == "fixed"
                                  and s["case"].get("objective") is not None for s in plan)
        chat = [r for r in chosen if r["kind"] != "loss"]
        verified = [r for r in chat if r.get("verified")]
        losses = [r["loss"] for r in chosen if r["kind"] == "loss" and r.get("verified")]
        tokens, nll = sum(r["predicted_tokens"] for r in losses), sum(r["summed_nll"] for r in losses)
        objectives = [r["objective"] for r in verified if r.get("objective") is not None]
        repeated = 0
        last = {}
        for row in verified:
            if row["kind"] == "rollout":
                key = row["case_id"]
                repeated += last.get(key) == row["result"]["output"]
                last[key] = row["result"]["output"]
        mean = nll / tokens if tokens else None
        metrics[variant] = {"identity": identities[variant], "expected_chat_slots": expected_chat,
            "guarded_pauses": sum(r["state"] == "paused" for r in verified),
            "pending_chat_slots": expected_chat - len(chat), "chat_states": dict(Counter(r["state"] for r in chat)),
            "chat_verified": len(verified), "finish_reasons": dict(Counter(r["result"]["finish_reason"] for r in verified)),
            "strict_json": {"expected_cases": expected_objectives, "evaluated": len(objectives), "literal_matches": sum(r["literal_match"] for r in objectives),
                            "literal_match_with_eos": sum(r["passed"] for r in objectives), "human_semantic_approval": False},
            "replies_with_repeated_4grams": sum(r["result"]["diagnostics"]["repetition"]["4_word_ngrams"]["repeated_positions"] > 0 for r in verified),
            "consecutive_identical_rollout_replies": repeated,
            "language_loss": {"expected_rows": expected_loss, "measured_rows": len(losses),
                              "complete_coverage": expected_loss > 0 and len(losses) == expected_loss,
                              "states": dict(Counter(r["state"] for r in chosen if r["kind"] == "loss")),
                              "predicted_tokens": tokens, "summed_nll": nll,
                              "token_weighted_mean_nll": mean, "exp_mean_nll": math.exp(mean) if mean is not None and mean < 709 else None}}
    for slot in plan:
        if slot["kind"] == "loss":
            continue
        record = records[slot["key"]]
        raw = (record or {}).get("result", {})
        reviews.append({"slot_id": slot["key"], "kind": KIND, "evaluation_type": slot["kind"],
            "cohort": slot["case"].get("cohort"), "case_id": slot["case_id"], "variant": slot["variant"],
            "turn": slot["turn"], "generation_policy": slot["policy"], "checkpoint": identities[slot["variant"]],
            "state": record["state"] if record else "pending", "verified": bool(record and record.get("verified")),
            "messages": (record or {}).get("request", {}).get("messages"), "output": raw.get("output"),
            "finish_reason": raw.get("finish_reason"), "complete": raw.get("complete"),
            "review_criteria": slot["case"].get("rubric", slot["case"].get("expected_meaning_checks")),
            "record": "journal/records/" + digest(slot["key"]) + ".json" if record else None,
            "human_review": {"status": "pending", "reviewer": None, "meaning": None, "requested_language": None,
                             "usefulness": None, "unsupported_claims": None, "notes": None}})
    policies = sorted({s["policy"] for s in plan})
    require(len(policies) == 1, "Raw and guarded results require separate executions")
    summary = {"kind": KIND, "generation_policy": policies[0], "binding": journal.binding, "provenance": evidence, "mechanical_metrics": metrics,
               "expected_slots": len(plan), "finalized_slots": sum(r is not None for r in records.values()),
               "human_quality": {"status": "pending", "semantic_quality_established": False,
                                 "native_human_certification": False, "clinical_certification": False}}
    save(folder / "summary.json", summary)
    save(folder / "review.json", {"kind": KIND, "rows": reviews})
    escape = lambda v: html.escape(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2))
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>R4 development review</title>",
             "<style>body{font:14px system-ui;margin:24px;color:#182021}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}th{text-align:left}summary{cursor:pointer}</style>",
             "<h1>R4 development review</h1><p>Development, not blind final. Generation policy: " + escape(policies[0]) + ". Human quality review pending; no native-human or clinical certification.</p>",
             "<details><summary>Dataset versions, actual consumption and checkpoint evidence</summary><pre>" + escape(evidence) + "</pre></details>",
             "<details><summary>Mechanical measurements, not semantic scores</summary><pre>" + escape(metrics) + "</pre></details>",
             "<table><thead><tr><th>Case / checkpoint</th><th>Exact messages</th><th>Raw output</th><th>Status / review</th></tr></thead><tbody>"]
    for row in reviews:
        link = "<a href='" + escape(row["record"]) + "'>Raw proof</a>" if row["record"] else ""
        parts.append("<tr><td><pre>" + escape({k: row[k] for k in ("case_id", "variant", "turn", "checkpoint")}) + "</pre>" + link
                     + "</td><td><pre>" + escape(row["messages"]) + "</pre></td><td><pre>" + escape(row["output"])
                     + "</pre></td><td><pre>" + escape({k: row[k] for k in ("state", "verified", "finish_reason", "complete", "review_criteria", "human_review")}) + "</pre></td></tr>")
    parts.append("</tbody></table></html>")
    (folder / "review.html").write_text("\n".join(parts))
    bound = files(folder)
    save(folder / "results-manifest.json", {"binding": journal.binding,
        "files": {p: s for p, s in bound.items() if p not in {"status.json", "results-manifest.json"}}, "kind": KIND})
    journal.commit()
    return summary


def terminal(call_id):
    import modal
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return True
    except (modal.exception.RemoteError, modal.exception.FunctionTimeoutError, modal.exception.UserCodeException):
        return True
    except (TimeoutError, modal.exception.TimeoutError, modal.exception.NotFoundError):
        return False


def launch(package, checksum, receipt, authorize, profile, previous=None):
    require(authorize and not Path(receipt).exists(), "Explicit GPU authorization and new receipt required")
    package = Path(package).resolve()
    verify(package, checksum)
    os.environ.update(GHA_R4_EVALUATION_PACKAGE=str(package), MODAL_PROFILE=profile)
    import modal
    identifier, attempt = "r4_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), 1
    if previous:
        old = read(previous)
        require(old["state"] == "submitted" and old["execution_sha256"] == checksum
                and old["profile"] == profile and terminal(old["call_id"]), "Previous call is active, ambiguous or differently bound")
        identifier, attempt = old["run_id"], old["attempt"] + 1
    record = {"state": "submitting", "run_id": identifier, "attempt": attempt, "execution_sha256": checksum,
              "package": str(package), "profile": profile, "volume": VOLUME, "remote_directory": PREFIX + "/" + identifier}
    save(receipt, record)
    worker = module_at(package / "code/evaluate_understanding_r4.py", "evaluate_understanding_r4", register=True)
    with modal.enable_output(), worker.app.run(detach=True):
        call = worker.evaluate.spawn(identifier, checksum, attempt)
        record.update(state="submitted", call_id=call.object_id)
        save(receipt, record)
    return record


def remote_status(receipt):
    value = read(receipt)
    require(value.get("call_id"), "Ambiguous submission; do not retry")
    os.environ["MODAL_PROFILE"] = value["profile"]
    import modal
    try:
        result = modal.FunctionCall.from_id(value["call_id"]).get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError) as error:
        if isinstance(error, (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError)):
            raise
        return {"state": "pending", "call_id": value["call_id"]}
    save(Path(receipt).with_suffix(".result.json"), result)
    return result


def collect(receipt, out):
    value, out = read(receipt), Path(out)
    require(not out.exists(), "Use a fresh collection directory")
    os.environ["MODAL_PROFILE"] = value["profile"]
    import modal
    require(terminal(value["call_id"]), "Collect only a terminal call for a consistent snapshot")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    prefix = value["remote_directory"]
    require(prefix == PREFIX + "/" + value["run_id"] and re.fullmatch(r"r4_eval_\d{8}T\d{6}Z", value["run_id"]), "Invalid collection prefix")
    out.mkdir(parents=True)
    for entry in volume.iterdir(prefix, recursive=True):
        if entry.type.name != "FILE":
            continue
        remote = "/" + entry.path.lstrip("/")
        require(remote.startswith(prefix + "/"), "Unexpected remote prefix")
        path = safe(out, remote[len(prefix) + 1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for chunk in volume.read_file(remote):
                stream.write(chunk)
    require(read(out / "journal/binding.json")["execution_sha256"] == value["execution_sha256"], "Collected execution differs")
    if (out / "results-manifest.json").exists():
        for name, checksum in read(out / "results-manifest.json")["files"].items():
            require(sha(safe(out, name)) == checksum, "Collected artifact differs")
    save(out / "collection.json", {"receipt": value, "files": files(out), "kind": KIND})
    return {"directory": str(out), "status": read(out / "status.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--tokenizer-bundle", type=Path, required=True)
    check.add_argument("--evaluation", type=Path, required=True)
    prep = sub.add_parser("prepare")
    for name in ("bundle", "evaluation", "training-inputs", "consumption", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--bundle-sha", required=True)
    prep.add_argument("--language-loss", action="store_true")
    prep.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(VARIANTS))
    prep.add_argument("--policy", choices=POLICIES, default="raw")
    for action in ("verify", "launch"):
        p = sub.add_parser(action)
        p.add_argument("--execution", type=Path, required=True)
        p.add_argument("--execution-sha", required=True)
        if action == "launch":
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--profile", required=True)
            p.add_argument("--previous-receipt", type=Path)
            p.add_argument("--authorize-gpu", action="store_true")
    for action in ("status", "collect"):
        p = sub.add_parser(action)
        p.add_argument("--receipt", type=Path, required=True)
        if action == "collect":
            p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight(args.tokenizer_bundle, args.evaluation)
    elif args.action == "prepare":
        result = prepare(args.bundle, args.bundle_sha, args.evaluation, args.training_inputs, args.consumption, args.out,
                         loss=args.language_loss, variants=args.variants, policy=args.policy)
    elif args.action == "verify":
        result = verify(args.execution, args.execution_sha)[0]
    elif args.action == "launch":
        result = launch(args.execution, args.execution_sha, args.receipt, args.authorize_gpu, args.profile, args.previous_receipt)
    elif args.action == "status":
        result = remote_status(args.receipt)
    else:
        result = collect(args.receipt, args.out)
    print(json.dumps(result, indent=2))
    if args.action == "preflight" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
