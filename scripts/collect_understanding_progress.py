#!/usr/bin/env python3
"""Read-only, bounded metadata collection for a private understanding training run.

No tensors are downloaded or deserialized. A checkpoint's seal, manifest identity
and state byte count can be checked; its state.pt content hash cannot. Counts are
report/step accounting, not proof of tensor contents or a model-quality metric.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from training_data_inputs import parse_json, read, require, sha, valid_hash

VERSION = "understanding-progress-v1"
METRICS = ("updates", "microbatches", "input_tokens", "effective_target_tokens")


def missing(error):
    if isinstance(error, FileNotFoundError):
        return True
    try:
        from modal.exception import NotFoundError
        return isinstance(error, NotFoundError)
    except ImportError:
        return False


def small(volume, path, optional=False):
    require(path.endswith((".json", ".jsonl")), "Only small JSON metadata may be read")
    data = bytearray()
    try:
        for chunk in volume.read_file(path):
            data.extend(chunk)
            require(len(data) <= 4 * 1024 * 1024, "Metadata exceeds 4 MiB bound")
    except Exception as error:
        if optional and missing(error):
            return None
        raise
    return bytes(data)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def inside(path, root, volume_id=None):
    prefixes = ["/artifacts" + root + "/"]
    if isinstance(volume_id, str) and re.fullmatch(r"vo-[A-Za-z0-9]+", volume_id):
        prefixes.append("/__modal/volumes/" + volume_id + root + "/")
    prefix = next((p for p in prefixes if isinstance(path, str) and path.startswith(p)), None)
    require(prefix is not None, "Report path outside requested run/volume")
    relative = path.removeprefix(prefix)
    require(".." not in Path(relative).parts and Path(relative).as_posix() == relative, "Unsafe report path")
    return root + "/" + relative


def listing(volume, root):
    try:
        entries = volume.listdir(root, recursive=True)
    except Exception as error:
        if missing(error):
            return {}
        raise
    require(len(entries) <= 10000, "Unexpectedly large metadata inventory")
    result = {}
    for entry in entries:
        path = "/" + str(entry.path).lstrip("/")
        require(path.startswith(root + "/") and ".." not in Path(path).parts, "Unexpected remote listing path")
        result[path] = entry
    return result


def checkpoint(volume, root, report, contract, inventory):
    path = report.get("checkpoint")
    if path is None:
        return {"status": "no_checkpoint_reported", "metadata_verified": False, "state_hash_verified": False}
    remote = inside(path, root, getattr(volume, "object_id", None))
    require(re.fullmatch(r"update-\d{8}", Path(remote).name), "Invalid checkpoint directory")
    update = int(Path(remote).name.removeprefix("update-"))
    require(update == report["checkpoint_completed_updates"] <= report["position"]["completed_updates"], "Checkpoint/report cursor mismatch")
    seal_raw = small(volume, remote + "/COMMITTED.json", optional=True)
    manifest_raw = small(volume, remote + "/manifest.json", optional=True)
    evidence = {"path": path, "reported_updates": update, "metadata_verified": False, "state_hash_verified": False}
    if seal_raw is None or manifest_raw is None:
        return {**evidence, "status": "incomplete_publication"}
    seal, manifest = parse_json(seal_raw.decode()), parse_json(manifest_raw.decode())
    require(set(seal) == {"manifest_sha256"} and seal["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest(), "Checkpoint manifest checksum mismatch")
    require(set(manifest) == {"version", "identity", "state_sha256", "state_bytes"}
            and manifest["version"] == 1 and manifest["identity"] == {
                "run_sha256": digest(contract), "data_sha256": report["data_sha256"]}, "Checkpoint identity mismatch")
    require(valid_hash(manifest["state_sha256"]) and type(manifest["state_bytes"]) is int and manifest["state_bytes"] > 0, "Invalid state metadata")
    entry = inventory.get(remote + "/state.pt")
    if entry is None:
        entry = listing(volume, remote).get(remote + "/state.pt")
    if entry is None:
        return {**evidence, "status": "state_not_visible"}
    require(entry.type.name == "FILE" and entry.size == manifest["state_bytes"], "Checkpoint state size/type mismatch")
    return {**evidence, "status": "committed_metadata_verified", "metadata_verified": True,
        "manifest_sha256": seal["manifest_sha256"], "state_sha256_declared": manifest["state_sha256"],
        "state_bytes_verified": entry.size,
        "limit": "COMMITTED binds metadata; state.pt bytes/hash and saved position were not inspected."}


def parse_steps(raw, planned):
    if raw is None:
        return [], False
    lines = raw.splitlines(keepends=True)
    trailing = bool(lines and not lines[-1].endswith(b"\n"))
    if trailing:
        lines.pop()
    values, previous = [], None
    for line in lines:
        step = parse_json(line.decode())
        require(integer(step["update"]) and 1 <= step["update"] <= planned["updates"], "Invalid logged update")
        require(previous is None or step["update"] == previous["update"] + 1, "Noncontiguous/duplicate step log")
        require(all(integer(step[k]) for k in ("start_index", "next_index", "microbatches", "input_tokens", "effective_target_tokens")), "Invalid step accounting")
        require(step["microbatches"] > 0 and step["next_index"] - step["start_index"] == step["microbatches"]
                and step["next_index"] <= planned["microbatches"]
                and (previous is None or step["start_index"] == previous["next_index"]), "Step cursor mismatch")
        require(step["effective_target_tokens"] > 0 and step["input_tokens"] >= step["effective_target_tokens"]
                and all(number(step[k]) for k in ("loss", "loss_sum", "seconds", "learning_rate")), "Invalid training loss/time")
        require(math.isclose(step["loss"], step["loss_sum"] / step["effective_target_tokens"], rel_tol=1e-8, abs_tol=1e-10), "Training loss denominator mismatch")
        values.append(step)
        previous = step
    return values, trailing


def project(planned, observed, steps, report):
    measured = [s for s in steps if s["seconds"] > 0]
    seconds = sum(s["seconds"] for s in measured)
    covered = [s for s in measured if s["update"] <= report["position"]["completed_updates"]]
    elapsed = report["elapsed_seconds"]
    wall_average = elapsed / len(covered) if covered and elapsed > 0 else None
    result = {"measured_updates": len(measured), "update_compute_seconds": seconds,
        "compute_updates_per_second": len(measured) / seconds if seconds else None,
        "reported_wall_updates_per_second": len(covered) / elapsed if covered and elapsed > 0 else None,
        "input_tokens_per_compute_second": sum(s["input_tokens"] for s in measured) / seconds if seconds else None,
        "target_tokens_per_compute_second": sum(s["effective_target_tokens"] for s in measured) / seconds if seconds else None}
    remaining = planned["updates"] - observed["updates"]
    if remaining == 0:
        estimate = {"seconds_low": 0.0, "seconds_high": 0.0, "basis": "No remaining optimizer updates; final save/upload may still be pending."}
    elif len(measured) < 3:
        estimate = {"seconds_low": None, "seconds_high": None, "basis": "Fewer than three measured updates; no useful projection yet."}
    else:
        sample = measured[-20:]
        low = min(s["seconds"] for s in sample)
        high = max([s["seconds"] for s in sample] + ([wall_average] if wall_average is not None else []))
        estimate = {"seconds_low": remaining * low, "seconds_high": remaining * high,
            "observed_updates_used": len(sample),
            "basis": "Remaining updates times recent observed fastest/slowest update durations; upper also includes measured average attempt overhead. Descriptive scenario range, not a confidence interval or deadline.",
            "limits": "Sequence lengths, future checkpoints, stalls and interruptions can change the rate. Applies only to this stage."}
    return result, estimate


def stage_report(volume, root, path, plan, execution_sha, data_sha, code_hashes, inventory):
    planned = {"updates": plan["optimizer_updates"], "microbatches": plan["scheduled_sequences"],
               "input_tokens": plan["sequence_tokens"], "effective_target_tokens": plan["effective_target_tokens"]}
    for _ in range(3):
        raw = small(volume, path)
        report = parse_json(raw.decode())
        require(report["run_sha256"] == execution_sha and report["data_sha256"] == data_sha
                and report["schedule_sha256"] == plan["checkpoint_schedule_sha256"], "Stage report identity mismatch")
        volume_id = getattr(volume, "object_id", None)
        require(inside(report["attempt"], root, volume_id) == str(Path(path).parent)
                and inside(report["steps_path"], root, volume_id) == str(Path(path).parent / "steps.jsonl"), "Stage report pointer mismatch")
        contract_raw = small(volume, str(Path(path).parent.parent / "contract.json"))
        contract = parse_json(contract_raw.decode())
        require(contract["run_sha256"] == execution_sha and contract["data_sha256"] == data_sha
                and contract["schedule_sha256"] == plan["checkpoint_schedule_sha256"]
                and digest(contract) == report["checkpoint_run_sha256"]
                and contract["core_sha256"] == code_hashes["modal/train/understanding_training_core.py"]
                and contract["resume_helper_sha256"] == code_hashes["scripts/response_resume_state.py"], "Stage contract mismatch")
        require({"updates": report["planned_updates"], "microbatches": report["planned_microbatches"],
                 "input_tokens": report["planned_input_tokens"], "effective_target_tokens": report["planned_effective_target_tokens"]} == planned,
                "Stage planned counts differ")
        require(number(report["elapsed_seconds"]), "Invalid reported elapsed time")
        position = report["position"]
        exposure = position["exposure"]
        counts = {"updates": position["completed_updates"], "microbatches": position["completed_microbatches"],
                  "input_tokens": exposure["input_tokens"], "effective_target_tokens": exposure["effective_target_tokens"]}
        require(all(integer(v) and v <= planned[k] for k, v in counts.items())
                and position["microbatches_in_update"] == 0, "Invalid reported update boundary")
        require(number(exposure["loss_sum"]), "Invalid cumulative training loss")
        steps, trailing = parse_steps(small(volume, inside(report["steps_path"], root, volume_id), optional=True), planned)
        proof = checkpoint(volume, root, report, contract, inventory)
        if raw == small(volume, path):
            break
    else:
        return {"attempt": str(Path(path).parent), "state": "concurrent_update", "planned": planned,
                "committed": None, "observed": None, "warning": "Report changed during all three read attempts; refresh."}
    observed = dict(counts)
    after_report = [s for s in steps if s["update"] > counts["updates"]]
    if after_report:
        require(after_report[0]["update"] == counts["updates"] + 1
                and after_report[0]["start_index"] == counts["microbatches"], "Step log skips reported cursor")
        observed["updates"] = after_report[-1]["update"]
        for key in METRICS[1:]:
            observed[key] += sum(s[key] for s in after_report)
    require(all(observed[k] <= planned[k] for k in METRICS), "Observed counters exceed plan")
    committed = None
    if proof["metadata_verified"]:
        cursor = proof["reported_updates"]
        tail = [s for s in steps if cursor < s["update"] <= counts["updates"]]
        require([s["update"] for s in tail] == list(range(cursor + 1, counts["updates"] + 1)), "Missing step evidence to separate committed exposure")
        committed = {"updates": cursor, **{k: counts[k] - sum(s[k] for s in tail) for k in METRICS[1:]}}
        require(all(integer(v) for v in committed.values()), "Invalid committed exposure")
    throughput, estimate = project(planned, observed, steps, report)
    return {"attempt": report["attempt"], "state": report["status"], "planned": planned,
        "reported": counts, "observed": observed, "committed": committed,
        "beyond_checkpoint": {k: observed[k] - committed[k] for k in METRICS} if committed is not None else None,
        "checkpoint": proof, "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_elapsed_seconds": report["elapsed_seconds"], "throughput": throughput,
        "projected_remaining": estimate, "partial_step_line_ignored": trailing,
        "loss_label": "Training loss (shifted causal cross-entropy); not evaluation accuracy or model quality.",
        "reported_mean_training_loss": exposure["loss_sum"] / counts["effective_target_tokens"] if counts["effective_target_tokens"] else None,
        "steps": [{"update": s["update"], "training_loss": s["loss"], "training_loss_sum": s["loss_sum"],
                   "input_tokens": s["input_tokens"], "effective_target_tokens": s["effective_target_tokens"],
                   "seconds": s["seconds"], "learning_rate": s["learning_rate"],
                   "checkpoint_covered": s["update"] <= committed["updates"] if committed is not None else None} for s in steps]}


def collect(receipt_path, volume, now=None):
    receipt = read(receipt_path)
    run_id = receipt["run_id"]
    require(isinstance(run_id, str) and re.fullmatch(r"understanding_\d{8}T\d{6}Z", run_id)
            and receipt["mode"] == "train" and valid_hash(receipt["execution_sha256"]), "Invalid training receipt identity")
    pack = Path(receipt["pack"])
    data_sha = sha(pack / "manifest.json")
    data = read(pack / "manifest.json")
    root = "/understanding-adaptation-r1/" + run_id
    current = now or datetime.now(timezone.utc)
    require(current.tzinfo is not None, "Timezone-aware collection time required")
    submitted = datetime.strptime(run_id.removeprefix("understanding_"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    result = {"version": VERSION, "run_id": run_id, "collected_at": current.astimezone(timezone.utc).isoformat(),
        "receipt_sha256": sha(receipt_path), "execution_sha256": receipt["execution_sha256"], "data_sha256": data_sha,
        "wall_since_run_id_timestamp_seconds": max(0.0, (current - submitted).total_seconds()),
        "wall_clock_note": "Includes submission/build/queue/loading; run-ID timestamp is not an exact training start timestamp.",
        "read_only": True, "gpu_launched": False, "tensor_bytes_downloaded": 0,
        "limits": ["Remote visibility may lag training until volume commit.",
                   "Checkpoint metadata verified does not mean state.pt content/hash or restoration verified.",
                   "Per-step loss is training loss, not evidence of semantic quality."]}
    execution_raw = small(volume, root + "/execution.json", optional=True)
    status_raw = small(volume, root + "/status.json", optional=True)
    if execution_raw is None or status_raw is None:
        return {**result, "state": "awaiting_remote_metadata", "identity_verified": False, "stages": [], "overall": None}
    require(hashlib.sha256(execution_raw).hexdigest() == receipt["execution_sha256"], "Execution checksum mismatch")
    execution, status = parse_json(execution_raw.decode()), parse_json(status_raw.decode())
    require(execution["mode"] == "train" and execution["data_sha256"] == data_sha, "Execution/data mismatch")
    require(status["mode"] == "train" and status["run_id"] == run_id
            and status["execution_sha256"] == receipt["execution_sha256"] and status["data_sha256"] == data_sha
            and status["remote_directory"] == "/artifacts" + root, "Remote status identity mismatch")
    remote_data = small(volume, root + "/inputs/manifest.json")
    require(hashlib.sha256(remote_data).hexdigest() == data_sha, "Remote data manifest mismatch")
    plans = execution["expected_stages"]
    require(plans == data["stages"] and [p["name"] for p in plans] == ["meaning", "responses"], "Stage plan mismatch")
    schedule_raw = small(volume, root + "/inputs/stages.json")
    require(hashlib.sha256(schedule_raw).hexdigest() == data["artifacts"]["stages.json"], "Schedule metadata checksum mismatch")
    schedules = parse_json(schedule_raw.decode())
    require(len(schedules) == len(plans), "Incomplete stage schedules")
    bound_plans = []
    for plan, schedule in zip(plans, schedules, strict=True):
        require({k: v for k, v in schedule.items() if k != "schedule"} == plan
                and isinstance(schedule["schedule"], list) and len(schedule["schedule"]) == plan["scheduled_sequences"]
                and all(integer(i) for i in schedule["schedule"])
                and digest(schedule["schedule"]) == plan["schedule_sha256"], "Schedule plan mismatch")
        # The source pack hashes the array; the trainer hashes {indices: array}.
        bound_plans.append({**plan, "checkpoint_schedule_sha256": digest({"indices": schedule["schedule"]})})
    inventory = listing(volume, root)
    stages = []
    for plan in bound_plans:
        paths = sorted(p for p in inventory if re.fullmatch(
            re.escape(root + "/" + plan["name"]) + r"/attempt-[A-Za-z0-9_-]+/report\.json", p))
        attempts = [stage_report(volume, root, p, plan, receipt["execution_sha256"], data_sha,
                                  execution["code_hashes"], inventory) for p in paths]
        planned = {"updates": plan["optimizer_updates"], "microbatches": plan["scheduled_sequences"],
                   "input_tokens": plan["sequence_tokens"], "effective_target_tokens": plan["effective_target_tokens"]}
        if attempts:
            selected = max(attempts, key=lambda a: ((a.get("committed") or {}).get("updates", -1),
                                                   (a.get("observed") or {}).get("updates", -1)))
            stages.append({"name": plan["name"], **selected, "attempts_visible": len(attempts),
                "selection_note": "Highest visible checkpoint then observed cursor; attempt names do not establish chronology."})
        else:
            stages.append({"name": plan["name"], "state": "no_report_visible", "planned": planned,
                           "committed": None, "observed": None, "steps": [], "attempts_visible": 0})
    overall = {"planned": {k: sum(s["planned"][k] for s in stages) for k in METRICS},
        "verified_checkpoint_accounting": {k: sum((s.get("committed") or {}).get(k, 0) for s in stages) for k in METRICS},
        "observed_accounting": {k: sum((s.get("observed") or {}).get(k, 0) for s in stages) for k in METRICS},
        "stages_without_verified_checkpoint": [s["name"] for s in stages if s.get("committed") is None],
        "projected_remaining_seconds_range": None,
        "projection_note": "Use stage-level ranges. Unobserved later stages have different workloads; no whole-run ETA is asserted."}
    return {**result, "state": status["state"], "identity_verified": True, "active_stage": status.get("stage"),
        "status_elapsed_seconds_at_last_write": status.get("elapsed_seconds"), "stages": stages, "overall": overall,
        "status_changed_during_collection": status_raw != small(volume, root + "/status.json")}


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".progress-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = read(args.receipt)
        os.environ["MODAL_PROFILE"] = receipt["profile"]
        import modal
        volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
        value = collect(args.receipt, volume)
        atomic_write(args.out, value)
        print(json.dumps({"state": value["state"], "identity_verified": value["identity_verified"],
                          "collected_at": value["collected_at"], "output": str(args.out), "gpu_launched": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"state": "blocked", "error": str(error), "previous_snapshot_preserved": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
