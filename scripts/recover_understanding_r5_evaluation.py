#!/usr/bin/env python3
"""Seal/verify an offline recovery controller; launch requires explicit GPU approval.

The original execution remains byte-identical. Only its runner.terminal callback
is replaced at runtime, with that control change and preserved journal hashes
written to a separate immutable sidecar before the sealed worker's local call.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil

import run_understanding_r5_evaluation as api

VERSION = "r5-evaluation-recovery-v1"
SDK = "1.4.3"
WORKER = "recover_understanding_r5_evaluation_worker.py"
CODE = ("recover_understanding_r5_evaluation.py", "run_understanding_r5_evaluation.py", "r5_retention.py", WORKER)
BASELINE = {"hash_concurrency": 8, "scope": "all_protected_files", "preserve_all_hashes": True}
PATCH = {"target": "sealed_worker.runner.terminal", "replacement": "controller_runner.terminal",
         "reason": "Recognize SDK-proven deserialized remote exceptions; unknown remains nonterminal",
         "inference_data_prompt_decoding_changes": False, "claim_and_journal_changes": False}


def verify_sdk():
    import modal
    # Modal mounts its SDK under /pkg in workers without distribution metadata.
    version = getattr(modal, "__version__", None)
    api.require(version == SDK, "Unverified Modal SDK version")
    return version


def previous_receipt(value, checksum):
    api.require(value["state"] == "submitted" and value["execution_sha256"] == checksum
                and type(value["attempt"]) is int and value["attempt"] >= 1
                and re.fullmatch(r"r5_eval_\d{8}T\d{6}Z", value["run_id"])
                and re.fullmatch(r"fc-[A-Z0-9]{26}", value["call_id"])
                and value["volume"] == api.VOLUME
                and value["remote_directory"] == api.PREFIX + "/" + value["run_id"]
                and isinstance(value["profile"], str) and value["profile"], "Invalid previous receipt binding")


def protected(name):
    return name.startswith(("journal/", "runtime/", "provenance/", "recovery-control/")) or name in {"execution.json", "context-preflight.json"}


def snapshot_inventory(snapshot, previous):
    collection = api.read(snapshot / "collection.json")
    api.require(collection["receipt"] == previous, "Collection is from another call")
    actual = api.files(snapshot)
    api.require({k: v for k, v in actual.items() if k != "collection.json"} == collection["files"],
                "Collected snapshot changed; use the original collection")
    binding = api.read(snapshot / "journal/binding.json")
    api.require(binding["run_id"] == previous["run_id"] and binding["execution_sha256"] == previous["execution_sha256"],
                "Journal binding differs")
    records, started, states, kinds = set(), set(), Counter(), Counter()
    for name in actual:
        if name.startswith("journal/records/"):
            envelope = api.read(api.safe(snapshot, name))
            record = envelope["record"]
            api.require(envelope["sha256"] == api.digest(record) and record["binding_sha256"] == api.digest(binding)
                        and Path(name).stem == api.digest(record["id"]) and record["id"] not in records,
                        "Corrupt saved record")
            proof = snapshot / "runtime" / ("attempt-" + str(record["attempt"]) + ".json")
            api.require(api.sha(proof) == record["runtime_proof_sha256"] and api.read(proof)["binding"] == binding,
                        "Saved runtime proof differs")
            records.add(record["id"])
            states[record["state"]] += 1
            kinds[record["kind"]] += 1
        elif name.startswith("journal/attempts/"):
            marker = api.read(api.safe(snapshot, name))
            api.require(marker["state"] == "started" and Path(name).stem == api.digest(marker["id"]), "Invalid start marker")
            started.add(marker["id"])
    api.require(records, "Recovery requires a nonempty verified snapshot")
    return {"collection_sha256": actual["collection.json"], "binding": binding,
            "files": {k: v for k, v in actual.items() if protected(k)},
            "finalized_records": len(records), "states": dict(states), "kinds": dict(kinds),
            "started_without_result": len(started - records)}


def prepare(package, checksum, previous, snapshot, out):
    package, snapshot, out = Path(package).resolve(), Path(snapshot).resolve(), Path(out).absolute()
    api.require(not out.exists() and not out.is_symlink() and not out.resolve().is_relative_to(package)
                and not out.resolve().is_relative_to(snapshot), "Use a new controller directory outside sealed inputs")
    execution = api.verify(package, checksum)[0]
    old = api.read(previous)
    previous_receipt(old, checksum)
    inventory = snapshot_inventory(snapshot, old)
    api.require(api.sha(api.retention.__file__) == execution["files"]["code/r5_retention.py"], "Retention helper differs")
    sources = {name: Path(__file__).with_name(name) for name in CODE if name != WORKER}
    sources[WORKER] = api.ROOT / "modal/train" / WORKER
    out.mkdir(mode=0o700)
    (out / "code").mkdir(mode=0o700)
    for name, source in sources.items():
        shutil.copyfile(source, out / "code" / name)
        (out / "code" / name).chmod(0o600)
    api.save(out / "previous.json", old)
    api.save(out / "snapshot.json", inventory)
    contract = {"version": VERSION, "execution_sha256": checksum, "original_worker_sha256": execution["files"]["code/evaluate_understanding_r5.py"],
                "original_runner_sha256": execution["files"]["code/run_understanding_r5_evaluation.py"],
                "run_id": old["run_id"], "attempt": old["attempt"] + 1, "previous_call_id": old["call_id"],
                "profile": old["profile"], "patch": PATCH, "modal_version": SDK, "baseline_verification": BASELINE,
                "resources": {"gpu": "H100", "maximum_gpu_workers": 1, "hard_seconds": 7200, "automatic_retries": 0},
                "files": api.files(out)}
    api.save(out / "controller.json", contract)
    checksum = api.sha(out / "controller.json")
    api.save(out / "SEALED.json", {"controller_sha256": checksum})
    for path in out.rglob("*.json"):
        path.chmod(0o600)
    return {"controller_sha256": checksum, "attempt": contract["attempt"], "snapshot": {
        k: inventory[k] for k in ("finalized_records", "states", "kinds", "started_without_result")}, "gpu_launched": False}


def verify(controller, controller_sha, package):
    controller, package = Path(controller).resolve(), Path(package).resolve()
    contract = api.read(controller / "controller.json")
    api.require(api.sha(controller / "controller.json") == controller_sha
                and api.read(controller / "SEALED.json") == {"controller_sha256": controller_sha}, "Controller seal differs")
    api.require(contract["version"] == VERSION and contract["patch"] == PATCH and contract["modal_version"] == SDK
                and contract["baseline_verification"] == BASELINE
                and contract["resources"] == {"gpu": "H100", "maximum_gpu_workers": 1, "hard_seconds": 7200, "automatic_retries": 0},
                "Recovery control contract differs")
    api.require(set(contract["files"]) == {"previous.json", "snapshot.json"} | {"code/" + n for n in CODE}
                and api.files(controller) == {**contract["files"], "controller.json": controller_sha,
                                              "SEALED.json": api.sha(controller / "SEALED.json")}, "Controller files changed")
    # The launcher itself, not only its uploaded copy, must be the sealed implementation.
    api.require(api.sha(__file__) == contract["files"]["code/recover_understanding_r5_evaluation.py"]
                and api.sha(api.__file__) == contract["files"]["code/run_understanding_r5_evaluation.py"], "Use the sealed controller code")
    execution = api.verify(package, contract["execution_sha256"])[0]
    api.require(execution["files"]["code/evaluate_understanding_r5.py"] == contract["original_worker_sha256"]
                and execution["files"]["code/run_understanding_r5_evaluation.py"] == contract["original_runner_sha256"]
                and execution["files"]["code/r5_retention.py"] == contract["files"]["code/r5_retention.py"] == api.sha(api.retention.__file__),
                "Frozen worker/runner/retention differs")
    old, snapshot = api.read(controller / "previous.json"), api.read(controller / "snapshot.json")
    previous_receipt(old, contract["execution_sha256"])
    api.require(contract["run_id"] == old["run_id"] and contract["attempt"] == old["attempt"] + 1
                and contract["previous_call_id"] == old["call_id"] and contract["profile"] == old["profile"]
                and snapshot["binding"]["execution_sha256"] == contract["execution_sha256"]
                and snapshot["binding"]["run_id"] == old["run_id"], "Recovery attempt differs")
    return contract, old, snapshot


def verify_baseline(folder, snapshot):
    paths = sorted(Path(folder).rglob("*"))
    api.require(not any(p.is_symlink() for p in paths), "Linked artifact")
    selected = [(p.relative_to(folder).as_posix(), p) for p in paths
                if p.is_file() and protected(p.relative_to(folder).as_posix())]
    # Verify every protected byte; bound parallel reads on the remote volume.
    def checksum(item):
        name, path = item
        return name, api.sha(path)
    with ThreadPoolExecutor(max_workers=BASELINE["hash_concurrency"]) as pool:
        actual = dict(pool.map(checksum, selected))
    api.require(actual == snapshot["files"], "Remote journal/provenance differs from collected snapshot")


def execute(worker, controller, controller_sha, package, call_id, *, artifacts_root=Path("/artifacts")):
    """Called only inside the separately authorized bounded wrapper container."""
    contract, old, snapshot = verify(controller, controller_sha, package)
    verify_sdk()
    api.require(worker.claims.get(old["run_id"] + ":binding") == old["execution_sha256"]
                and worker.claims.get(old["run_id"] + ":" + str(old["attempt"])) == old["call_id"], "Previous claim differs")
    api.require(worker.claims.get(old["run_id"] + ":" + str(contract["attempt"])) is None, "Attempt already claimed")
    proof = api.call_status(old["call_id"])
    api.require(proof["terminal"], "Previous call is active or ambiguous")
    worker.artifacts.reload()
    folder = Path(artifacts_root) / old["remote_directory"].lstrip("/")
    verify_baseline(folder, snapshot)
    sidecar = folder / "recovery-control" / ("attempt-" + str(contract["attempt"]))
    api.require(not sidecar.exists(), "Recovery attempt already recorded; no automatic retry")
    shutil.copytree(controller, sidecar)
    api.save(sidecar / "invocation.json", {"controller_sha256": controller_sha, "execution_sha256": old["execution_sha256"],
        "run_id": old["run_id"], "attempt": contract["attempt"], "call_id": call_id,
        "previous_terminal_proof": proof, "patch": PATCH, "snapshot_sha256": contract["files"]["snapshot.json"]})
    worker.artifacts.commit()
    original = worker.runner.terminal
    try:
        worker.runner.terminal = api.terminal
        # The original claim_attempt still atomically claims attempt N and probes N-1.
        return worker.evaluate.local(old["run_id"], old["execution_sha256"], contract["attempt"])
    finally:
        worker.runner.terminal = original


def launch(controller, checksum, package, receipt, authorize):
    api.require(authorize and not Path(receipt).exists(), "Explicit GPU authorization and new receipt required")
    controller, package = Path(controller).resolve(), Path(package).resolve()
    contract, old, _ = verify(controller, checksum, package)
    verify_sdk()
    os.environ.update(MODAL_PROFILE=old["profile"], GHA_R5_EVALUATION_PACKAGE=str(package), GHA_R5_RECOVERY_PACKAGE=str(controller))
    import modal
    proof = api.call_status(old["call_id"])
    api.require(proof["terminal"], "Previous call is active or ambiguous")
    record = {**old, "state": "submitting", "attempt": contract["attempt"], "package": str(package),
              "controller": str(controller), "controller_sha256": checksum, "previous_call_id": old["call_id"],
              "previous_terminal_proof": proof}
    record.pop("call_id")
    api.save(receipt, record)
    worker = api.module_at(controller / "code" / WORKER, WORKER[:-3], register=True)
    with modal.enable_output(), worker.app.run(detach=True):
        call = worker.recover.spawn(checksum)
        record.update(state="submitted", call_id=call.object_id)
        api.save(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for name in ("execution", "previous-receipt", "snapshot", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--execution-sha", required=True)
    for action in ("verify", "launch"):
        p = sub.add_parser(action)
        for name in ("controller", "execution"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--controller-sha", required=True)
        if action == "launch":
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--authorize-gpu", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.execution, args.execution_sha, args.previous_receipt, args.snapshot, args.out)
    elif args.action == "verify":
        result = verify(args.controller, args.controller_sha, args.execution)[0]
    else:
        result = launch(args.controller, args.controller_sha, args.execution, args.receipt, args.authorize_gpu)
    print(api.json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
