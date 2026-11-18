#!/usr/bin/env python3
"""Prepare all selected English histories, then explicitly launch/collect Modal triage."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from r7_english_review_controls import SYSTEM, VERSION as REVIEW_VERSION, assess, controls

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def prepare(source, qualification, out):
    from prepare_r7_source_inventory import verify, digest
    verify(source)
    if not assess(json.loads(qualification.read_text()))["qualified_for_english_source_triage_only"]:
        raise ValueError("English qualification failed")
    inputs, refs = [], []
    path = source / "oasst-candidates.jsonl"
    for line, raw in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(raw)
        if row["status"] != "eligible" or row["language"] != "en" or row["source_split"] != "train":
            raise ValueError("Wrong candidate scope")
        messages = row["messages"]
        if messages[-1]["role"] != "assistant" or row["messages_sha256"] != digest(messages):
            raise ValueError("Incomplete or changed original history")
        payload = {"history": messages[:-1], "assistant_target": messages[-1]["content"]}
        inputs.append({"id": row["id"], "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
        refs.append({"id": row["id"], "source_id": row["source_id"], "tree_id": row["tree_id"],
                     "candidate_line": line, "candidate_sha256": digest(row),
                     "coverage_bucket": row["coverage_bucket"], "training_eligible": False})
    if not 1 <= len(inputs) <= 5000 or len({r["id"] for r in inputs}) != len(inputs):
        raise ValueError("Invalid explicit all-candidate work plan")
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("requests.jsonl", inputs), ("source-refs.jsonl", refs)):
        with (out / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    plan = {"scope": "all_eligible_original_english_histories", "count": len(inputs),
        "source_manifest_sha256": sha(source / "manifest.json"), "candidate_file_sha256": sha(path),
        "qualification_path": str(qualification.resolve()), "qualification_sha256": sha(qualification),
        "requests_sha256": sha(out / "requests.jsonl"), "source_refs_sha256": sha(out / "source-refs.jsonl"),
        "training_eligible": False, "twi_validated": False, "human_reviewed": False}
    save(out / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "launch", "resume", "collect"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.source, args.qualification, args.out)))
        return
    import modal
    spec = importlib.util.spec_from_file_location("source_review", ROOT / "modal/train/r7_source_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt_path = args.out / "launch.receipt.json"
    if args.command == "collect":
        receipt = json.loads(receipt_path.read_text())
        graph = modal.FunctionCall.from_id(receipt["function_call_id"]).get_call_graph()
        result = json.loads(b"".join(module.teacher.artifacts.read_file(
            module.PREFIX + receipt["run_id"] + "/progress.json")))
        if result["requests_sha256"] != receipt["requests_sha256"]:
            raise ValueError("Collected evidence differs")
        save(args.out / "progress.json", result)
        lifecycle = {"function_call_id": receipt["function_call_id"], "collected_at_unix": time.time(),
            "remote_states": [node.status.name for node in graph], "checkpoint_sha256": sha(args.out / "progress.json")}
        save(args.out / "collection.receipt.json", lifecycle)
        print(json.dumps({"checkpoint_status": result["status"], "completed": len(result["results"]),
                          "total": result["total"], "remote_states": lifecycle["remote_states"]}))
        return
    plan = json.loads((args.out / "plan.json").read_text())
    if sha(args.out / "requests.jsonl") != plan["requests_sha256"] or sha(args.out / "source-refs.jsonl") != plan["source_refs_sha256"]:
        raise ValueError("Plan artifacts changed")
    qualification = Path(plan["qualification_path"])
    if sha(qualification) != plan["qualification_sha256"]:
        raise ValueError("Qualification changed")
    report = json.loads(qualification.read_text())
    if not assess(report)["qualified_for_english_source_triage_only"]:
        raise ValueError("Qualification failed")
    previous = None
    if args.command == "resume":
        old = json.loads(receipt_path.read_text())
        module.terminal_owner(old["function_call_id"])
        previous = old["function_call_id"]
        args.run_id = old["run_id"]
        save(args.out / ("previous-" + previous + ".receipt.json"), old)
    elif receipt_path.exists():
        raise ValueError("Existing launch receipt; collect or explicitly reconcile")
    import re
    if not isinstance(args.run_id, str) or not re.fullmatch(r"r7-source-[a-z0-9-]{3,80}", args.run_id):
        raise ValueError("Invalid immutable run ID")
    receipt = {"state": "preparing", "run_id": args.run_id, "requests_sha256": plan["requests_sha256"],
        "qualification_run": report["run_id"], "count": plan["count"], "detached": True,
        "created_at_unix": time.time(), "source_sha256": sha(ROOT / "modal/train/r7_source_review.py")}
    save(receipt_path, receipt)
    if previous is None:
        with module.teacher.artifacts.batch_upload() as batch:
            batch.put_file(args.out / "requests.jsonl", module.PREFIX + args.run_id + "/requests.jsonl")
    try:
        with modal.enable_output(), module.app.run(detach=True):
            preflight = module.preflight.remote()
            if (preflight["controls"] != len(controls()) or preflight["review_version"] != REVIEW_VERSION
                    or preflight["gpu_used"] is not False):
                raise ValueError("Worker import preflight failed")
            receipt["preflight"] = preflight
            receipt.update(state="submitting", app_id=module.app.app_id)
            save(receipt_path, receipt)
            call = module.review.spawn(args.run_id, plan["requests_sha256"], report["run_id"], previous)
            receipt.update(state="submitted", function_call_id=call.object_id, app_id=module.app.app_id)
            save(receipt_path, receipt)
            print(json.dumps(receipt))
    except Exception as error:
        receipt["submission_error_type"] = type(error).__name__
        if receipt["state"] == "preparing":
            receipt["state"] = "pre_submission_failure"
        save(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
