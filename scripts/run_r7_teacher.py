#!/usr/bin/env python3
"""Detached, receipt-first R7 teacher qualification and explicit result collection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("r7_teacher_app", ROOT / "modal/train/r7_teacher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "launch", "collect"])
    parser.add_argument("--profile", choices=["gemma31", "qwen35"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--kind", choices=["prepare", "qualify"], default="qualify",
                        help="Which receipt to collect")
    args = parser.parse_args()
    import modal
    module = load_module()
    args.out.mkdir(parents=True, exist_ok=True)
    kind = args.kind if args.command == "collect" else "prepare" if args.command == "prepare" else "qualify"
    receipt = args.out / (args.profile + "-" + kind + ".receipt.json")
    if args.command == "collect":
        value = json.loads(receipt.read_text())
        try:
            result = modal.FunctionCall.from_id(value["function_call_id"]).get(timeout=0)
        except TimeoutError:
            if kind == "qualify":
                volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
                remote = "/understanding-adaptation-r7/teacher/" + value["run_id"] + "/progress.json"
                result = json.loads(b"".join(volume.read_file(remote)))
                if result["run_id"] != value["run_id"] or result["profile"] != args.profile:
                    raise ValueError("Checkpoint identity differs")
                target = args.out / (args.profile + "-qualify-checkpoint.json")
                module.save(target, result)
                print(json.dumps({"state": "call_pending", "checkpoint_status": result["status"],
                                  "completed": len(result["results"]), "checkpoint": str(target)}))
                return
            print(json.dumps({"state": "pending", "function_call_id": value["function_call_id"]}))
            return
        target = args.out / (args.profile + "-" + kind + "-results.json")
        module.save(target, result)
        print(json.dumps({"results": str(target),
                          "status": result.get("status"), "count": len(result.get("results", []))}))
        return
    if receipt.exists():
        raise ValueError("Existing receipt; collect/reconcile, do not launch again")
    requests = json.loads(args.requests.read_text()) if args.requests else None
    if args.command == "launch":
        module.validate(args.run_id, args.profile, requests)
    evidence = {"state": "launching", "profile": args.profile, "run_id": args.run_id,
                "created_at_unix": time.time(), "source_sha256": hashlib.sha256(
                    (ROOT / "modal/train/r7_teacher.py").read_bytes()).hexdigest(),
                "requests_sha256": module.digest(requests), "detached": True}
    module.save(receipt, evidence)
    with modal.enable_output():
        with module.app.run(detach=True):
            call = (module.prepare.spawn(args.profile) if args.command == "prepare" else
                    module.qualify.spawn(args.run_id, args.profile, requests))
            evidence.update(state="submitted", function_call_id=call.object_id, app_id=module.app.app_id)
            module.save(receipt, evidence)
            print(json.dumps(evidence))


if __name__ == "__main__":
    main()
