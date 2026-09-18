"""Submit once to deployed Modal compute; local disconnect does not cancel work."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "response-v2": ("response-adaptation-v2", "ghana-response-adaptation-v2", "response_v2_"),
    "afrique-v3": ("afrique-response-v3", "ghana-afrique-response-v3", "afrique_v3_"),
    "semantic-v4": ("semantic-response-v4", "ghana-semantic-response-v4", "response_v4_"),
    "native-v5": ("native-understanding-v5", "ghana-native-understanding-v5", "response_v5_"),
    "balanced-v6": ("balanced-afrique-v6", "ghana-balanced-afrique-v6", "afrique_v6_"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "submit", "status"])
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="response-v2")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    directory, app, prefix = EXPERIMENTS[args.experiment]
    folder = ROOT / "tmp" / directory
    args.receipt = args.receipt or folder / "run-receipt.json"
    if args.action == "preflight":
        result = modal.Function.from_name(app, "preflight").remote()
        (folder / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rejected"}, indent=2))
    elif args.action == "submit":
        preflight = json.loads((folder / "preflight.json").read_text())
        run_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with args.receipt.open("x") as handle:
            json.dump({"run_id": run_id, "state": "submitting", "warning": "If no call ID is saved, inspect remote state before resubmitting."}, handle)
        call = modal.Function.from_name(app, "train").spawn(run_id, preflight["manifest_sha256"])
        receipt = {"run_id": run_id, "call_id": call.object_id, "app": app, "state": "submitted"}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
    else:
        receipt = json.loads(args.receipt.read_text())
        if receipt.get("app") != app or not receipt.get("run_id", "").startswith(prefix):
            raise ValueError("Receipt belongs to a different experiment")
        result_path = folder / (receipt["run_id"] + ".result.json")
        if receipt["state"] == "completed":
            data = result_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != receipt["result_sha256"]:
                raise ValueError("Saved completion result changed")
            print(data.decode())
            return
        if not receipt.get("call_id"):
            raise ValueError("Submission outcome unknown; inspect Modal before retrying")
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=5)
        except TimeoutError:
            print(json.dumps({**receipt, "state": "pending", "meaning": "Still running or queued, not failed"}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__, checked_at=datetime.now(timezone.utc).isoformat())
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        if result.get("run_id") != receipt["run_id"]:
            raise ValueError("Remote result belongs to another run")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        receipt.update(state="completed", result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                       checked_at=datetime.now(timezone.utc).isoformat())
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
