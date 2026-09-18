"""Durable receipts for a bounded, non-proprietary foundation comparison."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/stronger-response-comparison"
APP = "ghana-stronger-response-comparison"


def validate_result(report, receipt):
    if any(report.get(k) != receipt[k] for k in ("run_id", "model", "revision")):
        raise ValueError("Result identity differs from the submitted experiment")
    digest = hashlib.sha256(json.dumps(report["inputs"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if report["input_sha256"] != digest or (receipt.get("input_sha256") and receipt["input_sha256"] != digest):
        raise ValueError("Returned model inputs changed")
    ids = {r["id"] for r in report["inputs"]}
    if len(ids) != receipt["rows"] or len(ids) != len(report["inputs"]):
        raise ValueError("Missing or duplicate input identity")
    groups = (False, True) if len(receipt["modes"]) == 2 else (None,)
    expected = {(key, group) for key in ids for group in groups}
    actual = [(r["id"], r.get("thinking") if len(groups) == 2 else None) for r in report["results"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate comparison outputs")


def main():
    global OUT, APP
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "prepare-status", "submit", "status"])
    parser.add_argument("--wait-seconds", type=int, default=5)
    parser.add_argument("--attempt", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--model", choices=["qwen122", "qwen38", "oss120", "minicpm_twi"], default="qwen122")
    args = parser.parse_args()
    revision = "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
    gpu = "H100:2"
    if args.model == "qwen38":
        OUT = ROOT / "tmp/current-response-comparison"
        APP = "ghana-current-response-comparison"
        revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        gpu = "H100"
    elif args.model == "oss120":
        OUT = ROOT / "tmp/oss-response-comparison"
        APP = "ghana-oss-response-comparison"
        revision = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
        gpu = "H100"
    elif args.model == "minicpm_twi":
        OUT = ROOT / "tmp/native-fluency-comparison"
        APP = "ghana-native-fluency-comparison"
        revision = "d807ca1a3323972afafabff8f9affe2639e37b5c"
        gpu = "T4"
    if not 1 <= args.wait_seconds <= 1800:
        parser.error("wait-seconds must be between 1 and 1800")
    OUT.mkdir(parents=True, exist_ok=True)
    preparation = args.action.startswith("prepare")
    if preparation and args.attempt != 1:
        parser.error("Preparation has no retry override")
    suffix = "" if args.attempt == 1 else f".attempt-{args.attempt}"
    path = OUT / ("preparation.receipt.json" if preparation else "run" + suffix + ".receipt.json")
    if args.action in ("prepare", "submit"):
        if path.exists():
            raise ValueError("Receipt already exists; inspect status instead of duplicating")
        payload = []
        receipt = {"state": "submitting", "app": APP}
        if not preparation:
            if args.attempt > 1:
                prior_suffix = "" if args.attempt == 2 else f".attempt-{args.attempt - 1}"
                prior = json.loads((OUT / ("run" + prior_suffix + ".receipt.json")).read_text())
                if prior["state"] != "failed":
                    raise ValueError("Explicit retry requires a failed previous attempt")
            ready = json.loads((OUT / "preparation.results.json").read_text())
            if ready["revision"] != revision:
                raise ValueError("Unexpected model preparation")
            if args.model == "minicpm_twi":
                preflight = modal.Function.from_name(APP, "preflight").remote()
                if preflight["revision"] != revision or not preflight["finite_forward"]:
                    raise ValueError("Native-language CPU forward preflight failed")
                (OUT / ("preflight" + suffix + ".json")).write_text(json.dumps(preflight, indent=2) + "\n")
            rows = json.loads((ROOT / "tmp/general-foundation-comparison/cases-81a7aad880d9.json").read_text())
            if len(rows) != 30:
                raise ValueError("Expected the saved 30-case comparison")
            inputs = [{k: row[k] for k in ("id", "messages", "tools") if k in row} for row in rows]
            excluded = []
            if args.model == "minicpm_twi":
                excluded = [row["id"] for row in inputs if row.get("tools")]
                inputs = [row for row in inputs if not row.get("tools")]
            for row in inputs:
                row["messages"][0]["content"] += " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
            run_id = args.model + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            payload = [run_id, inputs]
            receipt.update(run_id=run_id, model=ready["model"], revision=ready["revision"], rows=len(inputs), attempt=args.attempt,
                           gpu=gpu, timeout_seconds=900 if args.model == "minicpm_twi" else 1800,
                           excluded_tool_case_ids=excluded,
                           modes=(["publisher non-thinking sampling"] if args.model == "minicpm_twi" else
                                  ["medium reasoning"] if args.model == "oss120" else ["publisher non-thinking", "publisher thinking"]),
                           reasoning_effort="medium" if args.model in ("qwen38", "oss120") else None)
            receipt["input_sha256"] = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with path.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        call = modal.Function.from_name(APP, "prepare" if preparation else "compare").spawn(*payload)
        receipt.update(state="submitted", call_id=call.object_id)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    else:
        receipt = json.loads(path.read_text())
        if not receipt.get("call_id"):
            raise ValueError("Submission uncertain; inspect Modal before retrying")
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=args.wait_seconds)
        except TimeoutError:
            print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__)
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        output = OUT / ("preparation.results.json" if preparation else "results" + suffix + ".json")
        if not preparation:
            validate_result(result, receipt)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        receipt.update(state="completed", result_file=output.name)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"state": "completed", "path": str(output), "elapsed_seconds": result.get("elapsed_seconds", result.get("seconds"))}))


if __name__ == "__main__":
    main()
