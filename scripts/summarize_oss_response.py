"""Export final answers and mechanical diagnostics, never private reasoning."""
import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from summarize_language_checkpoint import repetition_count


def summarize(report):
    inputs = {r["id"]: r for r in report["inputs"]}
    rows = {r["id"]: r for r in report["results"]}
    if not inputs or len(inputs) != len(report["inputs"]) or len(rows) != len(report["results"]) or inputs.keys() != rows.keys():
        raise ValueError("Incomplete comparison or duplicate identity")
    digest = hashlib.sha256(json.dumps(report["inputs"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != report["input_sha256"]:
        raise ValueError("Input identity changed")
    review, tool_errors = [], {}
    for key, row in rows.items():
        parsed = row.get("parsed", {})
        if row["answer"] != parsed.get("content", ""):
            raise ValueError("Display answer differs from parsed final content")
        tools = {t["function"]["name"]: t["function"]["parameters"] for t in inputs[key].get("tools", [])}
        calls, errors = parsed.get("tool_calls", []), []
        for call in calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name")
            if name not in tools:
                errors.append("unavailable_function")
            elif call.get("type") != "function" or not Draft202012Validator(tools[name]).is_valid(function.get("arguments")):
                errors.append("invalid_tool_arguments_or_type")
        if calls and (row["finish_reason"] != "stop" or not row["raw_output"].rstrip().endswith("<|call|>")):
            errors.append("incomplete_tool_handoff")
        if errors:
            tool_errors[key] = errors
        review.append({"id": key, "messages": inputs[key]["messages"], "answer": row["answer"],
                       "tool_calls": calls, "finish_reason": row["finish_reason"], "output_tokens": row["output_tokens"]})
    return {**{k: report[k] for k in ("run_id", "model", "revision", "input_sha256", "quantization", "reasoning_effort", "decoding", "load_seconds", "elapsed_seconds")},
        "rows": len(rows), "length_limit_ids": [k for k, r in rows.items() if r["finish_reason"] == "length"],
        "parse_error_ids": [k for k, r in rows.items() if r.get("parse_error")],
        "empty_answer_and_no_tool_ids": [k for k, r in rows.items() if not r["answer"].strip() and not r.get("parsed", {}).get("tool_calls")],
        "repetition_flag_ids": [k for k, r in rows.items() if repetition_count(r["answer"]) >= 4],
        "tool_request_ids": [k for k, r in rows.items() if r.get("parsed", {}).get("tool_calls")],
        "invalid_tool_requests": tool_errors, "review": review,
        "trained_by_project": False, "tools_executed": False, "production_promoted": False,
        "semantic_accuracy": None, "clinical_accuracy": None,
        "limitations": ["Mechanical output checks are not semantic or clinical scores.",
            "The same development cases informed earlier model decisions; they are not a fresh test set.",
            "The ambiguous eye/sleepiness fixture requires native interpretation.",
            "One sampling profile is not a paired reasoning or decoding ablation.",
            "Batch elapsed time is not interactive per-request latency."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = summarize(json.loads(raw))
    report["raw_result_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.input.with_suffix(".diagnostic.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "review"}, indent=2))


if __name__ == "__main__":
    main()
