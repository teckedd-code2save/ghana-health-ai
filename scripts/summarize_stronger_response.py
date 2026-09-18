"""Inspect foundation outputs without turning mechanical checks into quality scores."""
import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from summarize_language_checkpoint import repetition_count


def tool_errors(row, schemas):
    calls = row["parsed"].get("tool_calls", [])
    available = {s["function"]["name"]: s["function"]["parameters"] for s in schemas or []}
    errors = []
    for call in calls:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name")
        if name not in available:
            errors.append("unavailable_function")
        elif call.get("type") != "function":
            errors.append("invalid_call_type")
        elif not Draft202012Validator(available[name]).is_valid(fn.get("arguments")):
            errors.append("invalid_arguments")
    if calls and (row["finish_reason"] != "stop" or
                  row["raw_output"].count("</tool_call>") != len(calls) or
                  not row["raw_output"].endswith("<|im_end|>")):
        errors.append("incomplete_handoff")
    return errors


def summarize(report):
    inputs = {row["id"]: row for row in report["inputs"]}
    if not inputs or len(inputs) != len(report["inputs"]):
        raise ValueError("Empty or duplicate input IDs")
    groups = {False: {}, True: {}}
    for row in report["results"]:
        key, mode = row["id"], row["thinking"]
        if type(mode) is not bool or key not in inputs or key in groups[mode]:
            raise ValueError("Unknown case, mode, or duplicate result")
        if row["answer"] != row["parsed"].get("content", ""):
            raise ValueError("Display answer differs from parsed content")
        groups[mode][key] = row
    if any(group.keys() != inputs.keys() for group in groups.values()):
        raise ValueError("Comparison is incomplete; retain partial artifacts separately")
    output = {key: report[key] for key in ("run_id", "model", "revision", "input_sha256",
        "load_seconds", "elapsed_seconds", "quantization", "tensor_parallel_size")}
    output.update(cases_per_mode=len(inputs), modes={}, pairs=[], trained_by_project=False,
        tools_executed=False, semantic_grade=None, clinical_grade=None, production_promoted=False,
        limitations=["Mechanical flags are not semantic or native-language scores.",
            "Thinking comparison also changes sampling and output-token budget.",
            "Batch times are not interactive per-request latency.",
            "Tool schemas were supplied, but no tools were executed.",
            "These development cases are not an independent clinical test.",
            "The isolated eye/sleepiness fixture requires native interpretation."])
    for mode, rows in groups.items():
        label = "thinking" if mode else "non_thinking"
        output["modes"][label] = {
            "length_limit_ids": [key for key, r in rows.items() if r["finish_reason"] == "length"],
            "no_answer_or_tool_ids": [key for key, r in rows.items() if not r["answer"].strip() and not r["parsed"].get("tool_calls")],
            "repetition_flag_ids": [key for key, r in rows.items() if repetition_count(r["answer"]) >= 4],
            "tool_request_ids": [key for key, r in rows.items() if r["parsed"].get("tool_calls")],
            "invalid_tool_requests": {key: errors for key, r in rows.items()
                if (errors := tool_errors(r, inputs[key].get("tools")))},
            "decoding": next(iter(rows.values()))["decoding"],
        }
    for key, source in inputs.items():
        # Only final answer and tool intent are exported for review, never reasoning.
        pair = {"id": key, "messages": source["messages"]}
        for mode, rows in groups.items():
            row = rows[key]
            pair["thinking" if mode else "non_thinking"] = {
                "answer": row["answer"], "tool_calls": row["parsed"].get("tool_calls", []),
                "finish_reason": row["finish_reason"], "output_tokens": row["output_tokens"],
            }
        output["pairs"].append(pair)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    result = summarize(json.loads(data))
    result["raw_result_sha256"] = hashlib.sha256(data).hexdigest()
    path = args.input.with_suffix(".diagnostic.json")
    with path.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
