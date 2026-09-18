"""Measure prepared response supervision, separately from JSON and translations."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

RESPONSE_TASKS = {
    "general_conversation", "general_conversation_replay", "health_response",
    "grounded_question_answer", "native_source_context", "source_language_followup",
}


def coverage(rows, token_report):
    counts, multi = Counter(), Counter()
    for row in rows:
        if not row["messages"] or row["messages"][-1]["role"] != "assistant":
            raise ValueError("Expected an assistant training target")
        key = "train:" + row["task"] + ":" + row["language"]
        counts[key] += 1
        if sum(message["role"] == "assistant" for message in row["messages"]) > 1:
            multi[key] += 1
    recorded = {key: value for key, value in token_report["counts"].items() if key.startswith("train:")}
    tokens = {key: value for key, value in token_report["supervised_tokens"].items() if key.startswith("train:")}
    if (counts != recorded or counts.keys() != tokens.keys() or token_report.get("rejected")
            or token_report.get("truncated_targets", 0)):
        raise ValueError("Prepared row counts do not match untruncated tokenization evidence")
    if any(type(value) is not int or value <= 0 for value in tokens.values()):
        raise ValueError("Invalid target token count")
    total = sum(tokens.values())
    twi_response_keys = [key for key in counts if key.split(":")[1] in RESPONSE_TASKS and key.endswith(":tw")]
    response_tokens = sum(tokens[key] for key in twi_response_keys)
    health_tokens = tokens.get("train:health_response:tw", 0)
    twi_multi = sum(value for key, value in multi.items() if key.endswith(":tw"))
    return {
        "rows": len(rows), "prepared_target_tokens": total,
        "twi_response_task_rows": sum(counts[key] for key in twi_response_keys),
        "twi_response_task_tokens": response_tokens,
        "twi_response_fraction_of_all_target_tokens": response_tokens / total if total else 0,
        "health_fraction_of_twi_response_tokens": health_tokens / response_tokens if response_tokens else None,
        "twi_final_target_multiturn_rows": twi_multi,
        "structural_gaps": ["No multi-turn training rows with a Twi-labelled final target"] if not twi_multi else [],
        "by_task_and_language": {key: {"rows": counts[key], "target_tokens": tokens[key],
            "multiturn_rows": multi[key]} for key in sorted(counts)},
        "limitations": [
            "Language and task categories are declared labels, not automatic language or semantic verification.",
            "Counts describe the prepared pool, not actual sampled exposure in a partial-epoch run.",
            "Response tasks include short factual answers; none of these counts certify fluent conversation.",
            "Existing corpus rows and targets are not modified. No automatic training or model acceptance.",
        ],
    }


def report(folder, preflight):
    manifest_raw = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    train_raw = (folder / "train.jsonl").read_bytes()
    preflight_raw = preflight.read_bytes()
    token_report = json.loads(preflight_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    train_sha = hashlib.sha256(train_raw).hexdigest()
    train_artifact = next(item for item in manifest["artifacts"] if item["file"] == "train.jsonl")
    if train_artifact["sha256"] != train_sha or token_report["manifest_sha256"] != manifest_sha:
        raise ValueError("Training corpus or tokenization identity mismatch")
    rows = [json.loads(line) for line in train_raw.splitlines() if line.strip()]
    if train_artifact["rows"] != len(rows):
        raise ValueError("Training corpus row count mismatch")
    return {"experiment": manifest["experiment"], "input_sha256": {
        "manifest": manifest_sha, "train": train_sha,
        "tokenization": hashlib.sha256(preflight_raw).hexdigest()}, **coverage(rows, token_report)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.folder, args.preflight)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
