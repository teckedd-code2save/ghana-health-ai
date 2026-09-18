"""Describe raw paired diagnostics without inventing semantic accuracy scores."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def repetition_count(text):
    words = re.findall(r"\w+", text.casefold())
    return max(Counter(tuple(words[i:i + 8]) for i in range(len(words) - 7)).values(), default=0)


def summarize(events):
    provenance = [event for event in events if event.get("type") == "provenance"]
    if len(provenance) != 1 or provenance[0].get("comparison_scope") != "Paired frozen base and adapter":
        raise ValueError("Expected one paired diagnostic provenance record")
    groups = {variant: {} for variant in ("base", "adapter")}
    for event in events:
        if event.get("type") != "prediction":
            continue
        variant, key = event["variant"], event["id"]
        if variant not in groups or key in groups[variant]:
            raise ValueError("Unknown variant or duplicate prediction")
        groups[variant][key] = event
    if not groups["base"] or groups["base"].keys() != groups["adapter"].keys():
        raise ValueError("Incomplete paired comparison")
    output = {"provenance": provenance[0], "cases_per_variant": len(groups["base"]),
              "status": "Diagnostic only; no automatic semantic or clinical acceptance",
              "limitations": ["Repeated eight-word spans are an inspectable flag, not a semantic score.",
                              "Length-limit stops and repetitive outputs are reported separately.",
                              "The isolated eye/sleepiness fixture has an ambiguous rubric and needs native review."],
              "variants": {}, "pairs": []}
    for variant, rows in groups.items():
        output["variants"][variant] = {
            "limit_reached_ids": [key for key, row in rows.items() if row["limit_reached"]],
            "repetition_flag_ids": [key for key, row in rows.items() if repetition_count(row["prediction"]) >= 4],
            "empty_ids": [key for key, row in rows.items() if not row["prediction"].strip()],
        }
    for key in groups["base"]:
        a, b = groups["base"][key], groups["adapter"][key]
        if a["messages"] != b["messages"] or a.get("tools") != b.get("tools"):
            raise ValueError("Prompt or tool mismatch")
        output["pairs"].append({"id": key, "messages": a["messages"],
            **{variant: {"prediction": groups[variant][key]["prediction"],
                         "limit_reached": groups[variant][key]["limit_reached"],
                         "max_repeated_eight_word_span": repetition_count(groups[variant][key]["prediction"])}
               for variant in groups}})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    result = summarize([json.loads(line) for line in data.splitlines() if line.strip()])
    result["input_sha256"] = hashlib.sha256(data).hexdigest()
    with args.output.open("x") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
