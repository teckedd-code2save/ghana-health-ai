"""Validate V6 paired results and actual conversation history without scoring prose."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_language_checkpoint import repetition_count, summarize


def conversation_report(rows):
    grouped = {}
    for row in rows:
        if row["turn"] not in (0, 1) or row["turn"] in grouped.setdefault(row["id"], {}):
            raise ValueError("Duplicate or unknown conversation turn")
        grouped[row["id"]][row["turn"]] = row
    if len(grouped) != 6 or any(set(v) != {0, 1} for v in grouped.values()):
        raise ValueError("Incomplete six-conversation check")
    for turns in grouped.values():
        first, second = turns[0], turns[1]
        expected = [*first["messages"], {"role": "assistant", "content": first["prediction"]}]
        if second["messages"][:-1] != expected or second["messages"][-1]["role"] != "user":
            raise ValueError("Follow-up did not retain the model's own previous answer")
    return {"count": len(grouped), "turns": len(rows), "pairs": grouped,
        "limit_reached": [f"{r['id']}:{r['turn']}" for r in rows if r["limit_reached"]],
        "repetition_flag": [f"{r['id']}:{r['turn']}" for r in rows if repetition_count(r["prediction"]) >= 4],
        "empty": [f"{r['id']}:{r['turn']}" for r in rows if not r["prediction"].strip()]}


def report(folder):
    hashes = {}

    def read(name):
        data = (folder / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    result, manifest, status = read("summary.json"), read("manifest.json"), read("status.json")
    if (status.get("stage") != "completed" or status["run_id"] != result["run_id"]
            or manifest["experiment"] != "balanced_afrique_response_v6"):
        raise ValueError("Expected a completed, matching V6 run")
    provenance = {"type": "provenance", "comparison_scope": "Paired frozen base and adapter",
        "run_id": result["run_id"], "base_model": manifest["base_model"], "base_revision": manifest["base_revision"],
        "adapter_sha256": result["adapter_sha256"], "dataset_human_validated": False, "steps": result["steps"]}
    output = {"provenance": provenance, "paired": {}, "conversations": {}}
    for suite, filenames, expected in (("development", ("base.json", "adapter.json"), 30),
                                        ("source_validation", ("base-source-heldout.json", "source-heldout.json"), 48)):
        events = [provenance]
        for variant, filename in zip(("base", "adapter"), filenames, strict=True):
            rows = read(filename)
            if len(rows) != expected:
                raise ValueError("Incomplete " + suite)
            events.extend({**r, "variant": variant, "type": "prediction"} for r in rows)
        output["paired"][suite] = summarize(events)
    for variant in ("base", "adapter"):
        output["conversations"][variant] = conversation_report(read(variant + "-conversations.json"))
    base, adapter = (output["conversations"][v]["pairs"] for v in ("base", "adapter"))
    if base.keys() != adapter.keys():
        raise ValueError("Different conversation cases")
    for key in base:
        if base[key][0]["messages"] != adapter[key][0]["messages"] or base[key][1]["messages"][-1] != adapter[key][1]["messages"][-1]:
            raise ValueError("Conversation user inputs differ")
    output["artifact_hashes"] = hashes
    output["limitations"] = [
        "Untrained CPT base is not a capable-chat baseline; termination gain alone is not product success.",
        "The 30 development cases were reused during prior work and checkpoint probes; not untouched test accuracy.",
        "Six project-authored conversation checks are diagnostics, not the complete understanding benchmark.",
        "Follow-up inputs include each model's own previous reply, so second-turn full prompts deliberately differ.",
        "The health conversation says Menim (I know); do not score it as Minnim (I do not know). No age or diagnosis is supplied.",
        "Source-validation answers and rubric text are retained for inspection but are not included in model prompts.",
        "No automatic semantic, naturalness, medical-safety or executed-tool success claim.",
    ]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    output = report(args.folder)
    path = args.folder / "paired-diagnostic.json"
    with path.open("x") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"path": str(path), "paired": {k: v["variants"] for k, v in output["paired"].items()},
        "conversations": {k: {field: value for field, value in v.items() if field != "pairs"} for k, v in output["conversations"].items()}}, indent=2))


if __name__ == "__main__":
    main()
