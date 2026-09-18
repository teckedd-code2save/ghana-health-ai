"""Create a controlled tool-availability ablation without synthesizing Twi replies."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible(row):
    return (not row.get("tools") and row["task"] != "auxiliary_translation"
            and not any(char.isdigit() for message in row["messages"] if message["role"] == "user" for char in message["content"]))


def main():
    source = ROOT / "tmp/afrique-response-v3/corpus"
    destination = ROOT / "tmp/afrique-tool-context-v4/corpus"
    core = load_module("response_core", ROOT / "modal/train/response_adaptation_core.py")
    runtime = load_module("response_runtime", ROOT / "modal/response_runtime.py")
    manifest, splits = core.load_inputs(source, "afrique_response_sft_v3")
    artifacts, changes, counts = [], [], Counter()
    destination.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        selected = set()
        for language, maximum in (("tw", 400), ("en", 600)):
            pool = sorted((row for row in rows if row["language"] == language and eligible(row)),
                          key=lambda row: hashlib.sha256(("tool-context-v4:" + row["id"]).encode()).hexdigest())
            selected.update(row["id"] for row in pool[:maximum if split == "train" else 40])
        result = []
        for row in rows:
            if row["id"] in selected:
                row = {**row, "tools": runtime.TOOLS, "context_augmentation": {
                    "kind": "available_calculator_with_direct_source_answer", "original_messages_unchanged": True,
                    "reason": "No numeric arguments are supplied by the user; direct-answer supervision with an available tool"}}
                changes.append({"id": row["id"], "split": split, "language": row["language"]})
                counts[split + ":no_tool_with_schema:" + row["language"]] += 1
            result.append(row)
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result).encode()
        filename = split + ".jsonl"
        (destination / filename).write_bytes(data)
        artifacts.append({"file": filename, "rows": len(result), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {**manifest, "experiment": "afrique_tool_context_repair_v4", "artifacts": artifacts,
        "parent_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "context_augmentation_counts": dict(counts), "new_source_rows": 0,
        "limitations": [*manifest["limitations"], "Tool availability is augmented; no generated or rewritten source questions/answers.",
            "This addresses false tool calls only, not missing broad Twi dialogue or clinical understanding.",
            "Prepared ablation, not a submitted training run or accepted model."]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "changed-contexts.json").write_text(json.dumps(changes, indent=2) + "\n")
    print(json.dumps({"folder": str(destination), "new_source_rows": 0, "changed_contexts": counts}, indent=2))


if __name__ == "__main__":
    main()
