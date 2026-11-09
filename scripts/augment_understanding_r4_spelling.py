"""Reversible user-only ɛ -> 3 augmentation; no new meanings or answers."""
import argparse
import copy
import json
from pathlib import Path
from prepare_understanding_r3 import data, read_rows, verified_directory


def augment(record, source_ref):
    row = copy.deepcopy(record)
    edits = []
    for index, message in enumerate(row["messages"]):
        if message["role"] != "user":
            continue
        original = message["content"]
        positions = [i for i, c in enumerate(original) if c == "ɛ"]
        if positions:
            message["content"] = original.replace("ɛ", "3")
            restored = list(message["content"])
            for position in positions:
                restored[position] = "ɛ"
            data.require("".join(restored) == original, "Spelling transform is not reversible")
            edits.append({"message_index": index, "character_positions": positions,
                          "original_sha256": data.digest(original), "derived_sha256": data.digest(message["content"])})
    if not edits:
        return None
    data.require(all(a == b for a, b in zip(row["messages"], record["messages"], strict=True) if a["role"] != "user"),
                 "Augmentation modified assistant targets")
    row["id"] += ":ascii-input-v1"
    row["origin"] = "agent_translated_source_with_synthetic_user_spelling"
    row["original_ref"] = {**row["original_ref"], **source_ref, "parent_record_id": record["id"],
                           "parent_record_sha256": data.digest(record)}
    row["augmentation"] = {"method": "lowercase_epsilon_to_3_user_only_v1", "edits": edits,
        "assistant_targets_changed": False, "meaning_review_inherited": True,
        "native_texting_naturalness_reviewed": False, "independent_source": False}
    row["review"] = {**row["review"], "augmentation_note": "Parent meaning review inherited; this generated spelling is not a new native review."}
    return row


def build(source, out):
    data.require(not out.exists(), "Use a new immutable output")
    manifest = verified_directory(source, "READY.json")
    data.require(manifest["admission_status"] == "admitted" and manifest["private_research_only"] is True
                 and manifest["production_eligible"] is False, "Admitted research source required")
    rows, unchanged = [], []
    for record in read_rows(source / "conversations.jsonl"):
        data.require(record["split"] == "train" and record["training_eligible"] is True, "Unadmitted parent")
        derived = augment(record, {"parent_admission": str(source.resolve()),
                                  "parent_admission_sha256": data.sha(source / "manifest.json")})
        (rows if derived else unchanged).append(derived or {"id": record["id"], "reason": "no_lowercase_epsilon_in_user_turns"})
    out.mkdir(parents=True)
    data.write_rows(out / "conversations.jsonl", rows)
    data.write_rows(out / "unchanged.jsonl", unchanged)
    data.save(out / "report.json", {"parent_records": len(rows) + len(unchanged), "augmented_records": len(rows),
        "unchanged_records": len(unchanged), "new_independent_sources": 0, "new_assistant_answers": 0,
        "scope": "Synthetic spelling robustness; no claim of natural code-switching or new conversations."})
    data.save(out / "manifest.json", {"dataset_id": "r4-user-spelling-augmentation", "dataset_version": "v1",
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "parent_manifest_sha256": data.sha(source / "manifest.json"), "code_sha256": data.sha(__file__),
        "artifacts": {p.name: data.sha(p) for p in out.iterdir() if p.is_file()}})
    data.save(out / "READY.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return data.read(out / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out), indent=2))
