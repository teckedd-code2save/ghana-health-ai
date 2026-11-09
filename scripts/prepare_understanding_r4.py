"""Build a sealed R4 continuation dataset from R3 plus reviewed derivatives."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import prepare_understanding_r3 as previous
from prepare_understanding_run import HoldoutIndex
from check_afrique_training_contract import encode_complete, load_tokenizer

data = previous.data
R3_PACK_SHA = "2ab1f0c6f68572bf966c9d1cc7e604cb72bc570fe37f23b3c0c770f4bcda92f0"


def prepare(r3_pack, r2_pack, admissions, out, version):
    from run_understanding_r4 import INITIALIZATION, VERSION, verify_pack
    data.require(not out.exists() and data.valid_id(version), "Use a fresh path and explicit dataset version")
    data.require(data.sha(r3_pack / "manifest.json") == R3_PACK_SHA, "Preserved R3 reference changed")
    data.require(data.sha(r2_pack / "manifest.json") == previous.R2_SHA, "R2 replay pool changed")
    previous.verified_directory(r3_pack)
    previous.verified_directory(r2_pack)
    tokenizer = load_tokenizer(r3_pack / "tokenizer")
    held = previous.read_rows(r3_pack / "held-content.jsonl")
    holdouts = HoldoutIndex([(r["id"], r["text"]) for r in held])
    rows, witnesses, records, seen, duplicates, exclusions = [], [], {}, {}, [], []
    old_records = {}
    for ref in data.read(r3_pack / "source-manifests.json"):
        source_manifest = data.read(r3_pack / ref["manifest_path"])
        for r in previous.read_rows(r3_pack / source_manifest["records_path"]):
            old_records[(source_manifest["catalog_dataset_id"], r["id"])] = r

    def add(row, witness, record):
        key = data.digest({k: row[k] for k in ("objective", "input_ids", "labels")})
        if key in seen:
            duplicates.append({"id": row["id"], "retained_id": seen[key]})
            return
        seen[key] = row["id"]
        ref = previous.dataset_reference(row["collection"], version, record)
        records[(row["collection"], record["id"])] = record
        rows.append({**row, "dataset_ref": ref})
        witnesses.append({**witness, "dataset_ref": ref})

    r3_rows, r3_witnesses = (previous.read_rows(r3_pack / n) for n in ("rows.jsonl", "source-witnesses.jsonl"))
    for row, witness in zip(r3_rows, r3_witnesses, strict=True):
        if row["collection"] == "responses":
            add(row, witness, old_records[("responses", row["dataset_ref"]["record_id"])])

    added_sources, manifests = [], []
    for folder in admissions:
        manifest = previous.verified_directory(folder, "READY.json")
        data.require(manifest["admission_status"] == "admitted" and manifest["private_research_only"] is True
                     and manifest["production_eligible"] is False, "Research admission required")
        manifests.append({"path": str(folder.resolve()), "manifest_sha256": data.sha(folder / "manifest.json")})
        for record in previous.read_rows(folder / "conversations.jsonl"):
            data.require(record["training_eligible"] is True and record["split"] == "train"
                         and record["review"]["status"] == "agent_checked" and not record["review"]["findings"],
                         "Unreviewed derivative")
            added_sources.append(record)
    data.require(added_sources, "No new admitted conversation data; do not rerun R3 unchanged")
    replay_rows, replay_witnesses = (previous.read_rows(r2_pack / n) for n in ("rows.jsonl", "source-witnesses.jsonl"))
    blocked = previous.protected_groups(added_sources + replay_witnesses, holdouts)
    for record in added_sources:
        if record["id"] in blocked:
            exclusions.append({"id": record["id"], "reason": "protected_content", **blocked[record["id"]]})
            continue
        for turn, messages in enumerate(previous.prefixes(record)):
            identity = record["id"] + ":assistant-" + str(turn)
            try:
                encoded = encode_complete(tokenizer, {"id": identity, "messages": messages}, 4096)
            except ValueError as error:
                if "overlength_no_truncation" not in str(error):
                    raise
                exclusions.append({"id": identity, "reason": "overlength_no_truncation"})
                continue
            row = {"id": identity, "source_id": record["source_id"], "collection": "responses", "split": "train",
                "objective": "final_assistant", **encoded, "sequence_tokens": len(encoded["input_ids"]),
                "target_tokens": sum(t != -100 for t in encoded["labels"])}
            witness = {"id": identity, "source_id": record["source_id"], "collection": "responses", "messages": messages,
                "metadata": {"origin": record["origin"], "group_id": record["group_id"],
                    "full_history_id": record["id"], "assistant_index": turn, "full_history_sha256": data.digest(record["messages"])},
                "original_ref": record["original_ref"]}
            add(row, witness, record)
    new_views = len(rows) - sum(r["collection"] == "responses" for r in r3_rows)
    data.require(new_views > 0, "All new conversation views excluded; no unchanged training run")
    for row, witness in zip(replay_rows, replay_witnesses, strict=True):
        if row["collection"] == "responses":
            continue
        if witness["id"] in blocked:
            exclusions.append({"id": witness["id"], "reason": "protected_content", **blocked[witness["id"]]})
            continue
        record = {**witness, "split": "train", "training_eligible": True,
            "origin": witness["metadata"].get("origin", "inherited_r2_source_metadata"),
            "admission_provenance": {"mode": "unchanged_r2_admitted_view", "manifest_sha256": previous.R2_SHA,
                                     "witness_sha256": data.digest(witness)}}
        add(row, witness, record)
    stage = previous.replay_schedule(rows)
    selected = sorted(set(stage["schedule"]))
    selected_set = set(selected)
    selection_ledger = [{"id": r["id"], "source_id": r["source_id"], "collection": r["collection"],
                         "status": "selected" if i in selected_set else "available_not_selected",
                         "reason": "response_and_token_balanced_replay" if i in selected_set else "outside_replay_token_budget"}
                        for i, r in enumerate(rows)]
    remap = {old: new for new, old in enumerate(selected)}
    rows, witnesses = [rows[i] for i in selected], [witnesses[i] for i in selected]
    stage["schedule"] = [remap[i] for i in stage["schedule"]]
    out.mkdir(parents=True)
    snapshots = previous.register_datasets(out, rows, witnesses, records, version)
    # Dataset cards are generated afresh for this release; prior cards remain unchanged.
    for card in (out / "datasets").glob("*/README.md"):
        card.write_text(card.read_text().replace("R3 response adaptation from R2 meaning", "R4 response continuation from the final R3 adapter"))
    for name, values in (("rows.jsonl", rows), ("source-witnesses.jsonl", witnesses),
                         ("excluded.jsonl", exclusions), ("duplicate-prefixes.jsonl", duplicates), ("held-content.jsonl", held),
                         ("selection-ledger.jsonl", selection_ledger)):
        data.write_rows(out / name, values)
    data.save(out / "stages.json", [stage])
    data.save(out / "admission-manifests.json", manifests)
    data.save(out / "evaluation-manifest.json", data.read(r3_pack / "evaluation-manifest.json"))
    data.save(out / "overlap-audit.json", {"overlap_free": True, "protected_units": len(held),
        "method": "Inherited source grouping; normalized exact and eight-word Jaccard >= .85 checks",
        "limits": "Lexical protection does not prove unknown pretraining independence or semantic equivalence."})
    data.save(out / "selection-report.json", {"new_response_views": new_views,
        "inherited_response_views": sum(r["collection"] == "responses" for r in r3_rows),
        "new_derivative_records": len(added_sources), "new_unique_sources": len({r["source_id"] for r in added_sources}),
        "new_origin_counts": dict(Counter(r["origin"] for r in added_sources)),
        "selected_views": dict(Counter(r["collection"] for r in rows)),
        "excluded": len(exclusions), "duplicate_views": len(duplicates), "training_executed": False})
    shutil.copytree(r3_pack / "tokenizer", out / "tokenizer")
    manifest = {"version": VERSION, "model": previous.MODEL, "revision": previous.REVISION,
        "initialization": INITIALIZATION, "private_research_only": True, "production_eligible": False,
        "optimization": {"learning_rate": 1e-5, "grad_accum": 16, "warmup_fraction": .05,
            "checkpoint_every": 10, "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False},
        "optimization_rationale": "Continue the already response-adapted R3 conservatively at half its previous LR; save pass-one and final checkpoints for development selection.",
        "counts": dict(Counter(r["collection"] for r in rows)), "selected_rows": len(rows),
        "source_ids": len({r["source_id"] for r in rows}), "dataset_version": version,
        "multiturn_response_targets": sum(w["collection"] == "responses" and sum(m["role"] == "assistant" for m in w.get("messages", [])) > 1 for w in witnesses),
        "dataset_snapshots": snapshots, "r3_reference_manifest_sha256": R3_PACK_SHA,
        "r2_replay_manifest_sha256": previous.R2_SHA, "builder_sha256": data.sha(__file__),
        "stages": [{k: v for k, v in stage.items() if k != "schedule"}],
        "review_status": "Agent-checked research data; no new native-human or clinical certification",
        "artifacts": {p.relative_to(out).as_posix(): data.sha(p) for p in out.rglob("*") if p.is_file()}}
    data.save(out / "manifest.json", manifest)
    data.save(out / "PREPARED.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    verify_pack(out)
    return {k: manifest[k] for k in ("counts", "multiturn_response_targets", "stages")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r3-pack", "r2-pack", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission", action="append", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.r3_pack, args.r2_pack, args.admission, args.out, args.dataset_version), indent=2))
