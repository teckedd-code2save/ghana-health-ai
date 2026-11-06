#!/usr/bin/env python3
"""Build a source-bound conversational R3 pack without modifying R2 or reviews."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

SOURCE = Path(os.environ.get("GHA_R2_SOURCE_ROOT",
    "/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai"))
sys.path.extend([str(SOURCE / "scripts"), str(SOURCE / "modal/train")])
import training_data_inputs as data
from check_afrique_training_contract import ASSET_HASHES, MODEL, REVISION, encode_complete, load_tokenizer
from prepare_understanding_run import HoldoutIndex, content, effective_tokens
from run_understanding_r3 import INITIALIZATION, digest as record_digest

R2_SHA = "1e632f4d7382d6b7db910973a83d2483d4e1f843659335951fe59f81bd974084"
RATIOS = {"english_retention": 1 / 3, "meaning": 1 / 6, "language": 1 / 6}
PURPOSES = {
    "responses": "Direct Twi responses, useful follow-ups and instruction following.",
    "meaning": "Retain faithful bilingual and structured meaning during response adaptation.",
    "language": "Retain Twi language patterns during response adaptation.",
    "english_retention": "Retain English conversation capability during Twi adaptation.",
}


def read_rows(path):
    return [r for _, r in data.rows(path)]


def verified_directory(path, seal="PREPARED.json"):
    manifest = data.read(path / "manifest.json")
    data.require(data.read(path / seal)["manifest_sha256"] == data.sha(path / "manifest.json"), "Source seal changed")
    for name, checksum in manifest["artifacts"].items():
        data.require(data.sha(data.safe_path(path, name)) == checksum, "Source artifact changed: " + name)
    return manifest


def fields(witness):
    values = [witness["text"]] if "text" in witness else []
    for key in ("messages", "english_messages"):
        for message in witness.get(key, []):
            if message["role"] != "system":
                values.append(message["content"])
                if message["content"].lstrip().startswith("{"):
                    try:
                        parsed = data.parse_json(message["content"])
                    except ValueError:
                        continue
                    if isinstance(parsed, dict):
                        values.extend(parsed[k] for k in ("source_text", "text", "utterance", "transcript", "reference_english")
                                      if isinstance(parsed.get(k), str))
    values.append(content(witness))
    return sorted({v for v in values if v.strip()})


def prefixes(conversation):
    messages = conversation["messages"]
    start = int(messages[0]["role"] == "system")
    data.require((len(messages) - start) % 2 == 0, "Incomplete conversation")
    for end in range(start + 2, len(messages) + 1, 2):
        yield messages[:end]


def dataset_reference(collection, version, record):
    data.require(collection in PURPOSES, "Unknown dataset class")
    data.require(record["split"] == "train" and record["training_eligible"] is True,
                 "Unadmitted dataset record")
    return {"dataset_id": "gha-" + collection, "dataset_version": version,
            "record_id": record["id"], "source_id": record["source_id"],
            "record_sha256": record_digest(record)}


def register_datasets(out, rows, witnesses, records, version):
    """Publish the exact admitted source snapshots from which views are derived."""
    references = []
    for collection in sorted({r["collection"] for r in rows}):
        selected = {}
        for row, witness in zip(rows, witnesses, strict=True):
            if row["collection"] != collection:
                continue
            ref = row["dataset_ref"]
            source = records[(collection, ref["record_id"])]
            data.require(ref == witness["dataset_ref"] == dataset_reference(collection, version, source),
                         "Training view differs from its dataset record")
            selected[source["id"]] = source
        folder = out / "datasets" / collection
        folder.mkdir(parents=True)
        ordered = [selected[key] for key in sorted(selected)]
        records_path = folder / "records.jsonl"
        data.write_rows(records_path, ordered)
        manifest = {"dataset_id": "gha-" + collection, "dataset_version": version,
            "catalog_dataset_id": collection, "purpose": PURPOSES[collection],
            "admission_status": "admitted", "private_research_only": True,
            "production_eligible": False, "record_count": len(ordered),
            "records_path": records_path.relative_to(out).as_posix(),
            "records_sha256": data.sha(records_path),
            "unique_sources": len({r["source_id"] for r in ordered}),
            "origin_counts": dict(Counter(r.get("origin", "inherited_source_metadata") for r in ordered)),
            "review_note": "Preserves upstream review states. Registration grants no new human or clinical certification.",
            "scope": "Selected admitted research snapshot, not the full source inventory or an executed training receipt"}
        data.save(folder / "manifest.json", manifest)
        card = ["# " + manifest["dataset_id"], "", "Version: " + version, "", PURPOSES[collection], "",
            "## Contents", "", f"{len(ordered)} admitted source records; {manifest['unique_sources']} unique source IDs.",
            "Records retain complete source messages or text, source references, and inherited review metadata.",
            "Assistant-turn prefixes are training views, not additional independent conversations.", "",
            "## Preparation and use", "", "This is a private research selection for R3 response adaptation from R2 meaning.",
            "Source bytes and reviews are retained. Original spelling is not silently normalized.",
            "Exact duplicate training views and protected evaluation overlap are screened before selection.",
            "The tokenizer is unchanged. Context is masked; only the final assistant answer is supervised in chat views.",
            "Language replay uses causal language-model supervision.", "",
            "## Provenance and limitations", "", "Each record carries original_ref and admission evidence. Upstream source terms remain applicable.",
            "Synthetic/translated origins are retained, including paid request references where present.",
            "Agent checking does not establish native-human or clinical certification.",
            "This snapshot is admitted for the bounded research run, not production deployment or public redistribution.",
            "Recorded inventory is not proof of consumption; the trainer's checkpoint-backed consumption.json reports actual use.", ""]
        (folder / "README.md").write_text("\n".join(card), encoding="utf-8")
        references.append({"dataset_id": manifest["dataset_id"], "dataset_version": version,
            "manifest_path": (folder / "manifest.json").relative_to(out).as_posix(),
            "manifest_sha256": data.sha(folder / "manifest.json")})
    data.save(out / "source-manifests.json", references)
    return references


def source_keys(record):
    metadata = record.get("metadata", record)
    keys = {"source:" + record["source_id"]}
    for key in ("group_id", "family_id", "tree_id", "original_group_id"):
        if metadata.get(key):
            keys.add(key + ":" + metadata[key])
    keys.update("source:" + value for value in metadata.get("source_ids", []))
    return keys


def protected_groups(records, holdouts):
    """One matching branch protects every connected source/group version."""
    parents, blocked, hits = {}, set(), {}
    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for record in records:
        keys = sorted(source_keys(record))
        for key in keys[1:]:
            parents[root(key)] = root(keys[0])
    for record in records:
        hit = next((match for text in fields(record) if (match := holdouts.match(text))), None)
        if hit:
            group = root(next(iter(source_keys(record))))
            blocked.add(group)
            hits.setdefault(group, hit)
    return {record["id"]: hits[root(next(iter(source_keys(record))))]
            for record in records if root(next(iter(source_keys(record)))) in blocked}


def replay_schedule(rows, epochs=2):
    primary = [i for i, r in enumerate(rows) if r["collection"] == "responses"]
    data.require(primary and epochs in (1, 2), "Missing response examples or invalid epochs")
    schedule, passes = [], []
    used_replay = set()
    primary_tokens = sum(effective_tokens(rows[i]) for i in primary)
    for epoch in range(epochs):
        order = primary.copy()
        for collection, ratio in RATIOS.items():
            pool = [i for i, r in enumerate(rows) if r["collection"] == collection and i not in used_replay]
            random.Random(142 + epoch).shuffle(pool)
            budget, used = math.floor(primary_tokens * ratio), 0
            for i in pool:
                size = effective_tokens(rows[i])
                if used + size <= budget:
                    order.append(i)
                    used_replay.add(i)
                    used += size
            data.require(used > 0, "No replay for " + collection)
        # A pass snapshot must fall at a complete optimizer update. Trim only
        # replay (never a primary example), retain every replay category, and
        # allow the removed rows to be selected on the next pass.
        remainder = len(order) % 16
        replay_counts = Counter(rows[i]["collection"] for i in order if i not in primary)
        removable = [i for i in reversed(order) if i not in primary]
        removed = []
        for i in removable:
            category = rows[i]["collection"]
            if len(removed) < remainder and replay_counts[category] > 1:
                removed.append(i)
                replay_counts[category] -= 1
        if len(removed) == remainder and epoch == 0:
            order = [i for i in order if i not in removed]
            used_replay.difference_update(removed)
        random.Random(42 + epoch).shuffle(order)
        schedule.extend(order)
        passes.append({"pass": epoch + 1, "end_sequence": len(schedule),
                       "sequence_count": len(order), "target_tokens": sum(effective_tokens(rows[i]) for i in order)})
    totals = Counter()
    for i in schedule:
        totals[rows[i]["collection"]] += effective_tokens(rows[i])
    snapshots = [passes[0]["end_sequence"] // 16] if epochs == 2 and passes[0]["end_sequence"] % 16 == 0 else []
    return {"name": "responses", "schedule": schedule, "primary_collections": ["responses"],
            "snapshot_after_updates": snapshots, "primary_passes": epochs, "passes": passes,
            "scheduled_sequences": len(schedule), "optimizer_updates": math.ceil(len(schedule) / 16),
            "sequence_tokens": sum(len(rows[i]["input_ids"]) for i in schedule),
            "effective_target_tokens": sum(totals.values()), "target_tokens_by_collection": dict(totals),
            "target_fractions": {k: v / sum(totals.values()) for k, v in totals.items()},
            "maximum_row_exposures": max(Counter(schedule).values()),
            "mixing": "Full response passes; finite no-replacement replay targeting 60/20/10/10 response/English/meaning/Twi tokens"}


def prepare(r2_pack, admissions, evaluation, out, dataset_version, run_selection=None):
    data.require(not out.exists(), "Use a new immutable pack directory")
    data.require(data.valid_id(dataset_version), "Explicit safe dataset version required")
    data.require(data.sha(r2_pack / "manifest.json") == R2_SHA, "R2 source pack changed")
    verified_directory(r2_pack)
    eval_manifest = verified_directory(evaluation, "FROZEN.json")
    held = read_rows(r2_pack / "protection/held-content.jsonl") + read_rows(evaluation / "held-content.jsonl")
    holdouts = HoldoutIndex([(r["id"], r["text"]) for r in held])
    old_rows, old_witnesses = read_rows(r2_pack / "rows.jsonl"), read_rows(r2_pack / "source-witnesses.jsonl")
    data.require(len(old_rows) == len(old_witnesses), "R2 witness alignment changed")
    tokenizer = load_tokenizer(r2_pack / "tokenizer")
    rows, witnesses, excluded, duplicate_views, seen = [], [], [], [], {}
    admission_manifests, source_rows, source_records = [], [], {}
    for folder in admissions:
        manifest = verified_directory(folder, "READY.json")
        data.require(manifest["private_research_only"] is True and manifest["production_eligible"] is False,
                     "Explicit research admission required")
        admission_manifests.append({"path": str(folder.resolve()), "sha256": data.sha(folder / "manifest.json")})
        data.require("conversations.jsonl" in manifest["artifacts"], "Unbound conversations")
        for record in read_rows(folder / "conversations.jsonl"):
            data.require(record["split"] == "train" and record["training_eligible"] is True,
                         "Pending or protected source cannot enter training")
            data.require(record["review"]["status"] == "agent_checked"
                         and all(not record["review"].get(k) for k in ("findings", "unresolved_findings", "issues")),
                         "Unresolved review")
            source_rows.append((folder, record))
    deferred, deferred_targets = {}, {}
    if run_selection is not None:
        selection = data.read(run_selection)
        data.require(any(data.sha(folder / "manifest.json") == selection["input_manifest_sha256"]
                         and data.sha(folder / "conversations.jsonl") == selection["input_conversations_sha256"]
                         for folder in admissions), "Run selection refers to another admission")
        source_lookup = {record["id"]: record for _, record in source_rows}
        for decision in selection["recommended_exclusions"]:
            record = source_lookup[decision["record_id"]]
            data.require(data.digest(record) == decision["record_sha256"], "Run-selection record changed")
            deferred[record["id"]] = decision
            deferred_targets[data.digest(record["messages"])] = record["id"]

    def add(row, witness, source_record):
        data.require(row["id"] == witness["id"] and row["source_id"] == witness["source_id"], "Mismatched witness")
        key = data.digest({"objective": row["objective"], "input_ids": row["input_ids"], "labels": row["labels"]})
        if key in seen:
            duplicate_views.append({"id": row["id"], "retained_id": seen[key], "source_id": row["source_id"]})
            return
        seen[key] = row["id"]
        source_key = (row["collection"], source_record["id"])
        data.require(source_key not in source_records or source_records[source_key] == source_record,
                     "Conflicting dataset record versions")
        source_records[source_key] = source_record
        ref = dataset_reference(row["collection"], dataset_version, source_record)
        rows.append({**row, "dataset_ref": ref})
        witnesses.append({**witness, "dataset_ref": ref})

    grouped_blocks = protected_groups([r for _, r in source_rows] + old_witnesses, holdouts)
    # Build all prefixes only after the full source history has passed protection.
    for folder, record in source_rows:
        if record["id"] in deferred:
            excluded.append({"id": record["id"], "reason": "run_objective_deferred",
                "category": deferred[record["id"]]["category"], "detail": deferred[record["id"]]["rationale"],
                "corpus_eligibility_unchanged": True})
            continue
        hit = grouped_blocks.get(record["id"])
        if hit:
            excluded.append({"id": record["id"], "reason": "protected_content", **hit})
            continue
        for turn, messages in enumerate(prefixes(record)):
            identity = record["id"] + ":assistant-" + str(turn)
            if data.digest(messages) in deferred_targets:
                excluded.append({"id": identity, "reason": "deferred_assistant_target",
                    "same_target_as": deferred_targets[data.digest(messages)],
                    "note": "Preserved in source history as context, but not independently supervised."})
                continue
            try:
                encoded = encode_complete(tokenizer, {"id": identity, "messages": messages}, 4096)
            except ValueError as error:
                if "overlength_no_truncation" not in str(error):
                    raise
                excluded.append({"id": identity, "reason": "overlength_no_truncation"})
                continue
            metadata = {k: v for k, v in record.items() if k not in {"messages", "english_messages"}}
            metadata.update(assistant_index=turn, full_history_id=record["id"], full_history_sha256=data.digest(record["messages"]))
            row = {"id": identity, "source_id": record["source_id"], "collection": "responses", "split": "train",
                   "objective": "final_assistant", **encoded, "sequence_tokens": len(encoded["input_ids"]),
                   "target_tokens": sum(t != -100 for t in encoded["labels"])}
            add(row, {"id": identity, "source_id": record["source_id"], "collection": "responses",
                      "messages": messages, "metadata": metadata,
                      "original_ref": {"admission": str(folder.resolve()), "admission_sha256": data.sha(folder / "manifest.json"),
                                       "row_sha256": data.digest(record), **record["original_ref"]}}, record)

    for row, witness in zip(old_rows, old_witnesses, strict=True):
        if row["collection"] == "responses" and witness["metadata"].get("task_kind") != "twi_prose":
            excluded.append({"id": row["id"], "reason": "language_exercise_not_conversation"})
            continue
        hit = grouped_blocks.get(witness["id"])
        if hit:
            excluded.append({"id": row["id"], "reason": "protected_content", **hit})
            continue
        record = {**witness, "split": "train", "training_eligible": True,
                  "origin": witness["metadata"].get("origin", "inherited_r2_source_metadata"),
                  "admission_provenance": {"mode": "unchanged_r2_admitted_view",
                      "manifest_sha256": R2_SHA, "witness_sha256": data.digest(witness)},
                  "review_note": "Inherited R2 admission; this wrapper adds no review certification."}
        add(row, witness, record)

    stage = replay_schedule(rows)
    # Export only rows actually consumed, with explicit old-to-new index mapping.
    consumed = sorted(set(stage["schedule"]))
    remap = {old: new for new, old in enumerate(consumed)}
    rows, witnesses = [rows[i] for i in consumed], [witnesses[i] for i in consumed]
    stage["schedule"] = [remap[i] for i in stage["schedule"]]
    multi = [w for w in witnesses if w["collection"] == "responses" and len(w.get("messages", [])) >= 4 + int(w.get("messages", [{}])[0].get("role") == "system")]
    data.require(multi, "No multi-turn Twi supervision selected")
    out.mkdir(parents=True)
    dataset_manifests = register_datasets(out, rows, witnesses, source_records, dataset_version)
    selected_records = Counter((r["collection"], r["dataset_ref"]["record_id"]) for r in rows)
    exclusions = {r["id"]: r for r in excluded}
    duplicates = {r["id"]: r for r in duplicate_views}
    selection_ledger = []
    for collection, record in [("responses", r) for _, r in source_rows] + [(w["collection"], w) for w in old_witnesses]:
        key = (collection, record["id"])
        detail = exclusions.get(record["id"], duplicates.get(record["id"]))
        status = "selected" if key in selected_records else "excluded" if record["id"] in exclusions else "duplicate_view" if record["id"] in duplicates else "eligible_not_sampled_replay"
        if collection == "responses" and key not in selected_records and detail is None:
            prefix_decisions = [r for r in excluded + duplicate_views if r["id"].startswith(record["id"] + ":assistant-")]
            status = "no_distinct_fitting_prefix_selected"
            detail = {"prefix_decisions": prefix_decisions}
        selection_ledger.append({"collection": collection, "record_id": record["id"], "source_id": record["source_id"],
            "status": status, "selected_training_views": selected_records[key], "detail": detail})
    data.write_rows(out / "selection-ledger.jsonl", selection_ledger)
    data.write_rows(out / "rows.jsonl", rows)
    data.write_rows(out / "source-witnesses.jsonl", witnesses)
    data.write_rows(out / "excluded.jsonl", excluded)
    data.write_rows(out / "duplicate-prefixes.jsonl", duplicate_views)
    data.save(out / "stages.json", [stage])
    data.save(out / "evaluation-manifest.json", {"sha256": data.sha(evaluation / "manifest.json"), "manifest": eval_manifest})
    data.save(out / "overlap-audit.json", {"overlap_free": True, "protected_units": len(held),
        "excluded": len([r for r in excluded if r["reason"] == "protected_content"]),
        "method": "Inherited source-group admission plus normalized exact and 8-word Jaccard >= .85 field checks",
        "limits": "No claim of semantic independence or unknown base pretraining exclusion"})
    shutil.copytree(r2_pack / "tokenizer", out / "tokenizer")
    data.write_rows(out / "held-content.jsonl", held)
    data.save(out / "admission-manifests.json", admission_manifests)
    if run_selection is not None:
        data.save(out / "run-selection.json", {"decision": "parent_accepts_recommended_run_deferrals",
            "reviewer": "Codex parent", "source_sha256": data.sha(run_selection), "source": selection,
            "scope": "This bounded response experiment only; original corpus admission unchanged"})
    data.save(out / "selection-report.json", {
        "input_admitted_conversations": len(source_rows), "input_r2_training_views": len(old_rows),
        "selected_training_views": dict(Counter(r["collection"] for r in rows)),
        "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
        "duplicate_prefix_views": len(duplicate_views),
        "training_executed": False,
        "note": "Replay selects a finite subset from the R2 pool; unselected replay remains eligible in its original dataset."})
    manifest = {"version": "understanding-adaptation-r3", "model": MODEL, "revision": REVISION,
        "initialization": INITIALIZATION, "private_research_only": True, "production_eligible": False,
        "optimization": {"learning_rate": 2e-5, "grad_accum": 16, "warmup_fraction": .05,
            "checkpoint_every": 10, "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False},
        "counts": dict(Counter(r["collection"] for r in rows)), "selected_rows": len(rows),
        "source_ids": len({r["source_id"] for r in rows}), "multiturn_response_targets": len(multi),
        "unique_conversation_histories": len({w["metadata"].get("full_history_id") for w in multi}),
        "dataset_version": dataset_version, "dataset_snapshots": dataset_manifests,
        "r2_source_manifest_sha256": R2_SHA, "builder_sha256": data.sha(Path(__file__)),
        "stages": [{k: v for k, v in stage.items() if k != "schedule"}],
        "review_status": "Source-backed and agent-checked research material; no inferred native-human or clinical approval",
        "artifacts": {p.relative_to(out).as_posix(): data.sha(p) for p in out.rglob("*") if p.is_file()}}
    data.save(out / "manifest.json", manifest)
    data.save(out / "PREPARED.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-pack", type=Path, required=True)
    parser.add_argument("--admission", action="append", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--run-selection", type=Path)
    args = parser.parse_args()
    result = prepare(args.r2_pack, args.admission, args.evaluation, args.out, args.dataset_version, args.run_selection)
    from run_understanding_r3 import verify_pack
    verify_pack(args.out)
    print(json.dumps({k: result[k] for k in ("counts", "source_ids", "multiturn_response_targets", "stages")}, indent=2))
