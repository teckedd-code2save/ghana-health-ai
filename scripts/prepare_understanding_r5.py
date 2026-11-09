"""Build a larger, source-bound R5 response continuation with explicit replay."""
import argparse
from collections import Counter
import math
from pathlib import Path
import random
import shutil

import training_data_inputs as data
import prepare_understanding_r3 as previous
from prepare_understanding_run import HoldoutIndex, effective_tokens
from check_afrique_training_contract import encode_complete, load_tokenizer
from screen_understanding_r5_data import inspect_messages
from augment_understanding_r4_spelling import augment
from select_understanding_r5_sources import bound_package, bound_read, recheck

R4_SHA = "b71682b47512506f9621faba7370f9f2d941437acc44e4dbe6c2716c0bf50cc7"


def instruction_paths(folder, manifest):
    names = {name for name in manifest["artifacts"]
             if Path(name).parent == Path(".") and name.endswith(".train.jsonl")}
    data.require({p.name for p in folder.glob("*.train.jsonl")} == names,
                 "Unbound or missing instruction training file")
    return [data.safe_path(folder, name) for name in sorted(names)]


def protection_views(records):
    """Scan each original/derived context while connecting all ancestry locations."""
    keys = ("group_id", "family_id", "tree_id", "original_group_id", "source_ids")
    for record in records:
        identity = {"id": record["id"], "source_id": record["source_id"]}
        top = {key: record[key] for key in keys if key in record}
        yield {**record, "metadata": top}
        for location in ("metadata", "source_metadata"):
            if record.get(location):
                yield {**identity, "metadata": record[location]}
        for key in ("messages", "english_messages", "original_english_messages"):
            messages = record.get(key, [])
            if messages:
                yield {**identity, "metadata": top, "messages": messages}
                for end, message in enumerate(messages[:-1], 1):
                    if message["role"] == "assistant":
                        yield {**identity, "metadata": top, "messages": messages[:end]}


def review_resolutions(record):
    review = record.get("review", {})
    key = ("screened_english_messages_sha256"
           if record.get("origin") == "source_english_separately_reviewed_for_r5"
           else "screened_messages_sha256")
    decision = review.get("decision_sha256")
    if (review.get("status") != "agent_checked" or not data.valid_hash(decision)
            or decision != record.get("original_ref", {}).get("review_decision_sha256")
            or review.get(key) != data.digest(record["messages"])):
        return set()
    return {(f["code"], f["turn"]) for f in review.get("screen_resolutions", [])
            if isinstance(f.get("reason"), str) and f["reason"].strip()}


def schedule(rows):
    responses = [i for i, r in enumerate(rows) if r["collection"] == "responses"]
    new_conversations = [i for i in responses if rows[i].get("new_source_conversation") is True]
    data.require(responses and new_conversations, "New source-backed conversation targets required")
    used, order, passes = set(), [], []
    for epoch in range(2):
        primary = responses if epoch == 0 else new_conversations
        target = sum(effective_tokens(rows[i]) for i in primary)
        batch = primary.copy()
        for collection, ratio in previous.RATIOS.items():
            pool = [i for i, r in enumerate(rows) if r["collection"] == collection and i not in used]
            random.Random(142 + epoch).shuffle(pool)
            # Divide finite replay across passes instead of exhausting English in pass one.
            budget = min(math.floor(target * ratio), sum(effective_tokens(rows[i]) for i in pool) // (2 - epoch))
            tokens, selected = 0, []
            for i in pool:
                amount = effective_tokens(rows[i])
                if tokens + amount <= budget:
                    selected.append(i)
                    tokens += amount
            data.require(selected, "No finite replay for " + collection)
            batch.extend(selected)
            used.update(selected)
        if epoch == 0:
            remainder = len(batch) % 16
            counts = Counter(rows[i]["collection"] for i in batch if i not in primary)
            removed = []
            for i in reversed(batch):
                c = rows[i]["collection"]
                if len(removed) < remainder and c != "responses" and counts[c] > 1:
                    removed.append(i)
                    counts[c] -= 1
            data.require(len(removed) == remainder, "Cannot align pass without dropping primary examples")
            batch = [i for i in batch if i not in removed]
            used.difference_update(removed)
        random.Random(42 + epoch).shuffle(batch)
        order.extend(batch)
        passes.append({"pass": epoch + 1, "end_sequence": len(order), "sequence_count": len(batch),
                       "target_tokens": sum(effective_tokens(rows[i]) for i in batch)})
    return {"name": "responses", "schedule": order, "primary_collections": ["responses"],
        "snapshot_after_updates": [passes[0]["end_sequence"] // 16], "passes": passes,
        "response_exposure_policy": {"new_source_backed_conversation": 2, "inherited_and_project_authored": 1, "spelling_variant": 1},
        "mixing": "One complete response pass followed by a source-backed conversation consolidation pass. Finite distinct replay in each; desired 60/20/10/10 token mix is capped by available English and actual fractions reported. Project-authored repair examples and spelling variants are not multiplied."}


def response_coverage(rows, order):
    targets, views, sources = Counter(), {}, {}
    for index in order:
        row = rows[index]
        if row["collection"] != "responses":
            continue
        origin = row["preparation_origin"]
        group = ("new_source_backed_conversation" if row.get("new_source_conversation") is True
                 else "new_project_authored" if origin == "new_reviewed_translation"
                 else origin)
        targets[group] += effective_tokens(row)
        views.setdefault(group, set()).add(row["id"])
        sources.setdefault(group, set()).add(row["source_id"])
    total = sum(targets.values())
    return {"scheduled_response_target_tokens": total,
        "groups": {key: {"target_tokens": value, "response_token_share": value / total,
                         "distinct_views": len(views[key]), "source_ids": len(sources[key])}
                   for key, value in sorted(targets.items())},
        "source_counts_additive": False,
        "note": "Source IDs can recur across original and spelling-derived groups; repeated exposures are tokens, not new examples."}


def prepare(r4_pack, r2_pack, instruction, admissions, fresh, out, version):
    from run_understanding_r5 import INITIALIZATION, VERSION, verify_pack, verify_native_rows
    data.require(not out.exists() and not out.is_symlink() and data.valid_id(version), "Use a fresh path and version")
    pins = {}
    r4_manifest = bound_package(r4_pack, pins, {"source-manifests.json", "held-content.jsonl", "evaluation-manifest.json"}
        | {"tokenizer/" + name for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json")})
    bound_package(r2_pack, pins, {"rows.jsonl", "source-witnesses.jsonl"})
    instruction_manifest = bound_package(instruction, pins, set(), "SEALED.json")
    bound_package(fresh, pins, {"held-content.jsonl"}, "FROZEN.json")
    data.require(pins[str((r4_pack / "manifest.json").absolute())] == R4_SHA, "R4 source changed")
    data.require(pins[str((r2_pack / "manifest.json").absolute())] == previous.R2_SHA, "R2 source changed")
    instruction_files = instruction_paths(instruction, instruction_manifest)
    tokenizer = load_tokenizer(r4_pack / "tokenizer")
    held = bound_read(r4_pack / "held-content.jsonl", pins, rows=True) + bound_read(fresh / "held-content.jsonl", pins, rows=True)
    holdouts = HoldoutIndex([(r["id"], r["text"]) for r in held])
    candidates, provenance, exclusions, duplicates = [], [], [], []
    records, rows, witnesses, seen = {}, [], [], {}

    for ref in bound_read(r4_pack / "source-manifests.json", pins):
        data.require(ref["manifest_path"] in r4_manifest["artifacts"], "Unbound source manifest")
        manifest = bound_read(data.safe_path(r4_pack, ref["manifest_path"]), pins)
        data.require(ref["manifest_sha256"] == r4_manifest["artifacts"][ref["manifest_path"]], "Source manifest reference changed")
        collection = manifest["catalog_dataset_id"]
        if collection in {"responses", "english_retention"}:
            data.require(manifest["records_path"] in r4_manifest["artifacts"], "Unbound source records")
            for record in bound_read(data.safe_path(r4_pack, manifest["records_path"]), pins, rows=True):
                # The full instruction export below supplies canonical task records.
                if collection == "responses" and record.get("origin") == "inherited_r2_source_metadata":
                    continue
                candidates.append((collection, record, "inherited_r4"))
    for name in instruction_files:
        for record in bound_read(name, pins, rows=True):
            data.require(record["training_eligible"] and record["quality_status"] == "scoped_agent_checked_research", "Unadmitted instruction")
            record = {**record, "origin": record["target_origin"], "original_ref": {
                "release_path": str(instruction.resolve()), "release_manifest_sha256": data.sha(instruction / "manifest.json"),
                "file": name.name, "file_sha256": data.sha(name), "record_sha256": data.digest(record),
                "provenance": record["provenance"], "admission_sha256": record["admission_sha256"]}}
            candidates.append(("responses", record, "existing_instruction"))
    new_ids = set()
    for folder in admissions:
        manifest = bound_package(folder, pins, {"conversations.jsonl"}, "READY.json")
        data.require(manifest["admission_status"] == "admitted" and manifest["private_research_only"]
                     and not manifest["production_eligible"], "Private admission required")
        data.require(manifest["fresh_suite_sha256"] == data.sha(fresh / "manifest.json"), "Admission used another fresh suite")
        provenance.append({"path": str(folder.resolve()), "manifest_sha256": data.sha(folder / "manifest.json")})
        for record in bound_read(folder / "conversations.jsonl", pins, rows=True):
            data.require(record["training_eligible"] and record["split"] == "train"
                         and record["review"]["status"] == "agent_checked" and not record["review"]["findings"], "Missing target review")
            data.require(record["id"] not in new_ids, "Duplicate new admitted source")
            new_ids.add(record["id"])
            candidates.append(("responses", record, "new_reviewed_translation"))
            users = [m["content"] for m in record["messages"] if m["role"] == "user"]
            if int(data.digest(record["source_id"])[:8], 16) % 3 == 0 and not any(
                    any(marker in text.casefold() for marker in
                        ('`', '"', "'", '\u2018', '\u2019', '\u201c', '\u201d', '<', '>', 'http:', 'https:')) for text in users):
                derived = augment(record, {"parent_admission": str(folder.resolve()),
                    "parent_admission_sha256": data.sha(folder / "manifest.json"),
                    "augmentation_code_sha256": data.sha(Path(__file__).with_name("augment_understanding_r4_spelling.py"))})
                if derived:
                    candidates.append(("responses", derived, "new_spelling_augmentation"))
            if record["origin"].startswith("source_grounded_"):
                english = {**record, "id": record["id"] + ":english", "messages": record["english_messages"],
                    "language": "en", "origin": "source_english_separately_reviewed_for_r5",
                    "paired_twi_record_id": record["id"]}
                candidates.append(("english_retention", english, "reviewed_english_source"))
    data.require(new_ids, "No new reviewed source material")
    replay_rows = bound_read(r2_pack / "rows.jsonl", pins, rows=True)
    replay_witnesses = bound_read(r2_pack / "source-witnesses.jsonl", pins, rows=True)
    expanded = list(protection_views([r for _, r, _ in candidates] + replay_witnesses))
    blocked = previous.protected_groups(expanded, holdouts)

    def add(row, witness, record, origin):
        key = data.digest({k: row[k] for k in ("objective", "input_ids", "labels")})
        if key in seen:
            duplicates.append({"id": row["id"], "retained_id": seen[key], "class": row["collection"]})
            return
        seen[key] = row["id"]
        record_key = (row["collection"], record["id"])
        data.require(record_key not in records or records[record_key] == record, "Conflicting dataset identity")
        records[record_key] = record
        reference = previous.dataset_reference(row["collection"], version, record)
        rows.append({**row, "dataset_ref": reference, "preparation_origin": origin,
            "new_source_conversation": origin == "new_reviewed_translation" and record["origin"].startswith("source_grounded_")})
        witnesses.append({**witness, "dataset_ref": reference})

    for collection, record, origin in candidates:
        if record["id"] in blocked:
            exclusions.append({"id": record["id"], "reason": "protected_content", "hit": blocked[record["id"]]})
            continue
        findings = inspect_messages(record["messages"])
        # Existing approvals are not discarded; newly observed issues require a separate decision.
        resolutions = review_resolutions(record)
        unresolved = [f for f in findings if f["severity"] == "invalid" or (f["code"], f["turn"]) not in resolutions]
        if unresolved:
            exclusions.append({"id": record["id"], "reason": "screen_review_needed", "findings": findings})
            continue
        for turn, messages in enumerate(previous.prefixes(record)):
            identity = record["id"] + ":assistant-" + str(turn)
            try:
                encoded = encode_complete(tokenizer, {"id": identity, "messages": messages}, 4096)
            except ValueError as error:
                if str(error) != "overlength_no_truncation: " + identity:
                    raise
                exclusions.append({"id": identity, "reason": "overlength_no_truncation", "detail": str(error)})
                continue
            row = {"id": identity, "source_id": record["source_id"], "collection": collection, "split": "train",
                "objective": "final_assistant", **encoded, "sequence_tokens": len(encoded["input_ids"]),
                "target_tokens": sum(t != -100 for t in encoded["labels"])}
            witness = {"id": identity, "source_id": record["source_id"], "collection": collection, "messages": messages,
                "metadata": {"origin": record["origin"], "group_id": record.get("group_id", record.get("metadata", {}).get("group_id")),
                    "full_history_id": record["id"], "assistant_index": turn, "full_history_sha256": data.digest(record["messages"])},
                "original_ref": record["original_ref"]}
            add(row, witness, record, origin)
    for row, witness in zip(replay_rows, replay_witnesses, strict=True):
        if row["collection"] not in {"language", "meaning"}:
            continue
        if row["id"] in blocked:
            exclusions.append({"id": row["id"], "reason": "protected_content", "hit": blocked[row["id"]]})
            continue
        if "messages" in witness and inspect_messages(witness["messages"]):
            exclusions.append({"id": row["id"], "reason": "screen_review_needed"})
            continue
        record = {**witness, "split": "train", "training_eligible": True,
            "origin": witness["metadata"].get("origin", "inherited_r2_source_metadata"),
            "admission_provenance": {"mode": "unchanged_r2_admitted_view", "manifest_sha256": previous.R2_SHA,
                                     "witness_sha256": data.digest(witness)}}
        add(row, witness, record, "inherited_r2_replay")
    stage = schedule(rows)
    selected = sorted(set(stage["schedule"]))
    selected_set = set(selected)
    ledger = [{"id": r["id"], "source_id": r["source_id"], "collection": r["collection"],
               "status": "selected" if i in selected_set else "available_not_selected",
               "reason": "finite_token_balanced_schedule"} for i, r in enumerate(rows)]
    remap = {old: new for new, old in enumerate(selected)}
    rows, witnesses = [rows[i] for i in selected], [witnesses[i] for i in selected]
    stage["schedule"] = [remap[i] for i in stage["schedule"]]
    data.require(any(r["preparation_origin"] == "new_reviewed_translation" for r in rows), "All new data excluded")
    recheck(pins)
    instruction_paths(instruction, instruction_manifest)
    out.mkdir(parents=True, mode=0o700)
    snapshots = previous.register_datasets(out, rows, witnesses, records, version)
    for card in (out / "datasets").glob("*/README.md"):
        card.write_text(card.read_text().replace("R3 response adaptation from R2 meaning", "R5 response continuation from R4"))
    for name, values in (("rows.jsonl", rows), ("source-witnesses.jsonl", witnesses), ("excluded.jsonl", exclusions),
                         ("duplicate-prefixes.jsonl", duplicates), ("held-content.jsonl", held), ("selection-ledger.jsonl", ledger)):
        data.write_rows(out / name, values)
    data.save(out / "stages.json", [stage])
    data.save(out / "admission-manifests.json", provenance)
    data.save(out / "evaluation-manifest.json", {"development": bound_read(r4_pack / "evaluation-manifest.json", pins),
        "fresh_path": str(fresh.resolve()), "fresh_manifest_sha256": data.sha(fresh / "manifest.json")})
    data.save(out / "overlap-audit.json", {"overlap_free": True, "protected_units": len(held),
        "method": "Connected source groups; normalized exact and eight-word Jaccard >= .85; original and derivative text",
        "limits": "Does not establish unknown pretraining independence or native semantic certification."})
    shutil.copytree(r4_pack / "tokenizer", out / "tokenizer")
    native = verify_native_rows(out, rows)
    data.save(out / "native-encoding.json", native)
    totals = Counter()
    for i in stage["schedule"]:
        totals[rows[i]["preparation_origin"]] += effective_tokens(rows[i])
    report = {"candidate_histories": dict(Counter(origin for _, _, origin in candidates)),
        "selected_views": dict(Counter(r["collection"] for r in rows)), "new_admitted_records": len(new_ids),
        "new_selected_records": len({r["dataset_ref"]["record_id"] for r in rows if r["preparation_origin"] == "new_reviewed_translation"}),
        "spelling_augmentation_new_independent_answers": 0,
        "target_tokens_by_preparation_origin": dict(totals), "exclusions": dict(Counter(e["reason"] for e in exclusions)),
        "response_coverage": response_coverage(rows, stage["schedule"]),
        "duplicate_views": len(duplicates), "training_executed": False, "human_reviewed_new": 0}
    data.save(out / "selection-report.json", report)
    manifest = {"version": VERSION, "model": previous.MODEL, "revision": previous.REVISION,
        "initialization": INITIALIZATION, "private_research_only": True, "production_eligible": False,
        "optimization": {"learning_rate": 1e-5, "grad_accum": 16, "warmup_fraction": .05, "checkpoint_every": 10,
            "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False},
        "counts": dict(Counter(r["collection"] for r in rows)), "selected_rows": len(rows),
        "source_ids": len({r["source_id"] for r in rows}), "dataset_version": version, "dataset_snapshots": snapshots,
        "r4_reference_manifest_sha256": R4_SHA, "r2_replay_manifest_sha256": previous.R2_SHA,
        "builder_sha256": data.sha(__file__), "review_status": "Preserved upstream plus separate agent source/translation review; not human or clinical certification",
        "artifacts": {p.relative_to(out).as_posix(): data.sha(p) for p in out.rglob("*") if p.is_file()}}
    data.save(out / "manifest.json", manifest)
    recheck(pins)
    instruction_paths(instruction, instruction_manifest)
    # The verifier requires a seal; revoke it on any failure, including interruption.
    marker = out / "PREPARED.json"
    try:
        data.save(marker, {"manifest_sha256": data.sha(out / "manifest.json")})
        _, _, measured = verify_pack(out)
        recheck(pins)
        instruction_paths(instruction, instruction_manifest)
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return {"selection": report, "measured_stages": [{k: v for k, v in s.items() if k != "schedule"} for s in measured]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r4-pack", "r2-pack", "instruction", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission", type=Path, action="append", required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.r4_pack, args.r2_pack, args.instruction, args.admission, args.fresh, args.out, args.dataset_version), indent=2))
