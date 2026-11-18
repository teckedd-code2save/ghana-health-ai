#!/usr/bin/env python3
"""Assemble admitted R7 collections, preserving every source and training target.

This never approves candidates. It accepts sealed admitted research collections,
registers exact assistant-prefix views, screens protection, encodes without
truncation, and schedules one complete shuffled pass of the retained views.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import tempfile

import run_understanding_r7 as run
import training_data_inputs as data
from check_afrique_training_contract import ASSET_HASHES, load_tokenizer
from prepare_understanding_r2 import encode_language
from prepare_understanding_run import HoldoutIndex
from prepare_r7_source_inventory import Families, identity_keys
from r7_encoding import encode_complete, WITNESS_VERSION
from select_understanding_r5_sources import bound_package, bound_read, recheck


def source_release(folder, pins):
    folder = Path(folder).resolve(strict=True)
    seals = [name for name in ("READY.json", "SEALED.json", "PREPARED.json") if (folder / name).is_file()]
    data.require(len(seals) == 1, "One unambiguous source seal required")
    manifest = bound_package(folder, pins, {"source-manifests.json"}, seals[0])
    admitted = manifest.get("admission_status") == "admitted"
    if manifest.get("version") == "r7-alignment-research-admission-v1" and "admission_status" not in manifest:
        # The first immutable alignment release declares admission per dataset.
        # Its dedicated verifier must succeed before accepting that legacy shape.
        from admit_r7_alignment import verify
        verify(folder)
        admitted = True
    data.require(admitted and manifest.get("private_research_only") is True
                 and manifest.get("production_eligible") is False, "Unadmitted collection release")
    result = []
    for ref in bound_read(folder / "source-manifests.json", pins):
        data.require(manifest["artifacts"].get(ref["manifest_path"]) == ref["manifest_sha256"], "Unbound source manifest")
        source = bound_read(data.safe_path(folder, ref["manifest_path"]), pins)
        data.require((source["dataset_id"], source["dataset_version"]) == (ref["dataset_id"], ref["dataset_version"])
                     and source["admission_status"] == "admitted" and source["private_research_only"] is True
                     and source["production_eligible"] is False, "Unadmitted dataset")
        data.require(manifest["artifacts"].get(source["records_path"]) == source["records_sha256"], "Unbound dataset records")
        records = bound_read(data.safe_path(folder, source["records_path"]), pins, rows=True)
        data.require(len(records) == source["record_count"] and records, "Dataset count differs")
        for record in records:
            data.require(record.get("training_eligible") is True and record.get("split") == "train"
                         and record.get("collection", source.get("catalog_dataset_id")) in run.COLLECTIONS,
                         "Unadmitted training record or unknown collection")
            result.append((record, source, {"release": str(folder), "release_manifest_sha256": data.sha(folder / "manifest.json"),
                "dataset_id": ref["dataset_id"], "dataset_version": ref["dataset_version"],
                "dataset_manifest_sha256": ref["manifest_sha256"], "record_sha256": run.digest(record)}))
    return result


def prefix_views(record, source, upstream):
    data.require(all(isinstance(record.get(k), str) and record[k].strip() for k in ("id", "source_id")), "Missing source ID")
    data.require(all(isinstance(record.get(k), dict) and record[k] for k in ("metadata", "original_ref")), "Missing provenance")
    collection = record.get("collection", source.get("catalog_dataset_id"))
    if collection == "language":
        data.require(isinstance(record.get("text"), str) and record["text"].strip(), "Missing language text")
        targets = [(None, None)]
    else:
        messages = record["messages"]
        data.require(messages and messages[-1]["role"] == "assistant", "Incomplete admitted conversation")
        assistant_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        selected = record["metadata"].get("training_assistant_indices", assistant_indices)
        data.require(isinstance(selected, list) and selected and all(type(i) is int for i in selected)
                     and selected == sorted(set(selected)) and set(selected) <= set(assistant_indices),
                     "Invalid explicit assistant-target selection")
        targets = [(i, messages[:i + 1]) for i in selected]
    for turn, messages in targets:
        view = deepcopy(record)
        identity = data.digest([upstream["dataset_id"], upstream["dataset_version"], record["id"]])
        view.update(id="r7-view:" + identity + (":text" if turn is None else ":assistant-" + str(turn)), collection=collection)
        if messages is not None:
            view["messages"] = messages
        view["metadata"]["r7_view"] = {"parent_record_id": record["id"], "parent_record_sha256": upstream["record_sha256"],
            "assistant_message_index": turn, "derivation": "exact_full_history_prefix" if turn is not None else "unchanged_text"}
        view["original_ref"]["r7_upstream_admission"] = deepcopy(upstream)
        yield view


def content_views(record):
    if "text" in record:
        yield record["text"]
    history = []
    for message in record.get("messages", []):
        if message["role"] == "system":
            continue
        history.append(message["content"])
        if message.get("tool_calls"):
            history.append(json.dumps(message["tool_calls"], sort_keys=True, ensure_ascii=False))
        if message["role"] == "assistant":
            yield "\n".join(history)


def protected_records(records, held):
    index = HoldoutIndex((r["id"], r["text"]) for r in held)
    families, record_keys = Families(), {}
    blocked = set()
    for record in records:
        keys = identity_keys(record) | {"r7-record:" + record["id"]}
        families.connect(keys)
        record_keys[record["id"]] = keys
    for item in held:
        keys = identity_keys(item) | {"r7-held:" + item["id"]}
        families.connect(keys)
        blocked.update(keys)
    reasons = {}
    for record in records:
        hit = next((hit for text in content_views(record) if (hit := index.match(text))), None)
        if hit:
            blocked.update(record_keys[record["id"]])
            reasons[record["id"]] = hit
    roots = {families.root(key) for key in blocked}
    return {record["id"]: reasons.get(record["id"], {"kind": "connected_source_group"}) for record in records
            if any(families.root(key) in roots for key in record_keys[record["id"]])}


def one_pass_schedule(rows):
    data.require(rows and len(rows) <= run.MAX_SCHEDULE, "Explicit dataset selection must fit the finite schedule; no silent subsampling")
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    return {"name": "responses", "schedule": order, "snapshot_after_updates": [],
            "primary_collections": ["responses", "tools", "health", "commerce"],
            "selection_policy": "one_complete_pass_of_every_retained_registered_view",
            "sampling_seed": 42, "repeat_factor": 1}


def prepare(admissions, protection, tokenizer_folder, out, version):
    out = Path(out).absolute()
    data.require(not out.exists() and data.valid_id(version), "New version and output directory required")
    pins, sources, held, protected_refs = {}, [], [], []
    for folder in admissions:
        sources.extend(source_release(folder, pins))
    data.require(sources, "No admitted sources")
    for n, folder in enumerate(protection):
        folder = Path(folder).resolve(strict=True)
        seals = [s for s in ("READY.json", "SEALED.json", "PREPARED.json", "FROZEN.json") if (folder / s).is_file()]
        data.require(len(seals) == 1, "Unambiguous protection seal required")
        bound_package(folder, pins, {"held-content.jsonl"}, seals[0])
        units = bound_read(folder / "held-content.jsonl", pins, rows=True)
        data.require(units, "Empty protection input")
        held.extend({**row, "id": f"protection-{n}:" + row["id"]} for row in units)
        protected_refs.append({"path": str(folder), "manifest_sha256": data.sha(folder / "manifest.json"), "units": len(units)})
    data.require(held, "Actual inherited and fresh holdout content required")
    views = [view for record, source, upstream in sources for view in prefix_views(record, source, upstream)]
    data.require(len({v["id"] for v in views}) == len(views), "Repeated source admission")
    blocked = protected_records(views, held)
    tokenizer = load_tokenizer(Path(tokenizer_folder))
    rows, witnesses, retained, exclusions, duplicates, seen = [], [], [], [], [], {}
    for view in views:
        if view["id"] in blocked:
            exclusions.append({"id": view["id"], "reason": "held_out_source_or_content", "evidence": blocked[view["id"]]})
            continue
        objective = "causal_lm" if view["collection"] == "language" else "final_assistant"
        try:
            encoded = encode_language(tokenizer, view["text"]) if objective == "causal_lm" else encode_complete(tokenizer, view, 4096)
        except ValueError as error:
            exclusions.append({"id": view["id"], "reason": "native_encoding_rejected_no_truncation", "detail": str(error)})
            continue
        duplicate_key = run.digest([view["collection"], objective, encoded["input_ids"], encoded["labels"]])
        if duplicate_key in seen:
            duplicates.append({"id": view["id"], "retained_id": seen[duplicate_key], "collection": view["collection"]})
            continue
        seen[duplicate_key] = view["id"]
        reference = {"dataset_id": "gha-r7-" + view["collection"], "dataset_version": version,
            "record_id": view["id"], "source_id": view["source_id"], "record_sha256": run.digest(view)}
        row = {"id": view["id"], "source_id": view["source_id"], "collection": view["collection"],
            "split": "train", "objective": objective, "dataset_ref": reference, **encoded,
            "sequence_tokens": len(encoded["input_ids"]), "target_tokens": sum(t != -100 for t in encoded["labels"])}
        witness = {k: view[k] for k in ("id", "source_id", "collection", "metadata", "original_ref")}
        witness.update(version=WITNESS_VERSION, dataset_ref=reference)
        for key in ("text", "messages", "tools"):
            if key in view:
                witness[key] = view[key]
        rows.append(row); witnesses.append(witness); retained.append(view)
    stages = [one_pass_schedule(rows)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r7-pack-") as temporary:
        target = Path(temporary)
        for name, values in (("rows.jsonl", rows), ("source-witnesses.jsonl", witnesses),
                             ("held-content.jsonl", held), ("exclusions.jsonl", exclusions), ("duplicates.jsonl", duplicates)):
            data.write_rows(target / name, values)
        data.save(target / "stages.json", stages)
        data.save(target / "evaluation-manifest.json", {"protected_inputs": protected_refs, "unseen_pretraining_claim": False})
        data.save(target / "overlap-audit.json", {"protected_units": len(held), "overlap_free": True,
            "held_content_sha256": data.sha(target / "held-content.jsonl"),
            "source_witnesses_sha256": data.sha(target / "source-witnesses.jsonl"),
            "evaluation_manifest_sha256": data.sha(target / "evaluation-manifest.json")})
        references = []
        for collection in sorted({r["collection"] for r in retained}):
            subset = [r for r in retained if r["collection"] == collection]
            folder = target / "datasets" / collection
            folder.mkdir(parents=True)
            records_path = folder / "records.jsonl"
            data.write_rows(records_path, subset)
            manifest = {"dataset_id": "gha-r7-" + collection, "dataset_version": version,
                "catalog_dataset_id": collection, "admission_status": "admitted", "private_research_only": True,
                "production_eligible": False, "record_count": len(subset), "records_sha256": data.sha(records_path),
                "records_path": records_path.relative_to(target).as_posix(),
                "review_scope": "Inherited scoped research admission; exact derived views grant no new semantic or human certification."}
            data.save(folder / "manifest.json", manifest)
            references.append({"dataset_id": manifest["dataset_id"], "dataset_version": version,
                "manifest_path": (folder / "manifest.json").relative_to(target).as_posix(),
                "manifest_sha256": data.sha(folder / "manifest.json")})
        data.save(target / "source-manifests.json", references)
        (target / "tokenizer").mkdir()
        for name, checksum in ASSET_HASHES.items():
            path = Path(tokenizer_folder) / name
            data.require(data.sha(path) == checksum, "Pinned tokenizer changed")
            shutil.copyfile(path, target / "tokenizer" / name)
        report = {"source_records": len(sources), "derived_views_considered": len(views),
            "retained_views": len(rows), "excluded_views": len(exclusions), "duplicate_views": len(duplicates),
            "unique_source_ids": len({r["source_id"] for r in retained}), "training_executed": False,
            "views_by_collection": dict(Counter(r["collection"] for r in rows)), "input_files": pins}
        data.save(target / "report.json", report)
        manifest = {"version": run.VERSION, "model": run.MODEL, "revision": run.REVISION, "initialization": run.INITIALIZATION,
            "private_research_only": True, "production_eligible": False,
            "optimization": {"learning_rate": run.LEARNING_RATE, "grad_accum": 16, "warmup_fraction": .03,
                "checkpoint_every": 25, "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False},
            "artifacts": {p.relative_to(target).as_posix(): data.sha(p) for p in target.rglob("*") if p.is_file()}}
        data.save(target / "manifest.json", manifest)
        data.save(target / "PREPARED.json", {"manifest_sha256": data.sha(target / "manifest.json")})
        _, checked_rows, measured = run.verify_pack(target)
        run.verify_native_rows(target, checked_rows)
        recheck(pins)
        target.rename(out)
    return {**report, "manifest_sha256": data.sha(out / "manifest.json"),
            "measured_schedule": {k: v for k, v in measured[0].items() if k not in {"schedule", "row_exposures"}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, action="append", required=True)
    parser.add_argument("--protection", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.admission, args.protection, args.tokenizer, args.out, args.version), indent=2))


if __name__ == "__main__":
    main()
