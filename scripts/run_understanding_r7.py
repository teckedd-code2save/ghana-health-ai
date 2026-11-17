#!/usr/bin/env python3
"""Sealed R7 continuation from final R6; offline verification, explicit launch.

Contained versioned copy of the R6 runner: keep its sealed-pack, frozen-core,
checkpoint and bounded-execution contracts without mutating R2-R6 files/globals.

Pack builder API: verify_pack(folder), verify_native_rows(folder, rows), and
prepare(pack, out, mode, source, smoke_directory=None). Dataset record digests
use this module's digest (the sealed R2 canonical JSON convention). Witnesses
must carry WITNESS_VERSION and exact messages/tools/metadata/original_ref.
Overlap evidence binds held-content, witnesses and evaluation-manifest hashes;
held-content must be nonempty. No helper refreshes a seal or launches a worker.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import run_understanding_r2 as r2
from run_understanding_r2 import FROZEN, RUNTIME, digest, read, require, run_id, safe_file, save, sha
from r7_encoding import WITNESS_VERSION, validate_history

ROOT = Path(__file__).resolve().parents[1]
VERSION = "understanding-adaptation-r7"
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
SOURCE_RUN = "understanding-adaptation-r6/understanding_20261113T152209Z"
INITIAL_SHA = "6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c"
INITIALIZATION = {"adapter_directory": "/artifacts/" + SOURCE_RUN + "/responses-adapter",
                  "adapter_weights_sha256": INITIAL_SHA,
                  "adapter_config_sha256": "577b80656dbcff84886c4e402022d9507abbef34b0e0013bceab029ccf4c9403",
                  "new_optimizer": True, "new_scheduler": True}
# Completion metadata measured from R6 train-collected; adapter bytes verified before loading.
SOURCE_FILES = {
    "responses-adapter/adapter_model.safetensors": INITIAL_SHA,
    "responses-adapter/adapter_config.json": "577b80656dbcff84886c4e402022d9507abbef34b0e0013bceab029ccf4c9403",
    "status.json": "c38d953cb626ba49183f7f8b60aaab6dbfe19ee603d785dd0977bb5bf40356a9",
    "execution.json": "5461176e8a3a9fe67bdc59ca6a562b00691332cc54d6bb5cc528cb7ac85ebf4e",
    "adapters.json": "b62c3a84bd823a8c363a5008085c81d49fcd5185051f4a7ade27be2d2e9bdeeb",
    "consumption.json": "cca1fd6ac51b9d5449afd9060b9cbf08b639ba0534210504d11389327f34c074",
}
CODE = ("scripts/run_understanding_r7.py", "scripts/r7_encoding.py", "modal/train/train_understanding_r7.py")
REUSED = ("scripts/run_understanding_r2.py", "scripts/prepare_understanding_r2.py")
VOLUME, CLAIMS = r2.VOLUME, "ghana-understanding-r7-invocations"
BOUNDS = {"smoke": (780, 900), "train": (28200, 28800)}
LEARNING_RATE = 1e-5
MAX_SCHEDULE = 50000
OPT_KEYS = ("learning_rate", "grad_accum", "warmup_fraction", "checkpoint_every")
COLLECTIONS = {"responses", "tools", "health", "commerce", "meaning", "language", "english_retention"}


def json_rows(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate row key")
            value[key] = item
        return value
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line, object_pairs_hook=pairs,
                           parse_constant=lambda _: require(False, "Nonfinite row value")) for line in stream]


def admitted_sources(folder, rows, witnesses, artifacts):
    """Bind every encoded view to a versioned, admitted, pack-local source record."""
    references = read(safe_file(folder, "source-manifests.json"))
    require(isinstance(references, list) and references, "Missing admitted dataset snapshots")
    snapshots, records = [], {}
    for ref in references:
        key = (ref["dataset_id"], ref["dataset_version"])
        require(all(isinstance(v, str) and v.strip() for v in key) and key not in records, "Duplicate/unversioned dataset")
        require(ref["manifest_path"] in artifacts and artifacts[ref["manifest_path"]] == ref["manifest_sha256"], "Unbound dataset manifest")
        path = safe_file(folder, ref["manifest_path"])
        require(sha(path) == ref["manifest_sha256"], "Dataset manifest changed")
        source = read(path)
        require((source["dataset_id"], source["dataset_version"]) == key and source["admission_status"] == "admitted"
                and source["private_research_only"] is True and source["production_eligible"] is False, "Dataset is not admitted for research training")
        require(source["records_path"] in artifacts and artifacts[source["records_path"]] == source["records_sha256"], "Unbound dataset records")
        path = safe_file(folder, source["records_path"])
        require(sha(path) == source["records_sha256"], "Dataset records changed")
        source_rows = json_rows(path)
        require(type(source["record_count"]) is int and source["record_count"] == len(source_rows) and source_rows, "Dataset inventory count differs")
        records[key] = {}
        for record in source_rows:
            require(all(isinstance(record[k], str) and record[k].strip() for k in ("id", "source_id"))
                    and record["id"] not in records[key], "Duplicate/invalid dataset record")
            require(record["split"] == "train" and record["training_eligible"] is True, "Dataset record not admitted to training")
            records[key][record["id"]] = record
        snapshots.append({**ref, "admitted_record_count": len(source_rows)})
    for row, witness in zip(rows, witnesses, strict=True):
        require(witness.get("version") == WITNESS_VERSION, "Unversioned R7 source witness")
        ref = row.get("dataset_ref")
        require(isinstance(ref, dict) and set(ref) == {"dataset_id", "dataset_version", "record_id", "source_id", "record_sha256"}
                and ref == witness.get("dataset_ref"), "Missing/mismatched dataset reference")
        key = (ref["dataset_id"], ref["dataset_version"])
        require(key in records and ref["record_id"] in records[key], "Training row outside admitted snapshot")
        record = records[key][ref["record_id"]]
        require(ref["source_id"] == row["source_id"] == witness["source_id"] == record["source_id"]
                and digest(record) == ref["record_sha256"], "Dataset record/source identity differs")
        if row["objective"] == "final_assistant":
            messages = witness["messages"]
            require(isinstance(record.get("messages"), list) and messages == record["messages"],
                    "Chat messages differ from complete admitted record")
            require(("tools" in witness) == ("tools" in record) and witness.get("tools") == record.get("tools"),
                    "Tool schemas differ from admitted record")
        else:
            require(witness["text"] == record.get("text"), "Language text differs from admitted record")
        for key in ("metadata", "original_ref"):
            require(witness[key] == record.get(key), "Source provenance differs: " + key)
    return snapshots


def verify_overlap(folder, witnesses, artifacts):
    """Recompute lexical and source-family protection from sealed, nonempty inputs."""
    from prepare_understanding_run import HoldoutIndex
    audit = read(safe_file(folder, "overlap-audit.json"))
    for name, key in (("held-content.jsonl", "held_content_sha256"),
                      ("source-witnesses.jsonl", "source_witnesses_sha256"),
                      ("evaluation-manifest.json", "evaluation_manifest_sha256")):
        require(name in artifacts and audit.get(key) == artifacts[name], "Unbound overlap evidence: " + name)
    held = json_rows(safe_file(folder, "held-content.jsonl"))
    require(held and audit.get("protected_units") == len(held) and audit.get("overlap_free") is True,
            "Empty/uncleared held-out overlap protection")
    require(all(isinstance(r.get("id"), str) and r["id"].strip()
                and isinstance(r.get("text"), str) and r["text"].strip() for r in held)
            and len({r["id"] for r in held}) == len(held), "Invalid held-out content")
    identities = {"source_id", "source_ids", "canonical_source_id", "group_id", "group_aliases",
                  "family_id", "tree_id", "original_group_id"}

    def groups(value):
        found = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in identities:
                    values = child if isinstance(child, list) else [child]
                    found.update(v for v in values if isinstance(v, str) and v)
                elif isinstance(child, (dict, list)):
                    found.update(groups(child))
        elif isinstance(value, list):
            for child in value:
                found.update(groups(child))
        return found

    protected = set().union(*(groups(row) for row in held))
    index = HoldoutIndex((row["id"], row["text"]) for row in held)
    for witness in witnesses:
        require(not groups(witness) & protected, "Held-out source/group overlap: " + witness["id"])
        views = [witness["text"]] if "text" in witness else []
        for key in ("messages", "english_messages", "original_english_messages"):
            parts = []
            for message in witness.get(key, witness.get("metadata", {}).get(key, [])):
                if message["role"] != "system":
                    parts.append(message["content"])
                    if message.get("tool_calls"):
                        parts.append(json.dumps(message["tool_calls"], sort_keys=True, ensure_ascii=False))
                    if message["role"] == "assistant":
                        views.append("\n".join(parts))
        for text in views:
            require(index.match(text) is None, "Held-out content overlap: " + witness["id"])
    return {"protected_units": len(held), "checked_witnesses": len(witnesses), "recomputed": True}


def consumption_receipt(rows, stage, grad_accum, position, *, snapshots, mode, execution_sha, data_sha, checkpoint=None):
    """Account only the committed schedule prefix, never source/archive inventory."""
    order = stage["schedule"]
    cursor, updates = position["completed_microbatches"], position["completed_updates"]
    require(type(cursor) is int and type(updates) is int and 0 <= updates <= math.ceil(len(order) / grad_accum)
            and cursor == min(updates * grad_accum, len(order)) and position["microbatches_in_update"] == 0, "Uncommitted consumption cursor")
    visits = Counter(order[:cursor])
    exposure = position["exposure"]
    require(exposure["row_visits"] == {str(i): n for i, n in visits.items()}, "Consumed row visits differ from exact schedule prefix")
    per_row, by_class, by_dataset = [], {}, {}
    for i, n in sorted(visits.items()):
        row = rows[i]
        amounts = {"unique_rows": 1, "row_exposures": n, "input_tokens": len(row["input_ids"]) * n,
                   "effective_target_tokens": sum(t != -100 for t in row["labels"][1:]) * n}
        per_row.append({"row_index": i, "id": row["id"], "class": row["collection"],
                        "dataset_ref": row["dataset_ref"], **amounts})
        for groups, key in ((by_class, row["collection"]),
                            (by_dataset, (row["dataset_ref"]["dataset_id"], row["dataset_ref"]["dataset_version"]))):
            counter = groups.setdefault(key, Counter())
            counter.update(amounts)
    totals = {key: sum(row[key] for row in per_row) for key in ("unique_rows", "row_exposures", "input_tokens", "effective_target_tokens")}
    def record_key(row):
        return tuple(row["dataset_ref"][k] for k in ("dataset_id", "dataset_version", "record_id"))
    totals["unique_dataset_records"] = len({record_key(row) for row in per_row})
    for name, counter in by_class.items():
        counter["unique_dataset_records"] = len({record_key(row) for row in per_row if row["class"] == name})
    for key, counter in by_dataset.items():
        counter["unique_dataset_records"] = len({record_key(row) for row in per_row if record_key(row)[:2] == key})
    require(totals["row_exposures"] == cursor and totals["input_tokens"] == exposure["input_tokens"]
            and totals["effective_target_tokens"] == exposure["effective_target_tokens"], "Consumed token accounting differs")
    require(checkpoint is not None or cursor == 0, "Nonzero consumption requires checkpoint proof")
    return {"version": "r7-consumption-v1", "mode": mode, "scope": "cumulative_committed_schedule_prefix",
            "execution_sha256": execution_sha, "data_sha256": data_sha, "checkpoint": checkpoint,
            "completed_updates": updates, "schedule_complete": cursor == len(order),
            "training_complete": mode == "train" and cursor == len(order),
            "inventory": {"packed_rows": len(rows), "packed_rows_by_class": dict(Counter(r["collection"] for r in rows)),
                          "admitted_snapshots": snapshots, "archive_counts_included": False},
            "planned": {"row_exposures": len(order), "unique_rows": len(set(order)),
                        "schedule_sha256": digest(order),
                        "input_tokens": sum(len(rows[i]["input_ids"]) for i in order),
                        "effective_target_tokens": sum(sum(t != -100 for t in rows[i]["labels"][1:]) for i in order)},
            "consumed": {**totals, "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
                         "by_dataset": [{"dataset_id": k[0], "dataset_version": k[1], **dict(v)} for k, v in sorted(by_dataset.items())]},
            "rows": per_row}


def verify_pack(folder):
    """Verify bytes, full-history structure, masks and exact finite exposure counts."""
    folder = Path(folder).resolve(strict=True)
    manifest = read(safe_file(folder, "manifest.json"))
    require(read(safe_file(folder, "PREPARED.json")) == {"manifest_sha256": sha(folder / "manifest.json")}, "Pack seal differs")
    require(manifest["version"] == VERSION and manifest["model"] == MODEL and manifest["revision"] == REVISION
            and manifest["initialization"] == INITIALIZATION, "Wrong R7 foundation/initialization")
    require(manifest["private_research_only"] is True and manifest["production_eligible"] is False, "Research-only pack required")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked pack artifact")
    require(actual == set(manifest["artifacts"]) | {"manifest.json", "PREPARED.json"}, "Unbound or missing artifact")
    required = {"rows.jsonl", "source-witnesses.jsonl", "stages.json", "evaluation-manifest.json", "overlap-audit.json", "source-manifests.json", "held-content.jsonl"}
    required |= {"tokenizer/" + n for n in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json")}
    require(required <= set(manifest["artifacts"]), "Missing required pack evidence")
    for name, checksum in manifest["artifacts"].items():
        require(sha(safe_file(folder, name)) == checksum, "Pack artifact changed: " + name)
    require(read(folder / "evaluation-manifest.json"), "Missing evaluation provenance")
    opt = manifest["optimization"]
    require(type(opt["learning_rate"]) in (float, int) and opt["learning_rate"] == LEARNING_RATE, "Changed R7 learning rate")
    require(type(opt["warmup_fraction"]) in (float, int) and 0 <= opt["warmup_fraction"] <= .2, "Unsafe warmup")
    for key, maximum in (("grad_accum", 32), ("checkpoint_every", 25)):
        require(type(opt[key]) is int and 1 <= opt[key] <= maximum, "Unsafe " + key)
    require(opt["seed"] == 42 and opt["precision"] == "bf16_base_fp32_adapters"
            and opt["packing"] is False and opt["truncate"] is False, "Changed numerical contract")
    rows, witnesses = json_rows(folder / "rows.jsonl"), json_rows(folder / "source-witnesses.jsonl")
    require(rows and len(rows) == len(witnesses), "Missing rows/witnesses")
    seen, effective, multiturn = set(), [], set()
    for i, (row, witness) in enumerate(zip(rows, witnesses, strict=True)):
        require(all(isinstance(row[k], str) and row[k].strip() for k in ("id", "source_id", "collection")), "Invalid row identity")
        key = (row["collection"], row["id"])
        require(key not in seen and row["collection"] in COLLECTIONS, "Duplicate/unknown collection")
        seen.add(key)
        require(all(row[k] == witness[k] for k in ("id", "source_id", "collection")) and row["split"] == "train", "Witness/split differs")
        require(all(isinstance(witness[k], dict) and witness[k] for k in ("metadata", "original_ref")), "Missing source provenance")
        ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
        require(all(type(v) is list and all(type(t) is int for t in v) for v in (ids, labels, mask)), "Invalid token arrays")
        require(2 <= len(ids) <= 4096 and len(ids) == len(labels) == len(mask) and mask == [1] * len(ids)
                and all(t >= 0 for t in ids) and ids[-1] == 248046, "Invalid sequence/mask/EOS")
        count = sum(t != -100 for t in labels[1:])
        require(count > 0 and row["sequence_tokens"] == len(ids)
                and row["target_tokens"] == sum(t != -100 for t in labels), "Target accounting differs")
        effective.append(count)
        if row["objective"] == "causal_lm":
            require(row["collection"] == "language" and labels == ids and isinstance(witness["text"], str)
                    and witness["text"].strip() and "messages" not in witness and "tools" not in witness,
                    "Invalid language target")
        else:
            require(row["objective"] == "final_assistant", "Unknown objective")
            messages = witness["messages"]
            has_tools = validate_history(witness)
            require(row["collection"] != "tools" or has_tools, "Tools collection lacks schemas")
            first = next(j for j, t in enumerate(labels) if t != -100)
            require(first > 0 and labels[:first] == [-100] * first and labels[first:] == ids[first:]
                    and len(labels[first:]) > 1 and labels[first:].count(248046) == 1, "Invalid final-assistant supervision")
            if sum(m["role"] == "assistant" for m in messages) >= 2:
                multiturn.add(i)
    snapshots = admitted_sources(folder, rows, witnesses, manifest["artifacts"])
    verify_overlap(folder, witnesses, manifest["artifacts"])
    stages = read(folder / "stages.json")
    require(isinstance(stages, list) and len(stages) == 1 and stages[0]["name"] == "responses", "Expected one continuous response schedule")
    stage = dict(stages[0])
    order = stage["schedule"]
    require(type(order) is list and 1 <= len(order) <= MAX_SCHEDULE
            and all(type(i) is int and 0 <= i < len(rows) for i in order), "Invalid/beyond-budget schedule")
    visits = Counter(order)
    require(max(visits.values()) <= 4, "More than four row exposures")
    primary = stage.get("primary_collections", ["responses"])
    require(isinstance(primary, list) and primary and all(isinstance(x, str) and x in COLLECTIONS for x in primary)
            and len(primary) == len(set(primary)), "Invalid primary collections")
    require(any(rows[i]["collection"] in primary for i in order) and any(rows[i]["collection"] not in primary for i in order)
            and any(i in multiturn and rows[i]["collection"] in primary for i in order), "Scheduled conversation and retention required")
    measured = {"scheduled_sequences": len(order), "optimizer_updates": math.ceil(len(order) / opt["grad_accum"]),
                "sequence_tokens": sum(len(rows[i]["input_ids"]) for i in order),
                "effective_target_tokens": sum(effective[i] for i in order), "maximum_row_exposures": max(visits.values())}
    collections = sorted({rows[i]["collection"] for i in order})
    measured["collection_exposures"] = {c: sum(rows[i]["collection"] == c for i in order) for c in collections}
    measured["collection_input_tokens"] = {c: sum(len(rows[i]["input_ids"]) for i in order
                                                 if rows[i]["collection"] == c) for c in collections}
    measured["schedule_sha256"] = digest(order)
    measured["row_exposures"] = {str(i): n for i, n in sorted(visits.items())}
    measured["collection_target_tokens"] = {c: sum(effective[i] for i in order if rows[i]["collection"] == c) for c in collections}
    measured["collection_target_fractions"] = {c: tokens / measured["effective_target_tokens"]
                                               for c, tokens in measured["collection_target_tokens"].items()}
    measured["admitted_dataset_snapshots"] = snapshots
    for key, value in measured.items():
        require(key not in stage or stage[key] == value, "Stage accounting differs: " + key)
    stage.update(measured)
    boundaries = stage.get("snapshot_after_updates", [])
    require(type(boundaries) is list and all(type(n) is int and 0 < n < measured["optimizer_updates"] for n in boundaries)
            and boundaries == sorted(set(boundaries)), "Invalid pass snapshot boundaries")
    stage["snapshot_after_updates"] = boundaries
    stage["primary_collections"] = primary
    return manifest, rows, [stage]


def verify_native_rows(pack, rows):
    from check_afrique_training_contract import load_tokenizer
    from r7_encoding import encode_complete
    from prepare_understanding_r2 import encode_language
    tokenizer = load_tokenizer(Path(pack) / "tokenizer")
    witnesses = json_rows(Path(pack) / "source-witnesses.jsonl")
    evidence = Counter(chat_rows=0, language_rows=0, multiturn_chat_rows=0,
                       masked_context_tokens=0, masked_context_eos=0,
                       supervised_chat_eos=0, language_final_eos=0, tool_schema_rows=0,
                       tool_call_targets=0, tool_response_context_rows=0)
    for row, witness in zip(rows, witnesses, strict=True):
        encoded = encode_language(tokenizer, witness["text"]) if row["objective"] == "causal_lm" else encode_complete(tokenizer, witness, 4096)
        require(all(encoded[k] == row[k] for k in ("input_ids", "attention_mask", "labels")), "Native encoding differs: " + row["id"])
        if row["objective"] == "causal_lm":
            evidence["language_rows"] += 1
            evidence["language_final_eos"] += int(row["labels"][-1] == 248046)
        else:
            evidence["tool_schema_rows"] += bool(witness.get("tools"))
            evidence["tool_call_targets"] += bool(witness["messages"][-1].get("tool_calls"))
            evidence["tool_response_context_rows"] += any(m["role"] == "tool" for m in witness["messages"])
            evidence["chat_rows"] += 1
            evidence["multiturn_chat_rows"] += int(sum(m["role"] == "assistant" for m in witness["messages"]) > 1)
            evidence["masked_context_tokens"] += row["labels"].count(-100)
            evidence["masked_context_eos"] += sum(t == 248046 and label == -100
                                                   for t, label in zip(row["input_ids"], row["labels"], strict=True))
            evidence["supervised_chat_eos"] += row["labels"].count(248046)
    return {"all_rows_verified": len(rows), "full_history_preserved": True, "truncate": False,
            "basis": "exact_native_reencoding_of_all_packed_rows", "masking_eos": dict(evidence)}


def verify_initialization(artifacts_root):
    root = Path(artifacts_root).resolve(strict=True)
    for name, checksum in SOURCE_FILES.items():
        require(sha(safe_file(root, SOURCE_RUN + "/" + name)) == checksum, "R6 initializer/provenance changed: " + name)
    status = read(root / SOURCE_RUN / "status.json")
    require(status["state"] == "completed" and status["training_completed"] is True
            and status["mode"] == "train" and status["run_id"] == SOURCE_RUN.split("/")[-1]
            and status["execution_sha256"] == SOURCE_FILES["execution.json"]
            and status["consumption_sha256"] == SOURCE_FILES["consumption.json"], "R6 source run not complete/bound")
    receipt = read(root / SOURCE_RUN / "consumption.json")
    require(receipt["training_complete"] is True and receipt["schedule_complete"] is True,
            "R6 source consumption incomplete")
    return root / SOURCE_RUN / "responses-adapter"


def mounted_root(mount):
    root = Path(mount).resolve(strict=True) / VERSION
    require(not root.is_symlink(), "Linked R7 root")
    root.mkdir(exist_ok=True)
    return root


def compare_smoke(folder, contract):
    folder = Path(folder)
    old, status, smoke = (read(folder / name) for name in ("execution.json", "status.json", "smoke.json"))
    require(old["mode"] == "smoke" and status["state"] == "completed" and status["mode"] == "smoke"
            and status["execution_sha256"] == sha(folder / "execution.json") and status["data_sha256"] == contract["data_sha256"], "Smoke binding differs")
    for key in ("version", "data_sha256", "runtime", "frozen_code_hashes", "code_hashes", "reused_code_hashes", "expected_stages", "initialization", "source_files"):
        require(old[key] == contract[key], "Training differs from smoke: " + key)
    require(smoke["passed"] is True and smoke["real_weight_backward"] is True
            and smoke["full_state_recovery"]["equivalent"] is True and smoke["resume_parameter_and_buffer_bitwise_equal"] is True
            and smoke["initial_adapter_values_exact"] is True and smoke["native_encoding"]["all_rows_verified"] > 0, "Incomplete smoke proof")
    require(status["remote_directory"] == "/artifacts/" + VERSION + "/" + run_id(status["run_id"]), "Smoke outside R7")
    return {"run_id": status["run_id"], "execution_sha256": sha(folder / "execution.json"),
            "smoke_sha256": sha(folder / "smoke.json"), "status_sha256": sha(folder / "status.json")}


def prepare(pack, out, mode, source, smoke_directory=None):
    require(mode in BOUNDS, "Unknown execution mode")
    pack, out, source = Path(pack).resolve(strict=True), Path(out).absolute(), Path(source).resolve(strict=True)
    require(not out.exists() and not out.resolve().is_relative_to(pack), "Use a new execution directory outside the pack")
    r2.activate(source)
    from check_afrique_training_contract import ASSET_HASHES
    manifest, rows, stages = verify_pack(pack)
    for name, checksum in ASSET_HASHES.items():
        require(sha(safe_file(pack, "tokenizer/" + name)) == checksum, "Pinned tokenizer changed")
    verify_native_rows(pack, rows)
    contract = {"version": VERSION, "mode": mode, "data_sha256": sha(pack / "manifest.json"), "runtime": RUNTIME,
                "frozen_code_hashes": FROZEN, "code_hashes": {n: sha(ROOT / n) for n in CODE},
                "reused_code_hashes": {n: sha(ROOT / n) for n in REUSED}, "initialization": INITIALIZATION,
                "source_files": SOURCE_FILES, "soft_seconds": BOUNDS[mode][0], "hard_seconds": BOUNDS[mode][1],
                "maximum_gpu_workers": 1, "automatic_retries": 0, "private_research_only": True,
                "resources": {"gpu": "H100", "cpu": 4, "memory_mib": 49152},
                "expected_stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages]}
    if mode == "train":
        require(smoke_directory is not None, "Completed R7 real-weight/resume smoke required")
        contract["smoke"] = compare_smoke(smoke_directory, contract)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r7-package-") as temporary:
        staged = Path(temporary)
        for origin, names, group in ((source, FROZEN, "frozen"), (ROOT, CODE, "r7"), (ROOT, REUSED, "reused")):
            for name in names:
                target = staged / group / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_file(origin, name), target)
        shutil.copytree(pack, staged / "inputs")
        save(staged / "execution.json", contract, exclusive=True)
        verify_execution(staged)
        staged.rename(out)
    return contract


def verify_execution(package):
    package = Path(package)
    value = read(safe_file(package, "execution.json"))
    require(value["version"] == VERSION and value["runtime"] == RUNTIME and value["frozen_code_hashes"] == FROZEN
            and value["initialization"] == INITIALIZATION and value["source_files"] == SOURCE_FILES, "Execution identity differs")
    require(value["mode"] in BOUNDS and (value["soft_seconds"], value["hard_seconds"]) == BOUNDS[value["mode"]]
            and value["maximum_gpu_workers"] == 1 and value["automatic_retries"] == 0 and value["private_research_only"] is True
            and value["resources"] == {"gpu": "H100", "cpu": 4, "memory_mib": 49152}, "Unsafe execution bounds")
    require(set(value["code_hashes"]) == set(CODE) and set(value["reused_code_hashes"]) == set(REUSED), "Incomplete worker packaging")
    for group, files in (("frozen", FROZEN), ("r7", value["code_hashes"]), ("reused", value["reused_code_hashes"])):
        for name, checksum in files.items():
            require(sha(safe_file(package / group, name)) == checksum, "Packaged code changed: " + name)
    require(sha(package / "inputs/manifest.json") == value["data_sha256"], "Execution pack changed")
    _, _, stages = verify_pack(package / "inputs")
    require(value["expected_stages"] == [{k: v for k, v in s.items() if k != "schedule"} for s in stages], "Execution schedule changed")
    if value["mode"] == "train":
        proof = value["smoke"]
        run_id(proof["run_id"])
        require(all(re.fullmatch(r"[a-f0-9]{64}", proof[k]) for k in ("execution_sha256", "smoke_sha256", "status_sha256")), "Malformed smoke binding")
    return value


def launch(package, receipt, profile, authorize, resume_run=None):
    require(authorize, "Explicit --authorize-gpu required; never submit before pack authorization")
    package, receipt = Path(package).resolve(), Path(receipt)
    contract = verify_execution(package)
    require(not receipt.exists(), "Submission exists; poll it, never resubmit")
    os.environ.update(MODAL_PROFILE=profile, GHA_R7_PACKAGE=str(package), GHA_R2_SOURCE_ROOT=str(package / "frozen"))
    import modal
    if resume_run:
        require(contract["mode"] == "train", "Only R7 training can resume")
        claim = modal.Dict.from_name(CLAIMS, create_if_missing=False)[run_id(resume_run)]
        outcome = modal.FunctionCall.from_id(claim["call_id"]).get(timeout=0)
        r2.terminal_pause(claim, outcome, resume_run, sha(package / "execution.json"), contract["data_sha256"])
    identifier = "understanding_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"state": "submitting", "run_id": identifier, "mode": contract["mode"], "profile": profile,
              "execution_sha256": sha(package / "execution.json"), "data_sha256": contract["data_sha256"],
              "package": str(package), "resume_run_id": resume_run, "volume": VOLUME,
              "remote_directory": "/" + VERSION + "/" + identifier}
    save(receipt, record, exclusive=True)
    spec = importlib.util.spec_from_file_location("train_understanding_r7", package / "r7/modal/train/train_understanding_r7.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with modal.enable_output(), module.app.run(detach=True):
        record["modal_image_id"] = module.image.object_id
        save(receipt, record)
        call = (module.smoke if contract["mode"] == "smoke" else module.train).spawn(identifier, record["execution_sha256"], resume_run)
        record.update(state="submitted", call_id=call.object_id)
        save(receipt, record)
    return record


def collect(receipt, out):
    value, out = read(receipt), Path(out)
    require(value["remote_directory"] == "/" + VERSION + "/" + run_id(value["run_id"]), "Receipt path differs")
    os.environ["MODAL_PROFILE"] = value["profile"]
    import modal
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    files = {}
    for name in ("execution.json", "status.json", "loading.json", "smoke.json", "stages.json", "adapters.json", "resume.json", "restore-proof.json", "consumption.json",
                 "consumption-smoke-uninterrupted.json", "consumption-smoke-interrupted.json", "consumption-smoke-resumed.json"):
        try:
            raw = b"".join(volume.read_file(value["remote_directory"] + "/" + name))
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
        require(len(raw) < 10_000_000, "Metadata exceeds bound")
        parsed = json.loads(raw)
        if name == "execution.json":
            require(__import__("hashlib").sha256(raw).hexdigest() == value["execution_sha256"], "Remote execution differs")
        if name == "status.json":
            require(all(parsed[k] == value[k] for k in ("run_id", "execution_sha256", "data_sha256")), "Remote status differs")
        if name.startswith("consumption"):
            require(all(parsed[k] == value[k] for k in ("run_id", "execution_sha256", "data_sha256", "mode")), "Consumption identity differs")
        out.mkdir(parents=True, exist_ok=True)
        if (out / name).exists():
            require((out / name).read_bytes() == raw, "Use a new collection directory")
        else:
            with (out / name).open("xb") as stream:
                stream.write(raw)
        files[name] = sha(out / name)
    return {"directory": str(out), "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "prepare", "launch", "status", "collect"))
    for name in ("pack", "out", "execution", "receipt", "source", "smoke-directory"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=tuple(BOUNDS), default="smoke")
    parser.add_argument("--profile", default="createdliving1000")
    parser.add_argument("--resume-run")
    parser.add_argument("--authorize-gpu", action="store_true")
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()
    if args.action == "verify":
        manifest, rows, stages = verify_pack(args.pack)
        value = {"manifest_sha256": sha(args.pack / "manifest.json"), "rows": len(rows),
                 "stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages],
                 "native_encoding": verify_native_rows(args.pack, rows), "remote_called": False}
    elif args.action == "prepare":
        value = prepare(args.pack, args.out, args.mode, args.source, args.smoke_directory)
    elif args.action == "launch":
        value = launch(args.execution, args.receipt, args.profile, args.authorize_gpu, args.resume_run)
    elif args.action == "status":
        value = r2.status(args.receipt, args.wait)
    else:
        value = collect(args.receipt, args.out)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
