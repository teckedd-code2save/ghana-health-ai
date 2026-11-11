#!/usr/bin/env python3
"""Sealed R4 continuation from final R3; offline verification, explicit launch.

Contained versioned copy of the R3 runner: keep its sealed-pack, frozen-core,
checkpoint and bounded-execution contracts without mutating R3 module globals.
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

ROOT = Path(__file__).resolve().parents[1]
VERSION = "understanding-adaptation-r4"
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
SOURCE_RUN = "understanding-adaptation-r3/understanding_20261106T062502Z"
INITIAL_SHA = "5cc0537b21674231b640ae9109a08ffa6fab426c80ac1969705dfac9c9e6b874"
INITIALIZATION = {"adapter_directory": "/artifacts/" + SOURCE_RUN + "/responses-adapter",
                  "adapter_weights_sha256": INITIAL_SHA, "new_optimizer": True, "new_scheduler": True}
# Completion metadata hashes measured from the original repo's
# tmp/understanding-r3-20261106/train-v1/completed, not a reconstructed run.
SOURCE_FILES = {
    "responses-adapter/adapter_model.safetensors": INITIAL_SHA,
    "responses-adapter/adapter_config.json": "9a632e21e3d927d5cef610f5060961e8b18d2657ca22a42792a3821b1d514fa8",
    "status.json": "999c7c1a3fbd183c6f176386c2a3d5471a78cf5614ff4082ffefe706e2dafe45",
    "execution.json": "878f630c4501d5e05694befd8ad413d997b8bfd1a3030e5bc3e6e3fde7bced74",
    "adapters.json": "a79787a8449367eaececa7bb62fbfd11fba2289e8c5531fef92aeee0c17eed11",
}
CODE = ("scripts/run_understanding_r4.py", "modal/train/train_understanding_r4.py")
REUSED = ("scripts/run_understanding_r2.py", "scripts/prepare_understanding_r2.py")
VOLUME, CLAIMS = r2.VOLUME, "ghana-understanding-r4-invocations"
BOUNDS = {"smoke": (780, 900), "train": (10500, 10800)}
OPT_KEYS = ("learning_rate", "grad_accum", "warmup_fraction", "checkpoint_every")


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
            require(isinstance(record.get("messages"), list) and messages == record["messages"][:len(messages)], "Chat view is not an admitted exact prefix")
        else:
            require(witness["text"] == record.get("text"), "Language text differs from admitted record")
    return snapshots


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
    return {"version": "r4-consumption-v1", "mode": mode, "scope": "cumulative_committed_schedule_prefix",
            "execution_sha256": execution_sha, "data_sha256": data_sha, "checkpoint": checkpoint,
            "completed_updates": updates, "schedule_complete": cursor == len(order),
            "training_complete": mode == "train" and cursor == len(order),
            "inventory": {"packed_rows": len(rows), "packed_rows_by_class": dict(Counter(r["collection"] for r in rows)),
                          "admitted_snapshots": snapshots, "archive_counts_included": False},
            "planned": {"row_exposures": len(order), "unique_rows": len(set(order))},
            "consumed": {**totals, "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
                         "by_dataset": [{"dataset_id": k[0], "dataset_version": k[1], **dict(v)} for k, v in sorted(by_dataset.items())]},
            "rows": per_row}


def verify_pack(folder):
    """Verify bytes, full-history structure, masks and exact finite exposure counts."""
    folder = Path(folder).resolve(strict=True)
    manifest = read(safe_file(folder, "manifest.json"))
    require(read(safe_file(folder, "PREPARED.json")) == {"manifest_sha256": sha(folder / "manifest.json")}, "Pack seal differs")
    require(manifest["version"] == VERSION and manifest["model"] == MODEL and manifest["revision"] == REVISION
            and manifest["initialization"] == INITIALIZATION, "Wrong R4 foundation/initialization")
    require(manifest["private_research_only"] is True and manifest["production_eligible"] is False, "Research-only pack required")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked pack artifact")
    require(actual == set(manifest["artifacts"]) | {"manifest.json", "PREPARED.json"}, "Unbound or missing artifact")
    required = {"rows.jsonl", "source-witnesses.jsonl", "stages.json", "evaluation-manifest.json", "overlap-audit.json", "source-manifests.json"}
    required |= {"tokenizer/" + n for n in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json")}
    require(required <= set(manifest["artifacts"]), "Missing required pack evidence")
    for name, checksum in manifest["artifacts"].items():
        require(sha(safe_file(folder, name)) == checksum, "Pack artifact changed: " + name)
    require(read(folder / "overlap-audit.json")["overlap_free"] is True, "Held-out overlap not cleared")
    read(folder / "evaluation-manifest.json")
    opt = manifest["optimization"]
    require(type(opt["learning_rate"]) in (float, int) and 0 < opt["learning_rate"] <= 5e-5, "Unsafe learning rate")
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
        require(key not in seen and re.fullmatch(r"[a-z][a-z0-9_-]*", row["collection"]), "Duplicate/unsafe row")
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
            require(labels == ids and isinstance(witness["text"], str) and witness["text"].strip(), "Invalid language target")
        else:
            require(row["objective"] == "final_assistant", "Unknown objective")
            messages = witness["messages"]
            require(isinstance(messages, list) and messages, "Missing complete history")
            start = int(messages[0].get("role") == "system")
            require((len(messages) - start) % 2 == 0 and len(messages) - start >= 2, "Incomplete history")
            for j, message in enumerate(messages):
                role = "system" if start and j == 0 else "user" if (j - start) % 2 == 0 else "assistant"
                require(set(message) == {"role", "content"} and message["role"] == role
                        and isinstance(message["content"], str) and message["content"].strip(), "Invalid history roles/text")
            first = next(j for j, t in enumerate(labels) if t != -100)
            require(first > 0 and labels[:first] == [-100] * first and labels[first:] == ids[first:]
                    and len(labels[first:]) > 1 and labels[first:].count(248046) == 1, "Invalid final-assistant supervision")
            if len(messages) - start >= 4:
                multiturn.add(i)
    snapshots = admitted_sources(folder, rows, witnesses, manifest["artifacts"])
    stages = read(folder / "stages.json")
    require(isinstance(stages, list) and len(stages) == 1 and stages[0]["name"] == "responses", "Expected one continuous response schedule")
    stage = dict(stages[0])
    order = stage["schedule"]
    require(type(order) is list and 1 <= len(order) <= 10000
            and all(type(i) is int and 0 <= i < len(rows) for i in order), "Invalid/beyond-budget schedule")
    visits = Counter(order)
    require(max(visits.values()) <= 4, "More than four row exposures")
    primary = stage.get("primary_collections", ["responses"])
    require(isinstance(primary, list) and primary and all(isinstance(x, str) for x in primary), "Invalid primary collections")
    require(any(rows[i]["collection"] in primary for i in order) and any(rows[i]["collection"] not in primary for i in order)
            and any(i in multiturn and rows[i]["collection"] in primary for i in order), "Scheduled conversation and retention required")
    measured = {"scheduled_sequences": len(order), "optimizer_updates": math.ceil(len(order) / opt["grad_accum"]),
                "sequence_tokens": sum(len(rows[i]["input_ids"]) for i in order),
                "effective_target_tokens": sum(effective[i] for i in order), "maximum_row_exposures": max(visits.values())}
    collections = sorted({rows[i]["collection"] for i in order})
    measured["collection_exposures"] = {c: sum(rows[i]["collection"] == c for i in order) for c in collections}
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
    from check_afrique_training_contract import encode_complete, load_tokenizer
    from prepare_understanding_r2 import encode_language
    tokenizer = load_tokenizer(Path(pack) / "tokenizer")
    witnesses = json_rows(Path(pack) / "source-witnesses.jsonl")
    evidence = Counter(chat_rows=0, language_rows=0, multiturn_chat_rows=0,
                       masked_context_tokens=0, masked_context_eos=0,
                       supervised_chat_eos=0, language_final_eos=0)
    for row, witness in zip(rows, witnesses, strict=True):
        encoded = encode_language(tokenizer, witness["text"]) if row["objective"] == "causal_lm" else encode_complete(tokenizer, witness, 4096)
        require(all(encoded[k] == row[k] for k in ("input_ids", "attention_mask", "labels")), "Native encoding differs: " + row["id"])
        if row["objective"] == "causal_lm":
            evidence["language_rows"] += 1
            evidence["language_final_eos"] += int(row["labels"][-1] == 248046)
        else:
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
        require(sha(safe_file(root, SOURCE_RUN + "/" + name)) == checksum, "R3 initializer/provenance changed: " + name)
    status = read(root / SOURCE_RUN / "status.json")
    require(status["state"] == "completed" and status["training_completed"] is True, "R3 source run not complete")
    return root / SOURCE_RUN / "responses-adapter"


def mounted_root(mount):
    root = Path(mount).resolve(strict=True) / VERSION
    require(not root.is_symlink(), "Linked R4 root")
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
    require(status["remote_directory"] == "/artifacts/" + VERSION + "/" + run_id(status["run_id"]), "Smoke outside R4")
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
    contract = {"version": VERSION, "mode": mode, "data_sha256": sha(pack / "manifest.json"), "runtime": RUNTIME,
                "frozen_code_hashes": FROZEN, "code_hashes": {n: sha(ROOT / n) for n in CODE},
                "reused_code_hashes": {n: sha(ROOT / n) for n in REUSED}, "initialization": INITIALIZATION,
                "source_files": SOURCE_FILES, "soft_seconds": BOUNDS[mode][0], "hard_seconds": BOUNDS[mode][1],
                "maximum_gpu_workers": 1, "automatic_retries": 0, "private_research_only": True,
                "resources": {"gpu": "H100", "cpu": 4, "memory_mib": 49152},
                "expected_stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages]}
    if mode == "train":
        require(smoke_directory is not None, "Completed R4 real-weight/resume smoke required")
        contract["smoke"] = compare_smoke(smoke_directory, contract)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r4-package-") as temporary:
        staged = Path(temporary)
        for origin, names, group in ((source, FROZEN, "frozen"), (ROOT, CODE, "r4"), (ROOT, REUSED, "reused")):
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
    for group, files in (("frozen", FROZEN), ("r4", value["code_hashes"]), ("reused", value["reused_code_hashes"])):
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
    os.environ.update(MODAL_PROFILE=profile, GHA_R4_PACKAGE=str(package), GHA_R2_SOURCE_ROOT=str(package / "frozen"))
    import modal
    if resume_run:
        require(contract["mode"] == "train", "Only R4 training can resume")
        claim = modal.Dict.from_name(CLAIMS, create_if_missing=False)[run_id(resume_run)]
        outcome = modal.FunctionCall.from_id(claim["call_id"]).get(timeout=0)
        r2.terminal_pause(claim, outcome, resume_run, sha(package / "execution.json"), contract["data_sha256"])
    identifier = "understanding_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"state": "submitting", "run_id": identifier, "mode": contract["mode"], "profile": profile,
              "execution_sha256": sha(package / "execution.json"), "data_sha256": contract["data_sha256"],
              "package": str(package), "resume_run_id": resume_run, "volume": VOLUME,
              "remote_directory": "/" + VERSION + "/" + identifier}
    save(receipt, record, exclusive=True)
    spec = importlib.util.spec_from_file_location("train_understanding_r4", package / "r4/modal/train/train_understanding_r4.py")
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
                 "stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages], "remote_called": False}
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
