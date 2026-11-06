#!/usr/bin/env python3
"""Immutable, offline R2 language -> exact R1 meaning -> response data pack."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unicodedata

SOURCE_ROOT = Path(os.environ.get("GHA_R2_SOURCE_ROOT",
    "/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai"))
sys.path[:0] = [str(SOURCE_ROOT / "scripts"), str(SOURCE_ROOT / "modal/train")]
import training_data_inputs as data
import prepare_understanding_run as r1
from check_afrique_training_contract import ASSET_HASHES, EOS, MODEL, REVISION, encode_complete, load_tokenizer

VERSION = "understanding-adaptation-r2"
TARGET_MODULES = r1.TARGET_MODULES
R1_SHA = "bf7b246d1627eadb016105da1d2bb67391d1b3a76a9b66347b812aaef36643ed"
EVAL_SHA = "f48b3fe1c21e1e14d33100e4009e11cf7391e22fadfa73dc75942e01fffe6cec"
PREPARED_SHA = "f565971ddc2d2637d1565dd1794ff523cfb47f86fbe8ddc0c9d762c0ffbb70d6"
RUNTIME_HELPERS = (Path(data.__file__), Path(r1.__file__),
                  SOURCE_ROOT / "scripts/check_afrique_training_contract.py",
                  SOURCE_ROOT / "modal/train/afrique_response_core.py")


def content_units(row):
    """Protect actual fields, including quoted source_text inside chat wrappers."""
    values = [row["text"]] if "text" in row else []
    for message in row.get("messages", []):
        if message["role"] == "system":
            continue
        value = message["content"]
        # Class/entity targets can legitimately recur for different requests.
        # Protect the request and complete example, not the label by itself.
        if row.get("record_type") == "intent_example" and message["role"] == "assistant":
            parsed = data.parse_json(value)
            data.require(isinstance(parsed, dict) and set(parsed) == {"intent", "entities"} and
                         parsed == row.get("understanding"), "Unexpected intent target schema")
            continue
        values.append(value)
        if value.lstrip().startswith("{"):
            try:
                parsed = data.parse_json(value)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                values.extend(parsed[k] for k in ("source_text", "text", "utterance", "transcript", "reference_english")
                              if isinstance(parsed.get(k), str))
    values.append(r1.content(row))
    return sorted({unicodedata.normalize("NFKC", t) for t in values if isinstance(t, str) and t.strip()})


class Protection:
    def __init__(self, ledger, held):
        data.check_ledger(ledger)
        self.parents, self.blocked, self.matches = {}, {}, {}
        for entry in ledger:
            root = self.find(entry["keys"][0])
            for key in entry["keys"][1:]:
                self.parents[self.find(key)] = root
        for entry in ledger:
            if entry["split"] != "train":
                self.blocked[self.find(entry["keys"][0])] = {"reason": "held_source_group", "held_id": entry["source_id"]}
        self.index = r1.HoldoutIndex([(item["id"], unicodedata.normalize("NFKC", item["text"])) for item in held])

    def find(self, key):
        self.parents.setdefault(key, key)
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def register_overlap(self, collection, row):
        keys, _ = data.row_identity(row, collection)
        roots = {self.find(k) for k in keys}
        for text in content_units(row):
            hit = self.index.match(text)
            if hit:
                reason = {"reason": "holdout_content_overlap", **hit, "matched_source_id": row["source_id"]}
                for root in roots:
                    self.blocked.setdefault(root, reason)
                self.matches.setdefault((collection, row["id"]), reason)
                return

    def exclusion(self, collection, row):
        keys, _ = data.row_identity(row, collection)
        for key in keys:
            if self.find(key) in self.blocked:
                return self.blocked[self.find(key)]
        return None


def language_reason(metadata):
    if metadata.get("record_type") not in {"sentence", "utterance"}:
        return "excluded_record_type"
    if metadata.get("origin") not in {"speech_transcript", "compiled_parallel_text", "source_preserved"}:
        return "synthetic_or_unapproved_origin"
    if metadata.get("generated_target") is not False or metadata.get("language") != "tw":
        return "not_source_preserved_twi"
    if metadata["source_id"].startswith(("afriqa:", "pristine")):
        return "excluded_source"
    return None


def encode_language(tokenizer, text, max_length=4096):
    data.require(isinstance(text, str) and text.strip(), "Empty language source")
    data.require(not any(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cs"} or
                         (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text), "Invalid source Unicode")
    ids = tokenizer.encode(text, add_special_tokens=False)
    data.require(ids and not tokenizer.contract_special_ids.intersection(ids), "Reserved language token")
    data.require(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
                 "Language source changed")
    ids = ids + [EOS]
    data.require(len(ids) <= max_length, "overlength_no_truncation")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}


def language_schedule(rows):
    primary = [i for i, r in enumerate(rows) if r["collection"] == "language"]
    english = [i for i, r in enumerate(rows) if r["collection"] == "english_retention"]
    data.require(primary and english, "Empty language stage/replay pool")
    target = sum(r1.effective_tokens(rows[i]) for i in primary)
    random.Random(40).shuffle(english)
    replay, used = [], 0
    for i in english:
        size = r1.effective_tokens(rows[i])
        if used + size <= target // 4:
            replay.append(i)
            used += size
    data.require(replay, "No English language-stage replay")
    order = primary + replay
    random.Random(41).shuffle(order)
    return {"name": "language", "schedule": order, "epochs_primary": 1,
            "unique_primary_rows": len(primary), "english_rows": len(replay),
            "scheduled_sequences": len(order), "optimizer_updates": math.ceil(len(order) / 16),
            "effective_target_tokens": target + used, "primary_target_tokens": target,
            "english_target_tokens": used, "english_target_fraction": used / (target + used),
            "sequence_tokens": sum(len(rows[i]["input_ids"]) for i in order), "maximum_row_exposures": 1,
            "mixing": "At most 20% English causal target tokens; finite no-replacement replay",
            "schedule_sha256": data.digest(order)}


def schedules(rows):
    stages = [language_schedule(rows), r1.stage_schedule(rows, "meaning", 1, 42),
              r1.stage_schedule(rows, "responses", 2, 44)]
    for stage, seed in zip(stages, (40, 42, 44)):
        stage.update(seed=seed, supervision={stage["name"]: "causal_lm_plus_eos" if seed == 40 else "final_assistant_plus_eos",
                                            "english_retention": "final_assistant_plus_eos"})
        data.require(stage["english_target_fraction"] <= .2, "English target-token limit exceeded")
    return stages


def control_identity(rows, stages, old_rows, old_stages):
    retained = [r for r in rows if r["collection"] != "language"]
    data.require(retained == old_rows, "Frozen R1 rows/order changed")
    for actual, original in zip(stages[1:], old_stages):
        actual_ids = [rows[i]["id"] for i in actual["schedule"]]
        expected_ids = [old_rows[i]["id"] for i in original["schedule"]]
        data.require(actual_ids == expected_ids, "R1 scheduled IDs/order changed")
        data.require({k: v for k, v in actual.items() if k not in {"schedule", "schedule_sha256", "seed", "supervision"}} ==
                     {k: v for k, v in original.items() if k not in {"schedule", "schedule_sha256"}}, "R1 exposure changed")
    return {s["name"]: data.digest([rows[i]["id"] for i in s["schedule"]]) for s in stages[1:]}


def evaluation_units(folder, live):
    data.require(data.sha(folder / "manifest.json") == EVAL_SHA, "Original evaluation changed")
    manifest = data.read(folder / "manifest.json")
    for name, checksum in manifest["artifacts"].items():
        data.require(data.sha(data.safe_path(folder, name)) == checksum, "Evaluation artifact changed")
    held = []
    for case in data.read(folder / "cases.json"):
        for i, text in enumerate(content_units(case)):
            held.append({"id": "evaluation:" + case["id"] + ":" + str(i), "text": text})
        if case.get("objective"):
            held.append({"id": "evaluation:" + case["id"] + ":expected", "text":
                         __import__("json").dumps(case["objective"]["expected"], ensure_ascii=False, sort_keys=True)})
    for case in data.read(folder / "legacy-development.json")["cases"]:
        for i, text in enumerate(case.get("turns", [])):
            held.append({"id": "legacy:" + case["id"] + ":" + str(i), "text": text})
    live_files = sorted(live.glob("*.json"))
    data.require(live_files, "Missing live regression evidence")
    for path in live_files:
        spec = data.read(path)["spec"]
        data.require(spec["execution_sha"] == "a9a09a5ba056983f08977308beb6ae1e78667258deda7287f3ce45a7e8d2ff70",
                     "Unexpected live regression model binding")
        for i, text in enumerate(content_units(spec)):
            held.append({"id": "live:" + path.stem + ":" + str(i), "text": text})
    return held


def prepare(prepared, snapshot, assets, r1_pack, evaluation, live_regressions, out):
    import prepare_notebook_training_data as bridge
    data.require(not out.exists(), "Use a fresh immutable output directory")
    data.require(data.sha(prepared / "manifest.json") == PREPARED_SHA, "Unexpected prepared snapshot")
    data.require(data.sha(r1_pack / "manifest.json") == R1_SHA, "Unexpected R1 control")
    print("Verifying complete prepared snapshot and original encodings", flush=True)
    prepared_manifest = bridge.verify_prepared(prepared, snapshot, assets)
    old_manifest, old_rows, old_stages = r1.verify_pack(r1_pack)
    old_witnesses = [r for _, r in data.rows(r1_pack / "source-witnesses.jsonl")]
    wrapped = list(data.iter_snapshot_records(snapshot))
    originals = {(w["collection"], w["batch_id"], w["row"]["id"]): w["row"] for w in wrapped}
    ledger = [r for _, r in data.rows(snapshot / "split-ledger.jsonl")]
    held = evaluation_units(evaluation, live_regressions)
    for w in wrapped:
        if w["row"]["split"] != "train":
            for i, text in enumerate(content_units(w["row"])):
                held.append({"id": "snapshot:" + w["collection"] + ":" + w["row"]["id"] + ":" + str(i), "text": text})
    protection = Protection(ledger, held)
    print("Scanning cross-collection protected content and source groups", flush=True)
    for w in wrapped:
        if w["row"]["split"] == "train":
            protection.register_overlap(w["collection"], w["row"])
    conflicts = []
    for row in old_rows:
        source = originals[(row["collection"], row["batch_id"], row["id"])]
        reason = protection.exclusion(row["collection"], source)
        if reason:
            conflicts.append({"id": row["id"], "collection": row["collection"], **reason})
    if conflicts:
        out.parent.mkdir(parents=True, exist_ok=True)
        data.save(out.parent / "r1-control-conflicts.json", {"count": len(conflicts), "conflicts": conflicts})
        raise ValueError("Protected content conflicts with exact R1 control; see r1-control-conflicts.json")
    rows, witnesses, exclusions, duplicates, seen = [], [], [], [], set()
    for row in bridge.load_partition(prepared, "language", "train").to_pylist():
        metadata = data.parse_json(row["source_metadata"])
        source = originals[(row["collection"], row["batch_id"], row["id"])]
        reason = language_reason(metadata)
        blocked = {"reason": reason} if reason else protection.exclusion("language", source)
        if blocked:
            exclusions.append({"id": row["id"], "source_id": row["source_id"], "source_hash": metadata["source_hash"],
                               "record_type": metadata["record_type"], **blocked})
            continue
        key = data.digest(row["input_ids"])
        if key in seen:
            duplicates.append({"id": row["id"], "source_id": row["source_id"], "source_hash": metadata["source_hash"],
                               "reason": "duplicate_language_encoded_text"})
            continue
        seen.add(key)
        rows.append(row)
        witnesses.append({"id": row["id"], "collection": "language", "source_id": row["source_id"],
                          "text": source["text"], "metadata": metadata, "original_ref": row["original_ref"]})
    rows.extend(old_rows)
    for row, witness in zip(old_rows, old_witnesses):
        source = originals[(row["collection"], row["batch_id"], row["id"])]
        witnesses.append({**witness, **({"text": source["text"]} if "text" in source else {})})
    stages = schedules(rows)
    controls = control_identity(rows, stages, old_rows, old_stages)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r2-pack-") as temp:
        staged = Path(temp)
        # Bundle the original pack verbatim so remote verification has no local-source dependency.
        shutil.copytree(r1_pack, staged / "r1-control")
        shutil.copytree(evaluation, staged / "protection/evaluation")
        shutil.copytree(live_regressions, staged / "protection/live-regressions")
        shutil.copytree(snapshot, staged / "protection/snapshot")
        shutil.copyfile(snapshot / "split-ledger.jsonl", staged / "protection/split-ledger.jsonl")
        data.write_rows(staged / "protection/snapshot-witnesses.jsonl",
                        [w for w in wrapped if w["row"]["split"] != "train" or
                         (w["collection"], w["row"]["id"]) in protection.matches])
        data.write_rows(staged / "protection/held-content.jsonl", held)
        data.write_rows(staged / "rows.jsonl", rows)
        data.write_rows(staged / "source-witnesses.jsonl", witnesses)
        data.write_rows(staged / "excluded.jsonl", exclusions)
        data.write_rows(staged / "duplicate-views.jsonl", duplicates)
        data.save(staged / "stages.json", stages)
        (staged / "tokenizer").mkdir()
        for name in ASSET_HASHES:
            shutil.copyfile(assets / name, staged / "tokenizer" / name)
        data.save(staged / "prepared-manifest.json", prepared_manifest)
        language_rows = [r for r in rows if r["collection"] == "language"]
        types, sources = Counter(), Counter()
        for row in language_rows:
            metadata = data.parse_json(row["source_metadata"])
            types[metadata["record_type"]] += 1
            sources[metadata["source_family"]] += 1
        contract = {"version": VERSION, "model": MODEL, "revision": REVISION,
            "initialization": "fresh_lora_from_original_base_not_r1_adapter",
            "controlled_variable": "one_source_backed_twi_language_pass_before_unchanged_r1_stages",
            "r1_manifest_sha256": R1_SHA, "r1_schedule_ids_sha256": controls,
            "prepared_manifest_sha256": PREPARED_SHA, "snapshot_manifest_sha256": data.sha(snapshot / "manifest.json"),
            "artifacts": {p.relative_to(staged).as_posix(): data.sha(p) for p in sorted(staged.rglob("*")) if p.is_file()},
            "selected_rows": len(rows), "counts": dict(Counter(r["collection"] for r in rows)),
            "source_ids": len({r["source_id"] for r in rows}),
            "language": {"candidate_train_rows": 41316, "source_types": dict(types), "sources": dict(sources),
                         "excluded_reasons": dict(Counter(r["reason"] for r in exclusions)),
                         "duplicate_rows": len(duplicates), "accepted_rows": len(language_rows)},
            "stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages],
            "adaptation": old_manifest["adaptation"],
            "optimization": {**old_manifest["optimization"], "supervision": "per_row_causal_lm_or_final_assistant_plus_eos",
                             "stage_seeds": {"language": 40, "meaning": 42, "responses": 44}},
            "protection": {"snapshot_held_content": sum(r["id"].startswith("snapshot:") for r in held),
                           "evaluation_manifest_sha256": EVAL_SHA, "live_regression_files": len(list(live_regressions.glob("*.json"))),
                           "group_overlap_propagation": True, "exact_r1_control_conflicts": 0,
                           "intent_label_only_targets_not_duplicate_examples": True,
                           "near_match": "NFKC then normalized exact or 8-word shingles, at least 12 words, Jaccard >= 0.85",
                           "limits": "Lexical/source-group screening does not certify semantic independence or unseen pretraining."},
            "runtime_helper_hashes": {p.name: data.sha(p) for p in RUNTIME_HELPERS},
            "builder_sha256": data.sha(__file__), "restrictions": old_manifest["restrictions"],
            "launchable": False, "remaining_launch_checks": ["real_weight_smoke_and_resume", "bound_runtime_code_and_budget"]}
        data.save(staged / "manifest.json", contract)
        data.save(staged / "PREPARED.json", {"manifest_sha256": data.sha(staged / "manifest.json")})
        verify_pack(staged)
        out.mkdir()
        for name in sorted(contract["artifacts"]) + ["manifest.json", "PREPARED.json"]:
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(staged / name)
    return contract


def verify_pack(folder):
    """Self-contained offline checks; original corpus paths are provenance only."""
    contract = data.read(folder / "manifest.json")
    data.require(data.read(folder / "PREPARED.json") == {"manifest_sha256": data.sha(folder / "manifest.json")}, "Changed R2 seal")
    data.require(contract["version"] == VERSION and (contract["model"], contract["revision"]) == (MODEL, REVISION), "Wrong R2 base")
    data.require(contract["builder_sha256"] == data.sha(__file__), "R2 builder changed")
    data.require(contract["runtime_helper_hashes"] == {p.name: data.sha(p) for p in RUNTIME_HELPERS}, "R2 runtime helper changed")
    for name, checksum in contract["artifacts"].items():
        data.require(data.sha(data.safe_path(folder, name)) == checksum, "Changed R2 artifact: " + name)
    actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    data.require(actual_files == set(contract["artifacts"]) | {"manifest.json", "PREPARED.json"} and
                 not any(p.is_symlink() for p in folder.rglob("*")), "Unexpected/missing/linked R2 files")
    data.require(data.sha(folder / "r1-control/manifest.json") == R1_SHA, "Changed R1 control manifest")
    old_manifest, old_rows, old_stages = r1.verify_pack(folder / "r1-control")
    data.require(contract["adaptation"] == old_manifest["adaptation"], "LoRA contract differs")
    data.require(contract["initialization"] == "fresh_lora_from_original_base_not_r1_adapter" and
                 contract["optimization"] == {**old_manifest["optimization"],
                    "supervision": "per_row_causal_lm_or_final_assistant_plus_eos",
                    "stage_seeds": {"language": 40, "meaning": 42, "responses": 44}}, "Initialization/optimization differs")
    rows = [r for _, r in data.rows(folder / "rows.jsonl")]
    witnesses = [r for _, r in data.rows(folder / "source-witnesses.jsonl")]
    data.require(len(rows) == len(witnesses) == contract["selected_rows"], "Incomplete rows/witnesses")
    stages = schedules(rows)
    data.require(stages == data.read(folder / "stages.json"), "R2 schedule changed")
    data.require(contract["stages"] == [{k: v for k, v in s.items() if k != "schedule"} for s in stages], "R2 exposure report changed")
    data.require(control_identity(rows, stages, old_rows, old_stages) == contract["r1_schedule_ids_sha256"], "R1 order proof changed")
    held = [r for _, r in data.rows(folder / "protection/held-content.jsonl")]
    snapshot = folder / "protection/snapshot"
    data.require(data.sha(snapshot / "manifest.json") == contract["snapshot_manifest_sha256"], "Snapshot binding differs")
    snapshot_manifest = data.read(snapshot / "manifest.json")
    for name, checksum in snapshot_manifest["artifacts"].items():
        data.require(data.sha(data.safe_path(snapshot, name)) == checksum, "Snapshot artifact changed")
    data.require(data.sha(folder / "protection/split-ledger.jsonl") == data.sha(snapshot / "split-ledger.jsonl"), "Split ledger differs")
    pointers = {(r["collection"], r["batch_id"], r["id"]): r for _, r in data.rows(snapshot / "records.jsonl")}
    snapshot_witnesses = [w for _, w in data.rows(folder / "protection/snapshot-witnesses.jsonl")]
    expected_held = evaluation_units(folder / "protection/evaluation", folder / "protection/live-regressions")
    seen_held = set()
    for w in snapshot_witnesses:
        key = w["collection"], w["batch_id"], w["row"]["id"]
        data.require(key in pointers and data.digest(w["row"]) == pointers[key]["row_sha256"], "Protection source witness changed")
        if w["row"]["split"] != "train":
            data.require(key not in seen_held, "Duplicate held source witness")
            seen_held.add(key)
            for i, text in enumerate(content_units(w["row"])):
                expected_held.append({"id": "snapshot:" + w["collection"] + ":" + w["row"]["id"] + ":" + str(i), "text": text})
    data.require(seen_held == {k for k, v in pointers.items() if v["split"] != "train"}, "Incomplete held-source coverage")
    data.require(expected_held == held, "Protection content changed")
    protected = Protection([r for _, r in data.rows(folder / "protection/split-ledger.jsonl")], held)
    for w in snapshot_witnesses:
        if w["row"]["split"] == "train":
            protected.register_overlap(w["collection"], w["row"])
    tokenizer = load_tokenizer(folder / "tokenizer")
    keys = set()
    for row, witness in zip(rows, witnesses):
        metadata = data.parse_json(row["source_metadata"])
        data.require(row["split"] == "train" and metadata == witness["metadata"] and
                     (row["id"], row["collection"], row["source_id"], row["original_ref"]) ==
                     (witness["id"], witness["collection"], witness["source_id"], witness["original_ref"]), "Source witness differs")
        source = {**metadata, **{k: witness[k] for k in ("text", "messages") if k in witness}}
        data.require(data.digest(source) == row["original_ref"]["row_sha256"], "Source row reconstruction changed")
        pointer = pointers[row["collection"], row["batch_id"], row["id"]]
        data.require(pointer["row_sha256"] == row["original_ref"]["row_sha256"], "Source snapshot pointer differs")
        protected.register_overlap(row["collection"], source)
        data.require(protected.exclusion(row["collection"], source) is None, "Holdout overlap in admitted row")
        if row["collection"] == "language":
            data.require(language_reason(metadata) is None and row["objective"] == "causal_lm", "Ineligible language source")
            expected = encode_language(tokenizer, witness["text"])
        else:
            expected = encode_complete(tokenizer, source, 4096)
        data.require(all(row[k] == expected[k] for k in ("input_ids", "attention_mask", "labels")), "Native encoding mismatch")
        data.require(row["sequence_tokens"] == len(row["input_ids"]) and
                     row["target_tokens"] == sum(v != -100 for v in row["labels"]), "Stored token counts differ")
        key = row["collection"], data.digest(row["input_ids"])
        data.require(key not in keys, "Within-collection duplicate")
        keys.add(key)
    data.require(dict(Counter(r["collection"] for r in rows)) == contract["counts"], "Collection counts changed")
    exclusions = [r for _, r in data.rows(folder / "excluded.jsonl")]
    duplicates = [r for _, r in data.rows(folder / "duplicate-views.jsonl")]
    language_rows = [r for r in rows if r["collection"] == "language"]
    data.require(len(language_rows) + len(exclusions) + len(duplicates) == contract["language"]["candidate_train_rows"],
                 "Incomplete language accounting")
    data.require(contract["language"]["accepted_rows"] == len(language_rows) and
                 contract["language"]["excluded_reasons"] == dict(Counter(r["reason"] for r in exclusions)) and
                 contract["language"]["duplicate_rows"] == len(duplicates), "Language accounting report changed")
    return contract, rows, stages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "snapshot", "assets", "r1-pack", "evaluation", "live-regressions", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.prepared, args.snapshot, args.assets, args.r1_pack, args.evaluation, args.live_regressions, args.out)
    print(__import__("json").dumps({k: result[k] for k in ("counts", "language", "stages", "protection")}, indent=2))
