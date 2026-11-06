#!/usr/bin/env python3
"""Freeze the measured meaning -> response experiment without mutating its inputs."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re
import shutil
import unicodedata

import training_data_inputs as data
from check_afrique_training_contract import ASSET_HASHES, MODEL, REVISION

VERSION = "understanding-adaptation-r1"
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                  "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"]


def effective_tokens(row):
    labels = row["labels"]
    data.require(len(labels) == len(row["input_ids"]) and len(labels) > 1, "Invalid encoded row")
    value = sum(t != -100 for t in labels[1:])
    data.require(value > 0, "No causal targets")
    return value


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def content(row):
    return "\n".join(m["content"] for m in row.get("messages", []) if m["role"] != "system") or row.get("text", "")


def shingles(text):
    words = normalized(text).split()
    return {tuple(words[i:i + 8]) for i in range(len(words) - 7)} if len(words) >= 12 else set()


class HoldoutIndex:
    """Exact normalized content plus conservative lexical near-match protection."""
    def __init__(self, held):
        self.exact, self.postings, self.grams = {}, defaultdict(set), {}
        for key, text in held:
            self.exact.setdefault(normalized(text), key)
            grams = shingles(text)
            self.grams[key] = grams
            for gram in grams:
                self.postings[gram].add(key)

    def match(self, text):
        exact = self.exact.get(normalized(text))
        if exact is not None:
            return {"kind": "normalized_exact", "held_id": exact}
        grams = shingles(text)
        candidates = set().union(*(self.postings.get(g, set()) for g in grams)) if grams else set()
        for key in sorted(candidates):
            other = self.grams[key]
            ratio = len(grams & other) / len(grams | other)
            if ratio >= .85:
                return {"kind": "8_word_shingle_jaccard", "held_id": key, "similarity": ratio}
        return None


def select(rows):
    selected, excluded = [], []
    for row in rows:
        metadata = data.parse_json(row["source_metadata"])
        collection = row["collection"]
        include = (collection == "meaning" and metadata.get("record_type") in
                   {"sentence", "utterance", "intent_example"}) or (
                       collection == "responses" and metadata.get("task_kind") in
                       {"twi_prose", "language_exercise"}) or collection == "english_retention"
        if include:
            data.require(row["split"] == "train", "Only training records can enter a schedule")
            data.require(not row["source_id"].startswith("afriqa:"), "AfriQA is held out")
            selected.append(row)
        else:
            excluded.append({"id": row["id"], "collection": collection,
                             "reason": "objective_outside_this_run", "metadata_type": metadata.get("record_type", metadata.get("task_kind"))})
    return selected, excluded


def stage_schedule(rows, primary, epochs, seed):
    """Full primary passes, with one-pass English replay near 20% target tokens."""
    primary_indices = [i for i, r in enumerate(rows) if r["collection"] == primary]
    english = [i for i, r in enumerate(rows) if r["collection"] == "english_retention"]
    data.require(primary_indices and english and epochs in (1, 2), "Empty stage or invalid epochs")
    primary_schedule = primary_indices * epochs
    target = sum(effective_tokens(rows[i]) for i in primary_schedule)
    random.Random(seed).shuffle(english)
    budget, used, replay = round(target / 4), 0, []
    for i in english:
        size = effective_tokens(rows[i])
        if used + size <= budget:
            replay.append(i)
            used += size
    data.require(replay, "Empty English retention allocation")
    order = primary_schedule + replay
    random.Random(seed + 1).shuffle(order)
    counts = Counter(order)
    total = sum(effective_tokens(rows[i]) for i in order)
    return {"name": primary, "schedule": order, "epochs_primary": epochs,
            "unique_primary_rows": len(primary_indices), "english_rows": len(replay),
            "scheduled_sequences": len(order), "optimizer_updates": math.ceil(len(order) / 16),
            "effective_target_tokens": total, "primary_target_tokens": target,
            "english_target_tokens": used, "english_target_fraction": used / total,
            "sequence_tokens": sum(len(rows[i]["input_ids"]) for i in order),
            "maximum_row_exposures": max(counts.values()),
            "mixing": "20% English target-token budget; finite no-replacement replay per stage",
            "schedule_sha256": data.digest(order)}


def prepare(prepared, snapshot, assets, out):
    import prepare_notebook_training_data as bridge
    data.require(not out.exists(), "Use a new run package directory")
    manifest = bridge.verify_prepared(prepared, snapshot, assets)
    all_rows = [r for p in manifest["partitions"] if p["split"] == "train"
                for r in bridge.load_partition(prepared, p["collection"], "train").to_pylist()]
    selected, excluded = select(all_rows)
    wrapped = list(data.iter_snapshot_records(snapshot))
    originals = {(w["collection"], w["batch_id"], w["row"]["id"]): w["row"] for w in wrapped}
    held = [(w["collection"] + ":" + w["row"]["id"], content(w["row"])) for w in wrapped if w["row"]["split"] != "train"]
    holdouts = HoldoutIndex(held)
    rows, witnesses, duplicates, keys = [], [], [], set()
    for row in selected:
        source = originals[(row["collection"], row["batch_id"], row["id"])]
        overlap = holdouts.match(content(source))
        if overlap:
            excluded.append({"id": row["id"], "collection": row["collection"], "reason": "holdout_content_overlap", **overlap})
            continue
        key = row["collection"], data.digest(row["input_ids"])
        if key in keys:
            duplicates.append({"id": row["id"], "collection": row["collection"]})
            continue
        keys.add(key)
        rows.append(row)
        witnesses.append({"id": row["id"], "collection": row["collection"],
                          "source_id": row["source_id"], "messages": source["messages"],
                          "metadata": data.parse_json(row["source_metadata"]), "original_ref": row["original_ref"]})
    stages = [stage_schedule(rows, "meaning", 1, SEED), stage_schedule(rows, "responses", 2, SEED + 2)]
    out.mkdir(parents=True)
    data.write_rows(out / "rows.jsonl", rows)
    data.write_rows(out / "source-witnesses.jsonl", witnesses)
    data.write_rows(out / "excluded.jsonl", excluded)
    data.write_rows(out / "duplicate-views.jsonl", duplicates)
    data.save(out / "stages.json", stages)
    (out / "tokenizer").mkdir()
    for name in ASSET_HASHES:
        shutil.copyfile(assets / name, out / "tokenizer" / name)
    data.save(out / "prepared-manifest.json", manifest)
    files = {str(p.relative_to(out)): data.sha(p) for p in out.rglob("*") if p.is_file()}
    contract = {"version": VERSION, "model": MODEL, "revision": REVISION,
                "prepared_manifest_sha256": data.sha(prepared / "manifest.json"),
                "snapshot_manifest_sha256": data.sha(snapshot / "manifest.json"),
                "artifacts": files, "selected_rows": len(rows),
                "counts": dict(Counter(r["collection"] for r in rows)),
                "source_ids": len({r["source_id"] for r in rows}),
                "stages": [{k: v for k, v in s.items() if k != "schedule"} for s in stages],
                "protection": {"selected_snapshot_split_ledger_verified": True,
                    "additional_held_content_records": len(held), "excluded_rows": len(excluded),
                    "duplicate_views_removed": len(duplicates),
                    "near_match": "8-word shingles, at least 12 words, Jaccard >= 0.85",
                    "limits": "Lexical screening cannot certify semantic independence or unseen base-model pretraining."},
                "adaptation": {"method": "lora", "rank": 16, "alpha": 32, "dropout": .05,
                    "target_modules": TARGET_MODULES, "frozen_base": True, "new_vocabulary": False},
                "optimization": {"learning_rate": 5e-5, "grad_accum": 16, "warmup_fraction": .05,
                    "checkpoint_every": 25, "seed": SEED, "precision": "bf16_base_fp32_adapters",
                    "packing": False, "truncate": False, "reset_optimizer_between_stages": True,
                    "supervision": "final_assistant_plus_eos", "normalization": "effective_causal_target_tokens"},
                "restrictions": ["private_noncommercial_research", "source_restrictions_inherited",
                    "not_clinically_certified", "not_publication_or_production_approval"],
                "new_data_policy": "New prepared batches require a new run package; never mutate active run inputs.",
                "launchable": False,
                "remaining_launch_checks": ["frozen_evaluation", "real_weight_smoke_and_resume", "bound_runtime_code_and_budget"]}
    data.save(out / "manifest.json", contract)
    data.save(out / "PREPARED.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return contract


def verify_pack(folder):
    contract = data.read(folder / "manifest.json")
    data.require(data.read(folder / "PREPARED.json") == {"manifest_sha256": data.sha(folder / "manifest.json")}, "Changed run manifest")
    data.require(contract["version"] == VERSION and (contract["model"], contract["revision"]) == (MODEL, REVISION), "Wrong run base")
    for name, checksum in contract["artifacts"].items():
        data.require(data.sha(data.safe_path(folder, name)) == checksum, "Changed run artifact: " + name)
    rows = [r for _, r in data.rows(folder / "rows.jsonl")]
    data.require(len(rows) == contract["selected_rows"] and all(r["split"] == "train" for r in rows), "Invalid run rows")
    stages = data.read(folder / "stages.json")
    expected = [stage_schedule(rows, "meaning", 1, SEED), stage_schedule(rows, "responses", 2, SEED + 2)]
    data.require(stages == expected, "Schedule mismatch")
    data.require(contract["stages"] == [{k: v for k, v in s.items() if k != "schedule"} for s in stages], "Changed schedule report")
    return contract, rows, stages


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.prepared, a.snapshot, a.assets, a.out)
    verify_pack(a.out)
    print(__import__("json").dumps({k: result[k] for k in ("counts", "source_ids", "stages", "protection")}, indent=2))
