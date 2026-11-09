"""Freeze source-bound translation work without granting training admission."""
import argparse
from collections import Counter
from pathlib import Path

import training_data_inputs as data
import prepare_understanding_r3 as previous
from prepare_understanding_run import HoldoutIndex
from screen_understanding_r5_data import inspect_messages


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def bind_file(path, pins):
    path = Path(path)
    data.require(path.is_file() and not path.is_symlink(), "Use a regular immutable input file")
    checksum = data.sha(path)
    key = str(path.absolute())
    data.require(key not in pins or pins[key] == checksum, "Input changed during preparation")
    pins[key] = checksum
    return checksum


def bound_read(path, pins, rows=False):
    bind_file(path, pins)
    value = previous.read_rows(path) if rows else data.read(path)
    bind_file(path, pins)
    return value


def bound_package(root, pins, required, seal="PREPARED.json"):
    data.require(root.is_dir() and not root.is_symlink(), "Use a regular input directory")
    manifest = bound_read(root / "manifest.json", pins)
    marker = bound_read(root / seal, pins)
    data.require(marker.get("manifest_sha256") == pins[str((root / "manifest.json").absolute())],
                 "Source seal changed")
    artifacts = manifest.get("artifacts")
    data.require(isinstance(artifacts, dict) and set(required) <= set(artifacts),
                 "Required input is not manifest-bound")
    for name, checksum in artifacts.items():
        path = data.safe_path(root, name)
        data.require(data.valid_hash(checksum) and bind_file(path, pins) == checksum,
                     "Source artifact changed")
    return manifest


def recheck(pins):
    for name in list(pins):
        bind_file(Path(name), pins)


def unique_rows(rows, key, integer=False):
    data.require(isinstance(rows, list), "Expected input rows")
    result = {}
    for row in rows:
        data.require(isinstance(row, dict), "Invalid input row")
        identity = row.get(key)
        data.require((type(identity) is int and identity >= 0) if integer else nonblank(identity),
                     "Invalid row identity")
        data.require(identity not in result, "Duplicate input identity")
        result[identity] = row
    return result


def protected_text(root, pins, namespace):
    rows = bound_read(root / "held-content.jsonl", pins, rows=True)
    data.require(rows, "Empty protection inventory")
    unique_rows(rows, "id")
    data.require(all(nonblank(row.get("text")) for row in rows), "Invalid protection text")
    return [(namespace + ":" + row["id"], row["text"]) for row in rows]


def check_source(source):
    data.require(isinstance(source, dict) and nonblank(source.get("id"))
                 and nonblank(source.get("source_id")), "Missing source identity")
    metadata, ref = source.get("metadata"), source.get("original_ref")
    data.require(isinstance(metadata, dict) and nonblank(metadata.get("group_id"))
                 and isinstance(ref, dict) and ref, "Missing source provenance")
    origin = metadata.get("origin")
    data.require(origin in {"source_preserved", "project_authored", "project_authored_development_informed"},
                 "Unsupported source origin; cannot relabel as project-authored")
    for container in (source, metadata, ref):
        data.require("split" not in container or container["split"] == "train", "Protected source split")
    for key in ("family_id", "tree_id", "original_group_id"):
        data.require(key not in metadata or nonblank(metadata[key]), "Invalid source ancestry")
    ancestors = metadata.get("source_ids", [])
    data.require(isinstance(ancestors, list) and all(nonblank(value) for value in ancestors),
                 "Invalid source ancestry")


def source_components(records):
    parents = {}
    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for record in records:
        keys = sorted(previous.source_keys(record))
        for key in keys[1:]:
            parents[root(key)] = root(keys[0])
    return {record["id"]: root(sorted(previous.source_keys(record))[0]) for record in records}


def select(candidates, r2_pack, r4_pack, out, limit=240, shards=4):
    data.require(not out.exists() and not out.is_symlink(), "Use a new immutable work directory")
    data.require(type(limit) is int and type(shards) is int
                 and 0 < limit <= 2131 and 0 < shards <= 16, "Invalid finite work plan")
    pins = {}
    bound_package(r2_pack, pins, {"source-witnesses.jsonl"})
    bound_package(r4_pack, pins, {"held-content.jsonl"})
    actual = unique_rows(bound_read(r2_pack / "source-witnesses.jsonl", pins, rows=True), "id")
    index = HoldoutIndex(protected_text(r4_pack, pins, "r4"))
    proposed = bound_read(candidates, pins, rows=True)
    unique_rows(proposed, "id")
    # Include branches outside the requested subset and before screen exclusions.
    blocked = previous.protected_groups(list(actual.values()), index)
    components = source_components(list(actual.values()))
    sources, excluded = [], []
    for record in proposed:
        data.require(record["id"] in actual, "Candidate missing from source witnesses")
        source = actual[record["id"]]
        data.require(record["record_sha256"] == data.digest(source), "Audit/source mismatch")
        data.require(source["collection"] == "english_retention", "Wrong source collection")
        check_source(source)
        findings = inspect_messages(source["messages"])
        if findings:
            excluded.append({"id": source["id"], "reason": "source_screen_review", "findings": findings})
            continue
        sources.append(source)
    sources.sort(key=lambda r: (-len(r["messages"]), len(" ".join(m["content"] for m in r["messages"]).split()), r["id"]))
    chosen, groups = [], set()
    for source in sources:
        group = components[source["id"]]
        reason = ("protected_source" if source["id"] in blocked else
                  "same_source_tree" if group in groups else
                  "outside_finite_work_plan" if len(chosen) >= limit else None)
        if reason:
            excluded.append({"id": source["id"], "reason": reason, "evidence": blocked.get(source["id"])})
            continue
        groups.add(group)
        chosen.append({"index": len(chosen), "source_record": source,
                       "source_record_sha256": data.digest(source), "training_eligible": False,
                       "review_status": "source_review_and_translation_required"})
    recheck(pins)
    out.mkdir(parents=True)
    data.write_rows(out / "sources.jsonl", chosen)
    data.write_rows(out / "selection-ledger.jsonl", excluded)
    for shard in range(shards):
        data.write_rows(out / ("source-shard-" + str(shard) + ".jsonl"), chosen[shard::shards])
    recheck(pins)
    data.save(out / "manifest.json", {
        "schema": "r5-source-work-plan-v1", "private": True, "training_eligible": False,
        "candidate_file": str(candidates.resolve()), "candidate_sha256": pins[str(candidates.absolute())],
        "r2_manifest_sha256": pins[str((r2_pack / "manifest.json").absolute())],
        "r4_manifest_sha256": pins[str((r4_pack / "manifest.json").absolute())],
        "selected": len(chosen), "source_trees": len(groups),
        "multiturn_histories": sum(len(r["source_record"]["messages"]) > 2 for r in chosen),
        "assistant_occurrences": sum(m["role"] == "assistant" for r in chosen for m in r["source_record"]["messages"]),
        "excluded": dict(Counter(r["reason"] for r in excluded)),
        "fresh_holdout_check": "Required again at admission; selection does not authorize training",
        "inherited_training_clearance": "Not established by current selection; requires a separately bound R2/R3/R4 actual-training audit",
        "pretraining_independence_claimed": False,
        "input_files": pins,
        "artifacts": {p.name: data.sha(p) for p in out.iterdir() if p.is_file()},
        "code_sha256": data.sha(__file__)})
    return data.read(out / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "r2-pack", "r4-pack", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    import json
    print(json.dumps(select(args.candidates, args.r2_pack, args.r4_pack, args.out, args.limit, args.shards), indent=2))
