#!/usr/bin/env python3
"""Read a published sealed pack and write a separate descriptive exposure profile."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

VERSION = "r7-pack-profile-v1"
ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = ("responses", "tools", "health", "commerce", "meaning", "language", "english_retention")
TOKEN_FIELDS = ("input_tokens", "masked_context_tokens", "target_label_tokens", "effective_target_tokens")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            require(line.strip(), "Blank JSONL row")
            yield json.loads(line)


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def save_rows(path, values):
    with Path(path).open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def safe_path(folder, name):
    path = folder / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts
            and path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(), "Unsafe artifact path")
    return path


def sealed(folder, seal_name):
    folder = Path(folder).resolve(strict=True)
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Symlink in sealed input")
    manifest = read(folder / "manifest.json")
    require(read(folder / seal_name)["manifest_sha256"] == sha(folder / "manifest.json"), "Manifest seal differs")
    for name, checksum in manifest["artifacts"].items():
        require(sha(safe_path(folder, name)) == checksum, "Sealed artifact hash differs: " + name)
    return manifest


class Families:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        node = key
        while self.parent[node] != node:
            node = self.parent[node]
        while key != node:
            parent = self.parent[key]
            self.parent[key] = node
            key = parent
        return node

    def connect(self, keys):
        roots = sorted({self.root(k) for k in keys})
        for root in roots[1:]:
            self.parent[root] = roots[0]


def method(witness):
    metadata = witness["metadata"]
    if metadata.get("prior_version") in {"r5", "r6"}:
        return "inherited_replay"
    if metadata.get("task") == "source_preserved_bilingual_alignment":
        return "source_parallel_bidirectional"
    if metadata.get("origin") == "programmatically_augmented_original_parallel_sentences":
        return "controlled_alignment_augmentation"
    if metadata.get("admission_scope") == "executed_simulated_english_actions":
        return "controlled_simulated_tools"
    return "unknown"


def target_payload(witness):
    if witness["collection"] == "language":
        return {"text": witness["text"]}
    target = witness["messages"][-1]
    require(target["role"] == "assistant", "Target is not final assistant")
    return ({"assistant": target} if target.get("tool_calls") else {"text": target["content"]})


def language(witness):
    metadata = witness["metadata"]
    prior = metadata.get("prior_full_record", {})
    target = witness.get("messages", [{}])[-1]
    source = metadata.get("instruction_language", metadata.get("source_language", "unknown"))
    if target.get("tool_calls"):
        return source, "structured_tool_call", "native_tool_call_not_prose"
    if metadata.get("record_type") == "intent_example":
        return metadata.get("language", source), "structured_intent_label", "record_language_describes_input_not_label"
    if metadata.get("target_language"):
        return source, metadata["target_language"], "explicit_target_language"
    if metadata.get("translation_status") == "english_only_no_twi_claim":
        return "en", "en", "explicit_english_only_tool_admission"
    for label, obj in (("metadata", metadata), ("prior_full_record", prior)):
        if obj.get("language"):
            return source, obj["language"], label + "_record_language_not_segment_verified"
    return source, "unknown", "no_explicit_language_label_no_id_or_script_inference"


def provenance(witness):
    metadata, ref = witness["metadata"], witness["original_ref"]
    source_ids = {witness["source_id"]}
    files, pairs = set(), set()
    refs = [ref, ref.get("current", {}), ref.get("previous", {})]
    pair = metadata.get("alignment_pair", {})
    if pair:
        pairs.add(pair["source_id"])
        refs.append(pair.get("original_refs", {}))
    if metadata.get("task") == "bilingual_alignment":
        pairs.add(witness["source_id"])
    for reference in refs:
        candidate_id = reference.get("inventory_candidate_id", "")
        if candidate_id.startswith("parallel:"):
            source_id = candidate_id.removeprefix("parallel:")
            source_ids.add(source_id)
            pairs.add(source_id)
        source = reference.get("source", {})
        if source.get("file_sha256"):
            files.add((source.get("file", "unknown"), source["file_sha256"]))
    if metadata.get("source_file_sha256"):
        files.add((metadata.get("source_file", "unknown"), metadata["source_file_sha256"]))
    aliases = source_ids | set(metadata.get("group_aliases", []))
    if metadata.get("group_id"):
        aliases.add(metadata["group_id"])
    require(all(isinstance(k, str) and k for k in aliases), "Invalid declared source alias")
    upstream = ref["r7_upstream_admission"]
    parent = metadata["r7_view"]["parent_record_id"]
    return {"source_ids": sorted(source_ids), "pair_source_ids": sorted(pairs), "aliases": sorted(aliases),
            "source_files": sorted(files), "upstream_record_key": digest([upstream["release_manifest_sha256"],
                upstream["dataset_id"], upstream["dataset_version"], parent]),
            "upstream_record_id": parent, "upstream_dataset_id": upstream["dataset_id"],
            "upstream_dataset_version": upstream["dataset_version"],
            "registered_group_id": metadata.get("group_id", "unknown")}


def describe(row, witness, presentations):
    require(all(row[k] == witness[k] for k in ("id", "collection", "source_id", "dataset_ref")), "Row/witness identity differs")
    ids, labels, mask = row["input_ids"], row["labels"], row["attention_mask"]
    require(len(ids) == len(labels) == len(mask) and mask == [1] * len(ids)
            and all(label == -100 or label == token for token, label in zip(ids, labels)), "Invalid token arrays")
    target_count = sum(v != -100 for v in labels)
    require(target_count == row["target_tokens"] and len(ids) == row["sequence_tokens"], "Declared token count differs")
    source_lang, target_lang, language_scope = language(witness)
    metadata = witness["metadata"]
    return {"id": row["id"], "collection": row["collection"], "source_id": row["source_id"], **provenance(witness),
            "method": method(witness), "origin": metadata.get("origin", "unknown"),
            "prior_version": metadata.get("prior_version", "not_replay"),
            "record_type": metadata.get("record_type", metadata.get("prior_full_record", {}).get("task_kind", "unknown")),
            "generated_target_declared": metadata.get("generated_target", metadata.get("generated_twi_target", "unknown")),
            "quality_tier": metadata.get("quality_tier", metadata.get("prior_quality_tier", "unknown")),
            "transformation": metadata.get("transformation", "none"), "source_language_declared": source_lang,
            "target_language_declared": target_lang, "language_scope": language_scope,
            "objective": row["objective"], "scheduled_presentations": presentations,
            "input_tokens": len(ids), "target_label_tokens": target_count,
            "effective_target_tokens": sum(v != -100 for v in labels[1:]), "masked_context_tokens": len(ids) - target_count,
            "target_payload_sha256": digest(target_payload(witness)),
            "encoded_example_sha256": digest([row["objective"], ids, labels]),
            "supervised_tokens_sha256": digest([v for v in labels if v != -100])}


def aggregate(views, families):
    return {"retained_views": len(views), "unique_target_source_id_values": len({v["source_id"] for v in views}),
            "unique_source_id_values_including_context_refs": len({k for v in views for k in v["source_ids"]}),
            "unique_explicit_original_pair_source_ids": len({k for v in views for k in v["pair_source_ids"]}),
            "unique_upstream_admitted_records": len({v["upstream_record_key"] for v in views}),
            "unique_registered_group_id_values": len({v["registered_group_id"] for v in views}),
            "connected_declared_alias_groups": len({families.root(v["source_id"]) for v in views}),
            "unique_declared_source_file_hashes": len({k[1] for v in views for k in v["source_files"]}),
            "unique_exact_target_payloads": len({v["target_payload_sha256"] for v in views}),
            "scheduled_presentations": sum(v["scheduled_presentations"] for v in views),
            "tokens_one_pass": {k: sum(v[k] for v in views) for k in TOKEN_FIELDS},
            "tokens_scheduled": {k: sum(v[k] * v["scheduled_presentations"] for v in views) for k in TOKEN_FIELDS}}


def exposure_groups(views, keys):
    groups = defaultdict(list)
    for view in views:
        groups[tuple(view[k] for k in keys)].append(view)
    for identity, group in sorted(groups.items()):
        yield {**dict(zip(keys, identity)), "retained_views": len(group),
               "scheduled_presentations": sum(v["scheduled_presentations"] for v in group),
               "effective_target_tokens_scheduled": sum(v["effective_target_tokens"] * v["scheduled_presentations"] for v in group),
               "methods": dict(Counter(v["method"] for v in group)), "collections": dict(Counter(v["collection"] for v in group)),
               "target_source_ids": sorted({v["source_id"] for v in group}),
               "unique_upstream_records": len({v["upstream_record_key"] for v in group}),
               "view_ids": [v["id"] for v in group]}


def report_markdown(summary):
    lines = ["# R7 Pack Profile", "", "Descriptive source and exposure accounting, not a quality certificate or training-completion report.", "",
             "Pack manifest SHA256: `" + summary["pack_manifest_sha256"] + "`", "",
             "| Collection | Source ID values | Admitted records represented | Training views | Input tokens | Effective target tokens |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in summary["by_collection"].items():
        tokens = value["tokens_scheduled"]
        lines.append(f"| {name} | {value['unique_target_source_id_values']:,} | {value['unique_upstream_admitted_records']:,} | "
                     f"{value['retained_views']:,} | {tokens['input_tokens']:,} | {tokens['effective_target_tokens']:,} |")
    total = summary["total"]
    lines += ["", f"Total: {total['retained_views']:,} retained training views from "
              f"{total['unique_upstream_admitted_records']:,} represented admission records; "
              f"{total['unique_target_source_id_values']:,} target source-ID values and "
              f"{total['connected_declared_alias_groups']:,} connected declared alias groups.", "",
              "## Authoring Routes", "", "| Route | Views | Scheduled effective target tokens |", "|---|---:|---:|"]
    for name, value in summary["by_method"].items():
        lines.append(f"| {name} | {value['retained_views']:,} | {value['tokens_scheduled']['effective_target_tokens']:,} |")
    duplicates, reuse = summary["duplicates"], summary["target_reuse"]
    lines += ["", "## Reuse and Duplication", "",
              f"Pack assembly removed {duplicates['same_class_removed']:,} same-class exact encoded duplicates. "
              f"Remaining full-example duplicate groups: {duplicates['residual_same_class_groups']:,} within class and "
              f"{duplicates['residual_crossclass_groups']:,} across classes.", "",
              f"{reuse['repeated_exact_payload_groups']:,} exact target payloads appear in multiple views. "
              f"{reuse['cross_method_exact_payload_groups']:,} are shared across authoring routes; "
              f"{reuse['cross_method_same_source_target_groups']:,} also share the exact declared source ID.", "",
              "Exact equal text alone does not establish common provenance or semantic duplication. "
              "The sidecar ledgers retain all groups and view IDs, not a handpicked subset.", "", "## Limits", ""]
    lines += ["- " + value for value in summary["limits"]]
    return "\n".join(lines) + "\n"


def prepare(pack, out):
    pack, destination = Path(pack).resolve(strict=True), Path(out).absolute()
    require(not destination.resolve().is_relative_to(pack), "Profile must be outside sealed pack")
    working = destination.with_name(destination.name + ".building")
    require(not destination.exists() and not working.exists(), "New profile directory required; preserve existing outputs")
    manifest = sealed(pack, "PREPARED.json")
    pack_sha = sha(pack / "manifest.json")
    required = {"rows.jsonl", "source-witnesses.jsonl", "stages.json", "report.json", "duplicates.jsonl", "exclusions.jsonl"}
    require(required <= set(manifest["artifacts"]), "Missing profile input artifact")
    report, stages = read(pack / "report.json"), read(pack / "stages.json")
    schedule = Counter()
    for stage in stages:
        require(isinstance(stage["schedule"], list), "Explicit finite schedule required")
        for index in stage["schedule"]:
            require(type(index) is int and index >= 0, "Invalid schedule index")
            schedule[index] += 1
    families, views, context_appearances, seen_ids = Families(), [], Counter(), set()
    for index, (row, witness) in enumerate(zip_longest(rows(pack / "rows.jsonl"), rows(pack / "source-witnesses.jsonl"))):
        require(row is not None and witness is not None, "Row/witness count differs")
        require(row["id"] not in seen_ids, "Repeated view ID")
        seen_ids.add(row["id"])
        view = describe(row, witness, schedule[index])
        views.append(view)
        families.connect(view["aliases"])
        for message in witness.get("messages", [])[:-1]:
            if message["role"] == "assistant":
                payload = {"assistant": message} if message.get("tool_calls") else {"text": message["content"]}
                context_appearances[digest(payload)] += schedule[index]
    require(views and (not schedule or max(schedule) < len(views)), "Schedule points outside pack")
    require(len(views) == report["retained_views"] and dict(Counter(v["collection"] for v in views)) == report["views_by_collection"],
            "Pack report counts differ")
    duplicates = list(rows(pack / "duplicates.jsonl"))
    exclusions = list(rows(pack / "exclusions.jsonl"))
    require(len(duplicates) == report["duplicate_views"] and len(exclusions) == report["excluded_views"], "Pack disposition counts differ")
    by_id = {v["id"]: v for v in views}
    require(all(d["id"] not in by_id and d["retained_id"] in by_id and d["collection"] == by_id[d["retained_id"]]["collection"]
                for d in duplicates), "Duplicate ledger does not point to retained same-class view")
    targets = list(exposure_groups(views, ["target_payload_sha256"]))
    source_targets = list(exposure_groups(views, ["source_id", "target_payload_sha256"]))
    for target in targets:
        target["exact_nonfinal_assistant_context_appearances_scheduled"] = context_appearances[target["target_payload_sha256"]]
    encoded_groups = list(exposure_groups(views, ["encoded_example_sha256"]))
    residual = [g for g in encoded_groups if g["retained_views"] > 1]
    same_class = [g for g in residual if any(n > 1 for n in g["collections"].values())]
    crossclass = [g for g in residual if len(g["collections"]) > 1]
    detail_keys = ("collection", "method", "origin", "prior_version", "record_type", "generated_target_declared",
                   "quality_tier", "transformation", "source_language_declared", "target_language_declared", "language_scope")
    detail = defaultdict(list)
    for view in views:
        detail[tuple(view[k] for k in detail_keys)].append(view)
    breakdown = [{**dict(zip(detail_keys, key)), **aggregate(group, families)} for key, group in
                 sorted(detail.items(), key=lambda item: json.dumps(item[0], sort_keys=True))]
    methods = sorted({v["method"] for v in views})
    summary = {"version": VERSION, "pack": str(pack), "pack_manifest_sha256": pack_sha,
        "training_executed_by_profiler": False, "pack_report": {k: v for k, v in report.items() if k != "input_files"},
        "total": aggregate(views, families),
        "by_collection": {c: aggregate([v for v in views if v["collection"] == c], families) for c in COLLECTIONS},
        "by_method": {m: aggregate([v for v in views if v["method"] == m], families) for m in methods},
        "by_declared_target_language": {lang: aggregate([v for v in views if v["target_language_declared"] == lang], families)
                                        for lang in sorted({v["target_language_declared"] for v in views})},
        "duplicates": {"same_class_removed": len(duplicates), "removed_by_collection": dict(Counter(d["collection"] for d in duplicates)),
                       "residual_same_class_groups": len(same_class), "residual_crossclass_groups": len(crossclass),
                       "crossclass_duplicate_views": sum(g["retained_views"] for g in crossclass)},
        "target_reuse": {"repeated_exact_payload_groups": sum(g["retained_views"] > 1 for g in targets),
                         "cross_method_exact_payload_groups": sum(len(g["methods"]) > 1 for g in targets),
                         "cross_collection_exact_payload_groups": sum(len(g["collections"]) > 1 for g in targets),
                         "cross_method_same_source_target_groups": sum(len(g["methods"]) > 1 for g in source_targets),
                         "maximum_exact_target_presentations": max(g["scheduled_presentations"] for g in targets),
                         "exact_target_presentation_histogram": dict(sorted(Counter(g["scheduled_presentations"] for g in targets).items())),
                         "method_intersections": dict(Counter(" + ".join(sorted(g["methods"])) for g in targets if len(g["methods"]) > 1))},
        "scheduled_stages": [{k: v for k, v in stage.items() if k != "schedule"} | {"presentations": len(stage["schedule"])} for stage in stages],
        "limits": ["Training views are not conversations, independent source records, or independent semantic tasks.",
                   "Source-ID values and alias-connected groups are declared provenance identities, not proven statistical independence; group counts across collections are not additive.",
                   "Original pair IDs are counted only where explicit parallel-reference metadata exists; inherited source IDs can denote records, histories or project-authored units.",
                   "Input tokens include teacher-forced target tokens. Effective target tokens omit position zero because causal loss shifts labels; context-only assistant turns are not supervised.",
                   "Language labels come from explicit metadata only. Record-level language is not a per-token language assay; missing labels stay unknown and structured outputs are separate.",
                   "Repeated-target ledgers compare exact bytes or structured payloads, with no trimming or normalization. Equal text is not proof of the same original source.",
                   "Context exposure counts exact nonfinal assistant payload appearances only; embedded references inside user text and other substring occurrences are not counted.",
                   "Authoring routes and original origin labels are retained separately; inherited AI corrections are not relabeled source-reference gold.",
                   "Pack duplicate removals are reported from its sealed ledger, not reconstructed discarded targets. Residual duplicates are independently measured from retained token arrays.",
                   "Scheduled exposure is not evidence that optimization has executed. This sidecar changes no targets, admissions, schedule, builder, or sealed pack."]}
    working.mkdir(parents=True, mode=0o700)
    save(working / "summary.json", summary)
    save_rows(working / "breakdown.jsonl", breakdown)
    save_rows(working / "view-ledger.jsonl", views)
    save_rows(working / "target-exposures.jsonl", targets)
    save_rows(working / "source-target-exposures.jsonl", source_targets)
    save_rows(working / "residual-exact-duplicates.jsonl", residual)
    save_rows(working / "same-class-removals.jsonl", duplicates)
    (working / "REPORT.md").write_text(report_markdown(summary))
    with (working / "breakdown.csv").open("w", newline="") as stream:
        columns = list(detail_keys) + ["retained_views", "unique_target_source_id_values", "unique_upstream_admitted_records",
                                     "connected_declared_alias_groups", "scheduled_presentations"] + list(TOKEN_FIELDS)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group in breakdown:
            writer.writerow({**{k: group[k] for k in columns if k in group}, **group["tokens_scheduled"]})
    require(sha(pack / "manifest.json") == pack_sha, "Pack changed during profiling")
    sealed(pack, "PREPARED.json")
    save(working / "manifest.json", {"version": VERSION, "private_research_only": True, "production_eligible": False,
         "descriptive_only": True, "training_admission": False, "pack_path": str(pack), "pack_manifest_sha256": pack_sha,
         "profiler_sha256": sha(__file__), "artifacts": {p.name: sha(p) for p in working.iterdir() if p.is_file()}})
    save(working / "SEALED.json", {"manifest_sha256": sha(working / "manifest.json")})
    verify(working)
    working.rename(destination)
    return summary


def verify(out):
    manifest = sealed(Path(out), "SEALED.json")
    require(manifest["version"] == VERSION and manifest["descriptive_only"] is True and manifest["training_admission"] is False,
            "Wrong profile scope")
    require(sha(__file__) == manifest["profiler_sha256"], "Profiler code differs")
    pack = Path(manifest["pack_path"])
    require(sha(pack / "manifest.json") == manifest["pack_manifest_sha256"], "Profile references changed pack")
    sealed(pack, "PREPARED.json")
    summary = read(Path(out) / "summary.json")
    return {"verified": True, "pack_manifest_sha256": manifest["pack_manifest_sha256"],
            "retained_views": summary["total"]["retained_views"], "scheduled_tokens": summary["total"]["tokens_scheduled"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--pack", type=Path, default=ROOT / "tmp/r7-20261117/pack-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/r7-20261117/pack-profile-v1")
    args = parser.parse_args()
    result = prepare(args.pack, args.out) if args.action == "prepare" else verify(args.out)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
