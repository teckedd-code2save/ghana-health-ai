#!/usr/bin/env python3
"""Scoped original-reference research admission, with offline native encoding."""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys

import check_afrique_training_contract as contract
import prepare_r7_alignment_candidates as alignment
import prepare_r7_source_inventory as inventory
import r7_encoding
import run_understanding_r7 as runner
from prepare_understanding_run import HoldoutIndex

ROOT = Path(__file__).resolve().parents[1]
VERSION = "r7-alignment-research-admission-v1"
TIER = "source_reference_preserved_automatically_screened"
require, sha, digest = inventory.require, inventory.sha, inventory.digest


def bind_sealed(folder, seal_name, inputs):
    folder = Path(folder).resolve(strict=True)
    manifest = inputs.read(folder / "manifest.json")
    require(inputs.read(folder / seal_name)["manifest_sha256"] == sha(folder / "manifest.json"), "Changed upstream seal")
    for name, checksum in manifest["artifacts"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Escaping upstream artifact")
        inputs.bind(folder / name, checksum)
    return manifest


def held_sources(folder, manifest, inputs, references):
    """Export actual inherited held text and identities, never merely a clearance flag."""
    held = []
    def add(text, ref, **metadata):
        if isinstance(text, str) and text.strip():
            held.append({**metadata, "id": "held:" + digest([ref, text]), "text": text,
                         "original_ref": ref, "training_eligible": False})
    for name, meta in manifest["input_files"].items():
        path = Path(name)
        if path.name == "held-content.jsonl":
            for number, row in enumerate(references.load(path), 1):
                add(row["text"], {"file": name, "file_sha256": meta["sha256"], "line": number, "record_sha256": digest(row)},
                    **{k: v for k, v in row.items() if k not in {"id", "text", "original_ref", "training_eligible"}})
        elif path.suffix == ".parquet" and path.parent.name in {"oasst_validation", "afrihealth-qa"}:
            if path.parent.name == "oasst_validation":
                columns, fields = ("message_id", "message_tree_id", "text"), ("text",)
            elif path.stem in {"validation", "test_holdout"}:
                columns, fields = ("id", "question", "answer"), ("question", "answer")
            else:
                continue
            inputs.bind(path, meta["sha256"])
            for row in inputs.parquet(path, columns, "held"):
                identity = ({"source_id": row["message_id"], "group_id": row["message_tree_id"]}
                            if "message_id" in row else {"source_id": row["id"]})
                for field in fields:
                    add(row[field], {**row["source_ref"], "field": field}, **identity)
    for row in inputs.rows(folder / "parallel-meaning-decisions.jsonl"):
        if row["source_split"] not in {"train", "unassigned"}:
            for language, text in zip(("en", "tw"), references.pair(row, validate_inheritance=False), strict=True):
                add(text, {**row["source_ref"], "language": language}, source_id=row["source_id"], group_aliases=row["source_family_keys"])
    expected = {r["normalized_text_sha256"] for r in inputs.rows(folder / "protection-index.jsonl")}
    require(expected and {digest(inventory.normalize(r["text"])) for r in held} == expected,
            "Inherited protection fingerprints differ")
    return list({row["id"]: row for row in held}.values())


def evaluation_sources(folders, inputs):
    held, receipts = [], []
    for folder in folders:
        folder = Path(folder).resolve(strict=True)
        manifest = bind_sealed(folder, "FROZEN.json", inputs)
        require("held-content.jsonl" in manifest["artifacts"], "Evaluation has no bound held content")
        checksum = sha(folder / "manifest.json")
        for number, row in enumerate(inputs.rows(folder / "held-content.jsonl"), 1):
            require(isinstance(row.get("text"), str) and row["text"].strip(), "Empty evaluation protection")
            held.append({**row, "id": "eval:" + checksum[:16] + ":" + row["id"],
                         "original_ref": {"file": str(folder / "held-content.jsonl"), "line": number,
                                          "file_sha256": manifest["artifacts"]["held-content.jsonl"], "record_sha256": digest(row)},
                         "training_eligible": False})
        receipts.append({"path": str(folder), "manifest_sha256": checksum,
                         "held_content_sha256": manifest["artifacts"]["held-content.jsonl"]})
    require(receipts and held, "Frozen R7 evaluation protection is required")
    return held, receipts


def family_protection(pairs, held):
    graph = inventory.Families()
    aliases = {}
    for pair in pairs:
        keys = set(pair["source_family_keys"]) | {pair["source_id"]}
        keys |= {"alignment:" + language + ":" + digest(inventory.normalize(pair[field]))
                 for language, field in (("en", "english"), ("tw", "twi"))}
        aliases[pair["id"]] = keys
        graph.connect(keys)
    protected = set().union(*(inventory.identity_keys(row) for row in held))
    index = HoldoutIndex((r["id"], r["text"]) for r in held)
    blocked = {}
    for pair in pairs:
        root = graph.root(pair["source_id"])
        hit = next((hit for text in (pair["english"], pair["twi"], pair["english"] + "\n" + pair["twi"])
                    if (hit := index.match(text))), None)
        if hit:
            blocked[root] = hit
        elif aliases[pair["id"]] & protected:
            blocked[root] = {"kind": "protected_source_identity", "keys": sorted(aliases[pair["id"]] & protected)}
    # Resolve after all links are present so both directions and related classes stay together.
    roots = {p["id"]: graph.root(p["source_id"]) for p in pairs}
    return roots, aliases, blocked, index


def inherited_directions(pair, references):
    result = {}
    for ref in pair["original_refs"]["inherited"]:
        values = references.load(ref["file"])
        row = values[ref["line"] - 1]
        require(digest(row) == ref["record_sha256"] and row["source_id"] == pair["source_id"]
                and row["split"] == "train" and row["generated_target"] is False, "Inherited alignment changed")
        direction = (row["source_language"], row["target_language"])
        require(direction in {("en", "tw"), ("tw", "en")}, "Invalid inherited direction")
        texts = {"en": pair["english"], "tw": pair["twi"]}
        require(inventory.strict_json(row["messages"][-2]["content"])["source_text"] == texts[direction[0]]
                and row["messages"][-1]["content"] == texts[direction[1]], "Inherited wrapper changed original reference")
        require(direction not in result or result[direction]["messages"] == row["messages"], "Conflicting inherited wrappers")
        result[direction] = row
    require(set(result) == {("en", "tw"), ("tw", "en")}, "Incomplete original alignment directions")
    return result


def native_measure(tokenizer, row, max_length):
    """Measure the entire native sequence; reject length or boundary-trim changes."""
    try:
        encoded = r7_encoding.encode_complete(tokenizer, row, tokenizer.model_max_length)
    except ValueError as error:
        return None, {"status": "rejected", "reason": "native_encoding_rejected", "detail": str(error)}
    labels, ids = encoded["labels"], encoded["input_ids"]
    target = [t for t in labels if t != -100]
    measure = {"sequence_tokens": len(ids), "target_tokens": len(target), "prompt_tokens": len(ids) - len(target),
               "effective_causal_target_tokens": sum(t != -100 for t in labels[1:]),
               "target_eos_count": target.count(contract.EOS), "target_ends_eos": target[-1] == contract.EOS,
               "encoded_arrays_sha256": digest({k: encoded[k] for k in ("input_ids", "labels", "attention_mask")})}
    decoded = tokenizer.decode(target[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != row["messages"][-1]["content"]:
        return None, {**measure, "status": "rejected", "reason": "native_boundary_trim_would_change_original_target"}
    if len(ids) > max_length:
        return None, {**measure, "status": "rejected", "reason": "overlength_no_truncation"}
    return encoded, {**measure, "status": "passed", "exact_original_target": True, "truncated": False}


def token_summary(values):
    if not values:
        return {"views": 0}
    result = {"views": len(values)}
    for key in ("sequence_tokens", "target_tokens", "effective_causal_target_tokens"):
        ordered = sorted(row[key] for row in values)
        result[key] = {"sum": sum(ordered), "min": ordered[0], "max": ordered[-1],
                       "p50": ordered[(len(ordered) - 1) // 2], "p95": ordered[(len(ordered) - 1) * 95 // 100]}
    return result


def verify(out, *, native=False):
    out = Path(out).resolve(strict=True)
    inputs = inventory.Inputs()
    manifest = bind_sealed(out, "SEALED.json", inputs)
    for path, meta in manifest["input_files"].items():
        inputs.bind(path, meta["sha256"])
    require(manifest["version"] == VERSION and manifest["quality_tier"] == TIER
            and manifest["training_pack_ready"] is False and manifest["parent_boundary_approval"] is False,
            "Changed admission scope")
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in out.rglob("*")) and actual == set(manifest["artifacts"]) | {"manifest.json", "SEALED.json"},
            "Unbound/linked admission artifact")
    rows, witnesses = inputs.rows(out / "rows.jsonl"), inputs.rows(out / "source-witnesses.jsonl")
    runner.admitted_sources(out, rows, witnesses, manifest["artifacts"])
    runner.verify_overlap(out, witnesses, manifest["artifacts"])
    seen, pair_directions = set(), defaultdict(set)
    tokenizer = contract.load_tokenizer(Path(manifest["tokenizer_path"])) if native else None
    for row, witness in zip(rows, witnesses, strict=True):
        require(row["id"] not in seen and all(row[k] == witness[k] for k in ("id", "source_id", "collection")), "Invalid row/witness identity")
        seen.add(row["id"])
        require(row["collection"] == "meaning" and row["objective"] == "final_assistant" and row["split"] == "train", "Wrong alignment collection")
        ids, labels = row["input_ids"], row["labels"]
        first = next(i for i, t in enumerate(labels) if t != -100)
        require(0 < first < len(ids) <= manifest["max_length"] and labels == [-100] * first + ids[first:]
                and row["attention_mask"] == [1] * len(ids) and ids[-1] == contract.EOS and labels.count(contract.EOS) == 1,
                "Invalid native target mask/length/EOS")
        require(row["sequence_tokens"] == len(ids) and row["target_tokens"] == len(ids) - first, "Token ledger differs")
        metadata = witness["metadata"]
        require(metadata["quality_tier"] == TIER and not metadata["human_reviewed"] and not metadata["semantic_certification"]
                and not metadata["alignment_pair"]["review_flags"], "Unresolved flags or unsupported quality tier")
        pair_directions[metadata["alignment_pair"]["id"]].add((metadata["source_language"], metadata["target_language"]))
        if native:
            encoded, measured = native_measure(tokenizer, witness, manifest["max_length"])
            require(encoded is not None and all(encoded[k] == row[k] for k in ("input_ids", "labels", "attention_mask")),
                    "Native encoded bytes changed")
    require(all(d == {("en", "tw"), ("tw", "en")} for d in pair_directions.values())
            and len(rows) == 2 * len(pair_directions), "Incomplete/duplicated admitted pair directions")
    return {"verified": True, "native_reencoded": native, "admitted_pairs": len(pair_directions), "encoded_views": len(rows)}


def prepare(source_alignment, tokenizer_path, evaluations, out, max_length=4096):
    out, source_alignment = Path(out).absolute(), Path(source_alignment).resolve(strict=True)
    require(not out.exists() and not out.is_symlink(), "Use a new admission directory")
    require(type(max_length) is int and 1 <= max_length <= 4096, "Invalid training length ceiling")
    alignment_manifest = alignment.verify(source_alignment)
    inputs = inventory.Inputs()
    bind_sealed(source_alignment, "SEALED.json", inputs)
    for path, info in alignment_manifest["input_files"].items():
        inputs.bind(path, info["sha256"])
    for module in (sys.modules[__name__], inventory, alignment, contract, r7_encoding, runner):
        inputs.bind(module.__file__)
    inputs.bind(ROOT / "modal/train/afrique_response_core.py")
    inputs.bind(ROOT / "scripts/run_understanding_r2.py")
    inputs.bind(ROOT / "scripts/prepare_understanding_run.py")
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    for name, checksum in contract.ASSET_HASHES.items():
        inputs.bind(tokenizer_path / name, checksum)
    tokenizer = contract.load_tokenizer(tokenizer_path)
    inventory_folder = Path(alignment_manifest["source_inventory"])
    require(sha(inventory_folder / "manifest.json") == alignment_manifest["source_inventory_manifest_sha256"], "Source inventory identity differs")
    inventory_manifest = inventory.verify(inventory_folder)
    bind_sealed(inventory_folder, "SEALED.json", inputs)
    references = alignment.References(inventory_manifest, inputs)
    held = held_sources(inventory_folder, inventory_manifest, inputs, references)
    eval_held, eval_refs = evaluation_sources(evaluations, inputs)
    held += eval_held
    require(len({r["id"] for r in held}) == len(held), "Duplicate protected unit")
    pairs = inputs.rows(source_alignment / "source-pairs.jsonl")
    for pair in pairs:
        require(pair["record_sha256"] == digest({k: v for k, v in pair.items() if k != "record_sha256"}), "Alignment pair hash differs")
    roots, aliases, blocked, index = family_protection(pairs, held)
    wrappers = {}
    for pair in pairs:
        if pair["review_flags"]:
            continue
        wrappers[pair["id"]] = inherited_directions(pair, references)
        for inherited in wrappers[pair["id"]].values():
            text = "\n".join(m["content"] for m in inherited["messages"] if m["role"] != "system")
            if hit := index.match(text):
                blocked[roots[pair["id"]]] = {**hit, "kind": "protected_native_alignment_family", "lexical_kind": hit["kind"]}
    decisions, records, rows, witnesses, measurements = [], defaultdict(list), [], [], []
    print("Bound source, alignment, tokenizer and evaluation; measuring complete original pairs", file=sys.stderr, flush=True)
    for number, pair in enumerate(pairs, 1):
        decision = {"id": pair["id"], "source_id": pair["source_id"], "candidate_class": pair["candidate_class"],
                    "source_pair_sha256": pair["record_sha256"], "original_ref": pair["original_refs"],
                    "group_id": "r7-alignment-family:" + digest(roots[pair["id"]]),
                    "review_flags": pair["review_flags"], "training_eligible": False, "research_admitted": False,
                    "human_reviewed": False, "semantic_certification": False, "directions": []}
        if pair["review_flags"]:
            decision.update(status="deferred_review", reasons=["unresolved_alignment_review_flags_not_adjudicated"])
        elif roots[pair["id"]] in blocked:
            decision.update(status="excluded", reasons=["protected_shared_source_family"], protection_evidence=blocked[roots[pair["id"]]])
        else:
            pending, failed = [], []
            for (source_language, target_language), inherited in sorted(wrappers[pair["id"]].items()):
                messages = inherited["messages"]
                identity = pair["id"] + ":" + source_language + "-to-" + target_language
                record = {"id": identity, "source_id": pair["source_id"], "collection": "meaning", "split": "train",
                    "messages": messages, "training_eligible": True, "research_admitted": True,
                    "quality_tier": TIER, "quality_status": TIER, "private_research_only": True, "production_eligible": False,
                    "parent_boundary_verification_required": True,
                    "metadata": {"origin": "compiled_parallel_text", "generated_target": False,
                        "task": "source_preserved_bilingual_alignment", "record_type": pair["record_type"],
                        "quality_tier": TIER, "human_reviewed": False, "semantic_certification": False,
                        "clinical_certification": False, "use_restriction": "private_noncommercial_research",
                        "group_id": decision["group_id"], "group_aliases": sorted(aliases[pair["id"]]),
                        "source_language": source_language, "target_language": target_language,
                        "alignment_pair": pair, "inherited_metadata": {k: v for k, v in inherited.items() if k != "messages"}},
                    "original_ref": {**pair["original_refs"], "alignment_path": str(source_alignment),
                        "alignment_manifest_sha256": sha(source_alignment / "manifest.json"),
                        "source_pair_sha256": pair["record_sha256"], "evaluation_manifests": eval_refs}}
                encoded, measured = native_measure(tokenizer, record, max_length)
                measurement = {"id": identity, "source_id": pair["source_id"], "pair_id": pair["id"],
                    "candidate_class": pair["candidate_class"], "source_language": source_language, "target_language": target_language, **measured}
                measurements.append(measurement)
                decision["directions"].append(measurement)
                if encoded is None:
                    failed.append(measured["reason"])
                pending.append((record, encoded, measured))
            if failed:
                decision.update(status="excluded", reasons=sorted(set(failed)))
            else:
                decision.update(status="admitted", reasons=[TIER], training_eligible=True, research_admitted=True, quality_tier=TIER)
                kind = pair["candidate_class"]
                for record, encoded, measured in pending:
                    ref = {"dataset_id": "gha-r7-alignment-research-" + kind, "dataset_version": VERSION,
                           "record_id": record["id"], "source_id": record["source_id"], "record_sha256": runner.digest(record)}
                    records[kind].append(record)
                    common = {"id": record["id"], "source_id": record["source_id"], "collection": "meaning", "dataset_ref": ref}
                    rows.append({**common, "split": "train", "objective": "final_assistant",
                                 **{k: encoded[k] for k in ("input_ids", "labels", "attention_mask")},
                                 "sequence_tokens": measured["sequence_tokens"], "target_tokens": measured["target_tokens"]})
                    witnesses.append({**common, "version": r7_encoding.WITNESS_VERSION,
                                      **{k: record[k] for k in ("messages", "metadata", "original_ref")}})
        decisions.append(decision)
        if number % 5000 == 0:
            print(f"Checked {number}/{len(pairs)} pairs; {len(rows)} admitted views", file=sys.stderr, flush=True)
    require(rows, "No source-preserved pairs passed scoped admission")
    admitted_ids = {r["id"] for r in rows}
    counts = {}
    for kind in sorted(alignment.CLASSES):
        subset = [d for d in decisions if d["candidate_class"] == kind]
        counts[kind] = {"original_pairs": len(subset), "initial_no_flag_pairs": sum(not d["review_flags"] for d in subset),
                       "pair_statuses": dict(Counter(d["status"] for d in subset)), "admitted_views": len(records[kind]),
                       "reasons_by_pair": dict(Counter(reason for d in subset for reason in d["reasons"])),
                       "tokens": token_summary([m for m in measurements if m["candidate_class"] == kind and m["id"] in admitted_ids])}
    summary = {"version": VERSION, "quality_tier": TIER, "counts_by_class": counts,
        "original_pairs": len(pairs), "initial_no_flag_pairs": sum(not d["review_flags"] for d in decisions),
        "admitted_pairs": len(rows) // 2, "admitted_views": len(rows), "measured_views": len(measurements),
        "tokens": token_summary([m for m in measurements if m["id"] in admitted_ids]),
        "max_length": max_length, "native_targets_equal_exact_originals": True, "truncated_views": 0,
        "synthetic_twi_created": 0, "human_reviewed": False, "semantic_certification": False,
        "clinical_certification": False, "training_schedule_created": False, "training_pack_ready": False,
        "parent_boundary_approval": False, "gpu_calls": 0, "api_calls": 0,
        "limitations": ["Scoped source-reference admission is not human/native-language gold or semantic certification.",
                       "Unresolved duplicate/polysemy/possible-misalignment flags remain outside this admission, without an error verdict.",
                       "The parent must verify the assembled pack, final evaluation boundary, token mixture and finite schedule.",
                       "Token totals are one-pass candidate availability, not scheduled or consumed training exposure."]}
    inputs.recheck()
    out.mkdir(parents=True, mode=0o700)
    for name, values in (("admission-ledger", decisions), ("token-measurements", measurements), ("rows", rows),
                         ("source-witnesses", witnesses), ("held-content", held)):
        inventory.write_rows(out / (name + ".jsonl"), values)
    refs = []
    for kind, values in sorted(records.items()):
        if not values:
            continue
        target = out / "datasets" / kind
        target.mkdir(parents=True)
        inventory.write_rows(target / "records.jsonl", values)
        manifest = {"dataset_id": "gha-r7-alignment-research-" + kind, "dataset_version": VERSION,
            "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
            "quality_tier": TIER, "quality_status": TIER, "human_reviewed": False, "semantic_certification": False,
            "admission_scope": "Unchanged original bilingual references; automatic source/flag/holdout/native-encoding screens only",
            "record_type": alignment.CLASSES[kind], "collection": "meaning", "record_count": len(values),
            "source_pair_count": len(values) // 2, "records_path": f"datasets/{kind}/records.jsonl",
            "records_sha256": sha(target / "records.jsonl"), "parent_boundary_verification_required": True}
        inventory.write_json(target / "manifest.json", manifest)
        refs.append({k: manifest[k] for k in ("dataset_id", "dataset_version")} |
                    {"manifest_path": f"datasets/{kind}/manifest.json", "manifest_sha256": sha(target / "manifest.json")})
    inventory.write_json(out / "source-manifests.json", refs)
    inventory.write_json(out / "evaluation-manifest.json", {"r7_evaluations": eval_refs,
        "inherited_inventory_manifest_sha256": sha(inventory_folder / "manifest.json"), "pair_level_shared_family_protection": True})
    inventory.write_json(out / "overlap-audit.json", {"protected_units": len(held), "overlap_free": True,
        "held_content_sha256": sha(out / "held-content.jsonl"), "source_witnesses_sha256": sha(out / "source-witnesses.jsonl"),
        "evaluation_manifest_sha256": sha(out / "evaluation-manifest.json"), "final_parent_boundary_verification_required": True})
    inventory.write_json(out / "summary.json", summary)
    inputs.recheck()
    inventory.write_json(out / "manifest.json", {"version": VERSION, "quality_tier": TIER,
        "alignment_path": str(source_alignment), "alignment_manifest_sha256": sha(source_alignment / "manifest.json"),
        "source_inventory_manifest_sha256": sha(inventory_folder / "manifest.json"), "evaluations": eval_refs,
        "tokenizer_path": str(tokenizer_path), "model": contract.MODEL, "revision": contract.REVISION,
        "tokenizer_assets": contract.ASSET_HASHES, "max_length": max_length, "enable_thinking": False,
        "runtime": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "pyarrow")},
        "input_files": inputs.files, "artifacts": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "private_research_only": True, "production_eligible": False, "training_pack_ready": False, "parent_boundary_approval": False})
    inventory.write_json(out / "SEALED.json", {"manifest_sha256": sha(out / "manifest.json")})
    verify(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--alignment", type=Path, default=ROOT / "tmp/r7-20261117/alignment-candidates-v1")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tmp/r6-demo/bundle-v1/tokenizer")
    parser.add_argument("--evaluation", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/r7-20261117/alignment-admission-v1")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--native", action="store_true", help="Re-encode all admitted views during verification")
    args = parser.parse_args()
    result = (prepare(args.alignment, args.tokenizer, args.evaluation or [ROOT / "tmp/r7-20261117/behavior-eval-v1"], args.out, args.max_length)
              if args.action == "prepare" else verify(args.out, native=args.native))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
