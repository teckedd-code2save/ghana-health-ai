#!/usr/bin/env python3
"""Register original bilingual alignment views, never generated conversations."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import unicodedata

import corpus_record_types as types
import prepare_r7_source_inventory as inventory
from prepare_understanding_run import HoldoutIndex

ROOT = Path(__file__).resolve().parents[1]
VERSION = "r7-source-preserved-alignment-candidates-v1"
CLASSES = {"sentences": "sentence", "utterances": "utterance", "lexicon": "dictionary_entry"}
FIELDS = {"ghana_nlp_parallel": ("text", "label"), "community_parallel": ("eng", "twi")}
require, digest, sha = inventory.require, inventory.digest, inventory.sha


class References:
    def __init__(self, manifest, inputs):
        self.manifest, self.inputs, self.cache = manifest, inputs, {}

    def load(self, path):
        path = Path(path).absolute()
        key = str(path)
        require(key in self.manifest["input_files"], "Reference outside bound source inventory")
        if key not in self.cache:
            self.inputs.bind(path, self.manifest["input_files"][key]["sha256"])
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    value = list(csv.DictReader(stream))
            elif path.suffix == ".parquet":
                import pyarrow.parquet as pq
                columns = ("eng", "twi", "count", "source_files") if path.parent.name == "community_parallel" else (
                    ("text",) if path.parent.name == "oasst_validation" else ("question", "answer"))
                value = pq.read_table(path, columns=list(columns)).to_pylist()
            elif path.suffix == ".jsonl":
                value = self.inputs.rows(path)
            else:
                raise ValueError("Unsupported original reference format")
            self.cache[key] = value
        return self.cache[key]

    def pair(self, candidate, validate_inheritance=True):
        family, ref = candidate["source_family"], candidate["source_ref"]
        require(family in FIELDS, "Unknown parallel source family")
        require(self.manifest["input_files"].get(ref["file"], {}).get("sha256") == ref["file_sha256"], "Source file identity differs")
        records = self.load(ref["file"])
        index = ref["row_index"]
        require(type(index) is int and 0 <= index < len(records), "Invalid original row index")
        raw = records[index]
        checksum = ref.get("record_sha256", ref.get("projection_sha256"))
        require(digest(raw) == checksum, "Original source record hash differs")
        english_field, twi_field = FIELDS[family]
        english, twi = raw[english_field] or "", raw[twi_field] or ""
        require(candidate["source_id"] == family + ":" + (raw["id"] if family == "ghana_nlp_parallel" else str(index)),
                "Original source ID differs")
        require(digest(english) == candidate["english_sha256"] and digest(twi) == candidate["twi_sha256"], "Original bilingual reference changed")
        if validate_inheritance:
            semantic = candidate["semantic_reference"]
            assessment = types.classify(english, twi, family, candidate["source_genre"], "sentence", "compiled_parallel_text")
            require(candidate["type_assessment"] == assessment and assessment["content_sha256"] == semantic["original_pair_content_sha256"],
                    "Original source type/content assessment differs")
            require((semantic["english_field"], semantic["twi_field"]) == FIELDS[family], "Reference field mapping differs")
            require(candidate["inherited_refs"], "Missing inherited source references")
            directions = set()
            for inherited in candidate["inherited_refs"]:
                values = self.load(inherited["file"])
                line = inherited["line"]
                require(type(line) is int and 1 <= line <= len(values), "Invalid inherited row reference")
                row = values[line - 1]
                require(digest(row) == inherited["record_sha256"] and row["source_id"] == candidate["source_id"]
                        and row["split"] == "train" and row["generated_target"] is False,
                        "Inherited source identity/split/target origin differs")
                require(self.manifest["input_files"][inherited["file"]]["sha256"] == inherited["file_sha256"]
                        and row["source_file_sha256"] == ref["file_sha256"], "Inherited original file binding differs")
                expected = {"en": english, "tw": twi}
                require({row["source_language"], row["target_language"]} == {"en", "tw"}, "Unexpected alignment direction")
                require(inventory.strict_json(row["messages"][-2]["content"])["source_text"] == expected[row["source_language"]]
                        and row["messages"][-1]["content"] == expected[row["target_language"]],
                        "Inherited alignment differs from complete original pair")
                require(row["type_assessment"]["content_sha256"] == semantic["original_pair_content_sha256"]
                        == semantic["inherited_pair_content_sha256"] and row["source_hash"] == semantic["inherited_source_hash"],
                        "Inherited semantic reference binding differs")
                directions.add((row["source_language"], row["target_language"]))
            require(directions == {("en", "tw"), ("tw", "en")}, "Both inherited alignment directions required")
        return english, twi


def protection(folder, manifest, inputs, references):
    """Reconstruct every held-text fingerprint, then recheck the original pairs."""
    held = {}
    def add(text):
        if isinstance(text, str) and text.strip():
            held[digest(inventory.normalize(text))] = text
    for name in manifest["input_files"]:
        path = Path(name)
        if path.name == "held-content.jsonl":
            for row in references.load(path):
                add(row["text"])
        elif path.suffix == ".parquet" and path.parent.name == "oasst_validation":
            for row in references.load(path):
                add(row["text"])
        elif path.suffix == ".parquet" and path.parent.name == "afrihealth-qa" and path.stem in {"validation", "test_holdout"}:
            for row in references.load(path):
                add(row["question"]); add(row["answer"])
    for candidate in inputs.rows(folder / "parallel-meaning-decisions.jsonl"):
        if candidate["source_split"] not in ("train", "unassigned"):
            for text in references.pair(candidate, validate_inheritance=False):
                add(text)
    expected = {r["normalized_text_sha256"] for r in inputs.rows(folder / "protection-index.jsonl")}
    require(expected and set(held) == expected, "Could not reconstruct exact inventory holdout fingerprints")
    return HoldoutIndex(held.items()), {"distinct_normalized_held_texts": len(held),
        "reconstructed_inventory_fingerprints_exact": True,
        "source_family_exclusions": "inherited from sealed source inventory decisions",
        "lexical_recheck": "all English, Twi and concatenated original pairs; normalized exact and eight-word Jaccard >=0.85",
        "final_r7_evaluation_freeze_recheck_required": True}


def pair_flags(english, twi):
    flags = []
    for language, text in (("en", english), ("tw", twi)):
        if not text.strip():
            flags.append({"code": "empty_reference", "language": language})
        if any(c == "\ufffd" or unicodedata.category(c) in {"Cs", "Co"} or
               (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text):
            flags.append({"code": "invalid_unicode_or_control_character", "language": language})
        if unicodedata.normalize("NFC", text) != text:
            flags.append({"code": "non_nfc_preserved_without_normalization", "language": language})
        if any(unicodedata.category(c) == "Cf" for c in text):
            flags.append({"code": "unicode_format_character_review", "language": language})
        for code in inventory.contact_flags(text):
            flags.append({"code": code, "language": language})
        if inventory.CONTROL.search(text):
            flags.append({"code": "literal_model_control_token", "language": language})
        if len(text) > 4096:
            flags.append({"code": "reference_over_4096_characters_not_token_measurement", "language": language})
    lengths = [len(inventory.normalize(t).split()) for t in (english, twi)]
    if min(lengths) and max(lengths) / min(lengths) > 4:
        flags.append({"code": "possible_misalignment_word_length_ratio_over_four", "word_counts_en_tw": lengths})
    if inventory.normalize(english) == inventory.normalize(twi):
        flags.append({"code": "identical_normalized_languages_review_loanwords_or_misalignment"})
    numbers = [re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", t) for t in (english, twi)]
    if Counter(numbers[0]) != Counter(numbers[1]):
        flags.append({"code": "numeric_literal_difference_possible_misalignment", "en": numbers[0], "tw": numbers[1]})
    return flags


def mapping_review(pairs):
    exact, normalized, en_to_tw, tw_to_en = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in pairs:
        kind, english, twi = row["candidate_class"], row["english"], row["twi"]
        exact[(kind, digest([english, twi]))].append(row)
        normalized[(kind, digest([inventory.normalize(english), inventory.normalize(twi)]))].append(row)
        en_to_tw[(kind, digest(inventory.normalize(english)))].append(row)
        tw_to_en[(kind, digest(inventory.normalize(twi)))].append(row)
    reports = []
    for code, groups, target in (("duplicate_exact_pair", exact, None), ("duplicate_normalized_pair", normalized, None),
                                  ("english_maps_to_multiple_twi_review_polysemy_or_conflict", en_to_tw, "twi"),
                                  ("twi_maps_to_multiple_english_review_polysemy_or_conflict", tw_to_en, "english")):
        for (kind, key), rows in sorted(groups.items()):
            distinct = len({digest(inventory.normalize(r[target])) for r in rows}) if target else 1
            if (target and distinct <= 1) or (not target and len(rows) <= 1):
                continue
            proof = {"id": code + ":" + kind + ":" + key, "code": code, "candidate_class": kind,
                     "key_sha256": key, "pair_ids": [r["id"] for r in rows], "source_records": len(rows),
                     "distinct_target_values": distinct, "semantic_error_confirmed": False}
            reports.append(proof)
            for row in rows:
                row["review_flags"].append({"code": code, "review_group_id": proof["id"]})
    return reports


def views(pairs, kind, inventory_sha):
    for pair in pairs:
        if pair["candidate_class"] != kind:
            continue
        for source_language, target_language in (("en", "tw"), ("tw", "en")):
            languages = {"en": pair["english"], "tw": pair["twi"]}
            yield {"id": pair["id"] + ":" + source_language + "-to-" + target_language,
                "dataset_id": "gha-r7-alignment-" + kind, "dataset_version": VERSION,
                "candidate_class": kind, "record_type": pair["record_type"], "task": "source_preserved_bilingual_alignment",
                "source_id": pair["source_id"], "split": "train", "source_language": source_language,
                "target_language": target_language, "source_text": languages[source_language], "target_text": languages[target_language],
                "source_text_sha256": digest(languages[source_language]), "target_text_sha256": digest(languages[target_language]),
                "source_pair_id": pair["id"], "source_pair_sha256": pair["record_sha256"],
                "source_family": pair["source_family"], "source_genre": pair["source_genre"],
                "source_family_keys": pair["source_family_keys"], "type_assessment": pair["type_assessment"],
                "original_refs": pair["original_refs"], "semantic_reference": pair["semantic_reference"],
                "inventory_manifest_sha256": inventory_sha, "review_flags": pair["review_flags"],
                "origin": "compiled_parallel_text", "generated_target": False,
                "admission_status": "pending_quality_decision", "training_eligible": False,
                "semantic_certification": False, "production_eligible": False}


def verify(out):
    out = Path(out).resolve(strict=True)
    manifest = inventory.strict_json((out / "manifest.json").read_text())
    require(manifest["version"] == VERSION and manifest["training_eligible"] is False, "Wrong alignment candidate identity")
    require(inventory.strict_json((out / "SEALED.json").read_text()) == {"manifest_sha256": sha(out / "manifest.json")}, "Alignment seal differs")
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in out.rglob("*")), "Linked alignment artifact")
    require(actual == set(manifest["artifacts"]) | {"manifest.json", "SEALED.json"}, "Unbound alignment artifact")
    for name, checksum in manifest["artifacts"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(out / name) == checksum, "Alignment artifact changed")
    return manifest


def prepare(source_inventory, out):
    folder, out = Path(source_inventory).resolve(strict=True), Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "Use a new alignment candidate directory")
    manifest = inventory.verify(folder)
    inputs = inventory.Inputs()
    inputs.bind(__file__)
    inputs.bind(Path(inventory.__file__))
    for module in (types,):
        path = Path(module.__file__).absolute()
        require(str(path) in manifest["input_files"], "Unbound inherited routing policy")
        inputs.bind(path, manifest["input_files"][str(path)]["sha256"])
    inputs.bind(Path(__file__).with_name("prepare_understanding_run.py"))
    inventory_sha = inputs.bind(folder / "manifest.json")
    inputs.bind(folder / "SEALED.json")
    for name, checksum in manifest["artifacts"].items():
        inputs.bind(folder / name, checksum)
    candidates = inputs.rows(folder / "parallel-meaning-candidates.jsonl")
    require(candidates and len({r["id"] for r in candidates}) == len(candidates)
            and len({r["source_id"] for r in candidates}) == len(candidates), "Missing/duplicate candidate pairs")
    references = References(manifest, inputs)
    index, protection_receipt = protection(folder, manifest, inputs, references)
    pairs = []
    for candidate in candidates:
        kind = candidate["routing"]
        require(candidate["status"] == "eligible" and candidate["source_split"] == "train" and kind in CLASSES
                and candidate["generated_target"] is False and candidate["training_eligible"] is False
                and candidate["type_assessment"]["effective_type"] == CLASSES[kind], "Unqualified candidate source/type")
        english, twi = references.pair(candidate)
        require(all(index.match(text) is None for text in (english, twi, english + "\n" + twi)), "Protected content in alignment candidate")
        pairs.append({"id": "alignment-pair:" + candidate["source_id"], "source_id": candidate["source_id"],
            "candidate_class": kind, "record_type": CLASSES[kind], "english": english, "twi": twi,
            "english_sha256": digest(english), "twi_sha256": digest(twi), "source_split": "train",
            "original_refs": {"source": candidate["source_ref"], "inherited": candidate["inherited_refs"],
                              "inventory_candidate_id": candidate["id"], "inventory_candidate_sha256": digest(candidate)},
            "semantic_reference": candidate["semantic_reference"], "source_family_keys": candidate["source_family_keys"],
            "source_family": candidate["source_family"], "source_genre": candidate["source_genre"],
            "type_assessment": candidate["type_assessment"],
            "lengths": {"en_characters": len(english), "tw_characters": len(twi),
                        "en_words": len(inventory.normalize(english).split()), "tw_words": len(inventory.normalize(twi).split())},
            "review_flags": pair_flags(english, twi), "origin": "compiled_parallel_text", "generated_target": False,
            "admission_status": "pending_quality_decision", "training_eligible": False,
            "semantic_certification": False, "production_eligible": False})
    groups = mapping_review(pairs)
    for pair in pairs:
        pair["record_sha256"] = digest(pair)
    counts = {}
    for kind in sorted(CLASSES):
        subset = [r for r in pairs if r["candidate_class"] == kind]
        counts[kind] = {"original_pairs": len(subset), "derived_views": 2 * len(subset),
            "exact_distinct_pair_values": len({digest([r["english"], r["twi"]]) for r in subset}),
            "normalized_distinct_pair_values": len({digest([inventory.normalize(r["english"]), inventory.normalize(r["twi"])]) for r in subset}),
            "pairs_with_automatic_flags": sum(bool(r["review_flags"]) for r in subset),
            "pairs_without_automatic_flags_not_gold": sum(not r["review_flags"] for r in subset),
            "flags_by_pair_nonexclusive": dict(Counter(code for r in subset for code in {f["code"] for f in r["review_flags"]})),
            "maximum_lengths": {key: max((r["lengths"][key] for r in subset), default=0)
                                for key in ("en_characters", "tw_characters", "en_words", "tw_words")},
            "mapping_review_groups": dict(Counter(g["code"] for g in groups if g["candidate_class"] == kind))}
    summary = {"version": VERSION, "source_inventory_manifest_sha256": inventory_sha, "counts_by_class": counts,
        "original_pairs": len(pairs), "derived_views": len(pairs) * 2, "training_eligible_views": 0,
        "training_schedule_created": False, "exceeds_50000_training_schedule_cap": len(pairs) * 2 > 50000,
        "schedule_limit_note": "All derived views are registered, not scheduled. A later admitted finite mixture must obey the training cap.",
        "protection": protection_receipt, "token_lengths_measured": False, "sequence_fit_claimed": False,
        "targets_changed": 0, "synthetic_twi_created": 0, "api_calls": 0, "gpu_calls": 0,
        "limits": ["Every view remains pending a quality decision, including views with no automatic flags.",
                   "One-to-many lexical mappings can be legitimate polysemy, synonyms or source errors; flags do not adjudicate them.",
                   "Length, identical-language and numeric differences are possible-misalignment selectors, not semantic correctness tests.",
                   "PII screens detect contact patterns only; absence of flags is not a privacy or semantic certification.",
                   "No dictionary definition, conversation, intent, explanation or source meaning was invented.",
                   "Inherited noncommercial/attribution terms remain; no new commercial, clinical or native-language certification."]}
    examples = []
    for kind in sorted(CLASSES):
        subset = sorted((r for r in pairs if r["candidate_class"] == kind), key=lambda r: digest(r["id"]))
        selected = {r["id"]: r for r in subset[:3]}
        for code in counts[kind]["flags_by_pair_nonexclusive"]:
            for row in subset:
                if any(f["code"] == code for f in row["review_flags"]):
                    selected[row["id"]] = row
                    break
        examples.extend({"selection": "deterministic class sample plus one example per automatic flag", **r} for r in selected.values())
    inputs.recheck()
    out.mkdir(parents=True, mode=0o700)
    inventory.write_rows(out / "source-pairs.jsonl", pairs)
    inventory.write_rows(out / "mapping-review-groups.jsonl", groups)
    inventory.write_rows(out / "representative-examples.jsonl", examples)
    source_manifests = []
    for kind in sorted(CLASSES):
        target = out / "datasets" / kind
        target.mkdir(parents=True)
        inventory.write_rows(target / "views.jsonl", views(pairs, kind, inventory_sha))
        data = {"dataset_id": "gha-r7-alignment-" + kind, "dataset_version": VERSION,
                "candidate_class": kind, "record_type": CLASSES[kind], "original_pair_count": counts[kind]["original_pairs"],
                "record_count": counts[kind]["derived_views"], "records_path": f"datasets/{kind}/views.jsonl",
                "records_sha256": sha(target / "views.jsonl"), "source_pairs_path": "source-pairs.jsonl",
                "source_pairs_sha256": sha(out / "source-pairs.jsonl"), "split": "train",
                "admission_status": "pending_quality_decision", "training_eligible": False, "production_eligible": False}
        inventory.write_json(target / "manifest.json", data)
        source_manifests.append({"dataset_id": data["dataset_id"], "dataset_version": VERSION,
                                 "manifest_path": f"datasets/{kind}/manifest.json", "manifest_sha256": sha(target / "manifest.json")})
    inventory.write_json(out / "source-manifests.json", source_manifests)
    inventory.write_json(out / "summary.json", summary)
    inputs.recheck()
    inventory.write_json(out / "manifest.json", {"version": VERSION, "source_inventory": str(folder),
        "source_inventory_manifest_sha256": inventory_sha, "input_files": inputs.files,
        "artifacts": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "admission_status": "pending_quality_decision", "training_eligible": False, "production_eligible": False})
    inventory.write_json(out / "SEALED.json", {"manifest_sha256": sha(out / "manifest.json")})
    verify(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--inventory", type=Path, default=ROOT / "tmp/r7-20261117/source-inventory/v4")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/r7-20261117/alignment-candidates-v1")
    args = parser.parse_args()
    result = prepare(args.inventory, args.out) if args.action == "prepare" else {**verify(args.out), "verified": True}
    print(json.dumps({key: result[key] for key in ("version", "counts_by_class", "original_pairs", "derived_views", "training_eligible_views")
                      if key in result} | ({"verified": True} if args.action == "verify" else {}), indent=2))


if __name__ == "__main__":
    main()
