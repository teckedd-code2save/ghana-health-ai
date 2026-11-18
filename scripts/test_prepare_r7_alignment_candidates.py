"""CPU-only alignment registration tests using local, explicitly artificial fixtures."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import corpus_record_types as types
import prepare_r7_alignment_candidates as alignment
import prepare_r7_source_inventory as inventory


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def seal_inventory(folder, bound):
    manifest = {"version": inventory.VERSION, "training_eligible": False,
        "input_files": {str(path): {"sha256": inventory.sha(path), "bytes": path.stat().st_size} for path in bound},
        "artifacts": {p.name: inventory.sha(p) for p in folder.iterdir() if p.name not in {"manifest.json", "SEALED.json"}}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "SEALED.json").write_text(json.dumps({"manifest_sha256": inventory.sha(folder / "manifest.json")}))


def fixture(root, pairs=None, held_text="Held private validation example unrelated to the training references."):
    """Use source placeholders, not newly authored Twi supervision."""
    pairs = pairs or [("A complete original English sentence.", "[original-reference-A]"),
                      ("Another complete original English sentence.", "[original-reference-B]")]
    raw = [{"id": str(i), "text": en, "label": tw, "Unnamed: 3": ""} for i, (en, tw) in enumerate(pairs)]
    source = root / "ghana_nlp_parallel" / "original.csv"
    source.parent.mkdir()
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    inherited = root / "inherited.jsonl"
    old_rows, candidates = [], []
    for index, (en, tw) in enumerate(pairs):
        source_id = "ghana_nlp_parallel:" + str(index)
        assessment = types.classify(en, tw, "ghana_nlp_parallel", "professional_parallel", "sentence", "compiled_parallel_text")
        refs = []
        for a, b, x, y in (("en", "tw", en, tw), ("tw", "en", tw, en)):
            row = {"source_id": source_id, "split": "train", "generated_target": False,
                   "source_file_sha256": inventory.sha(source), "source_hash": inventory.digest(raw[index]),
                   "source_language": a, "target_language": b, "type_assessment": assessment,
                   "messages": [{"role": "user", "content": json.dumps({"source_text": x})},
                                {"role": "assistant", "content": y}]}
            old_rows.append(row)
            refs.append({"file": str(inherited), "line": len(old_rows), "record_sha256": inventory.digest(row)})
        candidates.append({"id": "parallel:" + source_id, "source_id": source_id, "source_family": "ghana_nlp_parallel",
            "source_genre": "professional_parallel", "source_split": "train", "status": "eligible",
            "routing": assessment["view"], "type_assessment": assessment, "generated_target": False,
            "training_eligible": False, "source_family_keys": [source_id], "inherited_refs": refs,
            "source_ref": {"file": str(source), "file_sha256": inventory.sha(source), "row_index": index,
                           "record_sha256": inventory.digest(raw[index])},
            "english_sha256": inventory.digest(en), "twi_sha256": inventory.digest(tw),
            "semantic_reference": {"english_field": "text", "twi_field": "label",
                "original_pair_content_sha256": assessment["content_sha256"],
                "inherited_pair_content_sha256": assessment["content_sha256"],
                "inherited_source_hash": inventory.digest(raw[index]), "meaning_correctness_certified": False}})
    write_rows(inherited, old_rows)
    for candidate in candidates:
        for ref in candidate["inherited_refs"]:
            ref["file_sha256"] = inventory.sha(inherited)
    held = root / "held-content.jsonl"
    write_rows(held, [{"id": "held", "text": held_text}])
    folder = root / "inventory"
    folder.mkdir()
    write_rows(folder / "parallel-meaning-candidates.jsonl", candidates)
    write_rows(folder / "parallel-meaning-decisions.jsonl", candidates)
    write_rows(folder / "protection-index.jsonl", [{"normalized_text_sha256": inventory.digest(inventory.normalize(held_text))}])
    bound = [source, inherited, held, Path(types.__file__).absolute()]
    seal_inventory(folder, bound)
    return folder, bound, candidates


class RegistrationTests(unittest.TestCase):
    def test_complete_originals_both_directions_no_conversation_no_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            originals = [("A complete original English sentence.\nPreserve whitespace!", " [reference-A]\n"),
                         ("Another complete original English sentence.", "[reference-B]")]
            folder, bound, candidates = fixture(root, originals)
            checksums = {p: inventory.sha(p) for p in bound}
            output = root / "output"
            summary = alignment.prepare(folder, output)
            self.assertEqual(summary["original_pairs"], 2)
            self.assertEqual(summary["derived_views"], 4)
            self.assertEqual(summary["training_eligible_views"], 0)
            self.assertFalse(summary["training_schedule_created"])
            self.assertFalse(summary["sequence_fit_claimed"])
            pairs = rows(output / "source-pairs.jsonl")
            exported = rows(output / "datasets/sentences/views.jsonl")
            for i, pair in enumerate(pairs):
                self.assertEqual((pair["english"], pair["twi"]), originals[i])
                self.assertEqual(pair["record_sha256"], inventory.digest({k: v for k, v in pair.items() if k != "record_sha256"}))
                self.assertEqual(pair["original_refs"]["inventory_candidate_sha256"], inventory.digest(candidates[i]))
                a, b = exported[2 * i:2 * i + 2]
                self.assertEqual((a["source_text"], a["target_text"]), originals[i])
                self.assertEqual((b["source_text"], b["target_text"]), originals[i][::-1])
                for view in (a, b):
                    self.assertNotIn("messages", view)
                    self.assertNotIn("intent", view)
                    self.assertNotIn("explanation", view)
                    self.assertFalse(view["training_eligible"] or view["generated_target"] or view["semantic_certification"])
                    self.assertEqual(view["source_pair_sha256"], pair["record_sha256"])
                    self.assertEqual(view["admission_status"], "pending_quality_decision")
            self.assertEqual(checksums, {p: inventory.sha(p) for p in bound})
            self.assertEqual(len(alignment.verify(output)["artifacts"]), 11)
            with self.assertRaisesRegex(ValueError, "new alignment"):
                alignment.prepare(folder, output)
            with (output / "datasets/sentences/views.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                alignment.verify(output)

    def test_original_source_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, _ = fixture(root)
            with bound[0].open("a") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                alignment.prepare(folder, root / "output")
            self.assertFalse((root / "output").exists())

    def test_inherited_targets_must_equal_full_originals_even_with_rebound_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, candidates = fixture(root)
            old = rows(bound[1])
            old[0]["messages"][-1]["content"] = "shortened reference"
            write_rows(bound[1], old)
            for candidate in candidates:
                for ref in candidate["inherited_refs"]:
                    ref.update(file_sha256=inventory.sha(bound[1]), record_sha256=inventory.digest(old[ref["line"] - 1]))
            write_rows(folder / "parallel-meaning-candidates.jsonl", candidates)
            seal_inventory(folder, bound)
            with self.assertRaisesRegex(ValueError, "complete original pair"):
                alignment.prepare(folder, root / "output")

    def test_missing_reverse_direction_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, candidates = fixture(root)
            candidates[0]["inherited_refs"] = candidates[0]["inherited_refs"][:1]
            write_rows(folder / "parallel-meaning-candidates.jsonl", candidates)
            seal_inventory(folder, bound)
            with self.assertRaisesRegex(ValueError, "Both inherited"):
                alignment.prepare(folder, root / "output")

    def test_changed_identity_type_split_or_admission_rejected(self):
        mutations = [{"source_split": "validation"}, {"training_eligible": True}, {"routing": "documents"},
                     {"generated_target": True}, {"source_id": "ghana_nlp_parallel:999"},
                     {"english_sha256": "0" * 64}]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                folder, bound, candidates = fixture(root)
                candidates[0].update(mutation)
                write_rows(folder / "parallel-meaning-candidates.jsonl", candidates)
                seal_inventory(folder, bound)
                with self.assertRaises(ValueError):
                    alignment.prepare(folder, root / "output")

    def test_holdout_rechecked_not_boolean_stamp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, _, _ = fixture(root, held_text="A complete original English sentence.")
            with self.assertRaisesRegex(ValueError, "Protected content"):
                alignment.prepare(folder, root / "output")

    def test_missing_holdout_fingerprint_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, _ = fixture(root)
            write_rows(folder / "protection-index.jsonl", [{"normalized_text_sha256": "0" * 64}])
            seal_inventory(folder, bound)
            with self.assertRaisesRegex(ValueError, "exact inventory holdout"):
                alignment.prepare(folder, root / "output")

    def test_inventory_artifacts_cannot_change_unsealed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, _, candidates = fixture(root)
            write_rows(folder / "parallel-meaning-candidates.jsonl", candidates[:1])
            with self.assertRaisesRegex(ValueError, "Inventory artifact changed"):
                alignment.prepare(folder, root / "output")

    def test_original_validation_sources_and_parallel_split_are_protected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, candidates = fixture(root)
            validation = copy.deepcopy(candidates[1])
            validation.update(source_split="validation", status="excluded")
            write_rows(folder / "parallel-meaning-candidates.jsonl", candidates[:1])
            write_rows(folder / "parallel-meaning-decisions.jsonl", [candidates[0], validation])
            source_texts = ["Protected OASST original answer", "Protected health question", "Protected health answer"]
            oasst = root / "oasst_validation" / "validation.parquet"
            oasst.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([{"text": source_texts[0], "user_id": "not-exported"}]), oasst)
            health = root / "afrihealth-qa" / "test_holdout.parquet"
            health.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([{"question": source_texts[1], "answer": source_texts[2]}]), health)
            protected_texts = [rows(bound[2])[0]["text"], *source_texts,
                               "Another complete original English sentence.", "[original-reference-B]"]
            write_rows(folder / "protection-index.jsonl", [
                {"normalized_text_sha256": inventory.digest(inventory.normalize(text))} for text in protected_texts])
            seal_inventory(folder, bound + [oasst, health])
            summary = alignment.prepare(folder, root / "output")
            self.assertEqual(summary["original_pairs"], 1)
            self.assertEqual(summary["protection"]["distinct_normalized_held_texts"], 6)
            exported = (root / "output/source-pairs.jsonl").read_text()
            self.assertNotIn("not-exported", exported)
            self.assertNotIn("[original-reference-B]", exported)

    def test_inherited_split_and_recomputed_assessment_not_caller_stamps(self):
        for change in ("split", "assessment"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                folder, bound, candidates = fixture(root)
                if change == "split":
                    inherited = rows(bound[1])
                    inherited[0]["split"] = "validation"
                    write_rows(bound[1], inherited)
                    for candidate in candidates:
                        for ref in candidate["inherited_refs"]:
                            ref.update(file_sha256=inventory.sha(bound[1]), record_sha256=inventory.digest(inherited[ref["line"] - 1]))
                else:
                    candidates[0]["type_assessment"]["english_word_count"] += 100
                write_rows(folder / "parallel-meaning-candidates.jsonl", candidates)
                seal_inventory(folder, bound)
                with self.assertRaises(ValueError):
                    alignment.prepare(folder, root / "output")

    def test_extra_or_linked_output_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, _ = fixture(root)
            output = root / "output"
            alignment.prepare(folder, output)
            extra = output / "extra.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unbound alignment"):
                alignment.verify(output)
            extra.unlink()
            extra.symlink_to(bound[0])
            with self.assertRaisesRegex(ValueError, "Linked alignment"):
                alignment.verify(output)

    def test_parquet_dictionary_projection_and_routing_kept_separate(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder, bound, candidates = fixture(root)
            source = root / "community_parallel" / "train.parquet"
            source.parent.mkdir()
            raw = {"eng": "a word", "twi": "[reference]", "count": 4, "source_files": "02_english-twi-dictionary"}
            pq.write_table(pa.Table.from_pylist([raw]), source)
            assessment = types.classify(raw["eng"], raw["twi"], "community_parallel", raw["source_files"], "sentence", "compiled_parallel_text")
            candidate = copy.deepcopy(candidates[0])
            candidate.update(id="parallel:community_parallel:0", source_id="community_parallel:0", source_family="community_parallel",
                             source_family_keys=["community_parallel:0"], source_genre=raw["source_files"], routing="lexicon",
                             type_assessment=assessment, english_sha256=inventory.digest(raw["eng"]), twi_sha256=inventory.digest(raw["twi"]),
                             source_ref={"file": str(source), "file_sha256": inventory.sha(source), "row_index": 0,
                                         "projection_sha256": inventory.digest(raw)})
            candidate["semantic_reference"].update(english_field="eng", twi_field="twi", inherited_source_hash=inventory.digest(raw),
                original_pair_content_sha256=assessment["content_sha256"], inherited_pair_content_sha256=assessment["content_sha256"])
            old = rows(bound[1])[:2]
            for i, row in enumerate(old):
                row.update(source_id=candidate["source_id"], source_file_sha256=inventory.sha(source), source_hash=inventory.digest(raw), type_assessment=assessment)
                x, y = (raw["eng"], raw["twi"]) if i == 0 else (raw["twi"], raw["eng"])
                row["messages"][0]["content"] = json.dumps({"source_text": x})
                row["messages"][1]["content"] = y
            write_rows(bound[1], old)
            for ref in candidate["inherited_refs"]:
                ref.update(file_sha256=inventory.sha(bound[1]), record_sha256=inventory.digest(old[ref["line"] - 1]))
            write_rows(folder / "parallel-meaning-candidates.jsonl", [candidate])
            write_rows(folder / "parallel-meaning-decisions.jsonl", [candidate])
            seal_inventory(folder, bound + [source])
            summary = alignment.prepare(folder, root / "output")
            self.assertEqual(summary["counts_by_class"]["lexicon"]["original_pairs"], 1)
            self.assertEqual(summary["counts_by_class"]["sentences"]["original_pairs"], 0)
            views = rows(root / "output/datasets/lexicon/views.jsonl")
            self.assertTrue(all(v["record_type"] == "dictionary_entry" for v in views))
            self.assertEqual(views[0]["target_text"], raw["twi"])


class ReviewTests(unittest.TestCase):
    def test_duplicates_and_mapping_conflicts_are_per_class_not_adjudications(self):
        values = [("bank", "reference-a"), ("bank", "reference-b"), ("shore", "reference-b"),
                  ("bank", "reference-a"), (" Bank! ", "REFERENCE-A")]
        pairs = [{"id": str(i), "candidate_class": "lexicon", "english": en, "twi": tw, "review_flags": []}
                 for i, (en, tw) in enumerate(values)]
        pairs.append({**pairs[0], "id": "sentence", "candidate_class": "sentences", "review_flags": []})
        reports = alignment.mapping_review(pairs)
        self.assertEqual({r["code"]: r["source_records"] for r in reports}, {
            "duplicate_exact_pair": 2, "duplicate_normalized_pair": 3,
            "english_maps_to_multiple_twi_review_polysemy_or_conflict": 4,
            "twi_maps_to_multiple_english_review_polysemy_or_conflict": 2})
        self.assertFalse(pairs[-1]["review_flags"])
        self.assertTrue(all(not r["semantic_error_confirmed"] for r in reports))
        self.assertEqual([(p["english"], p["twi"]) for p in pairs[:5]], values)

    def test_length_unicode_pii_and_possible_misalignment_are_review_only(self):
        text = "e\u0301\u200b\ufffd\x01 person@example.test 2 " + "longword " * 600
        codes = {f["code"] for f in alignment.pair_flags(text, "reference 5")}
        self.assertTrue({"non_nfc_preserved_without_normalization", "unicode_format_character_review",
                        "invalid_unicode_or_control_character", "possible_email",
                        "numeric_literal_difference_possible_misalignment",
                        "reference_over_4096_characters_not_token_measurement",
                        "possible_misalignment_word_length_ratio_over_four"} <= codes)
        self.assertEqual(len(alignment.pair_flags("loanword", "LOANWORD")), 1)
        self.assertEqual(alignment.pair_flags("ordinary reference", "separate source reference"), [])

    def test_full_view_generation_has_no_sample_or_schedule_cap(self):
        pair = {"id": "pair", "candidate_class": "lexicon", "record_type": "dictionary_entry", "english": "en", "twi": "tw",
                "source_id": "source", "record_sha256": "a" * 64, "original_refs": {}, "semantic_reference": {},
                "review_flags": [], "source_family": "family", "source_genre": "genre", "source_family_keys": [], "type_assessment": {}}
        self.assertEqual(sum(1 for _ in alignment.views([pair] * 28025, "lexicon", "b" * 64)), 56050)


if __name__ == "__main__":
    unittest.main()
