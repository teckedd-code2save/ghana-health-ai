"""Offline regression checks for corpus boundaries, provenance and field reuse."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from corpus_release import ROOT, DiskNearIndex, Groups, digest, flags_for, views_for, jsonl
from corpus_annotation import validate_fields, prompt
from corpus_schemas import LanguageView


def source(**changes):
    return {"id": "fixture:1", "source": "fixture", "revision": "test", "repo": "fixture",
            "text": "Medaase", "language": "tw", "record_type": "utterance",
            "origin": "source_preserved", **changes}


class CorpusContractTests(unittest.TestCase):
    def test_export_schema_rejects_missing_provenance_and_extra_targets(self):
        row = {"schema_version": 1, "source_id": "fixture", "source_hash": "a" * 64,
               "source_file": "fixture.jsonl", "source_file_sha256": "b" * 64,
               "source_row_index": 0, "source_revision": "pinned", "group_id": "c" * 64,
               "split": "train", "language": "tw", "record_type": "utterance",
               "origin": "source_preserved", "topic": "unclassified", "license": None,
               "human_reviewed": False, "production_eligible": False, "semantic_certification": False,
               "screening": "automated_source_preserved", "id": "fixture:language", "tokens": 3, "text": "Medaase"}
        LanguageView.model_validate(row)
        with self.assertRaises(ValueError): LanguageView.model_validate({**row, "source_hash": "unknown"})
        with self.assertRaises(ValueError): LanguageView.model_validate({**row, "invented_reply": "answer"})

    def test_short_useful_text_is_not_discarded(self):
        row = source()
        self.assertEqual(flags_for(row, set()), [])
        self.assertEqual(views_for(row, [])[0][0], "language/short_utterances")

    def test_uncertain_dialect_not_silently_twi(self):
        row = source(language="ak")
        flags = flags_for(row, set())
        self.assertIn("dialect_not_resolved", flags)
        self.assertEqual(views_for(row, flags), [])

    def test_synthetic_dialogue_is_not_conversation(self):
        row = source(origin="synthetic_gemini_news_conditioned", record_type="document", genre="Dialogue")
        self.assertEqual(views_for(row, []), [])

    def test_english_reference_untouched_and_numbers_checked(self):
        row = source(record_type="sentence", text="tomato 2 kg", english="tomatoes 2 kg")
        alignment = dict(views_for(row, []))["meaning/alignment"]
        self.assertEqual(alignment["english"], row["english"])
        row["english"] = "tomatoes 20 kg"
        self.assertNotIn("meaning/alignment", dict(views_for(row, [])))

    def test_medical_structural_pass_is_not_semantic_pass(self):
        self.assertEqual(views_for(source(record_type="health_qa", reference_answer="unverified"), []), [])

    def test_exact_entity_offsets_required(self):
        row = {"source_text": "tomato 2 kg", "missing_fields": ["source_grounded_entities"]}
        value = {"source_grounded_entities": [{"type": "quantity", "text": "2 kg", "start": 7, "end": 11}]}
        self.assertEqual(validate_fields(row, value), value)
        value["source_grounded_entities"][0]["start"] = 6
        with self.assertRaises(ValueError): validate_fields(row, value)

    def test_unrequested_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_fields({"missing_fields": ["ambiguity"]}, {"ambiguity": [], "reply": "invented"})

    def test_history_cannot_be_spliced(self):
        row = {"missing_fields": ["grounded_twi_derivative"], "messages": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]}
        with self.assertRaises(ValueError):
            validate_fields(row, {"grounded_twi_derivative": {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}})

    def test_related_groups_transitive(self):
        groups = Groups()
        for _ in range(5): groups.add()
        groups.join(1, 2); groups.join(3, 4); groups.join(2, 4)
        self.assertEqual(len({groups.root(i) for i in range(1, 5)}), 1)
        self.assertNotEqual(groups.root(0), groups.root(1))

    def test_disk_near_restarts_without_memory_state(self):
        with tempfile.TemporaryDirectory() as folder:
            db = sqlite3.connect(Path(folder) / "index.sqlite")
            text = " ".join("word" + str(i) for i in range(50))
            first = DiskNearIndex(db)
            self.assertEqual(first.add(0, text), ([], False))
            db.commit()
            restored = DiskNearIndex(db)
            matches, overflow = restored.add(1, text + " another")
            self.assertIn(0, matches)
            self.assertFalse(overflow)
            db.close()

    def test_identity_preserves_original_spelling(self):
        self.assertNotEqual(digest("ɛ"), digest("e"))

    def test_afriqa_numeric_and_string_ids_join(self):
        folder = ROOT / "tmp/semantic-response-v4/sources"
        for split in ("train", "dev", "test"):
            queries = {str(r["id"]): r for r in jsonl(folder / f"queries.afriqa.twi.en.{split}.json")}
            for row in jsonl(folder / f"gold_span_passages.afriqa.twi.en.{split}.json"):
                self.assertIn(str(row["id"]), queries)


if __name__ == "__main__": unittest.main()
