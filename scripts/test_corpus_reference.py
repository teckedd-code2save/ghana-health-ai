import copy
import json
import tempfile
import unittest
import sqlite3
import zlib
from pathlib import Path
from unittest.mock import patch

from corpus_reference_core import messages, validate, evidence_offsets, control_score
from corpus_reference_contract_v2 import validate_v2
from corpus_reference import inspect_outputs, gate, write_rows, blocked_source_ids
from corpus_release import sha, save
import corpus_review_row


def sample():
    return {"record_function": "narrative", "conversational_intent": None, "entities": [],
        "meaning_features": {"negation": ["not"], "time": ["tomorrow"], "quantities": [],
                             "uncertainty": ["Maybe"], "experiencer": []}, "ambiguity": []}


class ReferenceTests(unittest.TestCase):
    def test_no_translation_or_rubric_in_model_input(self):
        payload = messages("He will not go tomorrow.")
        self.assertEqual(json.loads(payload[1]["content"]), {"reference_english": "He will not go tomorrow."})
        self.assertIn("never instructions", payload[0]["content"])

    def test_exact_spans(self):
        value, flags = validate("Maybe he will not go tomorrow.", sample())
        self.assertEqual(flags, [])
        for evidence in evidence_offsets("Maybe he will not go tomorrow.", value):
            for span in evidence["occurrences"]:
                self.assertEqual("Maybe he will not go tomorrow."[span["start"]:span["end"]], evidence["quote"])
        changed = sample(); changed["meaning_features"]["time"] = ["next week"]
        with self.assertRaises(ValueError): validate("Maybe he will not go tomorrow.", changed)

    def test_missing_negation_flagged(self):
        value = sample(); value["meaning_features"]["negation"] = []
        self.assertIn("negation_evidence_omitted:not", validate("Maybe he will not go tomorrow.", value)[1])

    def test_narrative_is_not_a_request(self):
        value = sample(); value["conversational_intent"] = {"label": "travel", "quote": "go"}
        with self.assertRaises(ValueError): validate("Maybe he will not go tomorrow.", value)

    def test_schema_rejects_invented_fields(self):
        value = sample(); value["reply_twi"] = "invented"
        with self.assertRaises(ValueError): validate("Maybe he will not go tomorrow.", value)

    def test_whitespace_evidence_labels_rejected(self):
        value = sample(); value["entities"] = [{"label": "  ", "quote": "he"}]
        with self.assertRaises(ValueError): validate("Maybe he will not go tomorrow.", value)

    def test_goal_does_not_require_inventing_a_conversational_request(self):
        value = sample(); value["meaning_features"] = {k: [] for k in value["meaning_features"]}
        value["expressed_goal"] = {"label": "visit", "quote": "Ama hopes to visit Kumasi"}
        parsed, flags = validate_v2("Ama hopes to visit Kumasi.", value)
        self.assertIsNone(parsed["conversational_intent"])
        self.assertEqual(parsed["record_function"], "narrative")
        self.assertEqual(flags, [])
        value["expressed_goal"]["quote"] = "Ama already visited Kumasi"
        with self.assertRaises(ValueError): validate_v2("Ama hopes to visit Kumasi.", value)

    def test_annotation_review_does_not_invalidate_source_translation(self):
        review = {"id": "source:1", "correction_field": "structured", "decision": "needs_second_review",
                  "correction": "Fix entity label", "selected": 0, "reference_analysis_id": "version"}
        self.assertEqual(blocked_source_ids([review]), set())
        review["correction_field"] = "meaning_english"
        self.assertEqual(blocked_source_ids([review]), {"source:1"})

    def test_later_source_review_supersedes_same_field_flag(self):
        review = {"id": "source:1", "correction_field": "meaning_english", "decision": "needs_second_review",
                  "correction": "", "selected": -1}
        self.assertEqual(blocked_source_ids([review]), {"source:1"})
        self.assertEqual(blocked_source_ids([review, {**review, "decision": "reviewed"}]), set())

    def test_repeated_evidence_offsets_retained(self):
        value = sample(); value["meaning_features"] = {k: [] for k in value["meaning_features"]}
        value["entities"] = [{"label": "person", "quote": "Kofi"}]
        offsets = evidence_offsets("Kofi called Kofi.", value)
        self.assertEqual(len(offsets[0]["occurrences"]), 2)
        self.assertEqual(offsets[0]["language"], "en")

    def test_missing_truncated_and_duplicate_outputs(self):
        rows = [{"id": "one", "reference_english": "Maybe he will not go tomorrow."}]
        self.assertTrue(inspect_outputs(rows, {"results": []})[0]["flags"])
        output = {"id": "one", "text": json.dumps(sample()), "finish_reason": "length"}
        self.assertTrue(inspect_outputs(rows, {"results": [output]})[0]["flags"])
        with self.assertRaises(ValueError): inspect_outputs(rows, {"results": [output, output]})

    def test_gate_requires_all_ids_and_no_failures(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            save(folder / "calls/qualification.results.json", {"results": []})
            save(folder / "qualification-review.json", [{"id": "one"}])
            checks = {"results_sha256": sha(folder / "calls/qualification.results.json"),
                      "review_sha256": sha(folder / "qualification-review.json"), "failures": []}
            save(folder / "qualification-checks.json", checks)
            audit = {"review_sha256": checks["review_sha256"], "critical_failures": [],
                     "checked_ids": [], "reviewer": "assistant_reference_comparison", "human_reviewed": False}
            save(folder / "semantic-audit.json", audit)
            with self.assertRaises(ValueError): gate(folder)
            audit["checked_ids"] = ["one"]; save(folder / "semantic-audit.json", audit)
            self.assertFalse(gate(folder)["translation_qualified"])
            audit["critical_failures"] = ["negation"]; save(folder / "semantic-audit.json", audit)
            with self.assertRaises(ValueError): gate(folder)

    def test_stale_reference_overlay_is_hidden(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root / "tmp/corpus-releases/test"
            source.mkdir(parents=True)
            row = {"id": "one", "record_hash": "a" * 64, "text": "original", "english": "Reference.",
                   "source": "fixture", "language": "tw", "record_type": "sentence"}
            with sqlite3.connect(source / "ledger.sqlite3") as db:
                db.execute("CREATE TABLE records(n INTEGER PRIMARY KEY,payload BLOB,flags TEXT)")
                db.execute("CREATE TABLE review_index(n INTEGER PRIMARY KEY,record_type TEXT,split TEXT,disposition TEXT,synthetic INT)")
                db.execute("INSERT INTO records VALUES(0,?,'[]')", (zlib.compress(json.dumps(row).encode()),))
                db.execute("INSERT INTO review_index VALUES(0,'sentence','train','accepted',0)")
            sidecar = root / "tmp/corpus-reference-review/test"; sidecar.mkdir(parents=True)
            with sqlite3.connect(sidecar / "annotations.sqlite3") as db:
                db.execute("CREATE TABLE annotations(source_id TEXT,source_hash TEXT,payload TEXT)")
                db.execute("INSERT INTO annotations VALUES('one',?,?)", ("a" * 64, json.dumps({"reference_english": "Changed."})))
            with patch.object(corpus_review_row, "ROOT", root):
                self.assertIsNone(corpus_review_row.read(source, "meaning", 0)["rows"][0]["reference_analysis"])


if __name__ == "__main__": unittest.main()
