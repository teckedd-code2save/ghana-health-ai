import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from corpus_annotation import execute_batch, validate_fields
from corpus_release import digest, save
from corpus_tools import build, execute, verify_trajectory


class FieldTests(unittest.TestCase):
    def test_incorrect_types_are_not_accepted(self):
        for key, value in (("meaning_english", 9), ("record_function", "HEALTH"),
                           ("source_grounded_entities", None), ("ambiguity", "maybe"),
                           ("source_answer_support", {}), ("medical_contradictions", "none"),
                           ("grounded_twi_derivative", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_fields({"missing_fields": [key]}, {key: value})

    def test_medical_claims_require_source_spans(self):
        row = {"missing_fields": ["medical_contradictions"], "reference_answer": "Original answer"}
        with self.assertRaises(ValueError):
            validate_fields(row, {"medical_contradictions": [{"claim": "Invented dose", "concern": "Unsupported"}]})

    def test_unknown_meaning_stays_null(self):
        row = {"missing_fields": ["meaning_english", "ambiguity"]}
        value = {"meaning_english": None, "ambiguity": ["Missing context"]}
        self.assertEqual(validate_fields(row, value), value)

    def test_narrative_cannot_be_labelled_as_patient_request(self):
        row = {"missing_fields": ["record_function", "conversational_intent"]}
        with self.assertRaises(ValueError):
            validate_fields(row, {"record_function": "narrative", "conversational_intent": "request treatment"})

    def test_partial_batch_uses_cached_result_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            batch = [{"request_id": str(i), "source_id": str(i), "source_hash": digest(i),
                      "missing_fields": ["meaning_english"], "source_text": "source"} for i in range(2)]
            cache = folder / ("batch-" + digest(["0", "1"]) + ".receipt.results.json")
            report = {"model": "test", "revision": "pinned", "results": [
                {"id": str(i), "text": '{"meaning_english": "preserved meaning"}', "finish_reason": "stop"} for i in range(2)]}
            save(cache, report)
            save(folder / "0.json", {"request_id": "0", "source_hash": batch[0]["source_hash"], "sentinel": True})
            before = (folder / "0.json").read_bytes()
            # No Modal methods are present: any accidental request fails the test.
            self.assertTrue(execute_batch(batch, folder, SimpleNamespace()))
            self.assertEqual((folder / "0.json").read_bytes(), before)
            self.assertFalse(execute_batch(batch, folder, SimpleNamespace()))

    def test_tool_final_cannot_disagree_with_actual_result(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "tools"
            build(destination)
            rows = [json.loads(line) for line in (destination / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 36)
            for row in rows: verify_trajectory(row)
            rows[0]["messages"][-1]["content"] = "I placed your order for GHS 900."
            with self.assertRaises(ValueError): verify_trajectory(rows[0])
            with self.assertRaises(ValueError): execute("calculate_total", {"sku": "SIM-SOAP-A", "quantity": "0.5"})


if __name__ == "__main__": unittest.main()
