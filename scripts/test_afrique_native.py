import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare_afrique_native as client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal/train"))
from evaluate_afrique_native import validate_inputs


class AfriqueNativeTests(unittest.TestCase):
    def test_inputs_reject_answers_duplicates_and_malformed_rows(self):
        row = {"id": "a", "task": "nli", "prompt": "Question"}
        validate_inputs([row])
        for rows in ([], [None], [row, row], [{**row, "answer": "A"}], [{**row, "prompt": ""}]):
            with self.assertRaises(ValueError):
                validate_inputs(rows)

    def test_real_source_selection_and_no_heldout_answers(self):
        inputs, refs = client.prepare_cases()
        validate_inputs(inputs)
        self.assertEqual(len(inputs), 964)
        self.assertEqual(len(refs["nli"]), 900)
        self.assertEqual(len(refs["translation"]), 64)
        self.assertEqual(refs["translation_source_groups"], 8)
        source_ids = {r["source_record_id"] for r in refs["translation"]}
        self.assertFalse(source_ids & set(client.DEMO_SOURCE_IDS))
        self.assertEqual(len(source_ids), 32)
        by_id = {r["id"]: r for r in inputs}
        for row in refs["translation"]:
            question, _ = client.translation_text(row)
            self.assertTrue(by_id[row["id"]]["prompt"].endswith(question))
        for row in refs["nli"]:
            self.assertEqual(by_id[row["id"]]["prompt"], client.DEMONSTRATIONS + row["messages"][0]["content"] + "\nAnswer: ")
        self.assertFalse(refs["demonstrations_human_verified"])

    def test_reference_snapshot_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(client, "OUT", Path(directory)):
            raw = json.dumps({"nli": []}).encode()
            receipt = {"references_sha256": hashlib.sha256(raw).hexdigest()}
            path = Path(directory) / "references.json"
            path.write_bytes(raw)
            self.assertEqual(client.verified_snapshot(receipt, "references"), {"nli": []})
            path.write_text("{}")
            with self.assertRaises(ValueError):
                client.verified_snapshot(receipt, "references")

    def test_incomplete_results_are_not_scored(self):
        _, refs = client.prepare_cases()
        with self.assertRaises(ValueError):
            client.score(refs, {"base": {"results": []}, "afrique": {"results": []}})


if __name__ == "__main__":
    unittest.main()
