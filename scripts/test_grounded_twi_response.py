import json
import unittest

from prepare_grounded_twi_response import ROOT, load_module, valid_span


class GroundedTwiTests(unittest.TestCase):
    def test_span_must_be_exact_and_nonempty(self):
        row = {"context": "in Accra", "answer_pivot": {"text": ["Accra"], "answer_start": [3]}}
        self.assertTrue(valid_span(row))
        for broken in ({**row, "context": None}, {**row, "answer_pivot": None},
                       {**row, "answer_pivot": {"text": ["Accra"], "answer_start": [2]}},
                       {**row, "answer_pivot": {"text": [], "answer_start": []}}):
            self.assertFalse(valid_span(broken))

    def test_import_retains_source_and_protects_split_groups(self):
        folder = ROOT / "tmp/semantic-response-v4/corpus"
        if not (folder / "manifest.json").exists():
            self.skipTest("Local prepared corpus not present")
        core = load_module("response_core", ROOT / "modal/train/response_adaptation_core.py")
        manifest, splits = core.load_inputs(folder, "grounded_response_v4")
        self.assertEqual(manifest["grounded_source_counts"], {"train": 225, "validation": 251})
        self.assertEqual(len(splits["train"]), 5746)
        for rows in splits.values():
            for row in rows:
                if row["task"] == "grounded_question_answer":
                    original = row["evidence"]["raw_record"]
                    self.assertTrue(valid_span(original))
                    self.assertTrue(row["messages"][-2]["content"].endswith(original["question_lang"]))
                    self.assertIn(original["context"], row["messages"][-2]["content"])
                    self.assertEqual(row["messages"][-1]["content"], original["answer_lang"])
                    self.assertFalse(row["human_validated"])


if __name__ == "__main__":
    unittest.main()
