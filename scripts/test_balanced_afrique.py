import copy
import json
import unittest

import build_balanced_afrique as core


class BalancedAfriqueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.exclusions = core.build()

    def test_reserved_sources_and_complete_task_contract(self):
        reserved = core.keys([*core.read(core.PARALLEL, "validation"), *core.intent_rows("test"), *self.splits["validation"]])
        core.validate(self.splits, reserved)
        for r in self.splits["train"]:
            if r["task"] == "native_intent_entities":
                self.assertIn('"entities"', r["messages"][0]["content"])
                self.assertIn("Entity types:", r["messages"][0]["content"])

    def test_flags_and_rejected_source_family_excluded(self):
        hashes = {r["source_record_hash"] for r in map(json.loads, core.FLAGS.read_text().splitlines())}
        selection = json.loads(core.SELECTION.read_text())
        for r in self.splits["train"]:
            self.assertNotIn(r["source_record_hash"], hashes)
            self.assertNotEqual(r["task"], "general_source_response")
            self.assertNotIn(r.get("source_record_id"), selection["additional_source_exclusions"])

    def test_contextual_answers_are_unchanged_and_not_standalone(self):
        sources = {r["source"]["source_record_hash"]: r for r in json.loads(core.REVIEW.read_text())}
        rows = [r for r in self.splits["train"] if r["task"] in ("native_source_context", "source_language_followup")]
        self.assertEqual(len(rows), 34)
        self.assertEqual(len({r["source_record_hash"] for r in rows}), 17)
        for row in rows:
            source = sources[row["source_record_hash"]]
            self.assertIn(source["answer_en"], row["messages"][1]["content"])
            self.assertEqual(source["answer_tw"], row["messages"][2]["content"])
            self.assertFalse(row["human_validated"])
            if row["task"] == "source_language_followup":
                self.assertEqual(source["answer_en"], row["messages"][-1]["content"])

    def test_flagged_question_and_modified_answer_rejected(self):
        review, selection = json.loads(core.REVIEW.read_text()), json.loads(core.SELECTION.read_text())
        question_id = selection["selected_questions"][0]
        source = next(r for r in review if any(q["id"] == question_id for q in r["translated_question_candidates"]))
        for modification in ("answer", "flag"):
            edited = copy.deepcopy(review)
            row = next(r for r in edited if r["id"] == source["id"])
            if modification == "answer":
                row["answer_tw"] += " changed"
            else:
                next(q for q in row["translated_question_candidates"] if q["id"] == question_id)["agent_translation_finding"] = "bad"
            with self.assertRaises(ValueError):
                core.context_rows(edited, selection)

    def test_prepared_artifacts_match_and_have_no_duplicate_ids(self):
        manifest = json.loads((core.OUT / "manifest.json").read_text())
        for split, expected in self.splits.items():
            rows = core.read(core.OUT, split)
            self.assertEqual(rows, expected)
            self.assertEqual(len(rows), len({r["id"] for r in rows}))
        self.assertFalse(manifest["ready_for_production"])

    def test_conversation_checks_never_enter_training(self):
        cases = json.loads((core.ROOT / "data/response-adaptation/conversation-checks.v6.json").read_text())
        training = {m["content"] for r in self.splits["train"] for m in r["messages"]}
        for case in cases:
            self.assertEqual(len(case["turns"]), 2)
            self.assertTrue(set(case["turns"]).isdisjoint(training))


if __name__ == "__main__":
    unittest.main()
