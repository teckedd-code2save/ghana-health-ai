import json
import copy
from pathlib import Path
import sys
import unittest

import build_native_answers as client
from gate_native_answers import gate, question_issues
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal/train"))
from native_answer_core import validate_inputs, parse_output, messages, grounded_approval


class NativeAnswerTests(unittest.TestCase):
    def test_reviewer_controls_exclude_expected_labels(self):
        rows = [{"id": "control:" + str(i), "answer_en": answer, "questions": questions}
                for i, (answer, questions, _) in enumerate(client.REVIEW_CONTROLS)]
        validate_inputs(rows, "review")
        self.assertNotIn('"expected"', json.dumps(rows))
        self.assertEqual(sum(len(expected) for _, _, expected in client.REVIEW_CONTROLS), 14)

    def test_generation_never_accepts_twi_targets_or_reviews(self):
        row = {"id": "a", "answer_en": "The shop is closed."}
        validate_inputs([row], "generate")
        for rows in ([], [row, row], [{**row, "answer_tw": "private"}], [None]):
            with self.assertRaises(ValueError):
                validate_inputs(rows, "generate")
        self.assertEqual(json.loads(messages(row, "generate")[1]["content"]), {"answer_en": row["answer_en"]})

    def test_rejects_contradictory_generation_and_duplicate_review(self):
        for value in ({"suitable": True, "reason": "okay", "questions": []},
                      {"suitable": False, "reason": "no", "questions": ["Is it open?"]},
                      {"suitable": True, "reason": "yes", "questions": ["Is it open?"] * 2}):
            with self.assertRaises(ValueError):
                parse_output(json.dumps(value), "generate")
        item = {"index": 0, "valid": True, "reason": "matches"}
        with self.assertRaises(ValueError):
            parse_output(json.dumps({"reviews": [item, item]}), "review", 2)
        with self.assertRaises(ValueError):
            parse_output(json.dumps({"reviews": [item]}), "review", 2)

    def test_original_source_targets_and_quarantine_are_preserved(self):
        rows, report = client.prepare_sources(128)
        self.assertEqual(len(rows), 128)
        self.assertEqual(report["generated_twi_answers"], 0)
        self.assertGreaterEqual(report["exclusions"]["flagged_source"], 5)
        for row in rows:
            self.assertEqual(row["answer_tw"], row["source"]["messages"][-1]["content"])
            self.assertEqual(row["source"]["split"], "train")
            self.assertNotIn(row["source"]["source_record_id"], ("66498", "96852"))
            self.assertFalse(client.HEALTH.search(row["answer_en"]))
        self.assertEqual(len({r["answer_en"].casefold() for r in rows}), 128)
        self.assertEqual(len({r["answer_tw"].casefold() for r in rows}), 128)

    def test_topical_overlap_is_not_an_explanation(self):
        self.assertIn("reason_not_actually_stated", question_issues(
            "Why are people not getting married?", "People are not getting married for some reasons."))
        self.assertIn("vague_quantity_needs_review", question_issues("How many companies are there?", "There are a number of companies."))
        self.assertIn("context_dependent_question_needs_review", question_issues("Who did he sell it to?", "He sold it to his neighbor."))
        self.assertEqual(question_issues("How can I encourage reading?", "Share what you are reading to encourage others."), [])

    def test_source_flag_overrides_positive_model_vote(self):
        row = {"source": {"source_record_hash": "x"}, "answer_en": "Prices change.", "answer_tw": "original",
               "candidates": [{"question_en": "What happens to prices?", "english_match_approved": True, "duplicate_question": False}]}
        rows, summary = gate([row], [{"source_record_hash": "x", "issue": "mismatch"}])
        self.assertFalse(rows[0]["candidates"][0]["review_candidate"])
        self.assertFalse(rows[0]["train_eligible"])
        self.assertEqual(rows[0]["answer_tw"], "original")
        self.assertEqual(summary["sources_with_review_candidate"], 0)

    def test_grounding_requires_verbatim_evidence_and_every_gate(self):
        item = {"index": 0, "directly_answers": True, "self_contained": True,
                "preserves_speaker": True, "safe_instruction": True,
                "evidence_quote": "20 cedis", "reason": "Exact delivery cost."}
        answer = "Delivery costs 20 cedis."
        valid = parse_output(json.dumps({"reviews": [item]}), "ground", 1, answer)
        self.assertTrue(grounded_approval(valid["reviews"][0]))
        for update in ({"evidence_quote": "30 cedis"}, {"evidence_quote": ""},
                       {"directly_answers": False}, {"evidence_quote": "20 Cedis"}):
            with self.assertRaises(ValueError):
                parse_output(json.dumps({"reviews": [{**item, **update}]}), "ground", 1, answer)
        for key in ("directly_answers", "self_contained", "preserves_speaker", "safe_instruction"):
            self.assertFalse(grounded_approval({**item, key: False}))
        self.assertFalse(grounded_approval({}))

    def test_ground_controls_are_english_only_and_not_given_expected_answers(self):
        rows = [{"id": f"ground-control:{i}", "answer_en": answer, "questions": questions}
                for i, (answer, questions, _) in enumerate(client.GROUND_CONTROLS)]
        validate_inputs(rows, "ground")
        self.assertEqual(sum(len(expected) for _, _, expected in client.GROUND_CONTROLS), 28)
        self.assertTrue(any("some reasons" in r["answer_en"] for r in rows))
        for row in rows:
            sent = json.loads(messages(row, "ground")[1]["content"])
            self.assertEqual(set(sent), {"answer_en", "questions"})

    def test_edited_review_cannot_reuse_old_model_approval(self):
        source = {"id": "a", "answer_en": "Delivery costs 20 cedis.", "answer_tw": "unchanged",
                  "answer_tw_sha256": "hash", "source": {"source_record_id": "1"}}
        reviewed = {**copy.deepcopy(source), "candidates": [{"question_en": "What does delivery cost?"}]}
        sent = {"id": "a", "answer_en": source["answer_en"], "questions": ["What does delivery cost?"]}
        client.validate_review_snapshot([reviewed], [source], [sent])
        for key in ("answer_en", "answer_tw", "answer_tw_sha256", "source"):
            with self.assertRaises(ValueError):
                client.validate_review_snapshot([{**reviewed, key: "changed"}], [source], [sent])
        changed = {**reviewed, "candidates": [{"question_en": "When does delivery arrive?"}]}
        with self.assertRaises(ValueError):
            client.validate_review_snapshot([changed], [source], [sent])


if __name__ == "__main__":
    unittest.main()
