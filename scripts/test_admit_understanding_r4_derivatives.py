import copy
import unittest
from admit_understanding_r4_derivatives import materialize, protected_derivative_groups, HoldoutIndex, data


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.source = {"id": "one", "source_id": "one", "source_hash": "a" * 64, "group_id": "g",
            "source_ids": ["one"], "tags": ["correction"], "facts": [], "input_flags": [],
            "messages": [{"role": "user", "content": "One?"}, {"role": "assistant", "content": "Yes."}], "original_ref": {"path": "original"}}
        self.proposal = {"source_id": "one", "twi": ["Biako?", "Yiw."]}
        self.decision = {"source_id": "one", "proposal_record_sha256": data.digest(self.proposal), "decision": "accepted",
            "findings": [], "per_turn": [{"index": i, "back_translation": s, "note": "Meaning retained.", "findings": []}
                                        for i, s in enumerate(["One?", "Yes."])]}

    def test_preserves_original(self):
        before = copy.deepcopy(self.source)
        row = materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"})
        self.assertEqual(self.source, before)
        self.assertEqual(row["original_english_messages"], before["messages"])
        self.assertFalse(row["human_reviewed"])

    def test_changed_proposal_rejected(self):
        self.proposal["twi"][1] = "Dabi."
        with self.assertRaisesRegex(ValueError, "changed"):
            materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"})

    def test_protected_source_not_admitted(self):
        self.source["input_flags"] = ["heldout_span"]
        self.assertIsNone(materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"}))

    def test_incomplete_review_rejected(self):
        self.decision["per_turn"].pop()
        with self.assertRaisesRegex(ValueError, "Every turn"):
            materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"})

    def test_unresolved_meaning_not_admitted(self):
        self.decision["findings"] = ["negation changed"]
        self.assertIsNone(materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"}))

    def test_original_only_holdout_protects_sibling(self):
        row = materialize(self.proposal, self.decision, self.source, {"reviewer": "agent"})
        row["original_english_messages"] = [{"role": "user", "content": "This protected sentence belongs only to the original source wording."}]
        sibling = {**row, "id": "sibling", "original_english_messages": []}
        index = HoldoutIndex([("held", row["original_english_messages"][0]["content"])])
        self.assertEqual(set(protected_derivative_groups([row, sibling], index)), {row["id"], "sibling"})


if __name__ == "__main__":
    unittest.main()
