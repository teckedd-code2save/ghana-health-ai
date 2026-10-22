import copy
import json
from pathlib import Path
import tempfile
import unittest

from package_instruction_review import checked, digest, progress_html, project, sha


class InstructionReviewTests(unittest.TestCase):
    def setUp(self):
        original = {"instruction": "Fixture request", "input": "", "output": "Old fixture"}
        reference = {"instruction": "Fixture request", "input": "", "output": "Reference fixture"}
        target = {**original, "output": "Corrected fixture"}
        self.source = {"id": "fixture:1", "source_hash": digest(original), "reference_hash": digest(reference),
            "original_twi": original, "reference_english": reference, "split": "train",
            "parent_disposition": "review_needed", "group_id": "fixture-group"}
        self.item = {"id": "fixture:1", "source_hash": digest(original), "reference_hash": digest(reference),
            "reference_english": reference, "after": target, "after_hash": digest(target),
            "training_eligible": False, "human_reviewed": False,
            "decision": {"action": "replace", "scope": "Fixture correction", "remaining": "Naturalness check"}}
        self.review = {"id": "fixture:1", "source_hash": digest(original), "reference_sha256": digest(reference),
            "target_sha256": digest(target), "full_example": copy.deepcopy(self.item),
            "human_reviewed": False, "training_eligible": False, "review": {"outcome": "agent_pass_to_F"}}

    def project(self):
        return project([self.item], [], [self.review], {"fixture:1": self.source})

    def test_complete_projection_has_no_admission(self):
        row = self.project()[0]
        self.assertEqual(row["response"]["original"], self.source["original_twi"])
        self.assertEqual(row["response"]["proposed"], self.item["after"])
        self.assertFalse(row["training_eligible"])
        self.assertIn("pending", row["annotation_status"])

    def test_original_is_not_previous_working_copy(self):
        self.item["before"] = {"output": "A previous repair"}
        self.review["full_example"] = copy.deepcopy(self.item)
        self.assertEqual(self.project()[0]["response"]["original"], self.source["original_twi"])

    def test_stale_review_rejected(self):
        self.review["target_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Stale E target"):
            self.project()

    def test_missing_review_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            project([self.item], [], [], {"fixture:1": self.source})

    def test_protected_and_changed_sources_rejected(self):
        self.source["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Protected"):
            self.project()
        self.source["split"] = "train"
        self.source["original_twi"]["output"] = "Mutated"
        with self.assertRaisesRegex(ValueError, "Changed original"):
            self.project()

    def test_false_approval_rejected(self):
        self.review["human_reviewed"] = True
        with self.assertRaisesRegex(ValueError, "Unsupported meaning approval"):
            self.project()

    def test_duplicate_and_missing_inventory_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            project([self.item, self.item], [], [self.review], {"fixture:1": self.source})
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            project([self.item], [], [self.review], {})

    def test_pending_has_no_proposal(self):
        row = project([], [self.item], [], {"fixture:1": self.source})[0]
        self.assertIsNone(row["response"]["proposed"])
        self.assertEqual(len(row["alternatives"]), 1)

    def test_derivative_keeps_original_reference_and_binds_choice(self):
        previous = self.project()[0]["proposal_set_id"]
        derivative = {**self.item["reference_english"], "output": "Corrected English fixture"}
        adjudication = {"reference_derivative": derivative, "verification_sources": []}
        self.item["decision"]["source_adjudication"] = adjudication
        self.item["reference_derivative_sha256"] = digest(derivative)
        self.item["source_adjudication_sha256"] = digest(adjudication)
        self.review["full_example"] = copy.deepcopy(self.item)
        row = self.project()[0]
        self.assertEqual(row["response"]["english"], self.source["reference_english"])
        self.assertEqual(row["response"]["corrected_english"], derivative)
        self.assertNotEqual(row["proposal_set_id"], previous)
        derivative["output"] = "Stale correction"
        self.review["full_example"] = copy.deepcopy(self.item)
        with self.assertRaisesRegex(ValueError, "Changed adjudication"):
            self.project()

    def test_holds_are_not_labelled_passes(self):
        self.review["review"]["outcome"] = "source_hold"
        self.assertIn("source hold", self.project()[0]["annotation_status"])

    def test_manifest_checks_bytes_and_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "rows.json").write_text("[]")
            manifest = {"training_ready": False, "artifacts": {"rows.json": sha(folder / "rows.json")}}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            checked(folder, sha(folder / "manifest.json"))
            (folder / "rows.json").write_text("[1]")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                checked(folder)
            manifest["artifacts"] = {"../rows.json": "0" * 64}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
                checked(folder)

    def test_map_is_escaped_and_counts_not_approvals(self):
        html = progress_html({"as_of": "2026-10-20", "checkpoints": "<script>bad</script>",
            "D": {"pending_work_items": 1, "cumulative_source_adjudicated_derivatives": 0},
            "E": {"agent_assessments": 2, "pending_assessments": 0, "outcomes": {"agent_pass_to_F": 2}},
            "decisions": {"replace": 1, "retain": 1}, "D_manifest_sha256": "a" * 64, "E_manifest_sha256": "b" * 64})
        self.assertNotIn("<script>", html)
        self.assertIn("0 training admissions", html)
        self.assertIn('id="node-D"', html)
        self.assertIn("1 replaced", html)
        self.assertIn("No focused E check pending", html)
        self.assertIn("Phrase confirmations do not approve whole examples", html)


if __name__ == "__main__":
    unittest.main()
