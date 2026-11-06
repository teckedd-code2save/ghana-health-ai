import tempfile
import unittest
from pathlib import Path

import admit_understanding_r3_repairs as a


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.record = {"id": "r1", "source_id": "s1", "origin": "project_authored_ai_translated",
            "review_candidate": True, "review": {"status": "agent_checked", "findings": []},
            "split": "train", "training_eligible": False, "original_ref": {"source": "original"},
            "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
        a.data.write_rows(self.dataset / "review-candidates.jsonl", [self.record])
        a.data.write_rows(self.dataset / "r3-mapping.jsonl", [{"dataset_record_id": "r1", "record_sha256": a.data.digest(self.record)}])
        a.data.save(self.dataset / "manifest.json", {"dataset_id": "repair", "dataset_version": 1,
            "files": {p.name: {"sha256": a.data.sha(p)} for p in self.dataset.iterdir()}})
        self.review = {"dataset_manifest_sha256": a.data.sha(self.dataset / "manifest.json"),
            "reviewer": "parent agent", "kind": "agent_review", "human_reviewed": False,
            "records": [{"id": "r1", "record_sha256": a.data.digest(self.record), "decision": "admit", "note": "Checked source and target."}]}
        self.review_path = self.root / "review.json"

    def run_admission(self):
        a.data.save(self.review_path, self.review)
        return a.admit(self.dataset, self.review_path, self.root / "admitted")

    def test_exact_review_admits_without_mutating_source(self):
        before = a.data.sha(self.dataset / "review-candidates.jsonl")
        self.assertEqual(self.run_admission()["admitted_records"], 1)
        self.assertEqual(a.data.sha(self.dataset / "review-candidates.jsonl"), before)
        self.assertFalse(a.read_rows(self.dataset / "review-candidates.jsonl")[0]["training_eligible"])
        self.assertTrue(a.read_rows(self.root / "admitted/conversations.jsonl")[0]["training_eligible"])

    def test_changed_review_hash_rejected(self):
        self.review["records"][0]["record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            self.run_admission()

    def test_missing_decision_rejected(self):
        self.review["records"] = []
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_admission()

    def test_tampered_source_rejected(self):
        a.data.write_rows(self.dataset / "review-candidates.jsonl", [])
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_admission()


if __name__ == "__main__":
    unittest.main()
