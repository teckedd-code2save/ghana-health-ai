import tempfile
import unittest
from pathlib import Path

import prepare_understanding_r5 as builder
import training_data_inputs as data


def record(identity, source, **metadata):
    return {"id": identity, "source_id": source, "metadata": metadata}


def decision(value):
    return {"source_id": value["source_id"], "record_id": value["id"],
            "record_sha256": data.digest(value), "record": value,
            "reason": "Conservatively exclude an unresolved near candidate",
            "evidence": {"audit_sha256": "a" * 64}}


class AdditionalProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fresh = self.root / "fresh"
        self.fresh.mkdir()
        data.save(self.fresh / "manifest.json", {"version": "test-fresh"})
        self.plan = self.root / "plan"
        self.plan.mkdir()
        self.held = [{"id": "rollout-join", "text": "First user turn\nSecond user turn"}]
        self.decisions = [decision(record("source-view", "source-A"))]
        self.seal()

    def seal(self, **overrides):
        data.write_rows(self.plan / "held-content.jsonl", self.held)
        data.write_rows(self.plan / "excluded-sources.jsonl", self.decisions)
        data.save(self.plan / "manifest.json", {
            "version": "r5-additional-protection-v1",
            "r4_manifest_sha256": builder.R4_SHA,
            "fresh_manifest_sha256": data.sha(self.fresh / "manifest.json"),
            "artifacts": {name: data.sha(self.plan / name) for name in
                          ("held-content.jsonl", "excluded-sources.jsonl")}, **overrides})
        data.save(self.plan / "FROZEN.json", {"manifest_sha256": data.sha(self.plan / "manifest.json")})

    def test_bound_plan_preserves_context_and_decision_provenance(self):
        held, decisions, reference = builder.additional_protection(self.plan, self.fresh, {})
        self.assertEqual(held, self.held)
        self.assertEqual(decisions, self.decisions)
        self.assertEqual(reference["manifest_sha256"], data.sha(self.plan / "manifest.json"))

    def test_changed_protection_file_fails(self):
        data.write_rows(self.plan / "held-content.jsonl", [{"id": "x", "text": "changed"}])
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            builder.additional_protection(self.plan, self.fresh, {})

    def test_other_fresh_suite_fails(self):
        self.seal(fresh_manifest_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            builder.additional_protection(self.plan, self.fresh, {})

    def test_missing_or_changed_source_binding_fails(self):
        for key, value in (("source_id", "wrong"), ("record_id", "wrong"),
                           ("record_sha256", "c" * 64), ("reason", ""), ("evidence", {})):
            with self.subTest(key=key):
                self.decisions = [{**decision(record("source-view", "source-A")), key: value}]
                self.seal()
                with self.assertRaisesRegex(ValueError, "Unbound additional exclusion"):
                    builder.additional_protection(self.plan, self.fresh, {})

    def test_duplicate_decisions_and_held_ids_fail(self):
        self.decisions *= 2
        self.seal()
        with self.assertRaisesRegex(ValueError, "Duplicate additional exclusion"):
            builder.additional_protection(self.plan, self.fresh, {})
        self.decisions = []
        self.held *= 2
        self.seal()
        with self.assertRaisesRegex(ValueError, "Invalid additional holdouts"):
            builder.additional_protection(self.plan, self.fresh, {})

    def test_exclusion_reaches_bilingual_branch_and_augmentation(self):
        records = [record("original", "source-A", tree_id="tree"),
                   record("translation", "source-B", tree_id="tree"),
                   record("spelling", "source-C", source_ids=["source-B"]),
                   record("other", "source-D", tree_id="unrelated")]
        blocked = builder.audit_excluded_groups(records, self.decisions)
        self.assertEqual(set(blocked), {"original", "translation", "spelling"})
        self.assertEqual(blocked["translation"]["record_sha256"], self.decisions[0]["record_sha256"])

    def test_source_alias_is_a_seed(self):
        records = [record("translated", "new-id", source_ids=["source-A"])]
        self.assertEqual(set(builder.audit_excluded_groups(records, self.decisions)), {"translated"})

    def test_absent_plan_does_not_change_existing_behavior(self):
        self.assertEqual(builder.additional_protection(None, self.fresh, {}), ([], [], None))
        self.assertEqual(builder.audit_excluded_groups([record("r", "s")], []), {})


if __name__ == "__main__":
    unittest.main()
