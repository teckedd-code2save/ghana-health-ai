"""Admission regression tests with synthetic English text in all language fields."""
from copy import deepcopy
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import admit_understanding_r5 as admission
from test_select_understanding_r5_sources import seal, source


def bundle(records):
    sources, proposals, decisions = [], [], []
    for index, record in enumerate(records):
        item = {"index": index, "source_record": record,
                "source_record_sha256": admission.data.digest(record),
                "training_eligible": False, "review_status": "meaning_review_required"}
        proposal = {"index": index, "source_id": record["id"],
                    "source_record_sha256": item["source_record_sha256"],
                    "decision": "translate", "reason": "Synthetic schema fixture.",
                    "twi": ["Target fixture request " + record["id"] + ".",
                            "Target fixture response " + record["id"] + "."],
                    "english_overrides": {}}
        decision = {"index": index, "proposal_record_sha256": admission.data.digest(proposal),
                    "decision": "accepted", "reason": "Synthetic independent review.",
                    "source_review": "Both source turns were considered in this fixture.",
                    "source_findings": [], "findings": [],
                    "per_turn": [{"index": i, "back_translation": "Synthetic English reference.",
                                  "note": "Fixture review of the supplied turn.", "findings": []}
                                 for i in range(len(record["messages"]))]}
        sources.append(item)
        proposals.append(proposal)
        decisions.append(decision)
    return sources, proposals, decisions


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        sources, proposals, decisions = bundle([source("alpha")])
        self.item, self.proposal, self.decision = sources[0], proposals[0], decisions[0]
        self.refs = {"reviewer": "independent-fixture-reviewer"}

    def run_materialize(self):
        self.item["source_record_sha256"] = admission.data.digest(self.item["source_record"])
        self.proposal["source_record_sha256"] = self.item["source_record_sha256"]
        self.decision["proposal_record_sha256"] = admission.data.digest(self.proposal)
        return admission.materialize(self.item, self.proposal, self.decision, self.refs)

    def test_valid_review_preserves_original_and_is_not_native_certification(self):
        before = deepcopy(self.item)
        row = self.run_materialize()
        self.assertEqual(self.item, before)
        self.assertFalse(row["human_reviewed"])
        self.assertEqual(row["original_english_messages"], before["source_record"]["messages"])

    def test_missing_explicit_review_fields_fail_closed(self):
        for key in ("source_findings", "findings", "source_review"):
            with self.subTest(key=key):
                saved = self.decision.pop(key)
                with self.assertRaises(ValueError):
                    self.run_materialize()
                self.decision[key] = saved

    def test_falsy_findings_are_not_an_explicit_empty_list(self):
        for value in (None, False, "", {}):
            with self.subTest(value=value):
                self.decision["source_findings"] = value
                with self.assertRaises(ValueError):
                    self.run_materialize()

    def test_whitespace_reason_is_not_accountability(self):
        self.decision["reason"] = "  "
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_translation_must_be_list_not_equal_length_string(self):
        self.proposal["twi"] = "OK"
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_boolean_review_indices_are_not_integer_indices(self):
        self.decision["per_turn"][0]["index"] = False
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_known_protected_split_cannot_be_relabelled_train(self):
        self.item["source_record"]["metadata"]["split"] = "validation"
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_source_work_cannot_self_mark_eligible(self):
        self.item["training_eligible"] = True
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_ancestry_links_survive_materialization(self):
        metadata = self.item["source_record"]["metadata"]
        metadata.update(family_id="family-fixture", tree_id="tree-fixture",
                        original_group_id="original-fixture", source_ids=["ancestor-fixture"])
        row = self.run_materialize()
        self.assertTrue(admission.previous.source_keys(self.item["source_record"])
                        <= admission.previous.source_keys(row))

    def test_unknown_origin_does_not_become_project_authored(self):
        self.item["source_record"]["metadata"]["origin"] = "unknown_unreviewed_import"
        with self.assertRaises(ValueError):
            self.run_materialize()

    def test_repair_and_english_override_require_reviews(self):
        self.decision.update(decision="repair", corrected_twi=["Repaired fixture input.", "Repaired fixture reply."])
        self.proposal["english_overrides"] = {"1": "Amended source reply."}
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.proposal["amendment_reason"] = "Fixture source amendment."
        self.decision["amendment_review"] = "Fixture amendment independently checked."
        row = self.run_materialize()
        self.assertEqual(row["messages"][1]["content"], "Repaired fixture reply.")
        self.assertEqual(row["english_messages"][1]["content"], "Amended source reply.")

    def test_resolved_review_findings_retain_exact_decision_and_text_bindings(self):
        self.decision["screen_resolutions"] = [
            {"code": "synthetic_review", "turn": 1, "reason": "Explicit fixture-only decision."}]
        with patch.object(admission, "inspect_messages", return_value=[
                {"code": "synthetic_review", "turn": 1, "severity": "review"}]):
            row = self.run_materialize()
        review = row["review"]
        self.assertEqual(review["screen_resolutions"], self.decision["screen_resolutions"])
        self.assertEqual(review["source_review"], self.decision["source_review"])
        self.assertEqual(review["decision_sha256"], admission.data.digest(self.decision))
        self.assertEqual(review["screened_messages_sha256"], admission.data.digest(row["messages"]))
        self.assertEqual(review["screened_english_messages_sha256"], admission.data.digest(row["english_messages"]))
        self.decision["screen_resolutions"][0]["reason"] = "Changed after materialization."
        self.assertNotEqual(review["screen_resolutions"], self.decision["screen_resolutions"])

    def test_invalid_history_cannot_be_overridden_by_resolution(self):
        self.decision["screen_resolutions"] = [
            {"code": "invalid_unicode", "turn": 1, "reason": "Not a valid override."}]
        with patch.object(admission, "inspect_messages", return_value=[
                {"code": "invalid_unicode", "turn": 1, "severity": "invalid"}]):
            with self.assertRaisesRegex(ValueError, "Invalid history"):
                self.run_materialize()

    def test_resolution_must_name_current_finding_and_have_nonblank_reason(self):
        self.decision["screen_resolutions"] = [
            {"code": "another_finding", "turn": 1, "reason": "Different finding."}]
        with patch.object(admission, "inspect_messages", return_value=[
                {"code": "synthetic_review", "turn": 1, "severity": "review"}]):
            with self.assertRaisesRegex(ValueError, "Unresolved screen"):
                self.run_materialize()
        self.decision["screen_resolutions"][0]["reason"] = "  "
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.run_materialize()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.r4, self.fresh, self.out = [self.root / name for name in ("r4", "fresh", "out")]
        self.source_file = self.root / "sources.jsonl"
        self.proposal_file = self.root / "proposals.json"
        self.review_file = self.root / "review.json"
        self.sources, self.proposals, self.decisions = bundle([source("alpha"), source("beta"), source("gamma")])
        self.old_held = [{"id": "same-id", "text": "Older synthetic protected material."}]
        self.fresh_held = [{"id": "same-id", "text": "Separate synthetic protected material."}]
        self.agent, self.reviewer = "fixture-author", "fixture-reviewer"
        self.write_inputs()
        no_network = patch.object(socket, "create_connection", side_effect=AssertionError("No network"))
        no_network.start()
        self.addCleanup(no_network.stop)

    def write_inputs(self, omit_fresh=()):
        seal(self.r4, {"held-content.jsonl": self.old_held})
        seal(self.fresh, {"held-content.jsonl": self.fresh_held}, "FROZEN.json", omit=omit_fresh)
        for item, proposal, decision in zip(self.sources, self.proposals, self.decisions):
            item["source_record_sha256"] = admission.data.digest(item["source_record"])
            proposal["source_record_sha256"] = item["source_record_sha256"]
            decision["proposal_record_sha256"] = admission.data.digest(proposal)
        admission.data.write_rows(self.source_file, self.sources)
        admission.data.save(self.proposal_file, {"kind": "agent_translation_proposals", "agent": self.agent,
            "human_reviewed": False, "training_eligible": False, "source_file": str(self.source_file),
            "source_file_sha256": admission.data.sha(self.source_file), "records": self.proposals})
        admission.data.save(self.review_file, {"kind": "agent_review", "reviewer": self.reviewer,
            "human_reviewed": False, "proposal_sha256": admission.data.sha(self.proposal_file), "records": self.decisions})

    def run_admit(self):
        return admission.admit([(self.proposal_file, self.review_file)], self.r4, self.fresh, self.out)

    def test_offline_valid_admission_seals_outputs_without_changing_inputs(self):
        before = {p: admission.data.sha(p) for p in self.root.rglob("*") if p.is_file()}
        report = self.run_admit()
        self.assertEqual(report["admitted_histories"], 3)
        admission.previous.verified_directory(self.out, "READY.json")
        self.assertEqual(before, {p: admission.data.sha(p) for p in before})

    def test_blank_reviewer_or_author_rejected(self):
        for name in ("agent", "reviewer"):
            with self.subTest(name=name):
                setattr(self, name, " ")
                self.write_inputs()
                with self.assertRaises(ValueError):
                    self.run_admit()
                self.assertFalse(self.out.exists())
                setattr(self, name, "fixture-" + name)

    def test_unbound_fresh_holdout_file_rejected(self):
        self.write_inputs(omit_fresh=("held-content.jsonl",))
        with self.assertRaises(ValueError):
            self.run_admit()
        self.assertFalse(self.out.exists())

    def test_empty_fresh_holdout_rejected(self):
        self.fresh_held = []
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.run_admit()

    def test_duplicate_source_indices_not_silently_overwritten(self):
        admission.data.write_rows(self.source_file, self.sources + [self.sources[-1]])
        proposals = admission.data.read(self.proposal_file)
        proposals["source_file_sha256"] = admission.data.sha(self.source_file)
        admission.data.save(self.proposal_file, proposals)
        review = admission.data.read(self.review_file)
        review["proposal_sha256"] = admission.data.sha(self.proposal_file)
        admission.data.save(self.review_file, review)
        with self.assertRaises(ValueError):
            self.run_admit()

    def test_rejected_protected_source_still_blocks_connected_accepted_source(self):
        self.sources[1]["source_record"]["metadata"]["group_id"] = self.sources[0]["source_record"]["metadata"]["group_id"]
        self.old_held[0]["text"] = self.sources[0]["source_record"]["messages"][0]["content"]
        self.decisions[0]["decision"] = "reject"
        self.write_inputs()
        self.assertEqual(self.run_admit()["admitted_histories"], 1)

    def test_family_link_blocks_other_direct_group(self):
        for item in self.sources[:2]:
            item["source_record"]["metadata"]["family_id"] = "shared-family"
        self.fresh_held[0]["text"] = self.proposals[0]["twi"][0]
        self.write_inputs()
        self.assertEqual(self.run_admit()["admitted_histories"], 1)

    def test_original_full_context_remains_protected_after_amendment(self):
        original = [{"role": "user", "content": "Synthetic original prompt has several distinct English words."},
                    {"role": "assistant", "content": "Original answer has a separate set of words."}]
        self.sources[0]["source_record"]["messages"] = original
        self.fresh_held[0]["text"] = "\n".join(m["content"] for m in original)
        self.proposals[0]["english_overrides"] = {"0": "Revised prompt.", "1": "Revised answer."}
        self.proposals[0]["amendment_reason"] = "Synthetic amendment."
        self.decisions[0]["amendment_review"] = "Independently reviewed fixture amendment."
        self.write_inputs()
        self.assertEqual(self.run_admit()["admitted_histories"], 2)

    def test_cross_pack_holdout_ids_are_namespaced(self):
        original = admission.HoldoutIndex
        observed = []
        def capture(rows):
            observed.extend(key for key, _ in rows)
            return original(rows)
        with patch.object(admission, "HoldoutIndex", side_effect=capture):
            self.run_admit()
        self.assertEqual(len(observed), len(set(observed)))

    def test_mid_admission_source_mutation_aborts_without_ready_output(self):
        original = admission.materialize
        changed = False
        def mutate(*args):
            nonlocal changed
            result = original(*args)
            if not changed:
                admission.data.write_rows(self.source_file, self.sources + [self.sources[-1]])
                changed = True
            return result
        with patch.object(admission, "materialize", side_effect=mutate):
            with self.assertRaises(ValueError):
                self.run_admit()
        self.assertFalse(self.out.exists())

    def test_input_mutation_while_publishing_never_gets_a_ready_marker(self):
        original = admission.data.save
        def save(path, value):
            original(path, value)
            if path == self.out / "report.json":
                admission.data.write_rows(self.source_file, self.sources + [self.sources[-1]])
        with patch.object(admission.data, "save", side_effect=save):
            with self.assertRaises(ValueError):
                self.run_admit()
        self.assertFalse((self.out / "READY.json").exists())


if __name__ == "__main__":
    unittest.main()
