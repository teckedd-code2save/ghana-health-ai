"""Offline replay tests: inherited evidence, whole histories and real tokenizer."""
import copy
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest import mock

import admit_r7_replay as replay
import prepare_r7_source_inventory as inventory
import run_understanding_r7 as runner
from test_prepare_r7_alignment_candidates import fixture, rows, seal_inventory, write_rows
from test_admit_r7_alignment import TOKENIZER, frozen_eval


def record(identity="source", answer="Preserved answer", **extra):
    return {"id": identity, "source_id": "family:" + identity, "split": "train", "training_eligible": True,
            "messages": [{"role": "user", "content": "A complete original question about this subject."},
                         {"role": "assistant", "content": answer}], **extra}


def pack(folder, version, collections):
    folder.mkdir()
    refs, witnesses = [], []
    for collection, values in collections.items():
        target = folder / "datasets" / collection
        target.mkdir(parents=True)
        write_rows(target / "records.jsonl", values)
        source = {"dataset_id": "fixture-" + collection, "dataset_version": version, "admission_status": "admitted",
                  "private_research_only": True, "production_eligible": False, "record_count": len(values),
                  "records_path": f"datasets/{collection}/records.jsonl", "records_sha256": inventory.sha(target / "records.jsonl")}
        inventory.write_json(target / "manifest.json", source)
        refs.append({"dataset_id": source["dataset_id"], "dataset_version": version,
                     "manifest_path": f"datasets/{collection}/manifest.json", "manifest_sha256": inventory.sha(target / "manifest.json")})
        for row in values:
            witnesses.append({"id": row["id"], "source_id": row["source_id"], "collection": collection,
                "dataset_ref": {"dataset_id": source["dataset_id"], "dataset_version": version,
                                "record_id": row["id"], "source_id": row["source_id"], "record_sha256": runner.digest(row)},
                **replay.content(row), "metadata": {"origin": "prior_fixture"}, "original_ref": {"saved": True}})
    write_rows(folder / "source-witnesses.jsonl", witnesses)
    inventory.write_json(folder / "source-manifests.json", refs)
    inventory.write_json(folder / "manifest.json", {"version": "understanding-adaptation-" + version,
        "private_research_only": True, "production_eligible": False,
        "artifacts": {p.relative_to(folder).as_posix(): inventory.sha(p) for p in folder.rglob("*") if p.is_file()}})
    inventory.write_json(folder / "PREPARED.json", {"manifest_sha256": inventory.sha(folder / "manifest.json")})
    return folder


def local_inputs(root, old, new):
    folder, bound, _ = fixture(root)
    r5, r6 = pack(root / "r5", "r5", old), pack(root / "r6", "r6", new)
    for name in ("oasst-tree-index", "oasst-decisions", "r5-replay-decisions"):
        write_rows(folder / (name + ".jsonl"), [])
    write_rows(folder / "protection-index.jsonl", [{**r, "identity_keys": []} for r in rows(folder / "protection-index.jsonl")])
    seal_inventory(folder, bound + [r5 / "manifest.json", r6 / "manifest.json"])
    audit = root / "audit"
    audit.mkdir()
    inventory.write_json(audit / "manifest.json", {"fixture": True})
    return r5, r6, folder, audit, bound


class AuditTests(unittest.TestCase):
    def entry(self, classification="repeated_phrase_review", decision=None):
        finding = {"id": "finding", "classification": classification, "turn": 1, "linked_agent_review_ids": ["review"] if decision else []}
        return {"audit": {"findings": [finding]}, "audit_reviews": [{"review_id": "review", "decision": decision}] if decision else []}

    def test_genuine_concern_excludes_whole_history_even_if_other_turn_resolved(self):
        entry = self.entry("unsupported_execution_claim_review", "unsupported_search_claim_review")
        entry["audit_reviews"].append({"review_id": "other", "decision": "preserve_section_specific_repetition"})
        self.assertEqual(replay.audit_status(entry), ("unresolved_genuine_audit_finding", ["review"]))

    def test_resolved_structures_and_code_not_rebranded_as_errors(self):
        self.assertEqual(replay.audit_status(self.entry("protected_code_repetition")), (None, []))
        for decision in replay.RESOLVED:
            self.assertEqual(replay.audit_status(self.entry(decision=decision)), (None, []))
        problem, _ = replay.audit_status(self.entry())
        self.assertEqual(problem, "unresolved_audit_selector_not_proven_error")

    def test_dedupe_requires_same_collection_family_and_complete_content(self):
        first = {"collection": "responses", "family": "family", "record": record()}
        second = copy.deepcopy(first)
        second["record"]["id"] = "changed-id"
        self.assertEqual(replay.dedupe_key(first), replay.dedupe_key(second))
        second["record"]["messages"] += [{"role": "user", "content": "A followup"}, {"role": "assistant", "content": "Another answer"}]
        self.assertNotEqual(replay.dedupe_key(first), replay.dedupe_key(second))
        second = {**first, "collection": "english_retention"}
        self.assertNotEqual(replay.dedupe_key(first), replay.dedupe_key(second))
        second = {**first, "family": "different"}
        self.assertNotEqual(replay.dedupe_key(first), replay.dedupe_key(second))

    def test_inherited_tiers_keep_machine_corrections_distinct_from_originals(self):
        entry = {"rich": {"metadata": {"disposition": "correction_candidate", "generated_target": True}},
                 "prior_dataset": {"dataset_version": "r6-relaxed"}}
        self.assertEqual(replay.inherited_tier(entry), "correction_candidate")
        entry["rich"] = {"review": {"status": "agent_checked"}}
        self.assertEqual(replay.inherited_tier(entry), "agent_checked")


class ReplayIntegrationTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(replay, "integrate_findings_review", return_value={
            "release": {"path": str(replay.FINDINGS_REVIEW)}, "dispositions": [], "review_summary": {}})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_structured_meaning_text_field_is_metadata_not_language_objective(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            structured = record("intent", '{"intent":"original_label"}', text="Original utterance also stored outside messages")
            r5, r6, folder, audit, _ = local_inputs(root, {"meaning": [structured]}, {"responses": [record("other")]})
            with mock.patch.object(replay, "restore_r6"):
                replay.prepare(r5, r6, folder, audit, TOKENIZER, [frozen_eval(root)], root / "output")
            self.assertTrue(replay.verify(root / "output", native=True)["verified"])
            witness = next(r for r in rows(root / "output/source-witnesses.jsonl") if r["collection"] == "meaning")
            self.assertNotIn("text", witness)
            self.assertEqual(witness["metadata"]["prior_full_record"]["text"], structured["text"])
            self.assertEqual(witness["messages"], structured["messages"])
            encoded = next(r for r in rows(root / "output/rows.jsonl") if r["collection"] == "meaning")
            self.assertEqual(encoded["objective"], "final_assistant")
            self.assertEqual(encoded["labels"][0], -100)

    def test_restores_r5_without_overwriting_r6_and_preserves_full_saved_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            prior = record("same", review={"status": "agent_checked", "unresolved_findings": [], "human_reviewed": False}, origin="saved_agent_review")
            changed = record("same", "Updated answer")
            old_only = record("old-only", "A useful earlier corrected answer")
            duplicate = record("duplicate")
            r5, r6, folder, audit, bound = local_inputs(root, {"responses": [prior, old_only, duplicate]}, {"responses": [changed, duplicate]})
            checksums = {p: inventory.sha(p) for p in bound}
            with mock.patch.object(replay, "restore_r6"):
                summary = replay.prepare(r5, r6, folder, audit, TOKENIZER, [frozen_eval(root)], root / "output")
            self.assertEqual(summary["admitted_full_records"], 4)
            self.assertEqual(summary["prefix_views_created"], 0)
            self.assertEqual(summary["counts_by_collection"]["responses"]["admitted_by_prior_version"], {"r6": 2, "r5": 2})
            self.assertTrue(replay.verify(root / "output", native=True)["verified"])
            values = rows(root / "output/datasets/responses/records.jsonl")
            old = next(r for r in values if r["id"] == "replay:r5:same")
            self.assertEqual(old["metadata"]["prior_full_record"], prior)
            self.assertEqual(old["original_ref"]["prior_review"], prior["review"])
            self.assertEqual(old["messages"], prior["messages"])
            self.assertFalse(old["metadata"]["clinical_certification"])
            root_manifest = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(root_manifest["admission_status"], "admitted")
            self.assertFalse(root_manifest["parent_boundary_approval"])
            self.assertEqual(checksums, {p: inventory.sha(p) for p in bound})
            with mock.patch.object(replay, "restore_r6"), self.assertRaisesRegex(ValueError, "new replay"):
                replay.prepare(r5, r6, folder, audit, TOKENIZER, [], root / "output")
            with (root / "output/rows.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                replay.verify(root / "output")

    def test_language_native_causal_objective_and_complete_multiturn_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            language = {"id": "language", "source_id": "language", "text": "Complete original source text.", "split": "train", "training_eligible": True}
            dialogue = record()
            dialogue["messages"] += [{"role": "user", "content": "A complete followup."}, {"role": "assistant", "content": "Preserved final answer."}]
            r5, r6, folder, audit, _ = local_inputs(root, {"language": [language]}, {"responses": [dialogue]})
            with mock.patch.object(replay, "restore_r6"):
                result = replay.prepare(r5, r6, folder, audit, TOKENIZER, [frozen_eval(root)], root / "output")
            self.assertEqual(result["admitted_full_records"], 2)
            self.assertTrue(replay.verify(root / "output", native=True)["verified"])
            encoded = rows(root / "output/rows.jsonl")
            lang = next(r for r in encoded if r["collection"] == "language")
            self.assertEqual(lang["labels"], lang["input_ids"])
            history = next(r for r in rows(root / "output/source-witnesses.jsonl") if r["collection"] == "responses")
            self.assertEqual(len(history["messages"]), 4)

    def test_bad_earlier_turn_excludes_whole_r6_history_not_distinct_prior_r5(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            r5, r6, folder, audit, _ = local_inputs(root, {"responses": [record("same", "Earlier corrected target")]},
                                                  {"responses": [record("same", "Unresolved target")]})
            def restore(entries, *args):
                for e in entries:
                    if e["version"] == "r6":
                        e.update(AuditTests().entry("unsupported_execution_claim_review", "unsupported_search_claim_review"))
            with mock.patch.object(replay, "restore_r6", side_effect=restore):
                result = replay.prepare(r5, r6, folder, audit, TOKENIZER, [frozen_eval(root)], root / "output")
            self.assertEqual(result["admitted_full_records"], 1)
            self.assertEqual(result["counts_by_collection"]["responses"]["admitted_by_prior_version"], {"r5": 1})
            self.assertEqual(rows(root / "output/admission-ledger.jsonl")[0]["status"], "excluded")

    def test_fresh_holdout_protects_both_versions_and_related_families(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            r5, r6, folder, audit, _ = local_inputs(root, {"responses": [record("shared", "Earlier response")]},
                {"responses": [record("shared", "Different response"), record("allowed", "Separate response")]})
            evaluation = frozen_eval(root, source_id="family:shared")
            with mock.patch.object(replay, "restore_r6"):
                result = replay.prepare(r5, r6, folder, audit, TOKENIZER, [evaluation], root / "output")
            self.assertEqual(result["admitted_full_records"], 1)
            self.assertEqual(result["reasons_by_record"]["protected_source_family_or_content"], 2)

    def test_changed_prior_pack_source_is_not_resealed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder = pack(root / "r5", "r5", {"responses": [record()]})
            with (folder / "datasets/responses/records.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                replay.read_pack(folder, "r5", inventory.Inputs())


class FindingsReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = rows(replay.ROOT / "tmp/r7-20261117/response-target-audit-v1/record-audit.jsonl")
        reviews = {r["review_id"]: r for r in rows(replay.ROOT / "tmp/r7-20261117/response-target-audit-v1/agent-reviews.jsonl")}
        cls.entries = [{"version": "r6", "collection": "responses", "record": r["original_record"],
                        "rich": r["original_record"], "audit": r,
                        "audit_reviews": [reviews[k] for k in r["agent_review_ids"]], "extra_evidence": {}}
                       for r in cls.audit]
        cls.result = replay.integrate_findings_review(cls.entries, replay.FINDINGS_REVIEW, inventory.Inputs())
        cls.by_line = {e["audit"]["original_line"]: e for e in cls.entries}

    def test_full_review_coverage_and_issue_only_scope(self):
        self.assertEqual(self.result["review_summary"]["total_selected_findings"], 191)
        self.assertEqual(len(self.result["dispositions"]), 122)
        self.assertEqual(len(replay.RECORD_CONCERN_HOLDS | replay.INHERITED_LIMITATION_LINES), 109)
        self.assertTrue(all(d["whole_translation_gold"] is False for d in self.result["dispositions"]))

    def test_clearance_does_not_hide_record_concern(self):
        entry = self.by_line[929]
        self.assertTrue(all(r["decision"] == "clear_issue" for r in entry["issue_reviews_v2"]))
        self.assertEqual(replay.audit_status(entry)[0], "unresolved_record_scope_concern")
        self.assertEqual(replay.audit_status(self.by_line[20]), (None, []))
        self.assertEqual(replay.audit_status(self.by_line[22]), (None, []))

    def test_new_and_inherited_claim_holds_remain_whole_history_holds(self):
        for line in (188, 842, 355, 465, 809, 908):
            with self.subTest(line=line):
                self.assertEqual(replay.audit_status(self.by_line[line])[0], "unresolved_genuine_audit_finding")

    def test_two_exact_deletions_keep_all_other_turns_and_full_originals(self):
        for line in (132, 875):
            entry = self.by_line[line]
            self.assertEqual(replay.audit_status(entry), (None, []))
            original, candidate = entry["rich"], entry["correction_candidate"]
            projected = replay.checked_deletion(original, candidate)
            self.assertEqual(projected, replay.training_content(entry))
            self.assertEqual(candidate["english_messages"], original["english_messages"])
            self.assertEqual(projected["messages"][2:], original["messages"][2:])
            self.assertEqual(entry["record"], original)
            mutated = copy.deepcopy(candidate)
            mutated["messages"][-1]["content"] += " invented"
            with self.assertRaisesRegex(ValueError, "unapproved content"):
                replay.checked_deletion(original, mutated)

    def test_record_hold_is_not_overridden_by_valid_deletion(self):
        entry = copy.deepcopy(self.by_line[132])
        entry["record_concern_hold"] = True
        self.assertEqual(replay.audit_status(entry)[0], "unresolved_record_scope_concern")

    def test_source_or_coverage_drift_fails_closed(self):
        entry = copy.deepcopy(self.by_line[20])
        entry["audit"]["findings"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete issue-review coverage"):
            replay.integrate_findings_review([entry], replay.FINDINGS_REVIEW, inventory.Inputs())
        entry = copy.deepcopy(self.by_line[20])
        entry["rich"]["messages"][-1]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "source differs"):
            replay.integrate_findings_review([entry], replay.FINDINGS_REVIEW, inventory.Inputs())

    def test_mutated_sealed_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "review"
            shutil.copytree(replay.FINDINGS_REVIEW, folder)
            with (folder / "review-ledger.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                replay.integrate_findings_review([], folder, inventory.Inputs())

    def test_native_deletion_views_are_complete_final_target_only(self):
        tokenizer = replay.contract.load_tokenizer(TOKENIZER)
        for line in (132, 875):
            entry = self.by_line[line]
            encoded, measured = replay.measure(tokenizer, replay.training_content(entry), 4096)
            self.assertIsNotNone(encoded)
            self.assertFalse(measured["truncated"])
            first = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
            self.assertEqual(encoded["labels"], [-100] * first + encoded["input_ids"][first:])
            self.assertEqual(encoded["labels"].count(replay.contract.EOS), 1)
            self.assertEqual(replay.measure(tokenizer, replay.training_content(entry), len(encoded["input_ids"]) - 1)[1]["reason"],
                             "overlength_no_truncation")


if __name__ == "__main__":
    unittest.main()
