"""CPU-only coverage, source-binding and deletion replay tests for the review ledger.

Real-source snippets are attributed to OpenAssistant/oasst1 (Apache-2.0).
Synthetic test fixtures below are never emitted as corpus or training records.
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_r7_response_findings as review


def fixture():
    messages = [{"role": "user", "content": "Offline unit fixture; not a training example."},
                {"role": "assistant", "content": "Alpha - first\nBeta - second\nAlpha - first"}]
    original = {"messages": messages, "english_messages": copy.deepcopy(messages),
                "split": "dev", "group_id": "fixture-group"}
    row = {"original_line": 1, "id": "fixture-record", "source_id": "fixture-source",
           "source_group_id": "fixture-group", "original_record": original,
           "original_record_sha256": review.digest(original),
           "findings": [{"id": "fixture-finding", "turn": 1, "agent_reviewed": False}]}
    note = {"line": 1, "record_concerns": [], "reviews": [{"indices": [0], "decision": "clear_issue",
            "kind": "fixture_only", "reason": "Synthetic validator exercise only.",
            "source_anchor": {"turn": 1, "quote": "Beta - second"}}]}
    plan = {"line": 1, "turn": 1, "finding_indices": [0], "kind": "exact_duplicate_list_entry",
            "target_unit": "Alpha - first", "source_unit": "Alpha - first",
            "keep_occurrence": 1, "delete_occurrence": 2, "source_delete_occurrence": 2,
            "reason": "Identical-language synthetic deletion proof fixture only.", "record_concerns": []}
    return row, note, plan


class PortableContractTests(unittest.TestCase):
    def test_synthetic_deletion_exact_and_not_admitted(self):
        row, _, plan = fixture()
        candidate = review.deletion_candidate(row, plan)
        self.assertEqual(candidate["final_changed_target"], "Alpha - first\nBeta - second\n")
        self.assertFalse(candidate["training_ready"])
        self.assertEqual(candidate["split"], "dev")
        self.assertEqual(candidate["messages"][0], row["original_record"]["messages"][0])

    def test_original_hash_required(self):
        row, _, plan = fixture()
        row["original_record"]["messages"][0]["content"] += "changed"
        with self.assertRaisesRegex(ValueError, "original hash"):
            review.deletion_candidate(row, plan)

    def test_manual_coverage_and_anchor_required(self):
        row, note, _ = fixture()
        self.assertEqual(len(review.validate_notes([row], [note])), 1)
        for notes in ([], [note, note]):
            with self.assertRaises(ValueError):
                review.validate_notes([row], notes)
        note["reviews"][0]["source_anchor"]["quote"] = "fabricated"
        with self.assertRaisesRegex(ValueError, "anchor mismatch"):
            review.validate_notes([row], [note])

    def test_user_and_fenced_code_deletions_rejected(self):
        row, _, plan = fixture()
        plan["turn"] = 0
        with self.assertRaisesRegex(ValueError, "user prompts"):
            review.deletion_candidate(row, plan)
        plan["turn"] = 1
        row["original_record"]["messages"][1]["content"] = "```\n" + row["original_record"]["messages"][1]["content"]
        row["original_record_sha256"] = review.digest(row["original_record"])
        with self.assertRaisesRegex(ValueError, "fenced code"):
            review.deletion_candidate(row, plan)

    def test_duplicate_source_unit_required(self):
        row, _, plan = fixture()
        plan["source_unit"] = "Beta - second"
        with self.assertRaisesRegex(ValueError, "occurrence proof"):
            review.deletion_candidate(row, plan)

    def test_canonical_hash_and_utf8_offsets(self):
        self.assertEqual(review.digest({"b": 2, "a": 1}), review.digest({"a": 1, "b": 2}))
        self.assertEqual(review.occurrences("\u025b test \u025b", "\u025b"), [0, 7])
        with self.assertRaises(ValueError):
            review.encoded(float("nan"))


class ReviewFindingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [review.AUDIT / "manifest.json", review.OUT / "manual-decisions.jsonl",
                    review.OUT / "manual-repairs.jsonl"]
        if not all(p.exists() for p in required):
            raise unittest.SkipTest("Sealed local R6 audit evidence not installed; portable contract tests still run")
        if not all(Path(p).exists() for p in review.read_json(required[0])["inputs"]["files"]):
            raise unittest.SkipTest("Original local source evidence not installed")
        cls.rows, cls.prior, cls.inputs = review.sealed_audit()
        cls.notes = review.load_rows(review.OUT / "manual-decisions.jsonl")
        cls.plans = review.load_rows(review.OUT / "manual-repairs.jsonl")
        cls.by_line = {r["original_line"]: r for r in cls.rows}
        cls.artifacts = review.compile_artifacts(cls.rows, cls.prior, cls.notes, cls.plans)

    def artifact(self, name):
        return [json.loads(line) for line in self.artifacts[name].splitlines()]

    def test_complete_997_and_191_accounting(self):
        summary = json.loads(self.artifacts["summary.json"])
        self.assertEqual(summary["all_original_histories_accounted"], 997)
        self.assertEqual(summary["flagged_histories"], 122)
        self.assertEqual(summary["total_selected_findings"], 191)
        self.assertEqual(summary["prior_agent_reviewed_findings"], 27)
        self.assertEqual(summary["remaining_findings_directly_reviewed"], 164)
        self.assertEqual(summary["histories_read_for_remaining_findings"], 109)
        self.assertEqual(summary["remaining_findings_unreviewed"], 0)
        self.assertEqual(summary["decisions"], {"clear_issue": 159,
                         "hold_unresolved_meaning": 4, "hold_unsupported_claim": 1})

    def test_no_prior_new_double_count(self):
        ledger = self.artifact("review-ledger.jsonl")
        self.assertEqual(len({r["finding_id"] for r in ledger}), 191)
        old = [r for r in ledger if r["review_pass"] == "prior_27"]
        new = [r for r in ledger if r["review_pass"] == "remaining_164"]
        self.assertFalse({r["finding_id"] for r in old} & {r["finding_id"] for r in new})
        self.assertTrue(all(r["prior_reviews"] for r in old))
        self.assertTrue(all(not r["automatic_semantic_pass"] for r in new))

    def test_full_original_context_and_source_refs_retained(self):
        registry = self.artifact("source-registry.jsonl")
        for entry in registry:
            original = self.by_line[entry["original_line"]]
            self.assertEqual(entry["original_record"], original["original_record"])
            self.assertEqual(entry["english_original_ref"], original["english_original_ref"])
            self.assertEqual(review.digest(entry["original_record"]), entry["original_record_sha256"])
        self.assertEqual(len(registry), 122)

    def test_no_admission_or_translation_gold(self):
        for name in ("review-ledger.jsonl", "source-registry.jsonl", "issue-clearances.jsonl",
                     "unresolved-findings.jsonl", "candidate-corrections.jsonl", "correction-registry.jsonl"):
            for row in self.artifact(name):
                for flag in ("training_eligible", "training_ready", "whole_translation_gold",
                             "native_certified", "human_reviewed", "production_eligible"):
                    self.assertIs(row[flag], False, (name, flag))

    def test_selected_holds_distinct_from_record_concerns(self):
        holds = self.artifact("unresolved-findings.jsonl")
        self.assertEqual([(r["original_line"], r["turn"]) for r in holds],
                         [(355, 3), (465, 1), (809, 3), (809, 3), (908, 3)])
        line484 = next(r for r in self.artifact("issue-clearances.jsonl") if r["original_line"] == 484)
        self.assertEqual(line484["decision"], "clear_issue")
        self.assertIn("stack underflow", line484["record_concerns"][0])
        self.assertFalse(line484["training_ready"])

    def test_code_and_lyrics_are_not_edited(self):
        candidates = self.artifact("candidate-corrections.jsonl")
        self.assertEqual([r["original_line"] for r in candidates], [132, 875])
        for line in (47, 263, 532, 613, 614, 646, 702, 748, 909, 945):
            rows = [r for r in self.artifact("issue-clearances.jsonl") if r["original_line"] == line]
            self.assertTrue(rows, line)
            self.assertTrue(all(not r["candidate_correction_ids"] for r in rows))

    def test_source_anchors_exact(self):
        for row in self.artifact("issue-clearances.jsonl") + self.artifact("unresolved-findings.jsonl"):
            source = self.by_line[row["original_line"]]["original_record"]["english_messages"]
            anchor = row["source_anchor"]
            self.assertIn(anchor["quote"], source[anchor["turn"]]["content"])

    def test_missing_duplicate_prior_or_unknown_findings_rejected(self):
        cases = []
        missing = copy.deepcopy(self.notes)
        missing.pop()
        cases.append(missing)
        duplicate = copy.deepcopy(self.notes)
        duplicate.append(duplicate[0])
        cases.append(duplicate)
        unknown = copy.deepcopy(self.notes)
        unknown[0]["reviews"][0]["indices"] = [999]
        cases.append(unknown)
        prior = copy.deepcopy(self.notes)
        prior.append({"line": 132, "record_concerns": [], "reviews": [{
            "indices": [0], "decision": "clear_issue", "kind": "invalid", "reason": "invalid",
            "source_anchor": {"turn": 0, "quote": "Hey"}}]})
        cases.append(prior)
        for case in cases:
            with self.subTest(case=case[-1]["line"]), self.assertRaises(ValueError):
                review.validate_notes(self.rows, case)

    def test_unsupported_verdict_or_anchor_rejected(self):
        for mutation in ("verdict", "quote", "turn", "blank_reason", "bool_index"):
            notes = copy.deepcopy(self.notes)
            note = notes[0]["reviews"][0]
            if mutation == "verdict":
                note["decision"] = "translation_gold"
            elif mutation == "quote":
                note["source_anchor"]["quote"] = "not a source quote: fabricated"
            elif mutation == "turn":
                note["source_anchor"]["turn"] = -1
            elif mutation == "blank_reason":
                note["reason"] = " "
            else:
                note["indices"] = [True]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                review.validate_notes(self.rows, notes)

    def test_prior_review_binding_required(self):
        prior = copy.deepcopy(self.prior)
        prior[0]["id"] = "wrong-record"
        with self.assertRaisesRegex(ValueError, "prior review binding"):
            review.compile_artifacts(self.rows, prior, self.notes, self.plans)

    def test_exact_deletion_replays_without_any_insertion(self):
        for candidate in self.artifact("candidate-corrections.jsonl"):
            original = self.by_line[candidate["original_line"]]["original_record"]
            transform = candidate["transformation"]
            old, new = candidate["original_target"], candidate["final_changed_target"]
            start, end = transform["start"], transform["end"]
            self.assertEqual(old[start:end], transform["deleted_text"])
            self.assertEqual(new, old[:start] + old[end:])
            self.assertEqual(old, new[:start] + transform["deleted_text"] + new[start:])
            self.assertEqual(transform["inserted_characters"], 0)
            self.assertEqual(review.text_sha(new), candidate["changed_target_sha256"])
            self.assertEqual(review.digest(candidate["messages"]), candidate["messages_sha256"])
            self.assertEqual(candidate["english_messages"], original["english_messages"])
            for turn, message in enumerate(original["messages"]):
                if turn != candidate["turn"]:
                    self.assertEqual(candidate["messages"][turn], message)
            self.assertEqual(candidate["split"], original["split"])
            self.assertEqual(candidate["source_group_id"], original["group_id"])

    def test_name_list_all_unique_entries_and_followup_survive(self):
        candidate = self.artifact("candidate-corrections.jsonl")[0]
        old, new = candidate["original_target"], candidate["final_changed_target"]
        self.assertEqual(set(old.splitlines()), set(new.splitlines()))
        self.assertEqual(old.count("Elijah -"), 2)
        self.assertEqual(new.count("Elijah -"), 1)
        self.assertEqual(candidate["messages"][2:], self.by_line[132]["original_record"]["messages"][2:])

    def test_essay_conclusion_and_other_traditions_not_deleted(self):
        candidate = self.artifact("candidate-corrections.jsonl")[1]
        old, new = candidate["original_target"], candidate["final_changed_target"]
        self.assertEqual(old.split("\n\n")[-1], new.split("\n\n")[-1])
        self.assertEqual(len(old.split("\n\n")), len(new.split("\n\n")))
        self.assertEqual(candidate["messages"][2:], self.by_line[875]["original_record"]["messages"][2:])

    def test_bad_deletion_plans_rejected(self):
        for mutation in ("first", "missing", "source", "user", "code", "quote", "empty", "finding"):
            plan = copy.deepcopy(self.plans[0])
            row = copy.deepcopy(self.by_line[132])
            if mutation == "first":
                plan["delete_occurrence"] = 1
            elif mutation == "missing":
                plan["delete_occurrence"] = 99
            elif mutation == "source":
                plan["source_unit"] = "unbound invented source unit"
            elif mutation == "user":
                plan["turn"] = 0
            elif mutation == "code":
                row["original_record"]["messages"][1]["content"] = "```\n" + row["original_record"]["messages"][1]["content"]
                row["original_record_sha256"] = review.digest(row["original_record"])
            elif mutation == "quote":
                plan["target_unit"] = '"quoted"'
            elif mutation == "empty":
                plan["target_unit"] = ""
            else:
                plan["finding_indices"] = [999]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                review.deletion_candidate(row, plan)

    def test_deterministic_byte_replay(self):
        self.assertEqual(self.artifacts,
                         review.compile_artifacts(self.rows, self.prior, self.notes, self.plans))

    def test_sealed_input_hash_drift_rejected(self):
        real = review.file_sha
        with patch.object(review, "file_sha", side_effect=lambda p:
                          "0" * 64 if Path(p).name == "record-audit.jsonl" else real(p)):
            with self.assertRaisesRegex(ValueError, "sealed input drift"):
                review.sealed_audit()

    def test_package_verify_tamper_and_no_reseal(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for name in ("manual-decisions.jsonl", "manual-repairs.jsonl"):
                shutil.copyfile(review.OUT / name, out / name)
            review.package(out)
            review.package(out, verify=True)
            with self.assertRaisesRegex(ValueError, "sealed package"):
                review.package(out)
            for name in ("candidate-corrections.jsonl", "source-registry.jsonl", "manual-decisions.jsonl",
                         "manifest.json", "SEALED.json"):
                path = out / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "replay mismatch"):
                    review.package(out, verify=True)
                path.write_bytes(original)
            extra = out / "unregistered.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "unexpected/missing"):
                review.package(out, verify=True)


if __name__ == "__main__":
    unittest.main()
