import copy
from html import escape
from html.parser import HTMLParser
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from package_instruction_review import checked, digest, progress_html, project, sha


def progress_summary(release=None):
    summary = {"as_of": "2026-10-20", "checkpoints": "D/E fixture",
        "D": {"pending_work_items": 1, "cumulative_source_adjudicated_derivatives": 0},
        "E": {"agent_assessments": 2, "pending_assessments": 0,
              "outcomes": {"agent_pass_to_F": 2, "source_hold": 1}},
        "decisions": {"replace": 1, "retain": 1},
        "D_manifest_sha256": "a" * 64, "E_manifest_sha256": "b" * 64}
    if release is not None:
        summary["training_release"] = release
    return summary


def training_release(**updates):
    return {"fg_complete": True, "fg_admitted": 650, "fg_excluded": 47,
        "checks_complete": False, "comparison_records": 150123,
        "export_ready": False, "export_sources": 0, "export_excluded": 0,
        "admission_manifest_sha256": "c" * 64, "export_manifest_sha256": None, **updates}


class ProgressNodes(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.states = {}
        self.node = None
        self.in_state = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.node = dict(attrs)["id"]
        elif tag == "em":
            self.in_state = True

    def handle_endtag(self, tag):
        if tag == "li":
            self.node = None
        elif tag == "em":
            self.in_state = False

    def handle_data(self, data):
        if self.in_state:
            self.states[self.node] = data


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

    def test_map_without_training_release_preserves_exact_legacy_output(self):
        html = progress_html(progress_summary())
        # Captured from the original renderer before adding release support.
        self.assertEqual(digest(html), "501abf9ae92a9d2ce69bdcbcde36392409bcb6ed175888841dd25c1742400661")
        self.assertEqual(ProgressNodes(html).states["node-D"], "In progress")
        self.assertNotIn("Admission manifest:", html)
        self.assertNotIn("Export manifest:", html)

    def test_map_fg_only_keeps_validation_in_progress_and_final_counts_unsealed(self):
        summary = progress_summary(training_release(export_sources=620, export_excluded=30))
        original = copy.deepcopy(summary)
        html = progress_html(summary)
        states = ProgressNodes(html).states
        self.assertEqual(states["node-D"], "Repairable queue processed with source holds")
        self.assertEqual(states["node-F"], "Complete")
        self.assertEqual(states["node-G"], "Complete")
        self.assertEqual(states["node-I"], "In progress")
        self.assertEqual(states["node-J"], "Pending")
        self.assertEqual(states["node-K"], "Not started")
        for text in ("650 provisional research admissions", "47 admission exclusions",
                     "150123 comparison records", "Final eligible research count pending",
                     "620 unsealed export sources", "30 export exclusions", "0 human approvals",
                     "Noncommercial research only", "No training.", "No all-four readiness claim",
                     "Phrase confirmations do not approve whole examples", "confirmation remains narrow"):
            self.assertIn(text, html)
        self.assertNotIn("620 final eligible research sources", html)
        self.assertNotIn("0 training admissions", html)
        self.assertIn(f'Admission manifest: <code>{"c" * 64}</code>', html)
        self.assertIn("Export manifest: <code>Not provided</code>", html)
        self.assertEqual(summary, original)

    def test_map_sealed_j_distinguishes_provisional_admissions_and_final_eligibility(self):
        summary = progress_summary(training_release(checks_complete=True, export_ready=True,
            export_sources=610, export_excluded=40, export_manifest_sha256="d" * 64))
        html = progress_html(summary)
        states = ProgressNodes(html).states
        self.assertEqual(states["node-I"], "Complete")
        self.assertEqual(states["node-J"], "Sealed research partition")
        self.assertEqual(states["node-K"], "Not started")
        for text in ("650 provisional research admissions", "47 admission exclusions",
                     "610 final eligible research sources", "40 export exclusions",
                     "0 human approvals", "Noncommercial research only", "No training.",
                     "No all-four readiness claim", "not certification of the entire corpus"):
            self.assertIn(text, html)
        for label, checksum in (("D", "a" * 64), ("E", "b" * 64),
                                ("Admission", "c" * 64), ("Export", "d" * 64)):
            self.assertIn(f"{label} manifest: <code>{checksum}</code>", html)
        self.assertNotIn("Final eligible research count pending", html)
        self.assertEqual(html.count("<details>"), 11)
        self.assertEqual(html.count("Open complete examples"), 3)

    def test_map_completion_uses_flags_not_counts_or_manifest_presence(self):
        for fg, checks, export in product((False, True), repeat=3):
            with self.subTest(fg=fg, checks=checks, export=export):
                release = training_release(fg_complete=fg, checks_complete=checks, export_ready=export,
                    export_sources=610, export_excluded=40, export_manifest_sha256="d" * 64)
                states = ProgressNodes(progress_html(progress_summary(release))).states
                self.assertEqual(states["node-F"], "Complete" if fg else "Pending")
                self.assertEqual(states["node-G"], "Complete" if fg else "Pending")
                self.assertEqual(states["node-I"], "Complete" if checks else "In progress")
                self.assertEqual(states["node-J"], "Sealed research partition" if export else "Pending")
        html = progress_html(progress_summary(training_release(export_ready=True)))
        self.assertEqual(ProgressNodes(html).states["node-J"], "Sealed research partition")
        self.assertIn("Export manifest: <code>Not provided</code>", html)

    def test_map_release_hashes_and_counts_are_html_escaped(self):
        payload = '<script>alert("x")</script>&\''
        for ready in (False, True):
            with self.subTest(ready=ready):
                release = training_release(export_ready=ready, fg_admitted=payload, fg_excluded=payload,
                    comparison_records=payload, export_sources=payload, export_excluded=payload,
                    admission_manifest_sha256=payload, export_manifest_sha256=payload)
                html = progress_html(progress_summary(release))
                self.assertNotIn(payload, html)
                self.assertNotIn("<script>", html)
                self.assertIn(f"Admission manifest: <code>{escape(payload)}</code>", html)
                self.assertIn(f"Export manifest: <code>{escape(payload)}</code>", html)
                self.assertIn(f"{escape(payload)} provisional research admissions", html)
                self.assertIn(f"{escape(payload)} admission exclusions", html)
                self.assertIn(f"{escape(payload)} comparison records", html)
                self.assertIn(f"{escape(payload)} export exclusions", html)


if __name__ == "__main__":
    unittest.main()
