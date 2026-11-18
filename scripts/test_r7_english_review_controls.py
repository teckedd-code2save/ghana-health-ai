import copy
import json
import unittest
from collections import Counter

from r7_english_review_controls import (
    SYSTEM, REAL_SOURCE_EXAMPLES, controls, assess, digest, validate_review,
)


class ReviewControlTests(unittest.TestCase):
    def report(self):
        return {"status": "complete", "requests": [{"id": r["id"], "messages": r["messages"]} for r in controls()],
            "results": [{"id": r["id"], "finish_reason": "stop", "output": json.dumps({
                "verdict": r["expected_verdict"], "reason": "Test fixture, not a model qualification.",
                "evidence": r.get("required_evidence") or r["reference_evidence"]})} for r in controls()]}

    def test_coverage_and_labels_not_in_model_input(self):
        rows = controls()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r["id"] for r in rows}), 48)
        self.assertEqual(Counter(r["expected_verdict"] for r in rows),
                         {"accept": 20, "reject": 20, "needs_review": 8})
        self.assertTrue(all(not r["training_eligible"] for r in rows))
        for row in rows:
            self.assertEqual(set(json.loads(row["messages"][-1]["content"])), {"history", "assistant_target"})
            self.assertEqual(row["messages"][0]["content"], SYSTEM)

    def test_real_sources_are_hash_bound_full_histories(self):
        for row in REAL_SOURCE_EXAMPLES:
            payload, source = row["payload"], row["provenance"]
            messages = payload["history"] + [{"role": "assistant", "content": payload["assistant_target"]}]
            self.assertEqual(digest(messages), source["messages_sha256"])
            self.assertEqual(row["source_record_id"], source["id"])
            self.assertEqual(row["source_group"], source["source_id"])
            self.assertEqual(len(source["ancestry"]), len(messages))
            self.assertEqual(source["ancestry"][-1]["message_id"], source["id"].split(":", 1)[1])
        self.assertEqual(Counter(r["provenance"]["coverage_bucket"] for r in REAL_SOURCE_EXAMPLES),
                         {"category_list": 2, "correction_cue": 2, "followup": 2, "other": 2})

    def test_real_context_and_uncertainty_labels(self):
        self.assertEqual([r["expected_verdict"] for r in REAL_SOURCE_EXAMPLES],
                         ["accept", "reject", "accept"] + ["needs_review"] * 5)
        superhero, correction, roleplay = REAL_SOURCE_EXAMPLES[:3]
        self.assertIn("Imagine", superhero["required_evidence"][0]["quote"])
        self.assertIn("Douglas Adams", correction["required_evidence"][0]["quote"])
        self.assertEqual([e["message_index"] for e in roleplay["required_evidence"]], [0, None, 2])

    def test_pass_is_scoped_not_actual_qualification(self):
        result = assess(self.report())
        self.assertTrue(result["qualified_for_english_source_triage_only"])
        self.assertEqual(result["passed_controls"], 48)
        self.assertEqual(result["real_source_controls"], 8)
        for key in ("twi_qualified", "clinical_qualified", "automatic_training_admission_authorized"):
            self.assertFalse(result[key])

    def test_old_32_control_run_fails_closed(self):
        report = self.report()
        report["requests"] = report["requests"][:32]
        report["results"] = report["results"][:32]
        with self.assertRaises(ValueError):
            assess(report)

    def test_legacy_two_key_output_no_longer_qualifies(self):
        report = self.report()
        report["results"][0]["output"] = '{"verdict":"accept","reason":"No anchors"}'
        self.assertFalse(assess(report)["qualified_for_english_source_triage_only"])

    def test_wrong_verdict_or_truncation_blocks_qualification(self):
        for change in ("wrong", "truncated", "broken"):
            report = self.report()
            result = report["results"][0]
            if change == "wrong":
                value = json.loads(result["output"])
                value["verdict"] = "reject"
                result["output"] = json.dumps(value)
            elif change == "truncated":
                result["finish_reason"] = "length"
            else:
                result["output"] = "broken"
            self.assertFalse(assess(report)["qualified_for_english_source_triage_only"])

    def test_generic_existing_quotes_do_not_replace_decisive_frame(self):
        report = self.report()
        value = json.loads(report["results"][40]["output"])
        value["evidence"][0]["quote"] = "Let's have a laugh today."
        report["results"][40]["output"] = json.dumps(value)
        row = assess(report)["rows"][40]
        self.assertTrue(row["valid"])
        self.assertTrue(row["anchor_match_required"])
        self.assertFalse(row["reference_anchor_match"])
        self.assertFalse(row["pass"])

    def test_shorter_synthetic_critical_quotes_do_not_fail_correct_verdict(self):
        report = self.report()
        value = json.loads(report["results"][0]["output"])
        value["evidence"][0]["quote"] = "group X contains acorn, pebble, shell"
        value["evidence"][1]["quote"] = "Acorn, pebble, shell"
        report["results"][0]["output"] = json.dumps(value)
        row = assess(report)["rows"][0]
        self.assertTrue(row["valid"])
        self.assertFalse(row["anchor_match_required"])
        self.assertFalse(row["reference_anchor_match"])
        self.assertTrue(row["pass"])
        value["verdict"] = "reject"
        report["results"][0]["output"] = json.dumps(value)
        self.assertFalse(assess(report)["rows"][0]["pass"])

    def test_short_correction_numbers_keep_verdict_gate(self):
        report = self.report()
        index = next(i for i, r in enumerate(controls()) if r["id"] == "english-review-correction-accept")
        value = json.loads(report["results"][index]["output"])
        value["evidence"][0]["quote"] = "six crates for Q, not four for P"
        value["evidence"][1]["quote"] = "Six crates for Q"
        report["results"][index]["output"] = json.dumps(value)
        self.assertTrue(assess(report)["rows"][index]["pass"])
        value["evidence"][0]["quote"] = "seven crates for Q"
        report["results"][index]["output"] = json.dumps(value)
        self.assertFalse(assess(report)["rows"][index]["pass"])

    def test_altered_duplicate_missing_or_extra_requests_rejected(self):
        for change in ("altered", "duplicate", "missing", "unknown"):
            report = self.report()
            if change == "altered":
                report["requests"][0]["messages"][-1]["content"] = "different"
            elif change == "duplicate":
                report["results"].append(copy.deepcopy(report["results"][0]))
            elif change == "missing":
                report["results"].pop()
            else:
                report["results"][0]["id"] = "not-a-control"
            with self.assertRaises(ValueError):
                assess(report)

    def test_instances_do_not_mutate_embedded_references(self):
        first = controls()
        first[-1]["required_evidence"][0]["quote"] = "changed"
        self.assertNotEqual(controls()[-1]["required_evidence"][0]["quote"], "changed")


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.row = controls()[0]
        self.payload = json.loads(self.row["messages"][-1]["content"])
        self.review = {"verdict": "accept", "reason": "The supplied group matches.",
                       "evidence": self.row["reference_evidence"]}

    def test_all_verdicts_require_both_anchors(self):
        for verdict in ("accept", "reject", "needs_review"):
            value = copy.deepcopy(self.review)
            value["verdict"] = verdict
            self.assertEqual(validate_review(json.dumps(value), self.payload)["verdict"], verdict)
            value["evidence"] = value["evidence"][:1]
            with self.assertRaises(ValueError):
                validate_review(json.dumps(value), self.payload)

    def test_bad_anchor_locations_or_invented_quotes(self):
        for change in ({"quote": "invented"}, {"quote": " "}, {"quote": "x"*601},
                       {"message_index": True}, {"message_index": -1}, {"message_index": 99},
                       {"message_index": None}, {"location": "web"}, {"extra": 1}):
            value = copy.deepcopy(self.review)
            value["evidence"][0].update(change)
            with self.assertRaises(ValueError):
                validate_review(json.dumps(value), self.payload)

    def test_wrong_target_index_and_duplicate_quotes(self):
        for change in ("index", "duplicate", "no_target"):
            value = copy.deepcopy(self.review)
            if change == "index":
                value["evidence"][1]["message_index"] = 0
            elif change == "duplicate":
                value["evidence"].append(value["evidence"][0])
            else:
                value["evidence"][1] = {**value["evidence"][0], "quote": "fictional exercise"}
            with self.assertRaises(ValueError):
                validate_review(json.dumps(value), self.payload)

    def test_strict_bounded_schema(self):
        for change in ({"verdict": "maybe"}, {"reason": ""}, {"reason": "x"*2001},
                       {"reason": None}, {"evidence": {}}, {"evidence": []},
                       {"evidence": self.review["evidence"]*5}, {"extra": True}):
            with self.assertRaises(ValueError):
                validate_review(json.dumps({**self.review, **change}), self.payload)
        with self.assertRaises(ValueError):
            validate_review("x"*16001, self.payload)
        with self.assertRaises(ValueError):
            validate_review('{"verdict":"accept","verdict":"reject"}', self.payload)


if __name__ == "__main__":
    unittest.main()
