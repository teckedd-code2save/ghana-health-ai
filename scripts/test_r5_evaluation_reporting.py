"""Capability denominators from metadata, not inferred semantic scores."""
import unittest

import run_understanding_r5_evaluation as runner
import test_run_understanding_r5_evaluation as fixtures


class BenchmarkTests(unittest.TestCase):
    def test_all_states_retained_in_capabilities_and_groups_not_model_copies(self):
        states = ("complete", "incomplete", "paused", "error", "blocked", "indeterminate", "pending")
        cases = [{"id": state, "messages": [{"role": "user", "content": "synthetic"}],
                  "tags": ["correction", "meaning", "meaning"], "family": "synthetic-task",
                  "input_language": "tw", "response_language": "en", "source_group_id": "shared-source",
                  "objective": {} if state in {"complete", "error"} else None} for state in states]
        plan = list(runner.slots(cases, [], [], policy="guarded"))
        records = {}
        for slot in plan:
            state = slot["case_id"]
            verified = state in {"complete", "incomplete", "paused"}
            records[slot["key"]] = None if state == "pending" else {
                "state": state, "verified": verified,
                "result": {"finish_reason": {"complete": "eos", "incomplete": "max_new_tokens", "paused": "cancelled"}.get(state)},
                "objective": {"literal_match": True, "passed": True} if state == "complete" else None}
        report = runner.benchmark_report(plan, records, runner.KIND, "guarded")
        self.assertEqual(report["generation_policy"], "guarded")
        self.assertFalse(report["completion_is_accuracy"])
        self.assertEqual(report["meaning_assessment"]["status"], "human_review_pending")
        for variant in runner.VARIANTS:
            total = report["totals"][variant]
            self.assertEqual((total["expected_slots"], total["expected_cases"], total["source_group_count"]), (7, 7, 1))
            self.assertEqual(total["states"], dict.fromkeys(states, 1))
            self.assertEqual(sum(total["states"].values()), total["expected_slots"])
            self.assertEqual(total["strict_literal"], {"expected_slots": 2, "evaluated_slots": 1,
                                                       "literal_matches": 1, "literal_matches_with_eos": 1})
            self.assertEqual(total["mechanical_attention_slots"], 6)
            self.assertIsNone(total["meaning_accuracy"])
        for group in report["groups"]:
            for total in group["models"].values():
                self.assertEqual(total["expected_slots"], 7)
                self.assertEqual(total["source_group_count"], 1)
                self.assertEqual(total["states"]["error"], 1)
                self.assertEqual(total["states"]["pending"], 1)
        categories = [g for g in report["groups"] if g["dimension"] == "category"]
        self.assertEqual(len(categories), 2)
        self.assertEqual(sum(g["models"]["base"]["expected_slots"] for g in categories), 14)
        self.assertEqual(report["category_membership"], "multi_label_nonadditive_across_categories")

    def test_rollout_slots_languages_and_unclassified_capabilities(self):
        roll = {"id": "paired-en", "source_group_id": "paired-task", "language": "en",
                "phenomena": ["context"], "user_turns": ["first", "second", "third"],
                "output_languages": ["tw", "en", "mixed"]}
        fixed = {"id": "legacy", "messages": [{"role": "user", "content": "Do not classify me from text"}]}
        plan = list(runner.slots([fixed], [roll], []))
        report = runner.benchmark_report(plan, {s["key"]: None for s in plan}, runner.FRESH_KIND, "raw")
        self.assertEqual(report["totals"]["base"]["expected_slots"], 4)
        self.assertEqual(report["totals"]["base"]["expected_cases"], 2)
        self.assertEqual(report["totals"]["base"]["source_group_count"], 1)
        self.assertEqual(report["totals"]["base"]["unclassified_source_group_cases"], 1)
        tasks = [g for g in report["groups"] if g["dimension"] == "task"]
        self.assertEqual([g["labels"] for g in tasks], [{"task": "unclassified"}])
        output = {g["labels"]["output_language"]: g["models"]["base"]["expected_slots"]
                  for g in report["groups"] if g["dimension"] == "output_language"}
        self.assertEqual(output, {"tw": 1, "en": 1, "mixed": 1, "unclassified": 1})
        labels = runner.benchmark_labels(plan[0])
        self.assertEqual(labels["categories"], ["unclassified"])
        self.assertEqual(labels["input_language"], "unclassified")
        self.assertIsNone(labels["label_sources"]["task"])

    def test_real_development_metadata_all_75_slots_without_source_edits(self):
        before = runner.files(fixtures.EVALUATION)
        _, cases, rollouts, _ = runner.evaluation(fixtures.EVALUATION)
        plan = list(runner.slots(cases, rollouts, []))
        report = runner.benchmark_report(plan, {s["key"]: None for s in plan}, runner.KIND, "raw")
        for total in report["totals"].values():
            self.assertEqual((total["expected_slots"], total["expected_cases"]), (75, 56))
            self.assertEqual(total["states"]["pending"], 75)
        self.assertTrue(any(g["labels"] == {"task": "basket-arithmetic"} for g in report["groups"]))
        self.assertTrue(any(g["labels"] == {"category": "unclassified"} for g in report["groups"]))
        for dimension in ("task", "cohort", "input_language", "output_language", "evaluation_type"):
            self.assertEqual(sum(g["models"]["base"]["expected_slots"] for g in report["groups"]
                                 if g["dimension"] == dimension), 75)
        self.assertEqual(before, runner.files(fixtures.EVALUATION))

    def test_saved_report_has_policy_specific_breakdown_and_errors(self):
        engine = fixtures.EngineTests()
        engine.setUp()
        self.addCleanup(engine.doCleanups)
        engine.cases[0].update(tags=["synthetic"], family="task-a", input_language="en", response_language="tw")
        engine.plan = list(runner.slots(engine.cases, engine.rollouts, []))
        runtime = fixtures.Runtime()
        def fail(messages, variant):
            raise RuntimeError("synthetic failure")
        runtime.reply = fail
        engine.run_engine(runtime)
        result = runner.report(engine.root, engine.plan, engine.journal, {"kind": runner.KIND},
                               {v: runtime.identity(v) for v in runner.VARIANTS})
        saved = runner.read(engine.root / "benchmark.json")
        self.assertEqual(saved["generation_policy"], "raw")
        self.assertEqual(saved["kind"], runner.KIND)
        self.assertEqual(saved["totals"], result["benchmark"]["totals"])
        self.assertEqual(saved["totals"]["base"]["states"]["error"], 2)
        self.assertEqual(saved["totals"]["base"]["states"]["blocked"], 2)
        reviews = runner.read(engine.root / "review.json")["rows"]
        self.assertEqual(len(reviews), 12)
        self.assertEqual(reviews[0]["error_type"], "RuntimeError")
        self.assertEqual(reviews[0]["benchmark_labels"]["categories"], ["synthetic"])
        self.assertEqual(reviews[0]["output_language"], "tw")
        mixed = [{**s, "policy": "guarded"} if i else s for i, s in enumerate(engine.plan)]
        with self.assertRaisesRegex(ValueError, "separate executions"):
            runner.report(engine.root, mixed, engine.journal, {}, {v: runtime.identity(v) for v in runner.VARIANTS})


if __name__ == "__main__":
    unittest.main()
