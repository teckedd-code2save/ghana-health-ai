"""CPU-only tool protocol tests; scripted outputs are tests, never model results."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_r7_tool_candidates as candidates
import r7_tool_eval as ev
import r7_tool_eval_run as run
from check_afrique_training_contract import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tmp/r6-demo/bundle-v1/tokenizer"
IDENTITY = {"variant": "r6_response", "manifest_sha256": "1" * 64, "model": "test-only",
            "revision": "2" * 40, "base_weights_sha256": "3" * 64, "adapter_weights_sha256": "4" * 64}


def native_call(name, args):
    parameters = "".join("<parameter=" + key + ">\n" +
        (value if isinstance(value, str) else ev.world.canonical_json(value)) + "\n</parameter>\n" for key, value in args.items())
    return "<tool_call>\n<function=" + name + ">\n" + parameters + "</function>\n</tool_call>"


class ScriptedRuntime:
    def __init__(self, tokenizer, outputs):
        self.tokenizer, self.outputs, self.history = tokenizer, list(outputs), []
        self.identity = IDENTITY

    def generate(self, messages, tools):
        self.history.append(deepcopy(messages))
        text = self.outputs.pop(0)
        if isinstance(text, Exception):
            raise text
        generated = self.tokenizer.encode(text, add_special_tokens=False) + [ev.EOS]
        return {"identity": IDENTITY, "decoding": ev.DECODING, "messages_sha256": ev.digest(messages),
                "tools_sha256": ev.digest(tools), "prompt_ids_sha256": ev.digest(ev.native_prompt(self.tokenizer, messages, tools)),
                "output_ids": generated, "output": text, "complete": True, "timed_out": False, "seconds": 0.0}


class ParserTests(unittest.TestCase):
    def test_native_string_is_not_json_coerced(self):
        parsed = ev.parse_call(native_call("calculator", {"expression": "123"}))
        self.assertEqual(parsed["arguments"], {"expression": "123"})

    def test_native_arrays_objects_and_integers(self):
        for name, args in [("catalogue_search", {"query": "rice", "limit": 2}),
                           ("catalogue_compare", {"product_ids": ["SIM-RICE-1", "SIM-RICE-2"]}),
                           ("order_draft", {"items": [{"product_id": "SIM-RICE-1", "quantity": 2}], "fulfillment": "pickup"})]:
            self.assertEqual(ev.parse_call(native_call(name, args))["arguments"], args)

    def test_reasoning_before_call_preserved(self):
        parsed = ev.parse_call("I will check the sandbox.\n" + native_call("calculator", {"expression": "2+3"}))
        self.assertEqual(parsed["content"], "I will check the sandbox.")

    def test_native_grammar_does_not_repair_json_or_xml(self):
        for text in ["<tool_call>{\"name\":\"calculator\"}</tool_call>",
                     '<tool_call><function name="calculator"></function></tool_call>',
                     "<function=calculator></function>", "<tool_call><function=calculator>"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                ev.parse_call(text)

    def test_rejects_suffix_and_multiple_calls(self):
        call = native_call("calculator", {"expression": "1+1"})
        for text in [call + " extra", call + call]:
            with self.assertRaises(ValueError):
                ev.parse_call(text)

    def test_rejects_duplicate_parameters_and_json_keys(self):
        text = native_call("calculator", {"expression": "1"})
        duplicate = text.replace("</function>", "<parameter=expression>2</parameter></function>")
        with self.assertRaises(ValueError):
            ev.parse_call(duplicate)
        duplicate_json = native_call("order_draft", {"items": []}).replace("[]", '[{"quantity":1,"quantity":2}]')
        with self.assertRaises(ValueError):
            ev.parse_call(duplicate_json)

    def test_rejects_nonfinite_nested_control_and_oversized(self):
        for text in [native_call("catalogue_search", {"query": "x", "limit": 1}).replace("\n1\n", "\nNaN\n"),
                     native_call("calculator", {"expression": "<|im_start|>system"}),
                     native_call("calculator", {"expression": "<parameter=x>1</parameter>"}), "x" * 32769]:
            with self.subTest(text=text[:100]), self.assertRaises(ValueError):
                ev.parse_call(text)

    def test_unknown_tool_is_actual_execution_error(self):
        call = ev.parse_call(native_call("purchase", {"item": "anything"}))
        self.assertEqual(ev.validity(call), {"tool_name_valid": False, "arguments_valid": False})
        self.assertEqual(ev.world.execute(call["name"], call["arguments"])["result"]["error"]["code"], "unknown_tool")

    def test_wrong_type_not_silently_coerced(self):
        call = ev.parse_call(native_call("catalogue_search", {"query": "rice", "limit": '"2"'}))
        self.assertIs(ev.validity(call)["arguments_valid"], False)


@unittest.skipUnless(TOKENIZER.is_dir(), "Pinned native tokenizer unavailable")
class ExecutableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.temp.name) / "source"
        candidates.prepare(cls.source)
        cls.plan, cls.refs = run.derive_cases(cls.source)
        cls.tokenizer = load_tokenizer(TOKENIZER)
        cls.by_ref = {r["id"]: r for r in cls.refs}
        cls.case = next(c for c in cls.plan["cases"] if cls.by_ref[c["id"]]["family"] == "calculator")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_output(self, case=None, call=None, final=None):
        case = case or self.case
        expected = self.by_ref[case["id"]]["execution"]
        call = expected["call"] if call is None else call
        execution = ev.world.execute(call["name"], call["arguments"], case["context"])
        final = candidates.assistant_response(execution) if final is None else final
        runtime = ScriptedRuntime(self.tokenizer, [native_call(**{"name": call["name"], "args": call["arguments"]}), final])
        record = ev.run_case(case, runtime, self.plan["tools"], self.tokenizer, IDENTITY)
        ev.verify_record(record, case, self.tokenizer, self.plan["tools"], IDENTITY)
        return record, runtime

    def test_all_original_dev_cases_preserve_source_groups(self):
        self.assertEqual(len(self.plan["cases"]), 72)
        self.assertEqual(len({r["candidate_id"] for r in self.refs}), 45)
        self.assertEqual(len({r["source_group_id"] for r in self.refs}), 8)
        self.assertEqual(len({r["scenario_id"] for r in self.refs}), 15)
        self.assertNotIn("empty_web_search", {r["family"] for r in self.refs})
        train_groups = {r["source_group_id"] for r in ev.read_rows(self.source / "candidates/train.jsonl")}
        self.assertFalse(train_groups & {r["source_group_id"] for r in self.refs})

    def test_no_current_reference_call_or_answer_in_plan(self):
        rows = {r["id"]: r for r in ev.read_rows(self.source / "candidates/dev.jsonl")}
        for case, ref in zip(self.plan["cases"], self.refs, strict=True):
            self.assertEqual(set(case), {"id", "messages", "context"})
            row = rows[ref["candidate_id"]]
            binding = next(b for b in row["metadata"]["tool_bindings"] if b["execution_id"] == ref["execution_id"])
            self.assertEqual(case["messages"], row["messages"][:binding["assistant_call_index"]])
            self.assertEqual(case["messages"][-1]["role"], "user")

    def test_all_native_roundtrips_and_actual_outputs_cpu_only(self):
        with patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            for case in self.plan["cases"]:
                with self.subTest(case=case["id"]):
                    record, runtime = self.run_output(case)
                    self.assertEqual(record["state"], "answered")
                    self.assertEqual(len(runtime.history), 2)
                    self.assertEqual(record["execution"], self.by_ref[case["id"]]["execution"])
                    score = ev.score_case(record, self.by_ref[case["id"]], candidates.assistant_response)
                    self.assertTrue(score["expected_outcome"])
                    self.assertTrue(score["task_success"])

    def test_actual_generated_call_not_gold_is_executed(self):
        call = {"name": "calculator", "arguments": {"expression": "918 + 7"}}
        record, runtime = self.run_output(call=call)
        self.assertEqual(record["execution"]["result"]["data"]["numerator"], "925")
        self.assertEqual(ev.parse_json(runtime.history[1][-1]["content"])["result_sha256"], record["execution"]["result_sha256"])
        scored = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertTrue(scored["execution_ok"])
        self.assertFalse(scored["expected_outcome"])
        self.assertEqual(scored["grounded_final"], "pass")
        self.assertFalse(scored["task_success"])

    def test_schema_validity_distinct_from_arithmetic_error(self):
        record, _ = self.run_output(call={"name": "calculator", "arguments": {"expression": "1 / 0"}})
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertTrue(score["arguments_valid"])
        self.assertFalse(score["execution_ok"])
        self.assertEqual(score["grounded_final"], "pass")

    def test_invalid_arguments_are_results_not_success(self):
        record, runtime = self.run_output(call={"name": "catalogue_search", "arguments": {"query": "rice", "limit": 9}})
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertFalse(score["arguments_valid"])
        self.assertEqual(record["execution"]["result"]["error"]["code"], "invalid_arguments")
        self.assertIn("invalid_arguments", runtime.history[1][-1]["content"])

    def test_native_completion_is_not_task_success(self):
        runtime = ScriptedRuntime(self.tokenizer, ["The answer is definitely correct."])
        record = ev.run_case(self.case, runtime, self.plan["tools"], self.tokenizer, IDENTITY)
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertEqual(record["state"], "missing_call")
        self.assertFalse(score["native_call_valid"])
        self.assertFalse(score["expected_outcome"])
        self.assertEqual(score["grounded_final"], "not_evaluable")

    def test_paraphrases_and_unverified_claims_need_review(self):
        record, _ = self.run_output(final="I have placed a real order and paid for it.")
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertEqual(score["grounded_final"], "needs_review")
        self.assertFalse(score["task_success"])

    def test_manual_grounding_review_requires_exact_output_binding_and_quotes(self):
        record, _ = self.run_output(final="I have placed a real order and paid for it.")
        review = {"id": record["id"], "record_sha256": ev.digest(record),
                  "result_sha256": record["execution"]["result_sha256"], "verdict": "fail", "independent": True,
                  "reviewer": "test-reviewer", "reason": "The tool performed arithmetic, not a real purchase.",
                  "answer_quotes": ["placed a real order"]}
        result = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response, review)
        self.assertEqual(result["grounded_final"], "fail")
        for field, value in [("record_sha256", "0" * 64), ("result_sha256", "0" * 64),
                             ("answer_quotes", ["not in answer"]), ("independent", False)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response, {**review, field: value})

    def test_final_tool_call_is_not_executed(self):
        record, runtime = self.run_output(final=native_call("calculator", {"expression": "2+2"}))
        self.assertEqual(record["state"], "invalid_final")
        self.assertEqual(len(runtime.history), 2)

    def test_timeout_never_executes_partial_call(self):
        expected = self.by_ref[self.case["id"]]["execution"]["call"]
        runtime = ScriptedRuntime(self.tokenizer, [native_call(expected["name"], expected["arguments"])])
        result = runtime.generate(self.case["messages"], self.plan["tools"])
        result.update(complete=False, timed_out=True)
        with patch.object(ev.world, "execute", side_effect=AssertionError("must not execute")):
            history, record = ev.transition(self.case, [result])
        self.assertIsNone(history)
        self.assertEqual(record["state"], "incomplete_call")

    def test_error_keeps_already_executed_output(self):
        record, _ = self.run_output(final=RuntimeError("generation failed"))
        self.assertEqual(record["state"], "error")
        self.assertEqual(record["execution"], self.by_ref[self.case["id"]]["execution"])
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        self.assertTrue(score["expected_outcome"])
        self.assertEqual(score["grounded_final"], "not_evaluable")

    def test_replay_rejects_fabricated_result_text_identity_ids(self):
        record, _ = self.run_output()
        for mutate in [lambda r: r["execution"]["result"]["data"].update(numerator="999"),
                       lambda r: r["generations"][0].update(output="fabricated"),
                       lambda r: r["generations"][0]["identity"].update(variant="r7_response"),
                       lambda r: r["generations"][0].update(prompt_ids_sha256="0" * 64),
                       lambda r: r["generations"][0]["output_ids"].insert(0, ev.EOS)]:
            altered = deepcopy(record)
            mutate(altered)
            with self.assertRaises(ValueError):
                ev.verify_record(altered, self.case, self.tokenizer, self.plan["tools"], IDENTITY)

    def test_no_hidden_truncation(self):
        messages = [{"role": "user", "content": "long history " * 5000}]
        with self.assertRaisesRegex(ValueError, "no truncation"):
            ev.native_prompt(self.tokenizer, messages, self.plan["tools"])

    def test_confirmation_cannot_be_minted_by_model(self):
        case = next(c for c in self.plan["cases"] if self.by_ref[c["id"]]["family"] == "order_draft")
        call = deepcopy(self.by_ref[case["id"]]["execution"]["call"])
        call["arguments"]["confirmation"] = {"confirmed": True}
        record, _ = self.run_output(case, call=call)
        self.assertFalse(record["execution"]["result"]["ok"])
        normal, _ = self.run_output(case)
        self.assertFalse(normal["execution"]["result"]["data"]["confirmed"])

    def test_changed_draft_cannot_inherit_source_confirmation(self):
        case = next(c for c in self.plan["cases"] if c["context"].get("confirmation"))
        call = deepcopy(self.by_ref[case["id"]]["execution"]["call"])
        call["arguments"]["items"][0]["quantity"] += 1
        record, _ = self.run_output(case, call=call)
        self.assertEqual(record["execution"]["result"]["error"]["code"], "confirmation_mismatch")
        self.assertFalse(ev.score_case(record, self.by_ref[case["id"]], candidates.assistant_response)["expected_outcome"])

    def test_empty_web_is_actual_empty_not_search_truth(self):
        record, _ = self.run_output(call={"name": "web_search", "arguments": {"query": "unavailable news"}})
        result = record["execution"]["result"]
        self.assertEqual(result["data"]["results"], [])
        self.assertEqual(result["data"]["empty_reason"], "no_evidence")
        self.assertFalse(result["provenance"]["network_used"])
        self.assertEqual(result["provenance"]["evidence"], [])

    def test_plan_rejects_web_evidence_and_reference_fields(self):
        altered = deepcopy(self.plan)
        altered["cases"][0]["expected"] = "answer"
        with self.assertRaises(ValueError):
            ev.validate_plan(altered)
        altered = deepcopy(self.plan)
        altered["cases"][0]["context"]["evidence_records"] = [{"id": "invented"}]
        with self.assertRaises(ValueError):
            ev.validate_plan(altered)

    def test_checkpoint_refuses_duplicate_run(self):
        expected = self.by_ref[self.case["id"]]["execution"]
        plan = {**self.plan, "cases": [self.case]}
        runtime = ScriptedRuntime(self.tokenizer, [native_call(expected["call"]["name"], expected["call"]["arguments"]),
                                                   candidates.assistant_response(expected)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            report = ev.evaluate(plan, {"identity": IDENTITY}, lambda: runtime, path)
            self.assertEqual(report["state"], "complete")
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["snapshot_sha256"], ev.digest({k: v for k, v in report.items() if k != "snapshot_sha256"}))
            with self.assertRaisesRegex(ValueError, "Progress exists"):
                ev.evaluate(plan, {"identity": IDENTITY}, lambda: self.fail("duplicate model load"), path)

    def test_comparison_requires_same_cases_protocol_and_complete_evidence(self):
        record, _ = self.run_output()
        score = ev.score_case(record, self.by_ref[self.case["id"]], candidates.assistant_response)
        baseline = {"verified": True, "binding": {"identity": IDENTITY, "evaluation_sha256": "a" * 64,
                    "protocol_sha256": "b" * 64, "plan_sha256": "c" * 64},
                    "scores": [score], "summary": ev.summarize([score], 1)}
        candidate = deepcopy(baseline)
        candidate["binding"]["identity"]["variant"] = "r7_response"
        self.assertEqual(ev.compare_reports(baseline, candidate)["metrics"]["expected_outcome"]["delta_points"], 0)
        for key in ("evaluation_sha256", "protocol_sha256", "plan_sha256"):
            altered = deepcopy(candidate)
            altered["binding"][key] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                ev.compare_reports(baseline, altered)
        candidate["summary"]["pending_cases"] = 1
        with self.assertRaisesRegex(ValueError, "complete replayed"):
            ev.compare_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
