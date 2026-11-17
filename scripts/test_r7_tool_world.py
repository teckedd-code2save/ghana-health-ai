"""Offline contract tests. Snapshot fixtures below are NOT a source corpus."""

import ast
import copy
from decimal import getcontext, setcontext
from fractions import Fraction
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import r7_tool_world as world


def snapshot(record_id="fixture-a", text="Offline fixture: alpha beta.", **fields):
    # Synthetic byte-contract fixture only; never admitted as real research evidence.
    record = {"id": record_id, "source_type": "web_snapshot",
              "url": "https://example.invalid/offline-fixture", "title": "Offline test fixture",
              "captured_at": "2026-11-17T12:00:00Z", "text": text, **fields}
    record["sha256"] = world.sha256_json(record)
    return record


class ToolWorldTests(unittest.TestCase):
    def success(self, name, args, context=None):
        envelope = world.execute(name, args, context)
        self.assertTrue(envelope["result"]["ok"], envelope["result"])
        self.assert_record(envelope)
        return envelope["result"]["data"]

    def failure(self, name, args, code, context=None):
        envelope = world.execute(name, args, context)
        self.assertFalse(envelope["result"]["ok"], envelope)
        self.assertEqual(envelope["result"]["error"]["code"], code)
        self.assertIsNone(envelope["result"]["data"])
        self.assert_record(envelope)
        return envelope

    def assert_record(self, envelope):
        result = envelope["result"]
        self.assertEqual(envelope["result_sha256"], world.sha256_json(result))
        self.assertFalse(result["provenance"]["network_used"])
        self.assertEqual(result["provenance"]["side_effects"], [])
        if envelope["call"] is None:
            self.assertEqual(envelope["messages"], [])
            return
        self.assertEqual(result["provenance"]["call_sha256"], world.sha256_json(envelope["call"]))
        if type(envelope["call"]["arguments"]) is not dict:
            self.assertEqual(envelope["messages"], [])
            return
        assistant, response = envelope["messages"]
        call = assistant["tool_calls"][0]
        self.assertEqual(call["function"], envelope["call"])
        self.assertEqual(call["id"], response["tool_call_id"])
        self.assertEqual(response["name"], envelope["call"]["name"])
        self.assertEqual(json.loads(response["content"]),
                         {"result": result, "result_sha256": envelope["result_sha256"]})

    def test_function_schemas_and_fresh_copies(self):
        schemas = world.tools("qwen")
        self.assertEqual([s["function"]["name"] for s in schemas],
                         ["calculator", "catalogue_search", "catalogue_compare", "order_draft", "web_search"])
        self.assertEqual(schemas, world.tools("openai"))
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertFalse(schema["function"]["parameters"]["additionalProperties"])
        schemas[0]["function"]["parameters"]["required"].clear()
        self.assertEqual(world.tools()[0]["function"]["parameters"]["required"], ["expression"])
        with self.assertRaises(ValueError):
            world.tools("unsupported")

    def test_actual_outputs_not_supplied_by_model(self):
        envelope = world.execute("calculator", {"expression": "7 * 8"})
        self.assert_record(envelope)
        self.assertEqual(envelope["result"]["data"]["numerator"], "56")
        self.failure("calculator", {"expression": "7 * 8", "result": "99"}, "invalid_arguments")

    def test_determinism_json_object_equivalence_and_no_mutation(self):
        args = {"limit": 2, "query": "rice"}
        context = {"evidence_records": []}
        before = copy.deepcopy((args, context))
        first = world.execute("catalogue_search", args, context)
        second = world.execute("catalogue_search", '{"query":"rice","limit":2}', context)
        self.assertEqual(first, second)
        self.assertEqual(list(first["messages"][0]["tool_calls"][0]["function"]["arguments"]),
                         ["limit", "query"])
        self.assertEqual((args, context), before)
        first["result"]["data"]["products"][0]["price_minor"] = -1
        self.assertEqual(world.execute("catalogue_search", args, context), second)

    def test_hash_binds_actual_result_call_and_context(self):
        a = world.execute("calculator", {"expression": "1 + 2"})
        b = world.execute("calculator", {"expression": "3"})
        c = world.execute("calculator", {"expression": "3"}, {"evidence_records": []})
        self.assertEqual(len({x["result_sha256"] for x in (a, b, c)}), 3)
        changed = copy.deepcopy(a["result"])
        changed["data"]["numerator"] = "4"
        self.assertNotEqual(world.sha256_json(changed), a["result_sha256"])

    def test_exact_arithmetic(self):
        cases = {"0.1 + 0.2": Fraction(3, 10), "(2 + 3) * 4": Fraction(20),
                 "-7 // 3": Fraction(-3), "-7 % 3": Fraction(2),
                 "2 ** -3": Fraction(1, 8), "1 / 3": Fraction(1, 3),
                 "+.5 - 1.": Fraction(-1, 2), "2 ** 3 ** 2": Fraction(512)}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                data = self.success("calculator", {"expression": expression})
                self.assertEqual(Fraction(int(data["numerator"]), int(data["denominator"])), expected)
        self.assertFalse(self.success("calculator", {"expression": "1/3"})["decimal_is_exact"])
        self.assertTrue(self.success("calculator", {"expression": "1/8"})["decimal_is_exact"])

    def test_calculator_does_not_inherit_decimal_context(self):
        previous = getcontext().copy()
        try:
            expected = world.execute("calculator", {"expression": "1/7"})
            getcontext().prec = 2
            getcontext().rounding = "ROUND_DOWN"
            self.assertEqual(world.execute("calculator", {"expression": "1/7"}), expected)
        finally:
            setcontext(previous)

    def test_calculator_rejects_code_and_non_decimal_literals(self):
        for expression in ["__import__('os').system('id')", "open('/tmp/x')", "x + 1",
                           "[1][0]", "True", "1e10", "0xff", "1_000", "...", "1,2"]:
            with self.subTest(expression=expression):
                self.failure("calculator", {"expression": expression}, "invalid_expression")

    def test_calculator_errors_and_bounds(self):
        for expression in ["1/0", "1//0", "1%0", "0**-1"]:
            self.failure("calculator", {"expression": expression}, "division_by_zero")
        for expression in ["2**13", "2**-13", "2**0.5", "1 +", "9" * 33]:
            self.failure("calculator", {"expression": expression}, "invalid_expression")
        self.failure("calculator", {"expression": "9" * 257}, "invalid_arguments")
        self.failure("calculator", {"expression": "+" * 26 + "1"}, "size_limit")
        self.failure("calculator", {"expression": "+".join(["1"] * 50)}, "size_limit")
        self.failure("calculator", {"expression": "((99999999999999999999999999999999**12)**12)"},
                     "size_limit")

    def test_strict_argument_validation(self):
        cases = [("calculator", {}), ("calculator", {"expression": 1}),
                 ("calculator", {"expression": " "}),
                 ("catalogue_search", {"query": "rice", "limit": True}),
                 ("catalogue_search", {"query": "rice", "limit": 1.0}),
                 ("catalogue_search", {"query": "rice", "limit": 6}),
                 ("catalogue_search", {"query": "rice", "limit": 0}),
                 ("catalogue_search", {"query": "rice", "unknown": 1}),
                 ("web_search", {"query": "x", "url": "https://example.invalid"})]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                self.failure(name, args, "invalid_arguments")

    def test_invalid_json_is_recorded_without_invented_arguments(self):
        for args in ['{"query":"one","query":"two"}', '{"limit":NaN}', "{"]:
            result = self.failure("web_search", args, "invalid_json")
            self.assertEqual(result["call"]["arguments"], args)
        for args in [[], "[]", "null", 3, True]:
            self.failure("web_search", args, "invalid_arguments")

    def test_unrecordable_inputs_are_explicitly_rejected(self):
        cycle = {}
        cycle["cycle"] = cycle
        for args, code in [(cycle, "size_limit"), ({"x": "a" * 9000}, "size_limit"),
                           ({"x": float("nan")}, "invalid_json"), ({1: "x"}, "invalid_json"),
                           ({"x": object()}, "invalid_json"), ({"x": "\ud800"}, "invalid_json"),
                           ({"x": 1 << 5000}, "size_limit")]:
            result = self.failure("calculator", args, code)
            self.assertIsNone(result["call"])
            self.assertEqual(result["messages"], [])

    def test_unknown_tools_names_and_context_fields(self):
        self.failure("purchase", {}, "unknown_tool")
        self.failure(None, {}, "invalid_arguments")
        self.failure("x" * 65, {}, "invalid_arguments")
        self.failure("calculator", {"expression": "1"}, "invalid_arguments", {"api_key": "ignored"})
        self.failure("calculator", {"expression": "1"}, "invalid_arguments", [])

    def test_simulated_catalogue_search_and_no_matches(self):
        data = self.success("catalogue_search", {"query": "RICE", "limit": 2})
        self.assertTrue(data["simulated"])
        self.assertEqual(data["currency"], "GHS")
        self.assertEqual(data["price_scale"], 100)
        self.assertEqual([p["id"] for p in data["products"]], ["SIM-RICE-1", "SIM-RICE-2"])
        self.assertEqual([p["price_minor"] for p in data["products"]], [2500, 4600])
        for query in ["unlisted-product", "!!!"]:
            empty = self.success("catalogue_search", {"query": query})
            self.assertEqual(empty["products"], [])
            self.assertEqual(empty["empty_reason"], "no_match")

    def test_compare_preserves_order_and_package_units(self):
        ids = ["SIM-RICE-2", "SIM-RICE-1"]
        data = self.success("catalogue_compare", {"product_ids": ids})
        self.assertEqual([p["id"] for p in data["products"]], ids)
        self.assertEqual([p["unit"] for p in data["products"]], ["2 kg bag", "1 kg bag"])
        self.assertEqual(data["price_spread_minor"], 2100)
        self.assertIn("package sizes may differ", data["comparison_basis"])
        self.failure("catalogue_compare", {"product_ids": ["SIM-RICE-1", "missing"]}, "product_not_found")
        for invalid in [["SIM-RICE-1"], ["SIM-RICE-1"] * 2, ["x"] * 6, "SIM-RICE-1"]:
            self.failure("catalogue_compare", {"product_ids": invalid}, "invalid_arguments")

    def draft_args(self):
        return {"items": [{"product_id": "SIM-RICE-1", "quantity": 2},
                          {"product_id": "SIM-SOAP-1", "quantity": 1}], "fulfillment": "pickup"}

    def test_order_is_draft_only_and_requires_bound_confirmation(self):
        args = self.draft_args()
        data = self.success("order_draft", args)
        self.assertEqual(data["draft"]["total_minor"], 5700)
        self.assertEqual(data["draft_sha256"], world.sha256_json(data["draft"]))
        self.assertTrue(data["requires_confirmation"])
        self.assertFalse(data["confirmed"])
        self.assertEqual(data["status"], "awaiting_confirmation")
        context = {"confirmation": {"confirmed": True, "draft_sha256": data["draft_sha256"]}}
        confirmed = self.success("order_draft", args, context)
        self.assertTrue(confirmed["confirmed"])
        self.assertEqual(confirmed["status"], "confirmed_draft")
        self.assertEqual(confirmed["draft"], data["draft"])
        self.assertFalse(confirmed["purchases_enabled"])
        self.assertEqual(confirmed["side_effects"], [])
        self.assertEqual(self.success("order_draft", args), data)
        changed = copy.deepcopy(args)
        changed["items"][0]["quantity"] = 3
        self.failure("order_draft", changed, "confirmation_mismatch", context)

    def test_model_cannot_confirm_and_context_cannot_infer_consent(self):
        args = self.draft_args()
        self.failure("order_draft", {**args, "confirmed": True}, "invalid_arguments")
        for confirmation in [{"confirmed": False, "draft_sha256": "0" * 64},
                             {"confirmed": 1, "draft_sha256": "0" * 64},
                             {"confirmed": "yes", "draft_sha256": "0" * 64},
                             {"draft_sha256": "0" * 64}]:
            self.failure("order_draft", args, "invalid_arguments", {"confirmation": confirmation})

    def test_draft_validation_stock_and_no_cart_actions(self):
        for quantity in [True, 1.0, 0, 101]:
            self.failure("order_draft", {"items": [{"product_id": "SIM-RICE-1", "quantity": quantity}],
                                         "fulfillment": "pickup"}, "invalid_arguments")
        for product_id, quantity, code in [("SIM-WATER-1", 1, "insufficient_simulated_stock"),
                                          ("SIM-RICE-1", 21, "insufficient_simulated_stock"),
                                          ("missing", 1, "product_not_found")]:
            self.failure("order_draft", {"items": [{"product_id": product_id, "quantity": quantity}],
                                         "fulfillment": "pickup"}, code)
        args = self.draft_args()
        args["items"].append(args["items"][0])
        self.failure("order_draft", args, "invalid_arguments")
        for name in ["checkout", "add_to_cart", "pay", "place_order"]:
            self.failure(name, {}, "unknown_tool")

    def test_delivery_requires_explicit_location_and_confirmation_binds_it(self):
        args = {**self.draft_args(), "fulfillment": "delivery"}
        self.failure("order_draft", args, "invalid_arguments")
        args["location"] = "Simulated pickup point A"
        data = self.success("order_draft", args)
        context = {"confirmation": {"confirmed": True, "draft_sha256": data["draft_sha256"]}}
        args["location"] = "Simulated pickup point B"
        self.failure("order_draft", args, "confirmation_mismatch", context)

    def test_web_missing_evidence_and_no_match_are_empty(self):
        data = self.success("web_search", {"query": "anything"})
        self.assertEqual(data["results"], [])
        self.assertEqual(data["empty_reason"], "no_evidence")
        self.assertFalse(data["live_search"])
        data = self.success("web_search", {"query": "notpresent"}, {"evidence_records": [snapshot()]})
        self.assertEqual(data["results"], [])
        self.assertEqual(data["empty_reason"], "no_match")

    def test_web_ranking_exact_excerpts_provenance_and_limit(self):
        records = [snapshot("fixture-c", "alpha"), snapshot("fixture-b", "alpha beta"),
                   snapshot("fixture-a", "prefix " * 300 + "alpha beta " + "tail " * 300)]
        envelope = world.execute("web_search", {"query": "ALPHA beta", "limit": 2},
                                 {"evidence_records": records})
        self.assert_record(envelope)
        data = envelope["result"]["data"]
        self.assertEqual([row["id"] for row in data["results"]], ["fixture-a", "fixture-b"])
        for row in data["results"]:
            source = next(r for r in records if r["id"] == row["id"])
            self.assertEqual(row["excerpt"], source["text"][row["excerpt_start"]:row["excerpt_end"]])
            self.assertLessEqual(len(row["excerpt"]), 1200)
            self.assertEqual(row["snapshot_sha256"], source["sha256"])
        self.assertTrue(data["results"][0]["truncated"])
        self.assertIn("alpha", data["results"][0]["excerpt"])
        self.assertEqual(envelope["result"]["provenance"]["evidence"],
                         [{"id": r["id"], "sha256": r["sha256"]} for r in records])

    def test_web_checks_all_pins_even_nonmatching_evidence(self):
        for field, changed in [("text", "tampered"), ("title", "changed title"),
                               ("url", "https://example.invalid/other"),
                               ("captured_at", "2026-11-16T12:00:00Z"), ("id", "changed-id")]:
            record = snapshot()
            record[field] = changed
            self.failure("web_search", {"query": "no-match"}, "evidence_hash_mismatch",
                         {"evidence_records": [record]})

    def test_web_rejects_unpinned_generated_or_duplicate_records(self):
        record = snapshot()
        record.pop("sha256")
        self.failure("web_search", {"query": "alpha"}, "invalid_arguments", {"evidence_records": [record]})
        record = snapshot(source_type="generated")
        self.failure("web_search", {"query": "alpha"}, "invalid_arguments", {"evidence_records": [record]})
        self.failure("web_search", {"query": "alpha"}, "invalid_evidence",
                     {"evidence_records": [snapshot(), snapshot()]})

    def test_web_snapshot_url_and_capture_validation(self):
        for url in ["file:///etc/passwd", "https://user:pass@example.invalid/", "https:///missing",
                    "https://example.invalid:bad/", "https://example.invalid/#fragment",
                    "https://example.invalid/has space"]:
            self.failure("web_search", {"query": "alpha"}, "invalid_evidence",
                         {"evidence_records": [snapshot(url=url)]})
        for timestamp in ["2026-02-30T12:00:00Z", "2026-1-1T12:00:00Z"]:
            self.failure("web_search", {"query": "alpha"}, "invalid_evidence",
                         {"evidence_records": [snapshot(captured_at=timestamp)]})

    def test_web_unicode_exact_text_and_boundaries(self):
        record = snapshot(text="Fixture: \u025bs\u025b \u0254kyeame and ALPHA.")
        data = self.success("web_search", {"query": "\u0190S\u0190"}, {"evidence_records": [record]})
        self.assertEqual(data["results"][0]["excerpt"], record["text"])
        self.failure("web_search", {"query": "alpha"}, "invalid_arguments",
                     {"evidence_records": [snapshot(text="a" * 16_385)]})
        self.failure("web_search", {"query": "alpha"}, "invalid_arguments",
                     {"evidence_records": [snapshot(str(i)) for i in range(33)]})

    def test_total_context_budget_and_output_budget(self):
        records = [snapshot(str(i), text="\u025b" * 16_384) for i in range(32)]
        self.failure("web_search", {"query": "fixture"}, "size_limit", {"evidence_records": records})
        records = [snapshot(str(i), text="alpha " + "\u025b" * 16_378) for i in range(5)]
        envelope = world.execute("web_search", {"query": "alpha"}, {"evidence_records": records})
        self.assertTrue(envelope["result"]["ok"])
        self.assertLess(len(world.canonical_json(envelope["result"]["data"])), world.MAX_RESULT_BYTES)
        self.assert_record(envelope)

    def test_errors_are_deterministic_and_preserve_original_object_arguments(self):
        args = {"expression": "1/0"}
        first = world.execute("calculator", args)
        self.assertEqual(first, world.execute("calculator", args))
        self.assertEqual(first["call"]["arguments"], args)
        self.assert_record(first)
        self.assertEqual(json.loads(first["messages"][1]["content"])["result"]["error"]["code"],
                         "division_by_zero")

    def test_all_tools_work_with_network_and_file_access_disabled(self):
        context = {"evidence_records": [snapshot()]}
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")), \
                patch("builtins.open", side_effect=AssertionError("file access forbidden")):
            self.success("calculator", {"expression": "2 + 2"})
            self.success("catalogue_search", {"query": "rice"})
            self.success("catalogue_compare", {"product_ids": ["SIM-RICE-1", "SIM-RICE-2"]})
            self.success("order_draft", self.draft_args())
            self.success("web_search", {"query": "alpha"}, context)

    def test_source_has_no_dynamic_code_execution(self):
        tree = ast.parse(Path(world.__file__).read_text())
        forbidden = {"eval", "exec", "compile", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, forbidden)


if __name__ == "__main__":
    unittest.main()
