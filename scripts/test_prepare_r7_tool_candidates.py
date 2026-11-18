"""Offline provenance, execution and candidate-only tests for R7 tool sources."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import prepare_r7_tool_candidates as candidates
from r7_encoding import validate_history
import r7_tool_world as world


def rows(files, name):
    return [json.loads(line) for line in files[name].splitlines()]


class ToolCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = candidates.build_package()
        cls.scenarios = rows(cls.files, "scenarios.jsonl")
        cls.registry = rows(cls.files, "source-registry.jsonl")
        cls.executions = rows(cls.files, "executions.jsonl")
        cls.records = rows(cls.files, "candidates/train.jsonl") + rows(cls.files, "candidates/dev.jsonl")
        cls.receipts = {record["id"]: record for record in cls.executions}

    def test_meaningful_coverage_and_honest_counts(self):
        report = json.loads(self.files["coverage.json"])
        required = {"calculator", "calculator_correction", "calculator_followup", "catalogue_search",
                    "catalogue_search_correction", "catalogue_compare", "search_then_compare",
                    "compare_then_total", "order_draft", "order_draft_correction", "draft_confirmation",
                    "invalid_args_recovery", "empty_web_search"}
        self.assertEqual(set(report["families"]), required)
        self.assertGreaterEqual(report["scenario_count"], 50)
        self.assertEqual(report["source_groups"], len(self.registry))
        self.assertEqual(report["scenario_count"], len(self.scenarios))
        self.assertEqual(report["candidate_views"], len(self.records))
        self.assertEqual(report["execution_receipts"], len(self.executions))
        self.assertEqual(report["native_call_occurrences_in_views"], len(self.executions) * len(candidates.FORMS))
        self.assertLess(report["source_groups"], report["scenario_count"])
        self.assertTrue(report["views_are_not_independent_sources"])
        self.assertFalse(report["family_source_counts_are_additive"])
        self.assertGreater(report["distinct_user_prompts"], report["candidate_views"])
        self.assertEqual(report["distinct_user_prompts"], len({m["content"] for r in self.records
                                                              for m in r["messages"] if m["role"] == "user"}))

    def test_every_record_remains_english_synthetic_pending(self):
        manifest = json.loads(self.files["manifest.json"])
        for record in [manifest, *self.registry, *self.scenarios, *self.records]:
            self.assertIs(record["training_ready"], False)
            self.assertIs(record["training_eligible"], False)
            self.assertEqual(record["admission_status"], "candidate_pending_twi_qualification")
            self.assertIs(record["production_eligible"], False)
            self.assertIs(record["human_reviewed"], False)
        for record in self.records:
            self.assertEqual(record["language"], "en")
            self.assertTrue(record["metadata"]["synthetic"])
            self.assertFalse(record["metadata"]["twi_targets_present"])
            self.assertFalse(record["metadata"]["prior_twi_targets_imported"])
            self.assertEqual(record["metadata"]["translation_status"], "pending_twi_translation_and_qualification")
        self.assertEqual(manifest["legacy_sources_used"], [])
        self.assertEqual(manifest["teacher_calls"], 0)
        self.assertEqual(manifest["network_requests"], 0)

    def test_byte_determinism(self):
        self.assertEqual(self.files, candidates.build_package())

    def test_groups_and_splits_precede_prompt_augmentation(self):
        single = candidates.build_package(forms=("direct",))
        for name in ["scenarios.jsonl", "source-registry.jsonl", "executions.jsonl"]:
            self.assertEqual(single[name], self.files[name])
        subset = rows(single, "candidates/train.jsonl") + rows(single, "candidates/dev.jsonl")
        self.assertEqual({r["source_group_id"]: r["split"] for r in subset},
                         {r["source_group_id"]: r["split"] for r in self.records})
        train = {r["source_group_id"] for r in self.records if r["split"] == "train"}
        dev = {r["source_group_id"] for r in self.records if r["split"] == "dev"}
        self.assertTrue(train and dev)
        self.assertFalse(train & dev)
        self.assertEqual(len(self.records), len(subset) * 3)

    def test_overlap_is_measured_not_claimed_absent(self):
        report = json.loads(self.files["coverage.json"])
        for role in ("user", "assistant"):
            texts = [{m["content"] for r in self.records if r["split"] == split for m in r["messages"]
                      if m["role"] == role and m["content"]} for split in ("train", "dev")]
            self.assertEqual(report["exact_overlap_diagnostics"][role + "_exact_texts_shared_across_splits"],
                             len(texts[0] & texts[1]))
        self.assertGreater(report["exact_overlap_diagnostics"]["assistant_exact_texts_shared_across_splits"], 0)

    def test_identity_stability_under_rewording_reordering_and_extension(self):
        specs = candidates.scenario_specs()
        original = candidates.register_scenarios(specs)
        reworded = copy.deepcopy(specs)
        for spec in reworded:
            for step in spec["steps"]:
                step["purpose"] = "different wording, same semantic inputs"
                if "instruction" in step:
                    step["instruction"] = "Reworded instruction"
                if "correction" in step:
                    step["correction"] = "Reworded correction"
        changed = candidates.register_scenarios(list(reversed(reworded)))
        identity = lambda values: {(r["id"], r["source_group_id"], r["split"]) for r in values}
        self.assertEqual(identity(original), identity(changed))
        extended = specs + [{"family": "calculator", "steps": [
            candidates._step("calculator", {"expression": "543 + 987"})]}]
        self.assertTrue(identity(original) < identity(candidates.register_scenarios(extended)))
        with self.assertRaisesRegex(ValueError, "Duplicate underlying scenario"):
            candidates.register_scenarios(specs + specs[:1])

    def test_common_initial_requests_share_source_group(self):
        by_anchor = {}
        for scenario in self.scenarios:
            key = world.canonical_json(scenario["group_anchor"])
            self.assertEqual(by_anchor.setdefault(key, scenario["source_group_id"]), scenario["source_group_id"])
            self.assertEqual(scenario["split"], candidates.split_group(scenario["source_group_id"]))
        rice = [s for s in self.scenarios if s["group_anchor"]["tool"] == "order_draft"
                and s["group_anchor"]["args"] == candidates._draft([("SIM-RICE-1", 2)])]
        self.assertGreater(len(rice), 3)
        self.assertEqual(len({s["source_group_id"] for s in rice}), 1)

    def test_all_execution_receipts_replay_exactly(self):
        for receipt in self.executions:
            execution = receipt["execution"]
            self.assertEqual(execution, world.execute(execution["call"]["name"],
                                                     receipt["arguments"], receipt["context"]))
            self.assertEqual(execution["result_sha256"], world.sha256_json(execution["result"]))

    def test_native_messages_preserve_actual_calls_and_responses(self):
        for record in self.records:
            self.assertTrue(validate_history(record))
            ids = []
            for binding in record["metadata"]["tool_bindings"]:
                receipt = self.receipts[binding["execution_id"]]
                self.assertEqual(binding["execution_record_sha256"], world.sha256_json(receipt))
                actual_call, actual_response = receipt["execution"]["messages"]
                call = record["messages"][binding["assistant_call_index"]]
                response = record["messages"][binding["tool_response_index"]]
                self.assertEqual(call["tool_calls"][0]["function"], actual_call["tool_calls"][0]["function"])
                self.assertEqual(response["content"], actual_response["content"])
                self.assertEqual(response["tool_call_id"], binding["native_call_id"])
                self.assertEqual(call["tool_calls"][0]["id"], binding["native_call_id"])
                self.assertEqual(binding["executor_call_id"], actual_call["tool_calls"][0]["id"])
                ids.append(binding["native_call_id"])
            self.assertEqual(len(ids), len(set(ids)))

    def test_english_answers_are_bound_to_executed_results(self):
        for record in self.records:
            for derivation in record["metadata"]["assistant_derivations"]:
                receipt = self.receipts[derivation["execution_id"]]
                text = record["messages"][derivation["message_index"]]["content"]
                self.assertEqual(text, candidates.assistant_response(receipt["execution"]))
                self.assertEqual(derivation["content_sha256"], candidates.digest_bytes(text.encode()))
        first = world.execute("calculator", {"expression": "7 * 8"})
        second = world.execute("calculator", {"expression": "7 * 9"})
        self.assertIn("56", candidates.assistant_response(first))
        self.assertIn("63", candidates.assistant_response(second))
        self.assertNotEqual(candidates.assistant_response(first), candidates.assistant_response(second))
        self.assertIn("exactly 57.5", candidates.assistant_response(world.execute("calculator", {"expression": "50+7.5"})))
        self.assertIn("approximation", candidates.assistant_response(world.execute("calculator", {"expression": "1/3"})))

    def test_registry_and_scenario_references_are_hash_bound(self):
        registry = {r["id"]: r for r in self.registry}
        scenarios = {r["id"]: r for r in self.scenarios}
        for record in self.records:
            ref = record["original_ref"]
            self.assertEqual(ref["registry_record_sha256"], world.sha256_json(registry[record["source_group_id"]]))
            self.assertEqual(ref["scenario_sha256"], world.sha256_json(scenarios[record["scenario_id"]]))
            self.assertEqual(record["source_id"], record["source_group_id"])
            self.assertEqual(record["split"], registry[record["source_group_id"]]["split"])
        for source in registry.values():
            for ref in source["scenario_refs"]:
                self.assertEqual(ref["sha256"], world.sha256_json(scenarios[ref["id"]]))

    def test_error_recovery_has_actual_error_then_actual_success(self):
        scenarios = [s for s in self.scenarios if s["family"] == "invalid_args_recovery"]
        self.assertGreaterEqual(len(scenarios), 8)
        for scenario in scenarios:
            steps = [r for r in self.executions if r["scenario_id"] == scenario["id"]]
            self.assertFalse(steps[0]["execution"]["result"]["ok"])
            self.assertEqual(steps[0]["execution"]["result"]["error"]["code"], scenario["steps"][0]["expect_error"])
            self.assertTrue(steps[1]["execution"]["result"]["ok"])
            self.assertIn("rejected", candidates.assistant_response(steps[0]["execution"]))

    def test_followups_resolve_actual_prior_values(self):
        step = candidates._step("calculator", mode="followup", derive={"kind": "previous_calculation",
                                "operator": "/", "operand": "4"})
        args, context = candidates.resolve_call(step, world.execute("calculator", {"expression": "100 + 20"}))
        self.assertEqual(args, {"expression": "(120/1) / 4"})
        self.assertEqual(context, {})
        self.assertEqual(world.execute("calculator", args)["result"]["data"]["numerator"], "30")
        with self.assertRaisesRegex(ValueError, "successful preceding result"):
            candidates.resolve_call(step, world.execute("calculator", {"expression": "1/0"}))
        for scenario in self.scenarios:
            receipts = [r for r in self.executions if r["scenario_id"] == scenario["id"]]
            for index, item in enumerate(scenario["steps"]):
                if "derive" in item:
                    self.assertEqual(receipts[index]["derived_from_execution_id"], receipts[index - 1]["id"])

    def test_order_corrections_preserve_unmentioned_fields(self):
        for scenario in self.scenarios:
            if scenario["family"] != "order_draft_correction":
                continue
            before, after = [r["execution"]["result"]["data"]["draft"] for r in self.executions
                             if r["scenario_id"] == scenario["id"]]
            change = scenario["steps"][1]["derive"]["change"]
            if change["kind"] in {"location", "fulfillment"}:
                self.assertEqual(before["items"], after["items"])
            else:
                self.assertEqual(before["fulfillment"], after["fulfillment"])
                self.assertEqual(before["location"], after["location"])
            if change["kind"] == "quantity":
                self.assertEqual(after["items"][0]["quantity"], 3)
                self.assertEqual(before["items"][0]["product_id"], after["items"][0]["product_id"])
            self.assertEqual(after["total_minor"], sum(line["subtotal_minor"] for line in after["items"]))

    def test_confirmation_requires_explicit_user_turn_and_exact_draft(self):
        matching = [r for r in self.records if r["metadata"]["family"] == "draft_confirmation"]
        self.assertEqual(len(matching), 3)
        for record in matching:
            first, second = [self.receipts[b["execution_id"]] for b in record["metadata"]["tool_bindings"]]
            self.assertEqual(second["context"]["confirmation"]["draft_sha256"],
                             first["execution"]["result"]["data"]["draft_sha256"])
            self.assertTrue(second["explicit_confirmation"])
            user_index = record["metadata"]["tool_bindings"][1]["assistant_call_index"] - 1
            self.assertIn("I explicitly confirm", record["messages"][user_index]["content"])
            self.assertNotEqual(record["metadata"]["tool_bindings"][0]["native_call_id"],
                                record["metadata"]["tool_bindings"][1]["native_call_id"])
        for receipt in self.executions:
            result = receipt["execution"]["result"]
            if receipt["execution"]["call"]["name"] == "order_draft" and result["ok"]:
                self.assertFalse(result["data"]["purchases_enabled"])
                self.assertEqual(result["data"]["side_effects"], [])
                self.assertEqual(result["data"]["confirmed"], receipt["explicit_confirmation"])

    def test_web_and_empty_catalogue_never_invent_results(self):
        web = [r for r in self.executions if r["execution"]["call"]["name"] == "web_search"]
        self.assertEqual(len(web), 3)
        for receipt in web:
            self.assertEqual(receipt["context"], {})
            self.assertEqual(receipt["execution"]["result"]["data"]["results"], [])
            self.assertIn("No pinned web evidence", candidates.assistant_response(receipt["execution"]))
        empty = [r for r in self.executions if r["execution"]["call"]["name"] == "catalogue_search"
                 and r["execution"]["result"]["ok"] and not r["execution"]["result"]["data"]["products"]]
        self.assertTrue(empty)
        self.assertTrue(all("No simulated catalogue entries" in candidates.assistant_response(r["execution"])
                            for r in empty))

    def test_package_artifact_hashes_and_candidate_only_seal(self):
        manifest = json.loads(self.files["manifest.json"])
        self.assertEqual(set(self.files), set(manifest["artifacts"]) | {"manifest.json", "PREPARED.json"})
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(candidates.digest_bytes(self.files[name]), expected)
        seal = json.loads(self.files["PREPARED.json"])
        self.assertEqual(seal["manifest_sha256"], candidates.digest_bytes(self.files["manifest.json"]))
        self.assertIs(seal["training_ready"], False)
        self.assertTrue(seal["candidate_only"])

    def test_fresh_directory_and_full_replay_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "candidates"
            report = candidates.prepare(out)
            self.assertTrue(report["verified"])
            self.assertFalse(report["training_ready"])
            self.assertEqual(report, candidates.verify_package(out))
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                candidates.prepare(out)
            (out / "extra.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unregistered"):
                candidates.verify_package(out)

    def test_resealing_fabricated_answer_does_not_pass_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "candidates"
            candidates.prepare(out)
            path = out / "candidates/train.jsonl"
            changed = [json.loads(line) for line in path.read_text().splitlines()]
            changed[0]["messages"][-1]["content"] = "A fabricated final answer."
            path.write_bytes(candidates._rows_bytes(changed))
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["artifacts"]["candidates/train.jsonl"] = candidates.digest_bytes(path.read_bytes())
            (out / "manifest.json").write_bytes(candidates._json_bytes(manifest))
            (out / "PREPARED.json").write_bytes(candidates._json_bytes({
                "manifest_sha256": candidates.digest_bytes((out / "manifest.json").read_bytes()),
                "candidate_only": True, "training_ready": False}))
            with self.assertRaisesRegex(ValueError, "differs from executed source"):
                candidates.verify_package(out)

    def test_symlinks_and_unsupported_forms_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "linked"
            out.symlink_to(Path(folder), target_is_directory=True)
            with self.assertRaises(ValueError):
                candidates.prepare(out)
            with self.assertRaises(ValueError):
                candidates.verify_package(out)
        for forms in [(), ("direct", "direct"), ("invented",)]:
            with self.assertRaisesRegex(ValueError, "Invalid prompt forms"):
                candidates.build_package(forms=forms)

    def test_no_network_needed(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.files, candidates.build_package())


if __name__ == "__main__":
    unittest.main()
