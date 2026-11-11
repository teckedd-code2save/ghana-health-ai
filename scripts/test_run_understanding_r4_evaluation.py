"""Offline synthetic generations and receipts. No API, model load or GPU calls."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import run_understanding_r4_evaluation as runner
import r2_inference as r2
import r4_inference as r4
import prepare_understanding_r4_inference as builder
from test_prepare_understanding_r4_inference import fixture as training_fixture, REFERENCE, HELPERS

ORIGINAL = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai")
PARENT = ORIGINAL / "tmp/understanding-r2-20261105/eval-bundle-v2"
EVALUATION = ORIGINAL / "tmp/understanding-r3-20261106/evaluation-v2"


def frozen_helpers():
    # Import the same hash-pinned CPU Journal/completion implementation used remotely.
    saved = sys.modules.get("check_afrique_training_contract")
    sys.path.insert(0, str(PARENT / "code"))
    dependency = runner.module_at(PARENT / "code/check_afrique_training_contract.py", "check_afrique_training_contract")
    sys.modules["check_afrique_training_contract"] = dependency
    try:
        core = runner.module_at(PARENT / "code/afrique_instruction_baseline_core.py", "_test_frozen_core")
    finally:
        sys.path.pop(0)
        if saved is None:
            del sys.modules["check_afrique_training_contract"]
        else:
            sys.modules["check_afrique_training_contract"] = saved
    return core


CORE = frozen_helpers()
GUARD = runner.module_at(HELPERS / "scripts/response_loop_guard.py", "_r4_test_exact_guard")


class Tokenizer:
    contract_special_ids = {CORE.EOS}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return "".join("<eos>" if i == CORE.EOS else chr(i) for i in ids)

    def __len__(self):
        return CORE.EOS + 1


def prompt_ids(tokenizer, messages, settings):
    assert settings == {"context_tokens": 4096, "max_new_tokens": 256}
    return list(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())


class Runtime:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.tokenizer = Tokenizer()
        self.manifest = {"decoding": r2.DECODING}
        self.loaded = set()
        self.calls = []
        self.loss_calls = []

    def identity(self, variant):
        return {"variant": variant, "manifest_sha256": "f" * 64, "base_weights_sha256": r2.BASE_SHA,
                "adapter_weights_sha256": None if variant == "base" else runner.digest(variant)}

    def reply(self, messages, variant):
        self.calls.append((copy.deepcopy(messages), variant))
        self.loaded.add(variant)
        text, stop = variant + ":reply", "eos"
        if self.behavior:
            text, stop = self.behavior(messages, variant)
        ids = list(text.encode()) + ([CORE.EOS] if stop == "eos" else [])
        diagnostics = CORE.completion(self.tokenizer, ids, 256, stop == "time_limit", .01)
        return {"identity": self.identity(variant), "messages_sha256": runner.digest(messages),
                "prompt_ids_sha256": runner.digest(prompt_ids(self.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})),
                "decoding": r2.DECODING, "output_ids": ids, "output": diagnostics["prediction"],
                "seconds": .01, "diagnostics": diagnostics, "finish_reason": diagnostics["finish_reason"],
                "complete": diagnostics["complete"]}

    def language_loss(self, text, variant):
        self.loss_calls.append((text, variant))
        self.loaded.add(variant)
        return {"predicted_tokens": len(text.encode()), "summed_nll": len(text.encode()) * 2.}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commits = []
        self.journal = CORE.Journal(self.root / "journal", {"execution_sha256": "test"}, lambda: self.commits.append(True))
        self.core = types.SimpleNamespace(Journal=CORE.Journal, EOS=CORE.EOS, completion=CORE.completion, prompt_ids=prompt_ids)
        self.scoring = types.SimpleNamespace(objective_score=lambda case, raw: None)
        self.cases = [{"id": "fixed", "messages": [{"role": "user", "content": "First"},
                        {"role": "assistant", "content": "Original fixed reply"}, {"role": "user", "content": "Second"}]}]
        self.rollouts = [{"id": "roll", "user_turns": ["First", "Second", "Third"], "rubric": ["Human meaning check"]}]
        self.plan = list(runner.slots(self.cases, self.rollouts, []))

    def run_engine(self, runtime, attempt=1, deadline=100, plan=None, clock=lambda: 0):
        return runner.evaluate(runtime, self.core, self.scoring, plan or self.plan, self.journal, attempt,
                               "proof", deadline, clock=clock)

    def test_fixed_native_prompts_and_independent_rollouts(self):
        runtime = Runtime()
        self.assertEqual(self.run_engine(runtime)["state"], "completed")
        self.assertEqual(len(runtime.calls), 12)
        fixed = [self.journal.get(s["key"]) for s in self.plan[:3]]
        self.assertEqual(fixed[0]["prompt_ids"], fixed[1]["prompt_ids"])
        self.assertEqual(fixed[1]["prompt_ids"], fixed[2]["prompt_ids"])
        for messages, variant in runtime.calls[3:]:
            self.assertTrue(all(m["content"] == variant + ":reply" for m in messages if m["role"] == "assistant"))
        self.assertEqual(len(self.commits), 1 + 2 * 12)

    def test_raw_four_variants_keep_pass1_history_independent(self):
        runtime = Runtime()
        plan = list(runner.slots(self.cases, self.rollouts, [], runner.ALL_VARIANTS))
        self.run_engine(runtime, plan=plan)
        self.assertEqual(len(runtime.calls), 16)
        for messages, variant in runtime.calls[4:]:
            self.assertTrue(all(m["content"] == variant + ":reply" for m in messages if m["role"] == "assistant"))
        self.run_engine(runtime, plan=plan, attempt=2)
        self.assertEqual(len(runtime.calls), 16)

    def test_guarded_pause_is_not_eos_and_only_continues_own_history(self):
        variants = ("base", "r3_response", "r4_pass1")
        plan = list(runner.slots(self.cases, self.rollouts, [], variants, "guarded"))
        runtime = Runtime(lambda messages, variant: ((variant + " one two three four five six seven ") * 4, "unexpected_stop"))

        def guarded(runtime, messages, variant):
            result = runtime.reply(messages, variant)
            result.update(finish_reason="cancelled", complete=False,
                          loop_guard={"version": GUARD.VERSION, "trigger": GUARD.repeated_passage(result["output"])})
            return result

        with patch.dict(sys.modules, {"response_loop_guard": GUARD}), patch.object(runner, "guarded_reply", side_effect=guarded):
            self.run_engine(runtime, plan=plan)
            for slot in plan:
                record = self.journal.get(slot["key"])
                self.assertEqual(record["state"], "paused")
                self.assertFalse(record["result"]["complete"])
                self.assertEqual(record["result"]["diagnostics"]["finish_reason"], "unexpected_stop")
            for messages, variant in runtime.calls[3:]:
                self.assertTrue(all(m["content"].startswith(variant) for m in messages if m["role"] == "assistant"))
            self.run_engine(runtime, plan=plan, attempt=2)
            self.assertEqual(len(runtime.calls), 12)
            summary = runner.report(self.root, plan, self.journal, {}, {v: runtime.identity(v) for v in variants})
            self.assertEqual(summary["generation_policy"], "guarded")
            self.assertEqual(summary["mechanical_metrics"]["r4_pass1"]["guarded_pauses"], 4)
            raw_plan = list(runner.slots(self.cases, self.rollouts, [], variants))
            with self.assertRaisesRegex(ValueError, "history differs"):
                self.run_engine(runtime, plan=raw_plan, attempt=3)

    def test_guarded_wrapper_uses_pinned_stream_trigger_without_rewriting_text(self):
        runtime = Runtime(lambda messages, variant: (("one two three four five six seven eight " * 4), "unexpected_stop"))
        messages = [{"role": "user", "content": "test"}]

        def reply(runtime, messages, variant, on_text, cancelled):
            result = runtime.reply(messages, variant)
            on_text(result["output"])
            self.assertTrue(cancelled())
            result.update(finish_reason="cancelled", complete=False)
            return result

        with patch.dict(sys.modules, {"response_loop_guard": GUARD,
                                      "guarded_r3_runtime": types.SimpleNamespace(reply=reply)}):
            result = runner.guarded_reply(runtime, messages, "r4_response")
            self.assertEqual(result["loop_guard"]["trigger"], GUARD.repeated_passage(result["output"]))
            ids = prompt_ids(runtime.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": 256})
            runner.verify_reply(runtime, self.core, messages, "r4_response", ids, result, policy="guarded")
            with self.assertRaisesRegex(ValueError, "Raw result"):
                runner.verify_reply(runtime, self.core, messages, "r4_response", ids, result)
            result["loop_guard"]["trigger"] = None
            with self.assertRaisesRegex(ValueError, "completion proof"):
                runner.verify_reply(runtime, self.core, messages, "r4_response", ids, result, policy="guarded")

    def test_token_and_time_limit_continue_exact_partial_without_status_rewrite(self):
        for stop, output in (("max_new_tokens", "x" * 256), ("time_limit", "exact partial")):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as tmp:
                self.journal = CORE.Journal(Path(tmp), {"test": stop})
                runtime = Runtime(lambda messages, variant: (output, stop))
                self.run_engine(runtime)
                row = self.journal.get("rollout:roll:base:0")
                self.assertEqual(row["state"], "incomplete")
                self.assertFalse(row["result"]["complete"])
                self.assertEqual(row["result"]["finish_reason"], stop)
                second = self.journal.get("rollout:roll:base:1")
                self.assertEqual(second["request"]["messages"][1], {"role": "assistant", "content": output})
                self.assertEqual(len(runtime.calls), 12)

    def test_unsupported_or_empty_stops_block_only_own_descendants(self):
        for text, stop in (("partial", "unexpected_stop"), ("", "eos")):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as tmp:
                self.journal = CORE.Journal(Path(tmp), {"test": stop})
                runtime = Runtime(lambda messages, variant: (text, stop) if variant == "base" else ("ok", "eos"))
                self.run_engine(runtime)
                self.assertEqual(self.journal.get("rollout:roll:base:1")["state"], "blocked")
                self.assertEqual(self.journal.get("rollout:roll:base:2")["state"], "blocked")
                self.assertEqual(self.journal.get("rollout:roll:r4_response:2")["state"], "complete")
                self.assertEqual(len(runtime.calls), 10)

    def test_native_output_mismatch_keeps_raw_error_and_never_retries(self):
        runtime = Runtime()
        original = runtime.reply
        def wrong(messages, variant):
            result = original(messages, variant)
            result["prompt_ids_sha256"] = "wrong"
            return result
        runtime.reply = wrong
        self.run_engine(runtime)
        first = self.journal.get(self.plan[0]["key"])
        self.assertFalse(first["verified"])
        self.assertEqual(first["state"], "error")
        self.assertEqual(first["result"]["prompt_ids_sha256"], "wrong")
        count = len(runtime.calls)
        self.run_engine(runtime, attempt=2)
        self.assertEqual(len(runtime.calls), count)

    def test_result_text_cannot_diverge_from_native_token_decoding(self):
        runtime = Runtime()
        original = runtime.reply
        def wrong(messages, variant):
            result = original(messages, variant)
            result["output"] = "rewritten answer"
            return result
        runtime.reply = wrong
        self.run_engine(runtime)
        record = self.journal.get(self.plan[0]["key"])
        self.assertEqual(record["state"], "error")
        self.assertFalse(record["verified"])
        self.assertEqual(record["result"]["output"], "rewritten answer")
        self.assertNotEqual(record["result"]["output"], record["result"]["diagnostics"]["prediction"])

    def test_request_marker_is_committed_before_each_model_call(self):
        runtime = Runtime()
        original = runtime.reply
        def checking(messages, variant):
            slot = self.plan[len(runtime.calls)]
            marker = self.journal.folder / "attempts/1" / (runner.digest(slot["key"]) + ".json")
            self.assertTrue(marker.is_file())
            self.assertEqual(len(self.commits), 2 + len(runtime.calls) * 2)
            self.assertIsNone(self.journal.get(slot["key"]))
            return original(messages, variant)
        runtime.reply = checking
        self.run_engine(runtime)
        self.assertTrue(all(self.journal.get(s["key"])["verified"] for s in self.plan))

    def test_paused_resume_reuses_all_completed_slots(self):
        runtime = Runtime()
        outcome = self.run_engine(runtime, deadline=2, clock=lambda: len(runtime.calls))
        self.assertEqual(outcome["state"], "paused_budget")
        self.assertEqual(len(runtime.calls), 2)
        before = self.journal.path(self.plan[0]["key"]).read_bytes()
        self.run_engine(runtime, attempt=2)
        self.assertEqual(len(runtime.calls), 12)
        self.assertEqual(before, self.journal.path(self.plan[0]["key"]).read_bytes())
        self.run_engine(runtime, attempt=3)
        self.assertEqual(len(runtime.calls), 12)

    def test_interrupted_request_indeterminate_and_descendants_blocked(self):
        key = "rollout:roll:base:0"
        request = {"messages": [{"role": "user", "content": "First"}]}
        self.journal.begin(key, 1, runner.digest(request))
        runtime = Runtime()
        self.run_engine(runtime, attempt=2)
        self.assertEqual(self.journal.get(key)["state"], "indeterminate")
        self.assertEqual(self.journal.get("rollout:roll:base:1")["state"], "blocked")
        self.assertEqual(len(runtime.calls), 9)

    def test_changed_binding_prompt_or_record_rejected(self):
        runtime = Runtime()
        self.run_engine(runtime)
        with self.assertRaisesRegex(ValueError, "Resume identity"):
            CORE.Journal(self.journal.folder, {"execution_sha256": "different"})
        changed = copy.deepcopy(self.plan)
        changed[0]["case"]["messages"][0]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "history"):
            self.run_engine(runtime, attempt=2, plan=changed)
        path = self.journal.path(self.plan[0]["key"])
        record = runner.read(path)
        record["record"]["state"] = "edited"
        runner.save(path, record)
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.run_engine(runtime, attempt=2)

    def test_loss_is_last_and_protected_text_never_exported(self):
        protected = "PROTECTED-HELD-TEXT"
        plan = list(runner.slots(self.cases, self.rollouts, [{"id": "held", "text": protected}]))
        runtime = Runtime()
        original = runtime.language_loss
        def loss(text, variant):
            self.assertEqual(len(runtime.calls), 12)
            return original(text, variant)
        runtime.language_loss = loss
        self.run_engine(runtime, plan=plan)
        summary = runner.report(self.root, plan, self.journal, {"datasets": [], "consumed": {}},
                                {v: runtime.identity(v) for v in runner.VARIANTS})
        self.assertEqual(summary["mechanical_metrics"]["base"]["language_loss"]["token_weighted_mean_nll"], 2.)
        self.assertFalse(summary["human_quality"]["semantic_quality_established"])
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(protected, path.read_text())
        self.assertEqual(len(runner.read(self.root / "review.json")["rows"]), 12)
        self.run_engine(runtime, plan=plan, attempt=2)
        self.assertEqual(len(runtime.loss_calls), 3)

    def test_loss_failure_is_recorded_without_leaking_protected_text(self):
        protected = "PRIVATE-HOLDOUT-CONTENT"
        runtime = Runtime()
        def failure(text, variant):
            raise RuntimeError(text)
        runtime.language_loss = failure
        plan = list(runner.slots([], [], [{"id": "held", "text": protected}]))
        self.run_engine(runtime, plan=plan)
        for slot in plan:
            record = self.journal.get(slot["key"])
            self.assertEqual(record["state"], "error")
            self.assertNotIn(protected, json.dumps(record))

    def test_reports_include_pending_slots_and_escape_output(self):
        runtime = Runtime(lambda messages, variant: ("<script>raw</script>", "eos"))
        self.run_engine(runtime, deadline=1, clock=lambda: len(runtime.calls))
        result = runner.report(self.root, self.plan, self.journal, {"consumed": {"row_exposures": 8}},
                               {v: runtime.identity(v) for v in runner.VARIANTS})
        self.assertEqual(result["finalized_slots"], 1)
        self.assertEqual(result["kind"], runner.KIND)
        rows = runner.read(self.root / "review.json")["rows"]
        self.assertEqual(sum(r["state"] == "pending" for r in rows), 11)
        markup = (self.root / "review.html").read_text()
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;raw", markup)
        self.assertTrue(all(r["human_review"]["status"] == "pending" for r in rows))

    def test_attempt_claim_requires_terminal_previous_and_no_duplicates(self):
        class Store(dict):
            def put(self, key, value, skip_if_exists=False):
                if key in self and skip_if_exists:
                    return False
                self[key] = value
                return True
        store = Store()
        CORE.claim_attempt(store, "run", 1, "call-1", lambda _: False)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            CORE.claim_attempt(store, "run", 2, "call-2", lambda _: False)
        CORE.claim_attempt(store, "run", 2, "call-2", lambda call: call == "call-1")
        with self.assertRaisesRegex(ValueError, "claimed"):
            CORE.claim_attempt(store, "run", 2, "call-3", lambda _: True)

    def test_context_failure_is_recorded_without_call_or_truncation(self):
        runtime = Runtime()
        def overlength(*args):
            raise ValueError("Prompt plus complete output budget exceeds context; no truncation")
        self.core.prompt_ids = overlength
        self.run_engine(runtime)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(self.journal.get("fixed:fixed:base")["state"], "error")
        self.assertEqual(self.journal.get("rollout:roll:base:1")["state"], "blocked")


def prepared_fixture(root):
    _, _, readback, run = training_fixture(root)
    inputs = run / "inputs"
    dataset = {"dataset_id": "gha-responses", "dataset_version": "synthetic-test-only",
               "catalog_dataset_id": "responses", "purpose": "Synthetic offline test fixture",
               "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
               "record_count": 1, "records_path": "datasets/responses/records.jsonl", "records_sha256": "a" * 64,
               "origin_counts": {"synthetic": 1}}
    r2.save(inputs / "datasets/responses/manifest.json", dataset)
    ref = {"dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"],
           "manifest_path": "datasets/responses/manifest.json", "manifest_sha256": r2.sha(inputs / "datasets/responses/manifest.json")}
    r2.save(inputs / "evaluation-manifest.json", {"sha256": runner.EVALUATION_SHA, "manifest": runner.read(EVALUATION / "manifest.json")})
    data = r2.read(inputs / "manifest.json")
    data.update(dataset_snapshots=[ref], artifacts={ref["manifest_path"]: ref["manifest_sha256"],
        "evaluation-manifest.json": r2.sha(inputs / "evaluation-manifest.json"), dataset["records_path"]: dataset["records_sha256"]})
    r2.save(inputs / "manifest.json", data)
    data_sha = r2.sha(inputs / "manifest.json")
    execution = r2.read(run / "execution.json")
    execution["data_sha256"] = data_sha
    r2.save(run / "execution.json", execution)
    execution_sha = r2.sha(run / "execution.json")
    contract = {"run_sha256": execution_sha, "data_sha256": data_sha}
    r2.save(run / "responses/contract.json", contract)
    status = r2.read(run / "status.json")
    status.update(data_sha256=data_sha, execution_sha256=execution_sha)
    for export in status["adapters"]:
        cp = run / "responses/attempt-fixture" / ("update-%08d" % export["completed_updates"])
        cp_value = r2.read(cp / "manifest.json")
        cp_value["identity"] = {"run_sha256": r2.digest(contract), "data_sha256": data_sha}
        r2.save(cp / "manifest.json", cp_value)
        r2.save(cp / "COMMITTED.json", {"manifest_sha256": r2.sha(cp / "manifest.json")})
        export["checkpoint"].update(manifest_sha256=r2.sha(cp / "manifest.json"),
                                    contract_sha256=r2.sha(run / "responses/contract.json"))
    checkpoint = status["adapters"][0]["checkpoint"]
    r2.save(run / "adapters.json", status["adapters"])
    counts = {"unique_rows": 1, "row_exposures": 8, "input_tokens": 24, "effective_target_tokens": 16, "unique_dataset_records": 1}
    row = {"row_index": 0, "id": "fixture-row", "class": "responses", "dataset_ref": {
        "dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"],
        "record_id": "fixture", "source_id": "fixture", "record_sha256": "b" * 64},
        **{k: v for k, v in counts.items() if k != "unique_dataset_records"}}
    consumption = {"version": "r4-consumption-v1", "mode": "train", "scope": "cumulative_committed_schedule_prefix",
        "training_complete": True, "schedule_complete": True, "run_id": status["run_id"], "execution_sha256": execution_sha,
        "data_sha256": data_sha, "checkpoint": checkpoint, "completed_updates": 4,
        "inventory": {"admitted_snapshots": [{**ref, "admitted_record_count": 1}], "archive_counts_included": False},
        "consumed": {**counts, "by_class": {"responses": counts},
                     "by_dataset": [{"dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"], **counts}]},
        "rows": [row]}
    r2.save(run / "consumption.json", consumption)
    status["consumption_sha256"] = r2.sha(run / "consumption.json")
    r2.save(run / "status.json", status)
    r2.save(root / "terminal.json", status)
    receipt = r2.read(root / "receipt.json")
    receipt.update(data_sha256=data_sha, execution_sha256=execution_sha)
    r2.save(root / "receipt.json", receipt)
    bundle = root / "inference"
    result = builder.prepare(REFERENCE, r4.R3_BUNDLE_SHA, root / "receipt.json", root / "terminal.json", readback, bundle,
                             helpers_root=HELPERS, include_pass1=True)
    return bundle, result["manifest_sha256"], inputs, run / "consumption.json"


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle, self.bundle_sha, self.inputs, self.consumption = prepared_fixture(self.root)

    def build(self, loss=False, **options):
        context = {"ready": True, "evaluation_sha256": runner.EVALUATION_SHA, "context_tokens": 4096,
                   "reserved_generation_tokens": 256, "truncate": False, "synthetic_fixture": True}
        with patch.object(runner, "context_preflight", return_value=context):
            return runner.prepare(self.bundle, self.bundle_sha, EVALUATION, self.inputs, self.consumption,
                                  self.root / "execution", loss=loss, **options)

    def test_context_preflight_failure_cannot_seal_execution(self):
        with patch.object(runner, "context_preflight", return_value={"ready": False}):
            with self.assertRaisesRegex(ValueError, "context preflight failed"):
                runner.prepare(self.bundle, self.bundle_sha, EVALUATION, self.inputs, self.consumption, self.root / "execution")
        self.assertFalse((self.root / "execution").exists())

    def test_production_freeze_and_minimal_worker_package(self):
        result = self.build()
        package = self.root / "execution"
        actual = runner.verify(package, result["execution_sha256"])
        self.assertEqual(len(actual[3]), 46)
        self.assertEqual(len(actual[4]), 10)
        self.assertEqual(actual[5], [])
        self.assertEqual(actual[0]["expected_slots"], 225)
        self.assertFalse(result["gpu_launched"])
        self.assertEqual(runner.files(package / "inference"), runner.files(self.bundle))
        self.assertEqual(set(runner.files(package / "evaluation")), set(runner.EVAL_FILES))
        self.assertFalse(any(p.suffix in {".sqlite", ".sqlite3", ".db"} for p in package.rglob("*")))
        evidence = runner.read(package / "provenance/summary.json")
        self.assertEqual(evidence["consumed"]["row_exposures"], 8)
        self.assertEqual(evidence["datasets"][0]["dataset_version"], "synthetic-test-only")
        self.assertEqual(evidence["datasets"][0]["records_sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_optional_loss_bound_and_held_proofs_not_copied(self):
        result = self.build(loss=True)
        package = self.root / "execution"
        self.assertEqual(result["expected_slots"], 1254)
        self.assertEqual(len(runner.verify(package, result["execution_sha256"])[5]), 343)
        self.assertFalse((package / "evaluation/observed-feedback-proofs.json").exists())
        self.assertFalse((package / "evaluation/held-content.jsonl").exists())

    def test_four_variant_raw_loss_has_exact_frozen_slot_counts(self):
        result = self.build(loss=True, variants=runner.ALL_VARIANTS)
        contract, _, _, cases, rollouts, held = runner.verify(self.root / "execution", result["execution_sha256"])
        self.assertEqual(contract["expected_slots"], 1672)
        self.assertEqual(contract["generation_policy"], "raw")
        self.assertIsNone(contract["guard_policy"])
        plan = list(runner.slots(cases, rollouts, held, contract["variants"], "raw"))
        self.assertEqual(len(plan), 1672)
        self.assertEqual(len({s["key"] for s in plan}), 1672)
        for variant in runner.ALL_VARIANTS:
            self.assertEqual(sum(s["kind"] == "fixed" and s["variant"] == variant for s in plan), 46)
            self.assertEqual(sum(s["kind"] == "rollout" and s["variant"] == variant for s in plan), 29)
            self.assertEqual(sum(s["kind"] == "loss" and s["variant"] == variant for s in plan), 343)

    def test_guarded_chosen_pass_has_distinct_protocol_and_no_loss_duplication(self):
        result = self.build(policy="guarded", variants=("base", "r3_response", "r4_pass1"))
        contract = runner.verify(self.root / "execution", result["execution_sha256"])[0]
        self.assertEqual(contract["generation_policy"], "guarded")
        self.assertEqual(contract["guard_policy"], "repeat-block-v1")
        self.assertEqual(contract["expected_slots"], 225)
        with self.assertRaisesRegex(ValueError, "separate guarded"):
            runner.selection(runner.VARIANTS, "guarded", True, runner.ALL_VARIANTS)
        for selected in (("base", "r4_response"), ("base", "r3_response", "r3_response"),
                         ("base", "r3_response", "r2_meaning")):
            with self.assertRaises(ValueError):
                runner.selection(selected, "raw", False, runner.ALL_VARIANTS)

    def test_frozen_evaluation_and_previous_runner_configuration_remain_unchanged(self):
        original = runner.files(EVALUATION)
        import run_understanding_r3_evaluation as previous
        before = copy.deepcopy((previous.VERSION, previous.VARIANTS, previous.LIMITS, previous.PREFIX))
        self.build()
        self.assertEqual(original, runner.files(EVALUATION))
        self.assertEqual(before, (previous.VERSION, previous.VARIANTS, previous.LIMITS, previous.PREFIX))
        self.assertEqual(runner.EVALUATION_SHA, "413c6e3b7e5737ffa3360219202bfb4e8e7a86437eb0d84dc9e2c72385612a06")
        self.assertEqual(runner.read(EVALUATION / "manifest.json")["variants"], list(runner.FROZEN_VARIANTS))

    def test_missing_consumption_or_changed_checkpoint_rejected(self):
        receipt = runner.read(self.consumption)
        receipt["consumed"]["row_exposures"] = 9000
        runner.save(self.consumption, receipt)
        with self.assertRaisesRegex(ValueError, "consumption"):
            self.build()
        self.assertFalse((self.root / "execution").exists())

    def test_receipt_counts_and_checkpoint_are_checked_beyond_file_hash(self):
        original = runner.read(self.consumption)
        manifest = runner.read(self.bundle / "manifest.json")
        for field in ("checkpoint", "counts", "dataset"):
            with self.subTest(field=field):
                receipt = copy.deepcopy(original)
                if field == "checkpoint":
                    receipt["checkpoint"]["state_sha256"] = "0" * 64
                elif field == "counts":
                    receipt["rows"][0]["row_exposures"] += 1
                else:
                    receipt["inventory"]["admitted_snapshots"][0]["dataset_version"] = "wrong"
                runner.save(self.consumption, receipt)
                terminal_path = self.bundle / "evidence/terminal.json"
                terminal = runner.read(terminal_path)
                terminal["consumption_sha256"] = runner.sha(self.consumption)
                runner.save(terminal_path, terminal)
                with self.assertRaises(ValueError):
                    runner.provenance(self.bundle, manifest, self.inputs, self.consumption,
                                      runner.read(EVALUATION / "manifest.json"))

    def test_training_evaluation_binding_required(self):
        wrapper = self.inputs / "evaluation-manifest.json"
        value = runner.read(wrapper)
        value["sha256"] = "0" * 64
        runner.save(wrapper, value)
        with self.assertRaisesRegex(ValueError, "bound to this frozen"):
            self.build()

    def test_any_package_or_checkpoint_tampering_fails(self):
        result = self.build()
        package = self.root / "execution"
        (package / "code/evaluate_understanding_r4.py").write_text("raise Exception('must never execute')")
        with self.assertRaisesRegex(ValueError, "files changed"):
            runner.verify(package, result["execution_sha256"])
        with self.assertRaisesRegex(ValueError, "bundle hash"):
            runner.inference(self.bundle, "0" * 64)

    def test_extra_feedback_file_is_rejected_even_if_runtime_pack_resealed(self):
        self.build()
        package = self.root / "execution"
        runner.save(package / "evaluation/observed-feedback-proofs.json", {"private": "test"})
        contract = runner.read(package / "execution.json")
        contract["files"]["evaluation/observed-feedback-proofs.json"] = runner.sha(package / "evaluation/observed-feedback-proofs.json")
        runner.save(package / "execution.json", contract)
        checksum = runner.sha(package / "execution.json")
        runner.save(package / "SEALED.json", {"execution_sha256": checksum})
        with self.assertRaisesRegex(ValueError, "Unexpected runtime evaluation data"):
            runner.verify(package, checksum)

    def test_remote_terminal_probe_fails_closed_on_pending_or_expired(self):
        import modal
        call = unittest.mock.Mock()
        with patch.object(modal.FunctionCall, "from_id", return_value=call):
            for error in (TimeoutError(), modal.exception.TimeoutError(), modal.exception.OutputExpiredError(),
                          modal.exception.NotFoundError("missing")):
                call.get.side_effect = error
                self.assertFalse(runner.terminal("fc-fixture"))
            for error in (modal.exception.FunctionTimeoutError(), modal.exception.RemoteError("failed")):
                call.get.side_effect = error
                self.assertTrue(runner.terminal("fc-fixture"))
            call.get.side_effect = None
            call.get.return_value = {"state": "paused_budget"}
            self.assertTrue(runner.terminal("fc-fixture"))

    def test_no_gpu_without_explicit_gate(self):
        with patch.object(runner, "verify", side_effect=AssertionError("Should fail before imports")):
            with self.assertRaisesRegex(ValueError, "authorization"):
                runner.launch(self.root, "sha", self.root / "receipt-new.json", False, "profile")

    def test_launcher_registers_importable_frozen_worker_and_never_retries(self):
        # Entire remote surface is replaced; this exercises receipt/dispatch wiring only.
        call = types.SimpleNamespace(object_id="fc-synthetic-dispatch")
        worker = types.SimpleNamespace(app=types.SimpleNamespace(run=lambda **kwargs: nullcontext()),
            evaluate=types.SimpleNamespace(spawn=unittest.mock.Mock(return_value=call)))
        fake = types.SimpleNamespace(enable_output=lambda: nullcontext())
        receipt = self.root / "evaluation-receipt.json"
        with patch.dict(sys.modules, {"modal": fake}), patch.dict("os.environ", {}), \
             patch.object(runner, "verify"), patch.object(runner, "module_at", return_value=worker) as load:
            result = runner.launch(self.root, "synthetic-sha", receipt, True, "synthetic-profile")
            self.assertEqual(result["state"], "submitted")
            self.assertEqual(result["attempt"], 1)
            self.assertEqual(load.call_args.args[1], "evaluate_understanding_r4")
            self.assertEqual(load.call_args.kwargs, {"register": True})
            self.assertEqual(worker.evaluate.spawn.call_count, 1)
            with patch.object(runner, "terminal", return_value=False):
                with self.assertRaisesRegex(ValueError, "active, ambiguous"):
                    runner.launch(self.root, "synthetic-sha", self.root / "resume.json", True,
                                  "synthetic-profile", previous=receipt)
            self.assertFalse((self.root / "resume.json").exists())
            self.assertEqual(worker.evaluate.spawn.call_count, 1)

    def test_worker_is_private_and_bounded_and_has_no_retries(self):
        source = (runner.ROOT / "modal/train/evaluate_understanding_r4.py").read_text()
        self.assertIn("retries=0", source)
        self.assertIn("timeout=7200", source)
        self.assertIn("max_containers=1", source)
        self.assertIn("cache.read_only()", source)
        self.assertIn("single_use_containers=True", source)
        self.assertNotIn("web_endpoint", source)
        self.assertNotIn("fastapi_endpoint", source)
        self.assertNotIn(".deploy(", source)

    def test_full_worker_entrypoint_and_resume_with_synthetic_runtime(self):
        result = self.build()
        package = self.root / "execution"
        class Store(dict):
            def put(self, key, value, skip_if_exists=False):
                if skip_if_exists and key in self:
                    return False
                self[key] = value
                return True
        class Resource:
            commits = 0
            def pip_install(self, *args):
                return self
            def env(self, *args):
                return self
            def add_local_dir(self, *args):
                return self
            def read_only(self):
                return self
            def reload(self):
                pass
            def commit(self):
                self.commits += 1
        resource, store, options = Resource(), Store(), {}
        class App:
            def function(self, **kwargs):
                options.update(kwargs)
                return lambda function: function
        fake = types.SimpleNamespace(is_local=lambda: True, App=lambda name: App(),
            Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: resource),
            Dict=types.SimpleNamespace(from_name=lambda *args, **kwargs: store),
            Image=types.SimpleNamespace(debian_slim=lambda **kwargs: resource),
            current_function_call_id=lambda: "fc-synthetic-worker")
        with patch.dict(sys.modules, {"modal": fake}), patch.dict("os.environ", {"GHA_R4_EVALUATION_PACKAGE": str(package)}), patch.object(sys, "path", list(sys.path)):
            worker = runner.module_at(package / "code/evaluate_understanding_r4.py", "_synthetic_worker_test")
            self.assertEqual(options["retries"], 0)
            original_verify = worker.runner.verify
            runtimes = []
            def make_runtime(*args):
                runtime = Runtime()
                runtime.loading = {"strict_text_weights_complete": True}
                runtimes.append(runtime)
                return runtime
            def verify(*args):
                values = list(original_verify(*args))
                values[2] = types.SimpleNamespace(Runtime=make_runtime)
                return tuple(values)
            def path(value):
                if str(value) == "/artifacts" + runner.PREFIX:
                    return self.root / "worker-results"
                if str(value).endswith("/responses-adapter"):
                    return self.consumption.parent / "responses-adapter"
                return Path(value)
            core = types.SimpleNamespace(Journal=CORE.Journal, claim_attempt=CORE.claim_attempt,
                EOS=CORE.EOS, completion=CORE.completion, prompt_ids=prompt_ids)
            scoring = types.SimpleNamespace(objective_score=lambda case, raw: None)
            with patch.object(worker, "Path", side_effect=path), patch.object(worker.runner, "verify", side_effect=verify), \
                 patch.object(worker.runner, "activate", return_value=(core, scoring)), \
                 patch.object(worker.runner, "terminal", return_value=True), \
                 patch("importlib.metadata.version", side_effect=lambda name: r2.RUNTIME[name]):
                first = worker.evaluate("r4_eval_20261106T120000Z", result["execution_sha256"], 1)
                self.assertEqual(first["state"], "completed", first)
                self.assertEqual(first["finalized_slots"], 225)
                self.assertEqual(len(runtimes[0].calls), 225)
                second = worker.evaluate("r4_eval_20261106T120000Z", result["execution_sha256"], 2)
                self.assertEqual(second["state"], "completed", second)
                self.assertEqual(runtimes[1].calls, [])
            folder = self.root / "worker-results/r4_eval_20261106T120000Z"
            summary = runner.read(folder / "summary.json")
            self.assertEqual(summary["provenance"]["consumed"]["row_exposures"], 8)
            self.assertEqual(summary["kind"], runner.KIND)
            self.assertGreaterEqual(resource.commits, 451)
            for name, checksum in runner.read(folder / "results-manifest.json")["files"].items():
                self.assertEqual(runner.sha(folder / name), checksum)


if __name__ == "__main__":
    unittest.main()
