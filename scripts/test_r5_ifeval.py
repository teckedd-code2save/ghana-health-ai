"""Offline IFEval contracts and synthetic runtime tests. No model/GPU calls."""
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import r5_ifeval_source as source
import r5_ifeval_runtime as generation
import run_understanding_r5_ifeval as runner
import score_understanding_r5_ifeval as scoring
import test_run_understanding_r5_evaluation as fixtures

SOURCE = fixtures.ORIGINAL / "tmp/understanding-r5-20261109/ifeval-source-v1"
SOURCE_SHA = "35400682ad754f137519e3e4bbe1769ccd2b56e7a2a486781dc198338d3003cf"


def row(key=1, prompt="Exact synthetic prompt", instructions=None):
    ids = instructions or ["detectable_format:json_format"]
    return {"key": key, "prompt": prompt, "instruction_id_list": ids, "kwargs": [{} for _ in ids]}


def prompt_ids(tokenizer, messages, settings):
    assert settings == {"context_tokens": 16384, "max_new_tokens": 8192}
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return list(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())


CORE = types.SimpleNamespace(Journal=fixtures.CORE.Journal, EOS=fixtures.CORE.EOS, PAD=fixtures.CORE.PAD,
    completion=fixtures.CORE.completion, prompt_ids=prompt_ids, claim_attempt=fixtures.CORE.claim_attempt)


class Runtime(fixtures.Runtime):
    def __init__(self, behavior=None, manifest=None, checksum=None):
        super().__init__()
        self.behavior, self.bound_manifest, self.checksum = behavior, manifest, checksum
        self.loading = {"strict_text_weights_complete": True}

    def identity(self, variant):
        if self.bound_manifest is None:
            return super().identity(variant)
        m = self.bound_manifest
        return {"manifest_sha256": self.checksum, "variant": variant, "model": m["model"], "revision": m["revision"],
                "base_weights_sha256": m["base_sha256"], "adapter_weights_sha256": None if variant == "base" else
                m["variants"][variant]["files"]["adapter_model.safetensors"]}

    def reply(self, prompt, variant, policy):
        self.calls.append((prompt, variant, policy))
        self.loaded.add(variant)
        text, stop = self.behavior(prompt, variant) if self.behavior else ('{"ok":true}', "eos")
        if stop == "raise":
            raise RuntimeError(text)
        ids = list(text.encode()) + ([CORE.EOS] if stop == "eos" else [])
        diagnostics = CORE.completion(self.tokenizer, ids, 8192, stop == "time_limit", .01)
        result = {"identity": self.identity(variant), "messages_sha256": runner.digest(generation.messages(prompt)),
            "prompt_ids_sha256": runner.digest(generation.prompt_ids(CORE, self.tokenizer, prompt)),
            "protocol": copy.deepcopy(generation.PROTOCOL), "generation_policy": policy,
            "output_ids": ids, "output": diagnostics["prediction"], "seconds": .01, "diagnostics": diagnostics,
            "timed_out": stop == "time_limit", "guard_stopped": False, "finish_reason": diagnostics["finish_reason"],
            "complete": diagnostics["complete"], "cap_hit": diagnostics["finish_reason"] == "max_new_tokens"}
        if policy == "product_guarded":
            result.update(guard_version=generation.GUARD, guard_trigger=None)
        return result


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.journal = CORE.Journal(self.root / "journal", {"test": "synthetic"})
        self.rows = [row(1), row(2, "  Preserve whitespace\nSecond prompt  ")]

    def run_engine(self, runtime, **kwargs):
        return runner.evaluate(runtime, CORE, self.rows, self.journal, 1, "proof", "raw", 10000, clock=lambda: 0, **kwargs)

    def test_chunk_resumes_without_regeneration_or_hidden_messages(self):
        runtime = Runtime()
        first = self.run_engine(runtime, chunk_slots=2)
        self.assertEqual(first["state"], "paused_chunk")
        self.assertEqual(len(runtime.calls), 2)
        second = self.run_engine(runtime, chunk_slots=2)
        self.assertEqual(second["state"], "paused_chunk")
        third = self.run_engine(runtime, chunk_slots=2)
        self.assertEqual(third["state"], "completed")
        self.assertEqual(len(runtime.calls), 6)
        self.assertEqual(runtime.calls[3][0], self.rows[1]["prompt"])
        self.run_engine(runtime)
        self.assertEqual(len(runtime.calls), 6)
        self.assertEqual(fixtures.r2.DECODING["max_new_tokens"], 256)
        self.assertEqual(generation.PROTOCOL["max_new_tokens"], 8192)

    def test_interrupted_slot_is_indeterminate_not_retried(self):
        slot = runner.slots(self.rows)[0]
        self.journal.begin(slot["key"], 1, runner.digest(runner.request(slot)))
        runtime = Runtime()
        self.run_engine(runtime)
        self.assertEqual(len(runtime.calls), 5)
        self.assertEqual(self.journal.get(slot["key"])["state"], "indeterminate")
        self.run_engine(runtime)
        self.assertEqual(len(runtime.calls), 5)

    def test_errors_and_caps_count_in_full_plan(self):
        runtime = Runtime(lambda p, v: ("synthetic failure", "raise") if v == "base" else ("x" * 8192, "max_new_tokens"))
        self.run_engine(runtime)
        report = runner.report(self.root, self.rows, self.journal)
        self.assertEqual(report["expected_slots"], 6)
        self.assertEqual(report["generation"]["base"]["states"]["error"], 2)
        self.assertEqual(report["generation"]["r4_response"]["states"]["incomplete"], 2)
        self.assertEqual(report["generation"]["r4_response"]["cap_hits"], 2)
        self.assertIsNone(report["generation"]["r4_response"]["accuracy"])
        self.run_engine(runtime)
        self.assertEqual(len(runtime.calls), 6)

    def test_budget_does_not_shorten_per_case_allowance(self):
        runtime = Runtime()
        result = runner.evaluate(runtime, CORE, self.rows, self.journal, 1, "proof", "raw", 254, clock=lambda: 0)
        self.assertEqual(result["state"], "paused_budget")
        self.assertFalse(runtime.calls)
        self.assertFalse(list((self.journal.folder / "attempts").glob("*")))

    def test_saved_protocol_checkpoint_and_prompt_tampering_rejected(self):
        runtime = Runtime()
        self.run_engine(runtime)
        original = self.rows[0]["prompt"]
        self.rows[0]["prompt"] = "edited"
        with self.assertRaisesRegex(ValueError, "Saved IFEval request"):
            self.run_engine(runtime)
        self.rows[0]["prompt"] = original
        first = self.journal.get("1:base")["result"]
        for key, value in (("protocol", {**generation.PROTOCOL, "max_new_tokens": 256}),
                           ("output", "rewritten"), ("identity", {"variant": "wrong"})):
            changed = {**first, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                generation.verify_reply(runtime, CORE, original, "base", generation.prompt_ids(CORE, runtime.tokenizer, original), changed, "raw")

    def test_guarded_requires_separate_binding_and_raw_never_applies_guard(self):
        runtime = Runtime()
        self.run_engine(runtime)
        with self.assertRaisesRegex(ValueError, "Saved IFEval request"):
            runner.evaluate(runtime, CORE, self.rows, self.journal, 2, "proof", "product_guarded", 10000, clock=lambda: 0)
        value = runtime.reply("test", "base", "raw")
        value["guard_trigger"] = None
        with self.assertRaisesRegex(ValueError, "Raw result"):
            generation.verify_reply(runtime, CORE, "test", "base", generation.prompt_ids(CORE, runtime.tokenizer, "test"), value, "raw")

    def test_global_append_only_lease_prevents_overlap_and_ambiguous_resubmit(self):
        store = Store()
        first = runner.reserve_gpu(store, {"owner": "one"}, lambda _: False)
        self.assertEqual(first, 0)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            runner.reserve_gpu(store, {"owner": "two"}, lambda _: True)
        runner.bind_call(store, first, "one", "fc-one")
        with self.assertRaisesRegex(ValueError, "active"):
            runner.reserve_gpu(store, {"owner": "two"}, lambda _: False)
        self.assertEqual(runner.reserve_gpu(store, {"owner": "two"}, lambda call: call == "fc-one"), 1)
        self.assertEqual(store["terminal:0"]["call_id"], "fc-one")
        with self.assertRaisesRegex(ValueError, "call differs"):
            runner.bind_call(store, 0, "one", "different")

    def test_authorization_fails_before_any_remote_import_or_verification(self):
        with patch.object(runner, "verify", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(ValueError, "authorization"):
                runner.launch(self.root, "sha", self.root / "receipt.json", "profile", False)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [row(1), row(2, "Second", ["detectable_format:json_format", "punctuation:no_comma"])]
        self.generated = CORE.Journal(self.root / "generation", {"test": "generation"})
        self.scores = CORE.Journal(self.root / "scores/journal", {"test": "scores", "generation_policy": "raw"})

    def test_complete_incomplete_error_pending_and_scorer_error_denominators(self):
        for slot in runner.slots(self.rows):
            if slot["variant"] == "r5_response":
                continue
            value = {"verified": slot["variant"] != "r4_response", "state": "incomplete" if slot["variant"] == "base" else "error",
                     "result": {"output": "partial", "cap_hit": True, "finish_reason": "max_new_tokens"}}
            self.generated.finish(slot["key"], value)
        def score(row, response, mode):
            if mode == "loose":
                raise RuntimeError("synthetic scorer error")
            flags = [True] * len(row["instruction_id_list"])
            return {"prompt": row["prompt"], "response": response, "instruction_id_list": row["instruction_id_list"],
                    "follow_all_instructions": True, "follow_instruction_list": flags}
        scorer = types.SimpleNamespace(score=unittest.mock.Mock(side_effect=score))
        self.assertEqual(scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, clock=lambda: 0), "finalized")
        report = scoring.report(self.root / "scores", self.rows, self.scores, "finalized")
        base = report["models"]["base"]
        self.assertEqual(base["cap_hits"], 2)
        self.assertEqual(base["overall"]["strict"]["expected_instructions"], 3)
        self.assertEqual(base["overall"]["strict"]["instruction_accuracy"], 1.)
        self.assertIsNone(base["overall"]["loose"]["prompt_accuracy"])
        self.assertEqual(base["states"]["scoring_error"], 2)
        self.assertEqual(report["models"]["r4_response"]["states"]["generation_error"], 2)
        self.assertEqual(report["models"]["r5_response"]["states"]["generation_pending"], 2)
        self.assertFalse(report["full_suite_scoring_complete"])
        self.assertFalse(report["twi_ability_established"])
        scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, clock=lambda: 0)
        self.assertEqual(scorer.score.call_count, 4)

    def test_scoring_chunks_resume_and_detect_changed_generation(self):
        scorer = types.SimpleNamespace(score=unittest.mock.Mock())
        self.assertEqual(scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, max_records=1, clock=lambda: 0), "paused")
        self.assertEqual(scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, clock=lambda: 0), "finalized")
        self.assertFalse(scorer.score.called)
        self.generated.finish("1:base", {"verified": False, "state": "error"})
        with self.assertRaisesRegex(ValueError, "Changed generation"):
            scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, clock=lambda: 0)

    def test_scoring_interruption_is_not_a_silent_retry(self):
        slot = runner.slots(self.rows)[0]
        self.scores.begin(slot["key"], 1, runner.digest({"source_case": slot["case"], "generation_record": None}))
        scorer = types.SimpleNamespace(score=unittest.mock.Mock())
        scoring.score_slots(self.rows, self.generated, self.scores, scorer, 10000, clock=lambda: 0)
        self.assertEqual(self.scores.get(slot["key"])["state"], "scoring_indeterminate")
        self.assertFalse(scorer.score.called)


class NativeGenerationWiringTests(unittest.TestCase):
    def test_separate_generate_config_and_native_whitespace_without_old_reply(self):
        class Vector(list):
            def tolist(self): return list(self)
        class Tensor:
            device = "cuda"
            def __init__(self, values):
                self.values, self.shape = values, (len(values), len(values[0]))
            def __getitem__(self, key):
                return Vector(self.values[key[0]][key[1]])
        generated_configs = []
        torch = types.SimpleNamespace(long="long", bool="bool", tensor=lambda rows, **kwargs: Tensor(rows),
            ones_like=lambda inputs: "attention", manual_seed=lambda _: None, inference_mode=nullcontext,
            full=lambda shape, value, **kwargs: value,
            cuda=types.SimpleNamespace(synchronize=lambda: None, manual_seed_all=lambda _: None))
        transformers = types.SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list,
            GenerationConfig=lambda **kwargs: types.SimpleNamespace(**kwargs))
        engine = Runtime()
        engine.reply = unittest.mock.Mock(side_effect=AssertionError("old 256-token reply must never run"))
        def select(variant):
            engine.loaded.add(variant)
            return nullcontext()
        engine.select = select
        def generate(**kwargs):
            config = vars(kwargs["generation_config"])
            generated_configs.append(config)
            self.assertEqual(config["max_new_tokens"], 8192)
            self.assertEqual(config["eos_token_id"], CORE.EOS)
            self.assertIsNone(config["bos_token_id"])
            ids = kwargs["input_ids"].values[0]
            return types.SimpleNamespace(sequences=Tensor([ids + list(b'{"ok":true}') + [CORE.EOS]]))
        engine.model = types.SimpleNamespace(generate=generate)
        runtime = generation.Runtime(engine, CORE)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers,
                                      "numpy": types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _: None))}):
            result = runtime.reply("  Exact whitespace\n", "r4_response", "raw")
        self.assertTrue(result["complete"])
        self.assertEqual(result["protocol"], generation.PROTOCOL)
        self.assertEqual(result["messages_sha256"], runner.digest([{"role": "user", "content": "  Exact whitespace\n"}]))
        engine.reply.assert_not_called()
        self.assertEqual(len(generated_configs), 1)
        self.assertEqual(fixtures.r2.DECODING["max_new_tokens"], 256)

    def test_guard_pause_is_not_raw_or_eos_completion(self):
        runtime = Runtime(lambda p, v: ("one two three four five six seven eight " * 4, "unexpected"))
        result = runtime.reply("prompt", "base", "product_guarded")
        result.update(guard_stopped=True, guard_trigger=fixtures.GUARD.repeated_passage(result["output"]),
                      finish_reason="product_guard", complete=False)
        with patch.dict(sys.modules, {"response_loop_guard": fixtures.GUARD}):
            ids = generation.prompt_ids(CORE, runtime.tokenizer, "prompt")
            generation.verify_reply(runtime, CORE, "prompt", "base", ids, result, "product_guarded")
            with self.assertRaises(ValueError):
                generation.verify_reply(runtime, CORE, "prompt", "base", ids, result, "raw")


@unittest.skipUnless(SOURCE.is_dir(), "Fetch the pinned private IFEval source package first")
class SourceTests(unittest.TestCase):
    def test_full_source_license_resources_and_constraint_bound(self):
        manifest, rows = source.load(SOURCE, SOURCE_SHA)
        self.assertEqual(manifest["inventory"]["prompts"], 541)
        self.assertEqual(manifest["inventory"]["instruction_instances"], 834)
        self.assertEqual(manifest["inventory"]["maximum_minimum_words"], 1200)
        self.assertEqual(len(rows), 541)
        self.assertTrue((SOURCE / "upstream/LICENSE").exists())
        self.assertEqual((SOURCE / "manifest.json").stat().st_mode & 0o777, 0o600)
        for subset in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaisesRegex(ValueError, "541"):
                source.inventory(subset)

    def test_source_edits_fail_even_with_resealed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "source"
            shutil.copytree(SOURCE, dest)
            path = dest / "upstream" / source.DATA
            path.write_bytes(path.read_bytes() + b"\n")
            manifest = runner.read(dest / "manifest.json")
            manifest["files"]["upstream/" + source.DATA] = runner.sha(path)
            runner.save(dest / "manifest.json", manifest)
            checksum = runner.sha(dest / "manifest.json")
            runner.save(dest / "SEALED.json", {"manifest_sha256": checksum})
            with self.assertRaisesRegex(ValueError, "bytes"):
                source.load(dest, checksum)


def fake_preflight(bundle, source_dir, checksum):
    _, rows = source.load(source_dir, checksum)
    tokenizer = fixtures.Tokenizer()
    return {"source_sha256": checksum, "protocol": generation.PROTOCOL, "expected_prompts": 541, "ready": True,
            "cases": [{"key": r["key"], "ready": True, "prompt_tokens": len(generation.prompt_ids(CORE, tokenizer, r["prompt"])),
                       "prompt_ids_sha256": runner.digest(generation.prompt_ids(CORE, tokenizer, r["prompt"])),
                       "messages_sha256": runner.digest(generation.messages(r["prompt"]))} for r in rows]}


@unittest.skipUnless(SOURCE.is_dir(), "Fetch pinned IFEval source package first")
class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle, self.bundle_sha, self.inputs, self.consumption = fixtures.prepared_fixture(self.root)
        self.audit = self.root / "source-audit/audit.json"
        proof = self.audit.parent / "evidence/report.json"
        runner.save(proof, {"scope": "synthetic fixture only", "training_data_registered": True})
        runner.save(self.audit, {"version": "r5-evaluation-source-audit-v1", "overlap_free": True,
            "training_manifest_sha256": runner.sha(self.inputs / "manifest.json"), "evaluation_manifest_sha256": SOURCE_SHA,
            "evaluation_source_sha256": source.UPSTREAM[source.DATA], "checks": {
                "source_groups_disjoint": True, "exact_text_disjoint": True, "normalized_text_disjoint": True, "full_protection_coverage": True},
            "evidence": {"evidence/report.json": runner.sha(proof)}})

    def build(self, policy="raw"):
        with patch.object(runner, "preflight", side_effect=fake_preflight):
            return runner.prepare(self.bundle, self.bundle_sha, SOURCE, SOURCE_SHA, self.inputs, self.consumption,
                                  self.audit, self.root / "execution", policy)

    def test_source_audit_training_receipt_and_new_protocol_are_bound(self):
        before = runner.files(self.bundle)
        result = self.build()
        execution, manifest, _, rows = runner.verify(self.root / "execution", result["execution_sha256"])
        self.assertEqual(execution["expected_slots"], 1623)
        self.assertEqual(execution["protocol"]["max_new_tokens"], 8192)
        self.assertEqual(manifest["decoding"]["max_new_tokens"], 256)
        self.assertEqual(len(rows), 541)
        self.assertEqual(before, runner.files(self.bundle))
        self.assertEqual((self.root / "execution").stat().st_mode & 0o777, 0o700)
        self.assertFalse(result["gpu_launched"])
        audit = runner.read(self.audit)
        audit["checks"]["full_protection_coverage"] = False
        runner.save(self.audit, audit)
        with patch.object(runner, "preflight", side_effect=AssertionError("audit first")), self.assertRaisesRegex(ValueError, "clearance"):
            runner.prepare(self.bundle, self.bundle_sha, SOURCE, SOURCE_SHA, self.inputs, self.consumption,
                           self.audit, self.root / "another")

    def test_missing_or_wrong_audit_rejected_and_no_context_failures_hidden(self):
        with self.assertRaises(ValueError):
            runner.prepare(self.bundle, self.bundle_sha, SOURCE, SOURCE_SHA, self.inputs, self.consumption, None, self.root / "execution")
        with patch.object(runner, "preflight", return_value={"ready": False}), self.assertRaisesRegex(ValueError, "no truncation"):
            runner.prepare(self.bundle, self.bundle_sha, SOURCE, SOURCE_SHA, self.inputs, self.consumption, self.audit, self.root / "execution")
        self.assertFalse((self.root / "execution").exists())

    def test_resealed_short_budget_and_missing_case_cannot_pass(self):
        self.build()
        package = self.root / "execution"
        original = runner.read(package / "execution.json")
        for field in ("protocol", "context"):
            value = copy.deepcopy(original)
            if field == "protocol":
                value["protocol"]["max_new_tokens"] = 256
            else:
                context = runner.read(package / "context-preflight.json")
                context["cases"].pop()
                runner.save(package / "context-preflight.json", context)
                value["files"]["context-preflight.json"] = runner.sha(package / "context-preflight.json")
            runner.save(package / "execution.json", value)
            checksum = runner.sha(package / "execution.json")
            runner.save(package / "SEALED.json", {"execution_sha256": checksum})
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.verify(package, checksum)

    def test_private_worker_chunk_and_resume_without_real_remote_calls(self):
        built = self.build()
        package = self.root / "execution"
        stores, options = {}, {}
        class Resource:
            def pip_install(self, *args): return self
            def env(self, *args): return self
            def add_local_dir(self, *args): return self
            def read_only(self): return self
            def reload(self): pass
            def commit(self): pass
        class App:
            def function(self, **kwargs):
                options.update(kwargs)
                return lambda function: function
        resource = Resource()
        current = ["fc-synthetic-1"]
        fake = types.SimpleNamespace(is_local=lambda: True, App=lambda _: App(),
            Volume=types.SimpleNamespace(from_name=lambda *a, **k: resource),
            Dict=types.SimpleNamespace(from_name=lambda name, **k: stores.setdefault(name, Store())),
            Image=types.SimpleNamespace(debian_slim=lambda **k: resource), current_function_call_id=lambda: current[0])
        runtimes = []
        with patch.dict(sys.modules, {"modal": fake}), patch.dict(os.environ, {"GHA_R5_IFEVAL_PACKAGE": str(package)}), patch.object(sys, "path", list(sys.path)):
            worker = runner.common.module_at(package / "code/evaluate_understanding_r5_ifeval.py", "_ifeval_synthetic_worker")
            original_verify = worker.runner.verify
            def verify(*args):
                values = list(original_verify(*args))
                def make(*unused):
                    runtime = Runtime(manifest=values[1], checksum=values[0]["inference_sha256"])
                    runtimes.append(runtime)
                    return runtime
                values[2] = types.SimpleNamespace(Runtime=make)
                return values
            def path(value):
                if str(value) == "/artifacts" + runner.PREFIX:
                    return self.root / "worker"
                if str(value).endswith("/responses-adapter"):
                    return self.consumption.parent / "responses-adapter"
                return Path(value)
            run_id = "r5_ifeval_20261110T010203Z_12345678"
            ledger = stores[runner.LEDGER]
            with patch.object(worker, "Path", side_effect=path), patch.object(worker.runner, "verify", side_effect=verify), \
                 patch.object(worker.runner.common, "activate", return_value=(CORE, None)), \
                 patch.object(worker.runner.common, "terminal", return_value=True), \
                 patch.object(worker.runner.generation, "Runtime", side_effect=lambda engine, core: engine), \
                 patch("importlib.metadata.version", side_effect=lambda name: fixtures.r2.RUNTIME[name]):
                for attempt in (1, 2):
                    current[0] = "fc-synthetic-" + str(attempt)
                    ledger["lease:" + str(attempt)] = {"owner": "owner", "run_id": run_id,
                        "execution_sha256": built["execution_sha256"], "attempt": attempt}
                    result = worker.evaluate(run_id, built["execution_sha256"], attempt, 3, attempt, "owner")
                    self.assertEqual(result["state"], "paused_chunk", result)
                    self.assertEqual(result["finalized_slots"], attempt * 3)
                    self.assertEqual(len(runtimes[-1].calls), 3)
            self.assertEqual(options["retries"], 0)
            self.assertEqual(options["max_containers"], 1)
            self.assertEqual(options["timeout"], 7200)
            self.assertFalse(options["include_source"])
            with patch.object(runner.common, "activate", return_value=(CORE, None)):
                execution, rows, journal = runner.audit_results(package, built["execution_sha256"], self.root / "worker" / run_id)
                self.assertEqual(journal.binding["generation_policy"], "raw")
                self.assertEqual(len(rows), 541)

    def test_launcher_reserves_before_spawn_and_receipt_blocks_overlap(self):
        store = Store()
        events = []
        worker = types.SimpleNamespace(app=types.SimpleNamespace(run=lambda **kwargs: nullcontext()),
            evaluate=types.SimpleNamespace(spawn=lambda *args: (events.append(args) or types.SimpleNamespace(object_id="fc-synthetic"))))
        fake = types.SimpleNamespace(enable_output=nullcontext,
            Dict=types.SimpleNamespace(from_name=lambda *args, **kwargs: store))
        with patch.dict(sys.modules, {"modal": fake}), patch.object(runner, "verify"), \
             patch.object(runner.common, "module_at", return_value=worker), patch.object(runner.common, "terminal", return_value=False):
            receipt = self.root / "receipt-new.json"
            result = runner.launch(self.root / "package", "sha", receipt, "fixture-profile", True)
            self.assertEqual(result["state"], "submitted")
            self.assertEqual(len(events), 1)
            self.assertEqual(store["call:0"]["call_id"], "fc-synthetic")
            with self.assertRaisesRegex(ValueError, "active"):
                runner.launch(self.root / "package", "sha", self.root / "other.json", "fixture-profile", True)
            self.assertEqual(len(events), 1)
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_official_cli_scores_partial_snapshot_then_resumes_cpu_only(self):
        python = fixtures.ORIGINAL / "tmp/understanding-r5-20261109/ifeval-scorer-env-v1/bin/python"
        if not python.exists():
            self.skipTest("Pinned scorer environment not installed")
        built = self.build()
        package = self.root / "execution"
        execution, manifest, _, rows = runner.verify(package, built["execution_sha256"])
        folder = self.root / "generated"
        bound = runner.binding(execution, built["execution_sha256"], "r5_ifeval_20261110T010203Z_12345678")
        journal = CORE.Journal(folder / "journal", bound)
        runtime = Runtime(manifest=manifest, checksum=self.bundle_sha)
        proof = {"binding": bound, "runtime": execution["runtime"], "identities": {v: runtime.identity(v) for v in runner.VARIANTS}}
        runner.save(folder / "runtime/attempt-1.json", proof)
        runner.evaluate(runtime, CORE, rows, journal, 1, runner.sha(folder / "runtime/attempt-1.json"), "raw", 10000, 3, clock=lambda: 0)
        runner.report(folder, rows, journal)
        out = self.root / "official-score"
        command = [str(python), str(runner.ROOT / "scripts/score_understanding_r5_ifeval.py"),
            "--execution", str(package), "--execution-sha", built["execution_sha256"], "--results", str(folder),
            "--out", str(out), "--max-records", "2"]
        env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["state"], "paused")
        command[-1] = "1623"
        completed = subprocess.run(command + ["--resume"], env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = runner.read(out / "scores.json")
        self.assertEqual(report["state"], "finalized")
        self.assertFalse(report["full_suite_scoring_complete"])
        for variant in runner.VARIANTS:
            metric = report["models"][variant]["overall"]["strict"]
            self.assertEqual((metric["expected_prompts"], metric["expected_instructions"], metric["scored_prompts"]), (541, 834, 1))
            self.assertIsNone(metric["prompt_accuracy"])
            self.assertEqual(report["models"][variant]["states"]["generation_pending"], 540)
        self.assertEqual(out.stat().st_mode & 0o777, 0o700)
        self.assertEqual((out / "scores.json").stat().st_mode & 0o777, 0o600)


@unittest.skipUnless(SOURCE.is_dir() and importlib.util.find_spec("nltk") and importlib.util.find_spec("langdetect"),
                     "Use the pinned private scorer environment and source package")
class OfficialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = scoring.OfficialScorer(SOURCE, SOURCE_SHA)

    def test_official_strict_loose_and_empty_are_not_custom_regexes(self):
        sample = row()
        self.assertTrue(self.scorer.score(sample, '{"ok": true}', "strict")["follow_all_instructions"])
        self.assertFalse(self.scorer.score(sample, 'preamble\n{"ok": true}', "strict")["follow_all_instructions"])
        self.assertTrue(self.scorer.score(sample, 'preamble\n{"ok": true}', "loose")["follow_all_instructions"])
        self.assertFalse(self.scorer.score(sample, "", "loose")["follow_all_instructions"])

    def test_full_541_schema_and_all_official_instruction_arguments(self):
        _, rows = source.load(SOURCE, SOURCE_SHA)
        import socket
        with patch.object(socket.socket, "connect", side_effect=AssertionError("offline scorer attempted network")):
            for sample in rows:
                for mode in scoring.MODES:
                    result = self.scorer.score(sample, "", mode)
                    self.assertEqual(result["follow_instruction_list"], [False] * len(sample["instruction_id_list"]))
        self.assertEqual(len(rows), 541)

    def test_nltk_sentence_resource_and_dependency_proof(self):
        sample = row(instructions=["length_constraints:number_sentences"])
        sample["kwargs"] = [{"relation": "at least", "num_sentences": 2}]
        result = self.scorer.score(sample, "First sentence. Second sentence.", "strict")
        self.assertTrue(result["follow_all_instructions"])
        self.assertEqual(self.scorer.proof["dependencies"], source.SCORER_ENV)
        self.assertFalse(self.scorer.proof["scorer_modified"])


if __name__ == "__main__":
    unittest.main()
