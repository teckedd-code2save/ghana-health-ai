"""Offline retention contract tests; only synthetic logits/runtime, no model calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import r5_retention as retention
import run_understanding_r5_evaluation as runner
import test_run_understanding_r5_evaluation as fixtures


def english_row():
    return {"id": "synthetic-en", "loss_cohort": retention.ENGLISH, "messages": [
        {"role": "user", "content": "PRIVATE-FIRST-USER"},
        {"role": "assistant", "content": "PRIVATE-EARLIER-ASSISTANT"},
        {"role": "user", "content": "PRIVATE-FOLLOWUP"},
        {"role": "assistant", "content": "PRIVATE-FINAL-ANSWER"}]}


def encoded():
    return {"input_ids": [1, 2, 3, fixtures.CORE.EOS], "labels": [-100, -100, 3, fixtures.CORE.EOS],
            "attention_mask": [1, 1, 1, 1]}


class SourceTests(unittest.TestCase):
    def test_original_44_validation_rows_and_minimal_replay(self):
        paths = retention.sources(fixtures.R2_PACK, fixtures.ENGLISH_VALIDATION)
        before = {k: retention.sha(v) for k, v in paths.items()}
        rows, proof = retention.load(paths)
        self.assertEqual(len(rows), 44)
        self.assertEqual(proof["selected_test_rows"], 0)
        self.assertFalse(proof["training_eligible"])
        self.assertFalse(proof["pool_with_twi_causal_loss"])
        self.assertTrue(any(len(r["messages"]) > 2 for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            copies = {name: Path(tmp) / name for name in paths}
            for name, source in paths.items():
                shutil.copyfile(source, copies[name])
            self.assertEqual(retention.load(copies), (rows, proof))
            for name in copies:
                original = copies[name].read_bytes()
                copies[name].write_bytes(original + b"\n")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    retention.load(copies)
                copies[name].write_bytes(original)
        self.assertEqual(before, {k: retention.sha(v) for k, v in paths.items()})

    def test_required_source_and_test_substitution_fail_closed(self):
        for pack, validation in ((None, None), (fixtures.R2_PACK, None)):
            with self.assertRaisesRegex(ValueError, "44 heldout"):
                retention.sources(pack, validation)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.jsonl"
            path.write_text('{"split":"test"}\n')
            with self.assertRaisesRegex(ValueError, "English validation source differs"):
                retention.load(retention.sources(fixtures.R2_PACK, path))


class EncodingTests(unittest.TestCase):
    def test_english_uses_complete_native_history_and_only_final_targets(self):
        row = english_row()
        helper = types.SimpleNamespace(encode_complete=unittest.mock.Mock(return_value=encoded()))
        tokenizer = fixtures.Tokenizer()
        with patch.dict(sys.modules, {"check_afrique_training_contract": helper}):
            value, proof = retention.encode(tokenizer, row, fixtures.CORE.EOS)
        helper.encode_complete.assert_called_once_with(tokenizer, row, 4096)
        self.assertEqual(len(row["messages"]), 4)
        self.assertEqual(value, encoded())
        self.assertEqual(proof["predicted_tokens"], 2)
        self.assertEqual(proof["sequence_tokens"], 4)
        self.assertEqual(proof["labels_sha256"], runner.digest(encoded()["labels"]))

    def test_twi_causal_encoding_is_unchanged_and_never_uses_chat(self):
        row = {"id": "twi", "text": "synthetic text"}
        value, proof = retention.encode(fixtures.Tokenizer(), row, fixtures.CORE.EOS)
        self.assertEqual(value["input_ids"], list(row["text"].encode()) + [fixtures.CORE.EOS])
        self.assertEqual(value["input_ids"], value["labels"])
        self.assertEqual(proof["predicted_tokens"], len(row["text"].encode()))
        with self.assertRaisesRegex(ValueError, "no truncation"):
            retention.encode(fixtures.Tokenizer(), {"id": "long", "text": "x" * 4096}, fixtures.CORE.EOS)

    def test_preflight_counts_all_errors_without_content_leak(self):
        core = types.SimpleNamespace(EOS=fixtures.CORE.EOS)
        helper = types.SimpleNamespace(load_tokenizer=lambda _: fixtures.Tokenizer(),
            encode_complete=unittest.mock.Mock(side_effect=ValueError("PRIVATE-FINAL-ANSWER")))
        with patch.object(runner, "activate_code", return_value=(core, None)), \
                patch.dict(sys.modules, {"check_afrique_training_contract": helper}):
            result = runner.context_preflight(Path("synthetic"), [], [], [english_row(), {"id": "tw", "text": "ok"}])
        self.assertFalse(result["ready"])
        self.assertEqual(result["kind"], runner.KIND)
        self.assertEqual(result["language_loss"]["cohorts"][retention.ENGLISH]["error_rows"], 1)
        self.assertEqual(result["language_loss"]["cohorts"][retention.TWI]["ready_rows"], 1)
        self.assertNotIn("PRIVATE-FINAL-ANSWER", json.dumps(result))

    def test_shifted_masked_loss_counts_targets_not_prompt_and_chunks(self):
        calls = []
        class Scalar:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        class Vector(list):
            def __ne__(self, value): return Vector(x != value for x in self)
            def sum(self): return Scalar(sum(self))
        class Matrix:
            def __init__(self, values, logits=False):
                self.values = values
                self.shape = (1, len(values), 4) if logits else (1, len(values))
            def __getitem__(self, key):
                values = Vector(self.values[key[1]])
                values.float = lambda: values
                return values
        def cross_entropy(logits, labels, **kwargs):
            self.assertEqual(kwargs, {"ignore_index": -100, "reduction": "sum"})
            self.assertEqual(len(logits), len(labels))
            calls.append(list(labels))
            return Scalar(sum(math.log(4) for label in labels if label != -100))
        fake = types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=cross_entropy))
        # Uniform synthetic logits: masked prompt positions do not contribute.
        labels = [-100] * 129 + [2, 3]
        with patch.dict(sys.modules, {"torch.nn": fake}):
            result = retention.shifted_masked_nll(Matrix([[0.] * 4] * len(labels), True), Matrix(labels))
            with self.assertRaisesRegex(ValueError, "No shifted assistant"):
                retention.shifted_masked_nll(Matrix([[0.] * 4] * 3, True), Matrix([-100] * 3))
        self.assertEqual([len(c) for c in calls], [128, 2])
        self.assertEqual(result["predicted_tokens"], 2)
        self.assertAlmostEqual(result["summed_nll"], 2 * math.log(4))

    def test_measure_selects_variant_and_disables_gradients_cache(self):
        events = []
        class Context:
            def __init__(self, name): self.name = name
            def __enter__(self): events.append(self.name + ":enter")
            def __exit__(self, *args): events.append(self.name + ":exit")
        fake = types.SimpleNamespace(long="long", tensor=lambda value, **kwargs: (value, kwargs),
            ones_like=lambda value: ("ones", value), inference_mode=lambda: Context("inference"))
        model = unittest.mock.Mock(return_value=types.SimpleNamespace(logits="synthetic"))
        runtime = types.SimpleNamespace(select=lambda variant: Context(variant), model=model)
        with patch.dict(sys.modules, {"torch": fake}), patch.object(retention, "shifted_masked_nll", return_value={"predicted_tokens": 2, "summed_nll": 4.}) as loss:
            result = retention.measure(runtime, english_row(), "r4_response", encoded())
        self.assertEqual(result["predicted_tokens"], 2)
        self.assertEqual(events, ["r4_response:enter", "inference:enter", "inference:exit", "r4_response:exit"])
        self.assertFalse(model.call_args.kwargs["use_cache"])
        self.assertEqual(loss.call_args.args[1][0], [encoded()["labels"]])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = fixtures.EngineTests()
        self.engine.setUp()
        self.addCleanup(self.engine.doCleanups)
        self.runtime = fixtures.Runtime()
        self.rows = [{"id": "tw", "text": "secret twi"}, english_row()]
        self.plan = list(runner.slots([], [], self.rows))
        helper = types.SimpleNamespace(encode_complete=lambda *args: encoded())
        self.helper_patch = patch.dict(sys.modules, {"check_afrique_training_contract": helper})
        self.helper_patch.start()
        self.addCleanup(self.helper_patch.stop)

    def measured(self, runtime, row, variant, value):
        runtime.loss_calls.append((row["id"], variant))
        runtime.loaded.add(variant)
        count = sum(x != -100 for x in value["labels"][1:])
        return {"predicted_tokens": count, "summed_nll": count * (5. if retention.cohort(row) == retention.ENGLISH else 2.)}

    def report(self):
        preflight = [{"cohort": retention.cohort(row), "case_id": row["id"], "ready": True,
                      **retention.encode(self.runtime.tokenizer, row, fixtures.CORE.EOS)[1]} for row in self.rows]
        return runner.report(self.engine.root, self.plan, self.engine.journal, {},
            {v: self.runtime.identity(v) for v in runner.VARIANTS}, loss_preflight=preflight)

    def test_independent_cohorts_exact_masks_resume_and_private_cases(self):
        with patch.object(retention, "measure", side_effect=self.measured):
            self.engine.run_engine(self.runtime, plan=self.plan)
            self.engine.run_engine(self.runtime, plan=self.plan, attempt=2)
        self.assertEqual(len(self.runtime.loss_calls), 6)
        report = self.report()
        metrics = report["mechanical_metrics"]["base"]["language_loss"]
        self.assertNotIn("token_weighted_mean_nll", metrics)
        self.assertEqual(metrics[retention.TWI]["token_weighted_mean_nll"], 2.)
        self.assertEqual(metrics[retention.ENGLISH]["token_weighted_mean_nll"], 5.)
        self.assertEqual(metrics[retention.ENGLISH]["predicted_tokens"], 2)
        self.assertEqual(metrics[retention.ENGLISH]["expected_predicted_tokens"], 2)
        self.assertTrue(report["retention_paired_complete_coverage"][retention.ENGLISH])
        self.assertFalse(report["human_quality"]["semantic_quality_established"])
        for path in self.engine.root.rglob("*"):
            if path.is_file():
                self.assertNotIn("PRIVATE-", path.read_text())
                self.assertNotIn("secret twi", path.read_text())
        cases = runner.read(self.engine.root / "language-loss-cases.json")
        self.assertEqual(len(cases["rows"]), 6)
        self.assertFalse(cases["accuracy_established"])

    def test_errors_and_pending_remain_in_denominators(self):
        def measure(runtime, row, variant, value):
            if row["id"] == "synthetic-en":
                raise RuntimeError("PRIVATE-FINAL-ANSWER")
            return self.measured(runtime, row, variant, value)
        tick = iter([0, 0, 0, 0, 2])
        with patch.object(retention, "measure", side_effect=measure):
            self.engine.run_engine(self.runtime, plan=self.plan, deadline=1, clock=lambda: next(tick))
        report = self.report()
        base = report["mechanical_metrics"]["base"]["language_loss"][retention.ENGLISH]
        pending = report["mechanical_metrics"]["r4_response"]["language_loss"][retention.ENGLISH]
        self.assertEqual((base["expected_rows"], base["measured_rows"], base["error_rows"], base["expected_predicted_tokens"]), (1, 0, 1, 2))
        self.assertIsNone(base["token_weighted_mean_nll"])
        self.assertEqual(pending["pending_rows"], 1)
        self.assertFalse(report["retention_paired_complete_coverage"][retention.ENGLISH])
        cases = runner.read(self.engine.root / "language-loss-cases.json")["rows"]
        self.assertEqual(sum(c["state"] == "error" for c in cases), 1)
        self.assertEqual(sum(c["state"] == "pending" for c in cases), 2)
        self.assertNotIn("PRIVATE-", json.dumps(cases))

    def test_saved_mask_and_source_changes_are_rejected(self):
        with patch.object(retention, "measure", side_effect=self.measured):
            self.engine.run_engine(self.runtime, plan=self.plan)
        changed = copy.deepcopy(self.plan)
        changed[-1]["row"]["messages"][-1]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "Saved slot/history"):
            self.engine.run_engine(self.runtime, plan=changed, attempt=2)
        altered = encoded()
        altered["labels"] = [-100, 2, -100, fixtures.CORE.EOS]
        with patch.dict(sys.modules, {"check_afrique_training_contract": types.SimpleNamespace(encode_complete=lambda *args: altered)}):
            with self.assertRaisesRegex(ValueError, "Saved language-loss proof"):
                self.engine.run_engine(self.runtime, plan=self.plan, attempt=2)

    def test_started_english_request_is_indeterminate_never_retried(self):
        slot = self.plan[3]
        self.engine.journal.begin(slot["key"], 1, runner.digest(retention.request(slot["row"])))
        with patch.object(retention, "measure", side_effect=self.measured):
            self.engine.run_engine(self.runtime, plan=self.plan, attempt=2)
        self.assertEqual(self.engine.journal.get(slot["key"])["state"], "indeterminate")
        self.assertNotIn((slot["case_id"], slot["variant"]), self.runtime.loss_calls)
        metric = self.report()["mechanical_metrics"]["base"]["language_loss"][retention.ENGLISH]
        self.assertEqual(metric["indeterminate_rows"], 1)
        self.assertFalse(metric["complete_coverage"])


class NativeCPUChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional local CPU PyTorch environment required")
    def test_real_cross_entropy_shift_and_mask_without_a_model(self):
        import torch
        labels = torch.tensor([[-100] * 129 + [2, 3]], device="cpu")
        result = retention.shifted_masked_nll(torch.zeros((1, 131, 4), device="cpu"), labels)
        self.assertEqual(result["predicted_tokens"], 2)
        self.assertAlmostEqual(result["summed_nll"], 2 * math.log(4), places=6)
        logits = torch.tensor([[[20., 0., 0.], [0., 5., 0.], [0., 0., 2.], [20., 0., 0.]]], device="cpu")
        result = retention.shifted_masked_nll(logits, torch.tensor([[-100, -100, 1, 2]], device="cpu"))
        self.assertEqual(result["predicted_tokens"], 2)
        self.assertAlmostEqual(result["summed_nll"], math.log1p(2 * math.exp(-5)) + math.log1p(2 * math.exp(-2)), places=6)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Optional local native tokenizer environment required")
    def test_all_native_heldout_rows_and_exact_target_denominators(self):
        # Isolate the pinned helper imports from other tests' synthetic runtime modules.
        import os
        import subprocess
        program = """
import json, sys
from pathlib import Path
import run_understanding_r5_evaluation as runner
root = Path(sys.argv[1])
result = runner.preflight(root/'tmp/understanding-r2-20261105/eval-bundle-v2',
    root/'tmp/understanding-r3-20261106/evaluation-v2', r2_pack=root/'tmp/understanding-r2-20261105/pack',
    english_validation=root/'tmp/corpus-ready/twi-alignment-v3-20260929/english-retention/validation.jsonl')
assert result['ready'] and not result['model_loaded'] and not result['gpu_used']
print(json.dumps(result['language_loss']['cohorts']))
"""
        completed = subprocess.run([sys.executable, "-c", program, str(fixtures.ORIGINAL)],
            env={**os.environ, "PYTHONPATH": str(Path(runner.__file__).parent), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            check=True, capture_output=True, text=True, timeout=120)
        cohorts = json.loads(completed.stdout)
        for cohort, rows, tokens in ((retention.TWI, 343, 9315), (retention.ENGLISH, 44, 8534)):
            self.assertEqual(cohorts[cohort], {"expected_rows": rows, "ready_rows": rows,
                                             "error_rows": 0, "predicted_tokens": tokens})


if __name__ == "__main__":
    unittest.main()
