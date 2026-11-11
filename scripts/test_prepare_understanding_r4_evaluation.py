import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r4_evaluation as evaluation


class EvaluationPlanTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        evaluation.data.save(path, value)

    def seal(self, root, extra, field="artifacts", name="manifest.json"):
        files = {p.relative_to(root).as_posix(): evaluation.data.sha(p)
                 for p in root.rglob("*") if p.is_file() and p != root / name}
        self.write(root / name, {**extra, field: files})
        return evaluation.data.sha(root / name)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = [self.root / name for name in ("development", "protection", "bundle", "results", "tools")]
        for root in self.inputs:
            root.mkdir()
        dev, protection, bundle, results, tools = self.inputs
        self.sentinel = "PRIVATE_FIXTURE_TEXT_NOT_FOR_EXPORT"
        decoding = {"do_sample": False, "max_new_tokens": 256}
        self.write(dev / "cases.json", [{"id": "existing-dev", "messages": [{"role": "user", "content": self.sentinel}]}])
        self.write(dev / "rollouts.json", [{"id": "existing-rollout", "user_turns": [self.sentinel, self.sentinel]}])
        for name in ("held-content.jsonl", "language-heldout.json", "language-loss.json"):
            self.write(dev / name, [{"text": self.sentinel}])
        dev_sha = self.seal(dev, {"kind": "development_not_blind_final", "training_eligible": False,
            "decoding": decoding, "counts": {"fixed_prompts": 1, "rollout_scenarios": 1, "rollout_user_turns": 2}})
        snapshot = protection / "protection/snapshot"
        snapshot.mkdir(parents=True)
        pointers = [{"collection": "english_retention", "split": "validation", "id": "held-en"},
                    {"collection": "english_retention", "split": "test", "id": "final-en"},
                    {"collection": "language", "split": "validation", "id": "held-tw"}]
        evaluation.data.write_rows(snapshot / "records.jsonl", pointers)
        self.write(snapshot / "split-ledger.jsonl", {"protected": True})
        snapshot_sha = self.seal(snapshot, {"training_authorized": False})
        self.write(protection / "protection/split-ledger.jsonl", {"protected": True})
        for name in ("held-content.jsonl", "snapshot-witnesses.jsonl"):
            self.write(protection / "protection" / name, [{"text": self.sentinel}])
        protection_sha = self.seal(protection, {"snapshot_manifest_sha256": snapshot_sha})
        self.write(bundle / "evidence.json", {"fixture": True})
        bundle_sha = self.seal(bundle, {"variants": {"r3_response": {"files": {
            "adapter_model.safetensors": evaluation.R3_ADAPTER_SHA}}}, "private": True, "training": False,
            "model": "fixture-model", "revision": "fixture-revision", "base_sha256": "a" * 64,
            "runtime": {"fixture": True}, "decoding": decoding}, "files")
        self.write(results / "summary.json", {"fixture": True})
        results_sha = self.seal(results, {"binding": {"evaluation_sha256": dev_sha,
            "inference_sha256": bundle_sha}}, "files", "results-manifest.json")
        (tools / "r2_comparison.py").write_text(
            'HISTORY_POLICY = "independent_model_histories_v1"\nraise AssertionError("Do not execute")\n')
        (tools / "response_loop_guard.py").write_text('VERSION = "repeat-block-v1"\nraise AssertionError("Do not execute")\n')
        (tools / "probe_independent_r3_comparison.py").write_text('raise AssertionError("Do not probe")\n')
        self.pins = patch.multiple(evaluation, R3_EVALUATION_SHA=dev_sha, R2_PACK_SHA=protection_sha,
                                  R3_BUNDLE_SHA=bundle_sha, R3_RESULTS_SHA=results_sha)
        self.pins.start()
        self.addCleanup(self.pins.stop)
        self.out = self.root / "plan"

    def prepare(self, out=None):
        return evaluation.prepare(*self.inputs, out or self.out)

    def test_offline_contract_preserves_inputs_and_exports_no_prompts(self):
        before = {str(p): evaluation.data.sha(p) for root in self.inputs for p in root.rglob("*") if p.is_file()}
        with patch.object(socket, "create_connection", side_effect=AssertionError("No network")), \
             patch.dict("sys.modules", {"torch": None, "modal": None, "transformers": None}):
            contract = self.prepare()
            self.assertEqual(evaluation.verify(self.out), contract)
        self.assertEqual(before, {p: evaluation.data.sha(Path(p)) for p in before})
        self.assertEqual({p.name for p in self.out.iterdir()},
                         {"manifest.json", "PREPARED.json", "evaluation-contract.json", "provenance.json"})
        self.assertNotIn(self.sentinel, "".join(p.read_text() for p in self.out.iterdir()))
        self.assertFalse(contract["launchable"])
        self.assertEqual(contract["model_calls"], 0)
        self.assertIsNone(contract["results"])

    def test_history_and_guard_cells_are_separate(self):
        contract = self.prepare()
        cells = contract["comparisons"]["cells"]
        self.assertEqual({(c["history_mode"], c["runtime_mode"]) for c in cells},
                         {(h, m) for h in ("fixed_context", "own_history_rollout") for m in ("raw", "guarded")})
        self.assertEqual(contract["comparisons"]["runtime_modes"]["guarded"]["trigger_state"], "paused_not_complete")
        self.assertFalse(contract["release"]["runtime_guard_is_semantic_gate"])

    def test_final_tests_empty_and_existing_test_split_remains_protected(self):
        contract = self.prepare()
        self.assertEqual(contract["final_test"]["cases"], [])
        self.assertFalse(contract["final_test"]["execution_authorized"])
        english = contract["measurement"]["language_retention"]["english"]
        self.assertEqual((english["validation_records"], english["protected_test_records"], english["selected_test_records"]), (1, 1, 0))
        self.assertEqual(contract["development"]["case_ids"], ["existing-dev"])

    def test_semantic_scores_require_calibration_not_heuristics(self):
        semantic = self.prepare()["measurement"]["semantic"]
        self.assertEqual(semantic["ratings"], [])
        self.assertIsNone(semantic["calibration"]["thresholds"])
        self.assertEqual(semantic["calibration"]["minimum_independent_proficient_reviewers"], 2)
        self.assertIn("word_count", semantic["forbidden_semantic_proxies"])
        self.assertIn("unable_to_judge", semantic["scale"])

    def test_changed_development_artifact_rejected_before_writes(self):
        self.write(self.inputs[0] / "cases.json", [])
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_wrong_baseline_pin_rejected(self):
        with patch.object(evaluation, "R3_BUNDLE_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Pinned manifest changed"):
                self.prepare()
        self.assertFalse(self.out.exists())

    def test_changed_protected_witness_rejected(self):
        self.write(self.inputs[1] / "protection/snapshot-witnesses.jsonl", [])
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            self.prepare()

    def test_legacy_shared_history_policy_rejected(self):
        (self.inputs[4] / "r2_comparison.py").write_text('HISTORY_POLICY = "shared_candidate_history"\n')
        with self.assertRaisesRegex(ValueError, "own-history"):
            self.prepare()

    def test_unrecognized_guard_rejected(self):
        (self.inputs[4] / "response_loop_guard.py").write_text('VERSION = "different-policy"\n')
        with self.assertRaisesRegex(ValueError, "Guard policy changed"):
            self.prepare()

    def test_fresh_output_required_and_inputs_cannot_be_output(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "fresh plan"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "inside an input"):
            self.prepare(self.inputs[0] / "new-output")

    def test_symlink_artifact_rejected(self):
        p = self.inputs[0] / "cases.json"
        elsewhere = self.root / "elsewhere.json"
        p.rename(elsewhere)
        p.symlink_to(elsewhere)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_verification_detects_source_or_contract_changes(self):
        self.prepare()
        self.write(self.out / "evaluation-contract.json", {})
        with self.assertRaisesRegex(ValueError, "Plan artifact changed"):
            evaluation.verify(self.out)
        other = self.root / "other-plan"
        self.prepare(other)
        (self.inputs[4] / "r2_comparison.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Bound source changed"):
            evaluation.verify(other)


if __name__ == "__main__":
    unittest.main()
