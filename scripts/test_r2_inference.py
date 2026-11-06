"""Offline protocol, completion and causal-loss checks. No remote compute."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import r2_inference as r
from run_understanding_r2_evaluation import check_completion


class InferenceTests(unittest.TestCase):
    def test_saved_prompt_hash_accepts_both_schemas_but_not_conflicts(self):
        for value in ({"prompt_ids_sha256": "a"}, {"prompt_token_ids_sha256": "a"},
                      {"prompt_ids_sha256": "a", "prompt_token_ids_sha256": "a"}):
            self.assertEqual(r.saved_prompt_hash(value), "a")
        for value in ({}, {"prompt_ids_sha256": "a", "prompt_token_ids_sha256": "b"}):
            with self.assertRaises(ValueError):
                r.saved_prompt_hash(value)

    def test_history_preserves_roles_and_rejects_invalid_end(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                   {"role": "user", "content": "continue"}]
        before = deepcopy(history)
        r.validate_messages(history)
        self.assertEqual(before, history)
        for bad in ([], history[:2], [{"role": "assistant", "content": "hello"}],
                    [{"role": "user", "content": " "}], [{"role": "user", "content": "x" * 65537}]):
            with self.assertRaises(ValueError):
                r.validate_messages(bad)

    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ("../outside", "/tmp/outside", "."):
                with self.assertRaises(ValueError):
                    r.safe(tmp, path)

    def test_causal_loss_matches_full_cross_entropy(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(13)
        logits = torch.randn(1, 270, 19)
        inputs = torch.randint(19, (1, 270))
        value = r.shifted_nll(logits, inputs)
        self.assertEqual(value["predicted_tokens"], 269)
        expected = F.cross_entropy(logits[0, :-1], inputs[0, 1:], reduction="sum").item()
        self.assertAlmostEqual(value["summed_nll"], expected, places=3)
        with self.assertRaises(ValueError):
            r.shifted_nll(logits * float("nan"), inputs)

    def test_completed_training_requires_exact_exposure(self):
        stage = {"name": "language", "optimizer_updates": 2, "scheduled_sequences": 4,
                 "effective_target_tokens": 12, "sequence_tokens": 16}
        status = {"state": "completed", "training_completed": True, "mode": "train", "data_sha256": "a",
                  "stages": [{"stage": "language", "status": "completed", "completion_verified": True,
                      "position": {"completed_updates": 2, "completed_microbatches": 4, "microbatches_in_update": 0,
                                   "exposure": {"effective_target_tokens": 12, "input_tokens": 16}}}]}
        execution = {"mode": "train", "runtime": r.RUNTIME, "data_sha256": "a"}
        check_completion(status, execution, [stage])
        for key, wrong in (("state", "training"), ("training_completed", False), ("data_sha256", "b")):
            with self.assertRaises(ValueError):
                check_completion({**status, key: wrong}, execution, [stage])
        changed = deepcopy(status)
        changed["stages"][0]["position"]["exposure"]["effective_target_tokens"] -= 1
        with self.assertRaises(ValueError):
            check_completion(changed, execution, [stage])

    def test_bundle_hash_and_protocol_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r.save(root / "evidence.json", {"ok": True})
            spec = {"directory": "/artifacts/understanding-adaptation-r2/understanding_20261105T192103Z/meaning-adapter",
                    "files": {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}}
            value = {"version": "r2-paired-inference-v1", "model": r.MODEL, "revision": r.REVISION,
                "base_sha256": r.BASE_SHA, "runtime": r.RUNTIME, "decoding": r.DECODING,
                "variants": {v: None if v == "base" else spec for v in r.VARIANTS},
                "files": {"evidence.json": r.sha(root / "evidence.json")}, "private": True, "training": False}
            r.save(root / "manifest.json", value)
            checksum = r.sha(root / "manifest.json")
            self.assertEqual(r.load_bundle(root, checksum), value)
            with self.assertRaises(ValueError):
                r.load_bundle(root, "c" * 64)
            r.save(root / "evidence.json", {"ok": False})
            with self.assertRaises(ValueError):
                r.load_bundle(root, checksum)


if __name__ == "__main__":
    unittest.main()
