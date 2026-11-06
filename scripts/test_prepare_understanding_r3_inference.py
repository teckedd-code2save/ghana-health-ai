"""Synthetic offline receipts/readback only. No trained weights or remote calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r3_inference as builder
import r2_inference as r2
import r3_inference as r3
import understanding_r2_demo as demo
from test_understanding_r2_demo import manifest


def fixture(root):
    parent = root / "r2"
    parent.mkdir()
    (parent / "code").mkdir()
    (parent / "code/r2_inference.py").write_bytes(Path(r2.__file__).read_bytes())
    for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json"):
        r2.save(parent / "tokenizer" / name, {"synthetic_fixture": True})
    original = manifest()
    original["variants"]["r2_meaning"] = {"directory": r3.R2_DIRECTORY, "files": {
        "adapter_config.json": r3.R2_CONFIG, "adapter_model.safetensors": r3.R2_WEIGHTS}}
    original["remote_files"] = {}
    original["files"] = {p.relative_to(parent).as_posix(): r2.sha(p) for p in parent.rglob("*") if p.is_file()}
    r2.save(parent / "manifest.json", original)
    parent_sha = r2.sha(parent / "manifest.json")
    readback = root / "readback"
    identifier = "understanding_20261106T120000Z"
    remote = "/artifacts/understanding-adaptation-r3/" + identifier
    physical = "/__modal/volumes/synthetic-fixture/understanding-adaptation-r3/" + identifier
    run = readback / remote.removeprefix("/artifacts/")
    data = {"version": "understanding-adaptation-r3", "initialization": r3.INITIALIZATION}
    r2.save(run / "inputs/manifest.json", data)
    data_sha = r2.sha(run / "inputs/manifest.json")
    stage = {"name": "responses", "optimizer_updates": 4, "scheduled_sequences": 8,
             "effective_target_tokens": 16, "sequence_tokens": 24}
    execution = {"mode": "train", "version": "understanding-adaptation-r3", "runtime": r2.RUNTIME,
                 "initialization": r3.INITIALIZATION, "data_sha256": data_sha, "expected_stages": [stage]}
    r2.save(run / "execution.json", execution)
    execution_sha = r2.sha(run / "execution.json")
    r2.save(run / "loading.json", {"runtime": r2.RUNTIME, "weight_sha256": r2.BASE_SHA,
        "initialization": r3.INITIALIZATION, "initial_adapter_values_exact": True, "frozen_base": True,
        "r2_optimizer_loaded": False, "new_vocabulary": False})
    adapter = run / "responses-adapter"
    config = {"base_model_name_or_path": r2.MODEL, "revision": r2.REVISION, "peft_type": "LORA",
              "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "lora_dropout": .05,
              "bias": "none", "target_modules": sorted(r2.TARGETS)}
    r2.save(adapter / "adapter_config.json", config)
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic fixture bytes, not model weights")
    contract = {"run_sha256": execution_sha, "data_sha256": data_sha}
    r2.save(run / "responses/contract.json", contract)
    cp = run / "responses/attempt-fixture/update-00000004"
    r2.save(cp / "manifest.json", {"identity": {"run_sha256": r2.digest(contract), "data_sha256": data_sha},
                                   "state_sha256": "f" * 64})
    r2.save(cp / "COMMITTED.json", {"manifest_sha256": r2.sha(cp / "manifest.json")})
    checkpoint = {"directory": physical + "/responses/attempt-fixture/update-00000004",
                  "manifest_sha256": r2.sha(cp / "manifest.json"), "state_sha256": "f" * 64,
                  "contract_sha256": r2.sha(run / "responses/contract.json")}
    export = {"stage": "responses", "directory": physical + "/responses-adapter",
              "checkpoint_adapter_tensors_verified": True, "completed_updates": 4,
              "initialization": r3.INITIALIZATION, "checkpoint": checkpoint,
              "files": {p.name: r2.sha(p) for p in adapter.iterdir()}}
    r2.save(run / "adapters.json", [export])
    status = {"state": "completed", "training_completed": True, "mode": "train", "run_id": identifier,
              "execution_sha256": execution_sha, "data_sha256": data_sha, "remote_directory": remote,
              "physical_directory": physical, "initialization": r3.INITIALIZATION, "adapters": [export],
              "stages": [{"stage": "responses", "status": "completed", "completion_verified": True,
                          "checkpoint": checkpoint["directory"], "position": {"completed_updates": 4,
                          "completed_microbatches": 8, "microbatches_in_update": 0,
                          "exposure": {"effective_target_tokens": 16, "input_tokens": 24}}}]}
    r2.save(run / "status.json", status)
    r2.save(root / "terminal.json", status)
    r2.save(root / "receipt.json", {"state": "submitted", "mode": "train", "run_id": identifier,
        "call_id": "fc-synthetic", "volume": r3.VOLUME, "remote_directory": remote.removeprefix("/artifacts"),
        "execution_sha256": execution_sha, "data_sha256": data_sha})
    return parent, parent_sha, readback, run


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent, self.parent_sha, self.readback, self.run = fixture(self.root)
        self.pin = patch.object(r3, "R2_BUNDLE_SHA", self.parent_sha)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def build(self, name="bundle"):
        return builder.prepare(self.parent, self.parent_sha, self.root / "receipt.json", self.root / "terminal.json",
                               self.readback, self.root / name)

    def alter(self, path, change):
        value = r2.read(path)
        change(value)
        r2.save(path, value)

    def test_seal_preserves_original_and_binds_readback(self):
        before = {p.relative_to(self.parent): r2.sha(p) for p in self.parent.rglob("*") if p.is_file()}
        result = self.build()
        bundle = self.root / "bundle"
        actual = r3.load_bundle(bundle, result["manifest_sha256"])
        self.assertEqual(set(actual["variants"]), set(r3.VARIANTS))
        self.assertEqual(actual["variants"]["r3_response"]["files"]["adapter_model.safetensors"],
                         r2.sha(self.run / "responses-adapter/adapter_model.safetensors"))
        self.assertFalse(result["remote_called"])
        self.assertFalse(result["deployed"])
        self.assertEqual(actual["runtime"], r2.RUNTIME)
        self.assertEqual(actual["decoding"], r2.DECODING)
        self.assertFalse((bundle / "evaluation").exists())
        self.assertEqual(before, {p.relative_to(self.parent): r2.sha(p) for p in self.parent.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_changed_or_missing_local_weights_rejected(self):
        weights = self.run / "responses-adapter/adapter_model.safetensors"
        weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "readback differs"):
            self.build()
        self.assertFalse((self.root / "bundle").exists())
        weights.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_no_receipt_or_partial_training_is_not_completion(self):
        with self.assertRaises(FileNotFoundError):
            builder.prepare(self.parent, self.parent_sha, self.root / "missing.json", self.root / "terminal.json",
                            self.readback, self.root / "bundle")
        self.alter(self.run / "status.json", lambda v: v.update(state="paused_budget", training_completed=False))
        r2.save(self.root / "terminal.json", r2.read(self.run / "status.json"))
        with self.assertRaisesRegex(ValueError, "completed terminal"):
            self.build()

    def test_terminal_mismatch_and_unverified_export_fail_closed(self):
        self.alter(self.root / "terminal.json", lambda v: v.update(state="training"))
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.build()
        status = r2.read(self.run / "status.json")
        status["adapters"][0]["checkpoint_adapter_tensors_verified"] = False
        r2.save(self.run / "adapters.json", status["adapters"])
        r2.save(self.run / "status.json", status)
        r2.save(self.root / "terminal.json", status)
        with self.assertRaisesRegex(ValueError, "Unverified final"):
            self.build()

    def test_wrong_initializer_and_incomplete_exposure_fail(self):
        self.alter(self.run / "loading.json", lambda v: v.update(initial_adapter_values_exact=False))
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.build()
        self.alter(self.run / "loading.json", lambda v: v.update(initial_adapter_values_exact=True))
        status = r2.read(self.run / "status.json")
        status["stages"][0]["position"]["completed_updates"] = 3
        r2.save(self.run / "status.json", status)
        r2.save(self.root / "terminal.json", status)
        with self.assertRaisesRegex(ValueError, "exposure"):
            self.build()

    def test_pass_snapshot_and_export_outside_run_rejected(self):
        original = r2.read(self.run / "status.json")
        for changes in ({"stage": "responses-pass1"}, {"directory": original["physical_directory"] + "/../other/responses-adapter"}):
            status = copy.deepcopy(original)
            status["adapters"][0].update(changes)
            r2.save(self.run / "adapters.json", status["adapters"])
            r2.save(self.run / "status.json", status)
            r2.save(self.root / "terminal.json", status)
            with self.assertRaises(ValueError):
                self.build()

    def test_checkpoint_commit_and_parent_identity_rejected(self):
        self.alter(self.run / "responses/attempt-fixture/update-00000004/COMMITTED.json",
                   lambda v: v.update(manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.build()
        with self.assertRaisesRegex(ValueError, "identity"):
            builder.prepare(self.parent, "0" * 64, self.root / "receipt.json", self.root / "terminal.json",
                            self.readback, self.root / "bundle")

    def test_bundle_tampering_is_rejected_before_import(self):
        result = self.build()
        runtime = self.root / "bundle/code/r3_inference.py"
        runtime.write_text("raise AssertionError('changed code executed')")
        with self.assertRaisesRegex(ValueError, "code changed"):
            demo.bound_bundle(self.root / "bundle", result["manifest_sha256"])

    def test_protocol_or_missing_evidence_cannot_be_resealed(self):
        self.build()
        bundle = self.root / "bundle"
        original = r2.read(bundle / "manifest.json")
        changed = copy.deepcopy(original)
        changed["decoding"]["max_new_tokens"] = 512
        r2.save(bundle / "manifest.json", changed)
        r2.save(bundle / "SEALED.json", {"manifest_sha256": r2.sha(bundle / "manifest.json")})
        with self.assertRaisesRegex(ValueError, "protocol"):
            r3.load_bundle(bundle, r2.sha(bundle / "manifest.json"))
        changed = copy.deepcopy(original)
        del changed["files"]["evidence/adapter-readback.json"]
        r2.save(bundle / "manifest.json", changed)
        r2.save(bundle / "SEALED.json", {"manifest_sha256": r2.sha(bundle / "manifest.json")})
        with self.assertRaisesRegex(ValueError, "required bundle evidence"):
            r3.load_bundle(bundle, r2.sha(bundle / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
