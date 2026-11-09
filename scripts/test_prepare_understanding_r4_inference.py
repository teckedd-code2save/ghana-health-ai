"""Synthetic offline receipts/readback only. No trained weights or remote calls."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r4_inference as builder
import r2_inference as r2
import r4_inference as r4

REFERENCE = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai/tmp/understanding-r3-20261106/inferencebundle-v1")
HELPERS = Path("/Users/welcome/.codex/worktrees/r2-train/ghana-health-ai")


def fixture(root):
    if not REFERENCE.is_dir() or not (HELPERS / "scripts/guarded_r3_runtime.py").is_file():
        raise unittest.SkipTest("Original pinned R3 bundle and guarded helper source are required")
    parent, parent_sha = REFERENCE, r4.R3_BUNDLE_SHA
    readback = root / "readback"
    identifier = "understanding_20261106T120000Z"
    remote = "/artifacts/understanding-adaptation-r4/" + identifier
    physical = "/__modal/volumes/synthetic-fixture/understanding-adaptation-r4/" + identifier
    run = readback / remote.removeprefix("/artifacts/")
    for name in ("evaluation-manifest.json", "source-manifests.json", "stages.json"):
        r2.save(run / "inputs" / name, [])
    data = {"version": "understanding-adaptation-r4", "initialization": r4.INITIALIZATION,
            "dataset_snapshots": [], "artifacts": {p.name: r2.sha(p) for p in (run / "inputs").iterdir()}}
    r2.save(run / "inputs/manifest.json", data)
    data_sha = r2.sha(run / "inputs/manifest.json")
    stage = {"name": "responses", "optimizer_updates": 4, "scheduled_sequences": 8,
             "effective_target_tokens": 16, "sequence_tokens": 24, "snapshot_after_updates": [2]}
    execution = {"mode": "train", "version": "understanding-adaptation-r4", "runtime": r2.RUNTIME,
                 "initialization": r4.INITIALIZATION, "data_sha256": data_sha, "expected_stages": [stage]}
    r2.save(run / "execution.json", execution)
    execution_sha = r2.sha(run / "execution.json")
    r2.save(run / "loading.json", {"runtime": r2.RUNTIME, "weight_sha256": r2.BASE_SHA,
        "initialization": r4.INITIALIZATION, "initial_adapter_values_exact": True, "frozen_base": True,
        "r3_optimizer_loaded": False, "new_vocabulary": False})
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
              "initialization": r4.INITIALIZATION, "checkpoint": checkpoint,
              "files": {p.name: r2.sha(p) for p in adapter.iterdir()}}
    pass_adapter = run / "responses-pass1-adapter"
    r2.save(pass_adapter / "adapter_config.json", config)
    (pass_adapter / "adapter_model.safetensors").write_bytes(b"synthetic pass-one weights")
    pass_cp = run / "responses/attempt-fixture/update-00000002"
    r2.save(pass_cp / "manifest.json", {"identity": {"run_sha256": r2.digest(contract), "data_sha256": data_sha},
                                      "state_sha256": "e" * 64})
    r2.save(pass_cp / "COMMITTED.json", {"manifest_sha256": r2.sha(pass_cp / "manifest.json")})
    pass_export = {**export, "stage": "responses-pass1", "directory": physical + "/responses-pass1-adapter",
        "completed_updates": 2, "files": {p.name: r2.sha(p) for p in pass_adapter.iterdir()},
        "checkpoint": {**checkpoint, "directory": physical + "/responses/attempt-fixture/update-00000002",
                       "manifest_sha256": r2.sha(pass_cp / "manifest.json"), "state_sha256": "e" * 64}}
    exports = [export, pass_export]
    r2.save(run / "adapters.json", exports)
    consumption = {"version": "r4-consumption-v1", "run_id": identifier, "mode": "train",
        "training_complete": True, "schedule_complete": True, "completed_updates": 4,
        "checkpoint": checkpoint, "execution_sha256": execution_sha, "data_sha256": data_sha,
        "consumed": {"row_exposures": 8, "effective_target_tokens": 16, "input_tokens": 24}}
    r2.save(run / "consumption.json", consumption)
    status = {"state": "completed", "training_completed": True, "mode": "train", "run_id": identifier,
              "execution_sha256": execution_sha, "data_sha256": data_sha, "remote_directory": remote,
              "physical_directory": physical, "initialization": r4.INITIALIZATION, "adapters": exports,
              "consumption_sha256": r2.sha(run / "consumption.json"),
              "stages": [{"stage": "responses", "status": "completed", "completion_verified": True,
                          "checkpoint": checkpoint["directory"], "position": {"completed_updates": 4,
                          "completed_microbatches": 8, "microbatches_in_update": 0,
                          "exposure": {"effective_target_tokens": 16, "input_tokens": 24}}}]}
    r2.save(run / "status.json", status)
    r2.save(root / "terminal.json", status)
    r2.save(root / "receipt.json", {"state": "submitted", "mode": "train", "run_id": identifier,
        "call_id": "fc-synthetic", "volume": r4.VOLUME, "remote_directory": remote.removeprefix("/artifacts"),
        "execution_sha256": execution_sha, "data_sha256": data_sha})
    return parent, parent_sha, readback, run


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent, self.parent_sha, self.readback, self.run = fixture(self.root)

    def build(self, name="bundle", **options):
        return builder.prepare(self.parent, self.parent_sha, self.root / "receipt.json", self.root / "terminal.json",
                               self.readback, self.root / name, helpers_root=HELPERS, **options)

    def alter(self, path, change):
        value = r2.read(path)
        change(value)
        r2.save(path, value)

    def test_seal_preserves_original_and_binds_readback(self):
        before = {p.relative_to(self.parent): r2.sha(p) for p in self.parent.rglob("*") if p.is_file()}
        result = self.build()
        bundle = self.root / "bundle"
        actual = r4.load_bundle(bundle, result["manifest_sha256"])
        self.assertEqual(set(actual["variants"]), set(r4.VARIANTS))
        self.assertEqual(actual["variants"]["r4_response"]["files"]["adapter_model.safetensors"],
                         r2.sha(self.run / "responses-adapter/adapter_model.safetensors"))
        self.assertFalse(result["remote_called"])
        self.assertFalse(result["deployed"])
        self.assertEqual(actual["runtime"], r2.RUNTIME)
        self.assertEqual(actual["decoding"], r2.DECODING)
        self.assertEqual(actual["variants"]["r3_response"]["files"]["adapter_model.safetensors"], r4.R3_WEIGHTS)
        self.assertEqual(actual["r3_bundle_sha256"], r4.R3_BUNDLE_SHA)
        for name, checksum in r4.HELPER_HASHES.items():
            self.assertEqual(r2.sha(bundle / "code" / name), checksum)
            self.assertEqual((bundle / "code" / name).read_bytes(), (HELPERS / "scripts" / name).read_bytes())
        self.assertFalse((bundle / "evaluation").exists())
        self.assertEqual(before, {p.relative_to(self.parent): r2.sha(p) for p in self.parent.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_optional_pass1_has_independent_committed_proof(self):
        result = self.build(include_pass1=True)
        value = r4.load_bundle(self.root / "bundle", result["manifest_sha256"])
        self.assertEqual(set(value["variants"]), set(r4.VARIANTS) | {"r4_pass1"})
        self.assertEqual(value["r4_training"]["pass1"]["completed_updates"], 2)
        self.assertEqual(value["r4_training"]["completed_updates"], 4)
        self.assertNotEqual(value["variants"]["r4_response"], value["variants"]["r4_pass1"])
        self.alter(self.run / "responses/attempt-fixture/update-00000002/COMMITTED.json",
                   lambda v: v.update(manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.build("bad-pass1", include_pass1=True)
        self.build("final-without-pass1")

    def test_consumption_must_be_bound_to_final_checkpoint_and_counts(self):
        path = self.run / "consumption.json"
        original = r2.read(path)
        for mutation in (lambda v: v.update(training_complete=False),
                         lambda v: v["checkpoint"].update(state_sha256="0" * 64),
                         lambda v: v["consumed"].update(effective_target_tokens=17)):
            changed = copy.deepcopy(original)
            mutation(changed)
            r2.save(path, changed)
            status = r2.read(self.run / "status.json")
            status["consumption_sha256"] = r2.sha(path)
            r2.save(self.run / "status.json", status)
            r2.save(self.root / "terminal.json", status)
            with self.assertRaisesRegex(ValueError, "Consumption"):
                self.build()

    def test_bound_import_uses_sealed_sibling_and_does_not_write_bytecode(self):
        result = self.build()
        bundle = self.root / "bundle"
        with patch.dict(sys.modules, {"r2_inference": object()}), patch.object(sys, "dont_write_bytecode", False):
            manifest, module = r4.bound_bundle(bundle, result["manifest_sha256"])
        self.assertEqual(manifest["version"], r4.VERSION)
        self.assertEqual(Path(module.base.__file__).resolve(), (bundle / "code/r2_inference.py").resolve())
        self.assertFalse(list(bundle.rglob("__pycache__")))

    def test_actor_cannot_reuse_r2_or_r3_name(self):
        for name in ("ghana-understanding-r2-demo-private", "ghana-understanding-r3-demo-private"):
            with self.assertRaisesRegex(ValueError, "separate private R4"):
                self.build(app_name=name)

    def test_guarded_helpers_cannot_be_silently_replaced(self):
        helpers = self.root / "changed-helpers"
        for name in r4.HELPER_HASHES:
            path = helpers / "scripts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((HELPERS / "scripts" / name).read_bytes())
        (helpers / "scripts/response_loop_guard.py").write_text("raise AssertionError('unreviewed policy')")
        with self.assertRaisesRegex(ValueError, "helper changed"):
            builder.prepare(self.parent, self.parent_sha, self.root / "receipt.json", self.root / "terminal.json",
                            self.readback, self.root / "bundle", helpers_root=helpers)
        self.assertFalse((self.root / "bundle").exists())

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
                            self.readback, self.root / "bundle", helpers_root=HELPERS)
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
        with self.assertRaisesRegex(ValueError, "Unverified response"):
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
                            self.readback, self.root / "bundle", helpers_root=HELPERS)

    def test_bundle_tampering_is_rejected_before_import(self):
        result = self.build()
        runtime = self.root / "bundle/code/r4_inference.py"
        runtime.write_text("raise AssertionError('changed code executed')")
        with self.assertRaisesRegex(ValueError, "code changed"):
            r4.bound_bundle(self.root / "bundle", result["manifest_sha256"])

    def test_protocol_or_missing_evidence_cannot_be_resealed(self):
        self.build()
        bundle = self.root / "bundle"
        original = r2.read(bundle / "manifest.json")
        changed = copy.deepcopy(original)
        changed["decoding"]["max_new_tokens"] = 512
        r2.save(bundle / "manifest.json", changed)
        r2.save(bundle / "SEALED.json", {"manifest_sha256": r2.sha(bundle / "manifest.json")})
        with self.assertRaisesRegex(ValueError, "protocol"):
            r4.load_bundle(bundle, r2.sha(bundle / "manifest.json"))
        changed = copy.deepcopy(original)
        del changed["files"]["evidence/adapter-readback.json"]
        r2.save(bundle / "manifest.json", changed)
        r2.save(bundle / "SEALED.json", {"manifest_sha256": r2.sha(bundle / "manifest.json")})
        with self.assertRaisesRegex(ValueError, "required bundle evidence"):
            r4.load_bundle(bundle, r2.sha(bundle / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
