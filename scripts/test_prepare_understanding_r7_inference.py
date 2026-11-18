"""Synthetic future R7 proof fixtures; never a saved model or real training claim."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r7_inference as builder
import r2_inference as r2
import r7_inference as r7

def fixture(root):
    readback = root / "readback"
    identifier = "understanding_20990101T000000Z"
    remote = "/artifacts/understanding-adaptation-r7/" + identifier
    physical = "/__modal/volumes/synthetic-fixture/understanding-adaptation-r7/" + identifier
    run = readback / remote.removeprefix("/artifacts/")
    for name in ("evaluation-manifest.json", "source-manifests.json", "stages.json"):
        r2.save(run / "inputs" / name, [])
    data = {"version": "understanding-adaptation-r7", "model": r2.MODEL, "revision": r2.REVISION,
            "private_research_only": True, "production_eligible": False, "initialization": r7.INITIALIZATION,
            "artifacts": {p.name: r2.sha(p) for p in (run / "inputs").iterdir()}}
    r2.save(run / "inputs/manifest.json", data)
    data_sha = r2.sha(run / "inputs/manifest.json")
    smoke_id = "understanding_20981231T000000Z"
    smoke_root = "/artifacts/understanding-adaptation-r7/" + smoke_id
    smoke_dir = readback / smoke_root.removeprefix("/artifacts/")
    r2.save(smoke_dir / "execution.json", {"mode": "smoke", "version": "understanding-adaptation-r7",
        "runtime": r2.RUNTIME, "initialization": r7.INITIALIZATION, "data_sha256": data_sha})
    r2.save(smoke_dir / "status.json", {"state": "completed", "smoke_passed": True, "mode": "smoke",
        "run_id": smoke_id, "remote_directory": smoke_root, "data_sha256": data_sha,
        "execution_sha256": r2.sha(smoke_dir / "execution.json")})
    r2.save(smoke_dir / "smoke.json", {"passed": True, "real_weight_backward": True,
        "initial_adapter_values_exact": True, "native_encoding": {"all_rows_verified": 1}})
    stage = {"name": "responses", "optimizer_updates": 4, "scheduled_sequences": 8,
             "effective_target_tokens": 16, "sequence_tokens": 24, "snapshot_after_updates": [2]}
    execution = {"mode": "train", "private_research_only": True,
                 "smoke": {"run_id": smoke_id, "execution_sha256": r2.sha(smoke_dir / "execution.json"),
                           "status_sha256": r2.sha(smoke_dir / "status.json"), "smoke_sha256": r2.sha(smoke_dir / "smoke.json")},
                 "version": "understanding-adaptation-r7", "runtime": r2.RUNTIME,
                 "initialization": r7.INITIALIZATION, "data_sha256": data_sha, "expected_stages": [stage]}
    r2.save(run / "execution.json", execution)
    execution_sha = r2.sha(run / "execution.json")
    r2.save(run / "loading.json", {"runtime": r2.RUNTIME, "weight_sha256": r2.BASE_SHA,
        "initialization": r7.INITIALIZATION, "initial_adapter_values_exact": True, "frozen_base": True,
        "r6_optimizer_loaded": False, "r7_full_state_resume": False, "trainable_parameters": 43278336, "new_vocabulary": False})
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
              "initialization": r7.INITIALIZATION, "checkpoint": checkpoint,
              "files": {p.name: r2.sha(p) for p in adapter.iterdir()}}
    exports = [export]
    r2.save(run / "adapters.json", exports)
    consumption = {"version": "r7-consumption-v1", "run_id": identifier, "mode": "train",
        "training_complete": True, "schedule_complete": True, "completed_updates": 4,
        "checkpoint": checkpoint, "execution_sha256": execution_sha, "data_sha256": data_sha,
        "planned": {"row_exposures": 8, "effective_target_tokens": 16, "input_tokens": 24},
        "consumed": {"row_exposures": 8, "effective_target_tokens": 16, "input_tokens": 24}}
    r2.save(run / "consumption.json", consumption)
    status = {"state": "completed", "production_promoted": False, "training_completed": True, "mode": "train", "run_id": identifier,
              "execution_sha256": execution_sha, "data_sha256": data_sha, "remote_directory": remote,
              "physical_directory": physical, "initialization": r7.INITIALIZATION, "adapters": exports,
              "consumption_sha256": r2.sha(run / "consumption.json"),
              "stages": [{"stage": "responses", "status": "completed", "completion_verified": True,
                          "checkpoint": checkpoint["directory"], "position": {"completed_updates": 4,
                          "completed_microbatches": 8, "microbatches_in_update": 0,
                          "exposure": {"effective_target_tokens": 16, "input_tokens": 24}}}]}
    r2.save(run / "status.json", status)
    r2.save(root / "terminal.json", status)
    r2.save(root / "receipt.json", {"state": "submitted", "mode": "train", "run_id": identifier,
        "call_id": "fc-synthetic", "volume": r7.VOLUME, "remote_directory": remote.removeprefix("/artifacts"),
        "execution_sha256": execution_sha, "data_sha256": data_sha})
    return readback, run


class ProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.readback, self.run = fixture(self.root)
        self.receipt = self.root / "receipt.json"

    def proof(self):
        def fetch(path):
            file = r2.safe(self.readback, path.removeprefix("/artifacts/"))
            return r2.read(file), r2.sha(file)
        return r7.training_proof(r2.read(self.receipt), r2.read(self.root / "terminal.json"), fetch)

    def test_final_export_is_derived_from_verified_metadata_not_placeholder_hash(self):
        proof = self.proof()
        self.assertEqual(proof["completed_updates"], 4)
        self.assertEqual(proof["variant"]["files"]["adapter_model.safetensors"],
                         r2.sha(self.run / "responses-adapter/adapter_model.safetensors"))
        self.assertNotEqual(proof["variant"]["files"]["adapter_model.safetensors"], r7.R6_WEIGHTS)
        self.assertEqual(r7.INITIALIZATION["adapter_directory"], r7.R6_DIRECTORY)

    def test_placeholder_smoke_wrong_receipt_and_nonfinal_run_are_rejected(self):
        original = r2.read(self.receipt)
        for change in ({"run_id": "PENDING_R7"}, {"mode": "smoke"}, {"execution_sha256": "0" * 64},
                       {"remote_directory": "/understanding-adaptation-r6/" + original["run_id"]}):
            with self.subTest(change=change):
                r2.save(self.receipt, {**original, **change})
                with self.assertRaises(ValueError):
                    self.proof()
        r2.save(self.receipt, original)
        terminal = r2.read(self.root / "terminal.json")
        for change in ({"training_completed": False}, {"state": "paused_budget"}, {"production_promoted": True}):
            r2.save(self.root / "terminal.json", {**terminal, **change})
            with self.assertRaises(ValueError):
                self.proof()

    def test_changed_checkpoint_consumption_loading_and_initialization_rejected(self):
        mutations = [
            ("responses/attempt-fixture/update-00000004/COMMITTED.json", "manifest_sha256", "0" * 64),
            ("responses/contract.json", "data_sha256", "0" * 64),
            ("consumption.json", "training_complete", False),
            ("loading.json", "r6_optimizer_loaded", True),
            ("loading.json", "initial_adapter_values_exact", False),
            ("loading.json", "weight_sha256", "0" * 64),
            ("loading.json", "new_vocabulary", True),
        ]
        for name, key, value in mutations:
            path = self.run / name
            original = r2.read(path)
            r2.save(path, {**original, key: value})
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                self.proof()
            r2.save(path, original)

    def test_partial_export_and_unchanged_weights_cannot_be_final(self):
        path = self.run / "adapters.json"
        original = r2.read(path)
        terminal = r2.read(self.root / "terminal.json")
        for mutate in (
            lambda e: e.update(directory=e["directory"].replace("responses-adapter", "responses-pass1-adapter")),
            lambda e: e["files"].update({"adapter_model.safetensors": r7.R6_WEIGHTS}),
            lambda e: e.update(checkpoint_adapter_tensors_verified=False),
        ):
            changed = copy.deepcopy(original)
            mutate(changed[0])
            r2.save(path, changed)
            current = {**terminal, "adapters": changed}
            r2.save(self.run / "status.json", current)
            r2.save(self.root / "terminal.json", current)
            with self.assertRaises(ValueError):
                self.proof()

    def make_bundle(self):
        proof = self.proof()
        adapters = {"r7_response": proof["variant"], "r6_response": {
            "directory": r7.R6_DIRECTORY, "files": {"adapter_config.json": r7.R6_CONFIG,
                                                   "adapter_model.safetensors": r7.R6_WEIGHTS}}}
        r2.save(self.readback / "adapter-readback.json", {
            key: {**value, "verified": True, "bytes_read": {name: 1 for name in value["files"]},
                  "method": "SYNTHETIC CPU TEST ONLY; not remote evidence"}
            for key, value in adapters.items()})
        out = self.root / "synthetic-bundle"
        result = builder.prepare(builder.ROOT / "tmp/r6-demo/bundle-v1", self.receipt, self.readback, out,
                                 receipt_sha=r2.sha(self.receipt))
        return out, result

    def test_synthetic_bundle_roundtrip_and_r6_identity_unchanged(self):
        bundle, result = self.make_bundle()
        manifest = r7.load_bundle(bundle, result["manifest_sha256"])
        self.assertEqual(list(manifest["variants"]), ["base", "r6_response", "r7_response"])
        self.assertEqual(manifest["default_variant"], "r7_response")
        self.assertEqual(manifest["variants"]["r6_response"]["files"]["adapter_model.safetensors"], r7.R6_WEIGHTS)
        self.assertFalse(result["deployed"])
        self.assertFalse(result["remote_called"])
        self.assertEqual(manifest["decoding"], r2.DECODING)

    def test_receipt_seal_and_new_output_required_before_readback(self):
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            builder.collect(self.receipt, self.root / "unused", receipt_sha="0" * 64)
        out = self.root / "existing"
        out.mkdir()
        with self.assertRaisesRegex(ValueError, "new immutable"):
            builder.prepare(self.root / "missing", self.receipt, self.readback, out, receipt_sha=r2.sha(self.receipt))

    def test_inherited_pass_snapshot_is_not_mistaken_for_final_export(self):
        exports = r2.read(self.run / 'adapters.json')
        earlier = copy.deepcopy(exports[0])
        earlier['directory'] = '/__modal/volumes/synthetic-fixture/understanding-adaptation-r7/understanding_20981230T000000Z/responses-pass1-adapter'
        earlier['stage'] = 'responses-pass1'
        earlier['completed_updates'] = 2
        exports.insert(0, earlier)
        r2.save(self.run / 'adapters.json', exports)
        status = r2.read(self.run / 'status.json')
        status['adapters'] = exports
        r2.save(self.run / 'status.json', status)
        r2.save(self.root / 'terminal.json', status)
        self.assertEqual(self.proof()['completed_updates'], 4)

    def test_smoke_bytes_must_match_training_smoke_reference(self):
        path = self.readback / 'understanding-adaptation-r7/understanding_20981231T000000Z/smoke.json'
        value = r2.read(path)
        value['passed'] = False
        r2.save(path, value)
        with self.assertRaisesRegex(ValueError, 'smoke evidence'):
            self.proof()

    def test_altered_code_rejected_before_execution(self):
        bundle, result = self.make_bundle()
        (bundle / "code/r7_inference.py").write_text('raise AssertionError("MUST NOT RUN")\\n')
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            r7.bound_bundle(bundle, result["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
