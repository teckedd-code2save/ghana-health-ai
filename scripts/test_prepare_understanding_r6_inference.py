"""Synthetic R6 completion proof and preparation safety tests, CPU only."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r6_inference as builder
import r2_inference as r2
import r6_inference as r6

def fixture(root):
    readback = root / "readback"
    identifier = "understanding_20261113T152209Z"
    remote = "/artifacts/understanding-adaptation-r6/" + identifier
    physical = "/__modal/volumes/synthetic-fixture/understanding-adaptation-r6/" + identifier
    run = readback / remote.removeprefix("/artifacts/")
    for name in ("evaluation-manifest.json", "source-manifests.json", "stages.json"):
        r2.save(run / "inputs" / name, [])
    data = {"version": "understanding-adaptation-r6", "initialization": r6.INITIALIZATION,
            "artifacts": {p.name: r2.sha(p) for p in (run / "inputs").iterdir()}}
    r2.save(run / "inputs/manifest.json", data)
    data_sha = r2.sha(run / "inputs/manifest.json")
    stage = {"name": "responses", "optimizer_updates": 4, "scheduled_sequences": 8,
             "effective_target_tokens": 16, "sequence_tokens": 24, "snapshot_after_updates": [2]}
    execution = {"mode": "train", "version": "understanding-adaptation-r6", "runtime": r2.RUNTIME,
                 "initialization": r6.INITIALIZATION, "data_sha256": data_sha, "expected_stages": [stage]}
    r2.save(run / "execution.json", execution)
    execution_sha = r2.sha(run / "execution.json")
    r2.save(run / "loading.json", {"runtime": r2.RUNTIME, "weight_sha256": r2.BASE_SHA,
        "initialization": r6.INITIALIZATION, "initial_adapter_values_exact": True, "frozen_base": True,
        "r4_optimizer_loaded": False, "r5_full_state_resume": False, "new_vocabulary": False})
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
              "initialization": r6.INITIALIZATION, "checkpoint": checkpoint,
              "files": {p.name: r2.sha(p) for p in adapter.iterdir()}}
    exports = [export]
    r2.save(run / "adapters.json", exports)
    consumption = {"version": "r5-consumption-v1", "run_id": identifier, "mode": "train",
        "training_complete": True, "schedule_complete": True, "completed_updates": 4,
        "checkpoint": checkpoint, "execution_sha256": execution_sha, "data_sha256": data_sha,
        "consumed": {"row_exposures": 8, "effective_target_tokens": 16, "input_tokens": 24}}
    r2.save(run / "consumption.json", consumption)
    status = {"state": "completed", "training_completed": True, "mode": "train", "run_id": identifier,
              "execution_sha256": execution_sha, "data_sha256": data_sha, "remote_directory": remote,
              "physical_directory": physical, "initialization": r6.INITIALIZATION, "adapters": exports,
              "consumption_sha256": r2.sha(run / "consumption.json"),
              "stages": [{"stage": "responses", "status": "completed", "completion_verified": True,
                          "checkpoint": checkpoint["directory"], "position": {"completed_updates": 4,
                          "completed_microbatches": 8, "microbatches_in_update": 0,
                          "exposure": {"effective_target_tokens": 16, "input_tokens": 24}}}]}
    r2.save(run / "status.json", status)
    r2.save(root / "terminal.json", status)
    r2.save(root / "receipt.json", {"state": "submitted", "mode": "train", "run_id": identifier,
        "call_id": "fc-synthetic", "volume": r6.VOLUME, "remote_directory": remote.removeprefix("/artifacts"),
        "execution_sha256": execution_sha, "data_sha256": data_sha})
    return readback, run


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.readback, self.run = fixture(self.root)
        for name, filename in (("R6_CONFIG", "adapter_config.json"), ("R6_WEIGHTS", "adapter_model.safetensors")):
            patcher = patch.object(r6, name, r2.sha(self.run / "responses-adapter" / filename))
            patcher.start()
            self.addCleanup(patcher.stop)

    def proof(self):
        def fetch(path):
            file = r2.safe(self.readback, path.removeprefix("/artifacts/"))
            return r2.read(file), r2.sha(file)
        return r6.training_proof(r2.read(self.root / "receipt.json"), r2.read(self.root / "terminal.json"), fetch)

    def test_completed_r6_proof_is_bound_to_r5_initialization(self):
        proof = self.proof()
        self.assertEqual(proof["completed_updates"], 4)
        self.assertEqual(proof["variant"]["directory"], r6.R6_DIRECTORY)
        self.assertEqual(r6.INITIALIZATION["adapter_directory"], r6.R5_DIRECTORY)
        self.assertNotEqual(r6.R6_WEIGHTS, r6.R5_WEIGHTS)

    def test_mismatched_receipt_or_terminal_is_rejected(self):
        for name, key, value in (("receipt.json", "run_id", "understanding_20261110T103734Z"),
                                 ("receipt.json", "execution_sha256", "f" * 64),
                                 ("terminal.json", "training_completed", False)):
            file = self.root / name
            original = file.read_bytes()
            changed = r2.read(file)
            changed[key] = value
            r2.save(file, changed)
            with self.assertRaises(ValueError):
                self.proof()
            file.write_bytes(original)

    def test_checkpoint_commitment_and_consumption_changes_are_rejected(self):
        for relative in ("responses/attempt-fixture/update-00000004/COMMITTED.json",
                         "responses/contract.json", "consumption.json"):
            path = self.run / relative
            original = path.read_bytes()
            value = r2.read(path)
            key = next(iter(value))
            value[key] = "tampered"
            r2.save(path, value)
            with self.assertRaises(ValueError):
                self.proof()
            path.write_bytes(original)

    def test_inherited_optimizer_or_wrong_base_is_rejected(self):
        path = self.run / "loading.json"
        original = r2.read(path)
        for field, value in (("r5_full_state_resume", True), ("frozen_base", False),
                             ("initial_adapter_values_exact", False), ("weight_sha256", "f" * 64)):
            r2.save(path, {**original, field: value})
            with self.assertRaises(ValueError):
                self.proof()
        r2.save(path, original)

    def test_existing_bundle_is_never_overwritten(self):
        out = self.root / "existing"
        out.mkdir()
        marker = out / "keep"
        marker.write_text("old evidence")
        with self.assertRaisesRegex(ValueError, "new immutable"):
            builder.prepare(self.root / "missing", self.root / "receipt.json", self.readback, out)
        self.assertEqual(marker.read_text(), "old evidence")

    def test_parent_protocol_and_adapter_are_exact(self):
        parent = {"version": "r5-comparison-inference-v1", "model": r2.MODEL, "revision": r2.REVISION,
            "base_sha256": r2.BASE_SHA, "runtime": r2.RUNTIME, "decoding": r2.DECODING,
            "variants": {"base": None, "r5_response": {"directory": r6.R5_DIRECTORY, "files": {
                "adapter_config.json": r6.R5_CONFIG, "adapter_model.safetensors": r6.R5_WEIGHTS}}}}
        r6.check_parent(parent, r6.R5_BUNDLE_SHA)
        for mutate in (lambda m: m.update(revision="main"),
                       lambda m: m["decoding"].update(repetition_penalty=1.1),
                       lambda m: m["variants"]["r5_response"].update(directory=r6.R6_DIRECTORY)):
            changed = copy.deepcopy(parent)
            mutate(changed)
            with self.assertRaises(ValueError):
                r6.check_parent(changed, r6.R5_BUNDLE_SHA)
        with self.assertRaises(ValueError):
            r6.check_parent(parent, "f" * 64)


if __name__ == "__main__":
    unittest.main()
