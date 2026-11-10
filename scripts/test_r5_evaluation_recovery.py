"""Offline terminal-proof and immutable recovery-controller tests. No Modal calls."""
from pathlib import Path
from contextlib import nullcontext
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import modal
import run_understanding_r5_evaluation as runner
import recover_understanding_r5_evaluation as recovery


def remote_error(kind=ValueError):
    try:
        exec(compile("raise error", "<ta-" + "0" * 26 + ">:sealed/core.py", "exec"), {"error": kind("Attempt already claimed")})
    except Exception as error:
        return error


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.call = Mock()
        self.patch = patch.object(modal.FunctionCall, "from_id", return_value=self.call)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_deserialized_remote_exceptions_prove_failure(self):
        for kind in (ValueError, RuntimeError, TimeoutError, ConnectionError):
            with self.subTest(kind=kind):
                self.call.get.side_effect = remote_error(kind)
                proof = runner.call_status("fc-test")
                self.assertTrue(proof["terminal"])
                self.assertTrue(proof["remote_traceback"])
                self.assertEqual(proof["state"], "remote_failed")

    def test_same_local_errors_and_expired_missing_transport_fail_closed(self):
        for error in (ValueError("Attempt already claimed"), RuntimeError("network"), ConnectionError("down"),
                      modal.exception.NotFoundError("missing"), modal.exception.OutputExpiredError("expired"),
                      modal.exception.ExecutionError("deserialize failed")):
            with self.subTest(error=error):
                self.call.get.side_effect = error
                proof = runner.call_status("fc-test")
                self.assertFalse(proof["terminal"])
                self.assertEqual(proof["state"], "unknown")
        self.call.get_call_graph.assert_not_called()

    def test_polling_timeout_is_pending_not_terminal(self):
        for error in (TimeoutError(), modal.exception.TimeoutError()):
            self.call.get.side_effect = error
            self.assertEqual(runner.call_status("fc-test")["state"], "pending")
            self.assertFalse(runner.terminal("fc-test"))

    def test_remote_wrappers_and_returned_failure_remain_terminal(self):
        for error in (modal.exception.RemoteError("failed"), modal.exception.FunctionTimeoutError("deadline")):
            self.call.get.side_effect = error
            self.assertTrue(runner.terminal("fc-test"))
        self.call.get.side_effect = None
        self.call.get.return_value = {"state": "failed", "error": "intentional fixture"}
        self.assertTrue(runner.terminal("fc-test"))

    def test_missing_traceback_helper_does_not_guess_from_error_message(self):
        self.call.get.side_effect = remote_error()
        with patch("modal._traceback.traceback_contains_remote_call", side_effect=AttributeError("unsupported")):
            self.assertFalse(runner.terminal("fc-test"))

    def test_status_persists_remote_failure_but_not_ambiguous_local_error(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / "launch.json"
            runner.save(receipt, {"call_id": "fc-test", "profile": "fixture"})
            with patch.dict("os.environ", {}):
                self.call.get.side_effect = ConnectionError("offline")
                self.assertEqual(runner.remote_status(receipt)["state"], "unknown")
                self.assertFalse(receipt.with_suffix(".result.json").exists())
                self.call.get.side_effect = remote_error()
                self.assertEqual(runner.remote_status(receipt)["state"], "remote_failed")
                self.assertTrue(runner.read(receipt.with_suffix(".result.json"))["terminal"])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / "execution"
        self.package.mkdir()
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.checksum = "a" * 64
        self.previous = {"state": "submitted", "run_id": "r5_eval_20261110T112211Z", "attempt": 1,
            "call_id": "fc-" + "0" * 26, "execution_sha256": self.checksum, "package": str(self.package),
            "profile": "fixture", "volume": runner.VOLUME,
            "remote_directory": runner.PREFIX + "/r5_eval_20261110T112211Z"}
        runner.save(self.root / "launch.json", self.previous)
        binding = {"run_id": self.previous["run_id"], "execution_sha256": self.checksum}
        runner.save(self.snapshot / "journal/binding.json", binding)
        proof_path = self.snapshot / "runtime/attempt-1.json"
        runner.save(proof_path, {"binding": binding})
        record = {"id": "finished", "attempt": 1, "kind": "fixed", "state": "complete",
                  "binding_sha256": runner.digest(binding), "runtime_proof_sha256": runner.sha(proof_path)}
        runner.save(self.snapshot / "journal/records" / (runner.digest(record["id"]) + ".json"),
                    {"record": record, "sha256": runner.digest(record)})
        runner.save(self.snapshot / "journal/attempts/1" / (runner.digest("interrupted") + ".json"),
                    {"id": "interrupted", "state": "started", "inputs_sha256": "b" * 64})
        runner.save(self.snapshot / "collection.json", {"receipt": self.previous, "files": runner.files(self.snapshot)})
        self.execution = {"files": {"code/r5_retention.py": runner.sha(runner.retention.__file__),
                                    "code/evaluate_understanding_r5.py": "c" * 64,
                                    "code/run_understanding_r5_evaluation.py": "d" * 64}}
        patcher = patch.object(runner, "verify", return_value=(self.execution,))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.controller = self.root / "controller"
        self.prepared = recovery.prepare(self.package, self.checksum, self.root / "launch.json", self.snapshot, self.controller)
        self.sha = self.prepared["controller_sha256"]

    def test_offline_prepare_and_verify_bind_original_execution_and_start_markers(self):
        contract, _, inventory = recovery.verify(self.controller, self.sha, self.package)
        self.assertEqual(contract["attempt"], 2)
        self.assertEqual(contract["execution_sha256"], self.checksum)
        self.assertEqual(inventory["finalized_records"], 1)
        self.assertEqual(inventory["started_without_result"], 1)
        self.assertFalse(self.prepared["gpu_launched"])
        self.assertEqual(contract["patch"], recovery.PATCH)

    def test_controller_bytes_cannot_be_silently_replaced(self):
        (self.controller / "code" / recovery.WORKER).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Controller files changed"):
            recovery.verify(self.controller, self.sha, self.package)

    def test_collected_data_cannot_be_mutated_or_dropped(self):
        for path in (self.snapshot / "journal/records").glob("*.json"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            recovery.snapshot_inventory(self.snapshot, self.previous)

    def test_remote_baseline_requires_exact_finished_and_inflight_inventory(self):
        snapshot = runner.read(self.controller / "snapshot.json")
        recovery.verify_baseline(self.snapshot, snapshot)
        runner.save(self.snapshot / "journal/attempts/1/extra.json", {})
        with self.assertRaisesRegex(ValueError, "Remote journal"):
            recovery.verify_baseline(self.snapshot, snapshot)

    def worker(self):
        folder = self.root / "artifacts" / self.previous["remote_directory"].lstrip("/")
        shutil.copytree(self.snapshot, folder)
        store = {self.previous["run_id"] + ":binding": self.checksum,
                 self.previous["run_id"] + ":1": self.previous["call_id"]}
        return folder, SimpleNamespace(claims=store, artifacts=Mock(), runner=SimpleNamespace(terminal=lambda _: False),
                                       evaluate=SimpleNamespace(local=Mock()))

    def execute(self, worker):
        return recovery.execute(worker, self.controller, self.sha, self.package, "fc-new",
                                artifacts_root=self.root / "artifacts")

    def test_only_terminal_callback_changes_and_sidecar_precedes_original_claim(self):
        from test_run_understanding_r5_evaluation import CORE
        folder, worker = self.worker()
        original = worker.runner.terminal
        before = {k: v for k, v in runner.files(folder).items() if recovery.protected(k)}
        class Store(dict):
            def put(self, key, value, skip_if_exists=False):
                if key in self:
                    return False
                self[key] = value
                return True
        worker.claims = Store(worker.claims)
        def local(identifier, checksum, attempt):
            self.assertIs(worker.runner.terminal, runner.terminal)
            sidecar = folder / "recovery-control/attempt-2"
            invocation = runner.read(sidecar / "invocation.json")
            self.assertEqual(invocation["controller_sha256"], self.sha)
            self.assertEqual(runner.sha(sidecar / "code" / recovery.WORKER),
                             runner.read(sidecar / "controller.json")["files"]["code/" + recovery.WORKER])
            CORE.claim_attempt(worker.claims, identifier, attempt, "fc-new", worker.runner.terminal)
            self.assertEqual((identifier, checksum, attempt), (self.previous["run_id"], self.checksum, 2))
            return {"state": "synthetic_only"}
        worker.evaluate.local.side_effect = local
        with patch.object(runner, "call_status", return_value={"terminal": True, "state": "remote_failed"}):
            self.assertEqual(self.execute(worker)["state"], "synthetic_only")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                self.execute(worker)
        self.assertEqual(worker.evaluate.local.call_count, 1)
        self.assertIs(worker.runner.terminal, original)
        self.assertEqual({k: v for k, v in runner.files(folder).items() if k in before}, before)
        self.assertTrue(recovery.protected("recovery-control/attempt-2/invocation.json"))

    def test_previous_active_or_wrong_claim_never_enters_sealed_worker(self):
        folder, worker = self.worker()
        with patch.object(runner, "call_status", return_value={"terminal": False, "state": "unknown"}):
            with self.assertRaisesRegex(ValueError, "active or ambiguous"):
                self.execute(worker)
        worker.claims[self.previous["run_id"] + ":1"] = "fc-other"
        with self.assertRaisesRegex(ValueError, "Previous claim differs"):
            self.execute(worker)
        worker.evaluate.local.assert_not_called()
        self.assertFalse((folder / "recovery-control").exists())

    def test_callback_restored_on_failure_and_same_sidecar_never_retried(self):
        _, worker = self.worker()
        original = worker.runner.terminal
        worker.evaluate.local.side_effect = ValueError("synthetic local failure")
        with patch.object(runner, "call_status", return_value={"terminal": True}):
            with self.assertRaisesRegex(ValueError, "synthetic local failure"):
                self.execute(worker)
            self.assertIs(worker.runner.terminal, original)
            with self.assertRaisesRegex(ValueError, "Remote journal/provenance differs"):
                self.execute(worker)
        self.assertEqual(worker.evaluate.local.call_count, 1)

    def test_no_launch_without_explicit_gpu_gate(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            recovery.launch(self.controller, self.sha, self.package, self.root / "new.json", False)
        self.assertFalse((self.root / "new.json").exists())

    def test_launch_writes_bound_receipt_before_one_dispatch(self):
        receipt = self.root / "attempt-2.json"
        def spawn(checksum):
            saved = runner.read(receipt)
            self.assertEqual(saved["state"], "submitting")
            self.assertNotIn("call_id", saved)
            self.assertEqual(saved["execution_sha256"], self.checksum)
            self.assertEqual(saved["controller_sha256"], checksum)
            return SimpleNamespace(object_id="fc-new")
        remote = SimpleNamespace(app=SimpleNamespace(run=lambda **_: nullcontext()),
                                 recover=SimpleNamespace(spawn=Mock(side_effect=spawn)))
        with patch.object(runner, "call_status", return_value={"terminal": True}), \
             patch.object(runner, "module_at", return_value=remote), \
             patch.object(modal, "enable_output", return_value=nullcontext()), patch.dict("os.environ", {}):
            result = recovery.launch(self.controller, self.sha, self.package, receipt, True)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["call_id"], "fc-new")
        remote.recover.spawn.assert_called_once_with(self.sha)

    def test_ambiguous_previous_call_prevents_submission_receipt(self):
        with patch.object(runner, "call_status", return_value={"terminal": False}), \
             patch.object(runner, "module_at") as load, patch.dict("os.environ", {}):
            with self.assertRaisesRegex(ValueError, "active or ambiguous"):
                recovery.launch(self.controller, self.sha, self.package, self.root / "attempt-2.json", True)
        load.assert_not_called()
        self.assertFalse((self.root / "attempt-2.json").exists())

    def test_wrapper_uses_one_bounded_gpu_and_no_remote_nested_launch(self):
        source = (self.controller / "code" / recovery.WORKER).read_text()
        for text in ('gpu="H100"', "retries=0", "max_containers=1", "timeout=7200", "single_use_containers=True", "cache.read_only()"):
            self.assertIn(text, source)
        self.assertNotIn(".spawn(", source)
        self.assertNotIn(".remote(", source)


if __name__ == "__main__":
    unittest.main()
