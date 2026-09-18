import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_response_adaptation as runner


class TrainingReceiptTests(unittest.TestCase):
    def test_completion_is_cached_and_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "tmp/balanced-afrique-v6"
            folder.mkdir(parents=True)
            run_id = "afrique_v6_20260910T222142Z"
            receipt = folder / "run-receipt.json"
            receipt.write_text(json.dumps({"run_id": run_id, "app": "ghana-balanced-afrique-v6", "state": "submitted", "call_id": "test-call"}))
            with patch.object(runner, "ROOT", root), patch("sys.argv", ["runner", "status", "--experiment", "balanced-v6"]), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(runner.modal.FunctionCall, "from_id") as remote:
                    remote.return_value.get.return_value = {"run_id": run_id, "steps": 400}
                    runner.main()
                    self.assertEqual(json.loads(receipt.read_text())["state"], "completed")
                    remote.reset_mock()
                    runner.main()
                    remote.assert_not_called()
                    (folder / (run_id + ".result.json")).write_text("{}")
                    with self.assertRaisesRegex(ValueError, "changed"):
                        runner.main()

    def test_foreign_experiment_receipt_rejected_before_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "tmp/balanced-afrique-v6"
            folder.mkdir(parents=True)
            (folder / "run-receipt.json").write_text(json.dumps({"run_id": "wrong", "app": "another-app", "state": "submitted"}))
            with patch.object(runner, "ROOT", root), patch("sys.argv", ["runner", "status", "--experiment", "balanced-v6"]):
                with self.assertRaisesRegex(ValueError, "different experiment"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
