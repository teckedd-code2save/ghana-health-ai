import importlib.util
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from r7_english_review_controls import controls, SYSTEM

spec = importlib.util.spec_from_file_location("source_review_test", Path(__file__).resolve().parents[1] / "modal/train/r7_source_review.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class SourceReviewTests(unittest.TestCase):
    def raw(self):
        rows = [{"id": r["id"], "messages": r["messages"]} for r in controls()]
        return b"\n".join(json.dumps(r).encode() for r in rows)

    def test_explicit_inventory_is_bound(self):
        raw = self.raw()
        self.assertEqual(len(worker.checked_rows(raw, hashlib.sha256(raw).hexdigest(), SYSTEM)), len(controls()))
        with self.assertRaises(ValueError):
            worker.checked_rows(raw, "changed", SYSTEM)

    def test_unqualified_prompt_and_duplicates_rejected(self):
        raw = self.raw()
        with self.assertRaises(ValueError):
            worker.checked_rows(raw, hashlib.sha256(raw).hexdigest(), "Other prompt")
        raw = raw + b"\n" + raw
        with self.assertRaises(ValueError):
            worker.checked_rows(raw, hashlib.sha256(raw).hexdigest(), SYSTEM)

    def test_active_or_unreachable_owner_not_taken_over(self):
        with patch.object(worker.modal.FunctionCall, "from_id") as call:
            call.return_value.get_call_graph.return_value = [SimpleNamespace(status=SimpleNamespace(name="PENDING"), children=[])]
            with self.assertRaises(ValueError):
                worker.terminal_owner("old-call")
        with patch.object(worker.modal.FunctionCall, "from_id") as call:
            call.return_value.get_call_graph.side_effect = worker.modal.exception.ConnectionError("offline")
            with self.assertRaises(worker.modal.exception.ConnectionError):
                worker.terminal_owner("old-call")

    def test_terminal_owner_can_resume(self):
        for state in ("SUCCESS", "TIMEOUT", "INIT_FAILURE"):
            with patch.object(worker.modal.FunctionCall, "from_id") as call:
                call.return_value.get_call_graph.return_value = [SimpleNamespace(status=SimpleNamespace(name=state), children=[])]
                worker.terminal_owner("old-call")


if __name__ == "__main__":
    unittest.main()
