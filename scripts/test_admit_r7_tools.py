import json
from pathlib import Path
import tempfile
import unittest

import admit_r7_tools as admission
import prepare_r7_tool_candidates as candidates


class ToolAdmissionTests(unittest.TestCase):
    def test_invalid_probe_calls_are_context_only(self):
        files = candidates.build_package()
        scenarios = {r["id"]: r for r in map(json.loads, files["scenarios.jsonl"].splitlines())}
        rows = list(map(json.loads, files["candidates/train.jsonl"].splitlines()))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "manifest.json").write_bytes(files["manifest.json"])
            for row in rows:
                result = admission.admitted_record(row, scenarios, source)
                self.assertEqual(result["messages"], row["messages"])
                chosen = set(result["metadata"]["training_assistant_indices"])
                skipped = set(result["metadata"]["context_only_assistant_indices"])
                self.assertFalse(chosen & skipped)
                self.assertIn(len(result["messages"]) - 1, chosen)
                if row["metadata"]["family"] == "invalid_args_recovery":
                    self.assertTrue(skipped)
                    self.assertEqual(skipped, {2})
                self.assertFalse(row["training_eligible"])
                self.assertTrue(result["training_eligible"])


if __name__ == "__main__":
    unittest.main()
