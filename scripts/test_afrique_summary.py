import json
import tempfile
import unittest
from pathlib import Path

from summarize_afrique_response import report


class AfriqueSummaryTests(unittest.TestCase):
    def test_pairs_and_unpaired_holdout_are_not_conflated(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def write(name, value):
                (folder / name).write_text(json.dumps(value))
            write("summary.json", {"run_id": "test", "steps": 691, "adapter_sha256": "a" * 64})
            write("manifest.json", {"base_model": "test-base", "base_revision": "b" * 40})
            write("status.json", {"stage": "completed", "run_id": "test"})
            rows = [{"id": str(i), "messages": [{"role": "user", "content": "test"}],
                     "prediction": "fixture output", "limit_reached": False} for i in range(30)]
            write("base.json", rows)
            write("adapter.json", rows)
            write("source-heldout.json", [{**row, "id": "heldout-" + row["id"]} for row in rows[:24]])
            result = report(folder)
            self.assertEqual(result["cases_per_variant"], 30)
            self.assertFalse(result["unpaired_source_checks"]["paired_with_base"])
            write("adapter.json", [{**row, "messages": []} for row in rows])
            with self.assertRaises(ValueError):
                report(folder)
            write("adapter.json", rows[:-1])
            with self.assertRaises(ValueError):
                report(folder)
            write("status.json", {"stage": "failed", "run_id": "test"})
            with self.assertRaises(ValueError):
                report(folder)


if __name__ == "__main__":
    unittest.main()
