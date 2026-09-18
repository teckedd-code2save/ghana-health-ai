import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("general_benchmark", ROOT / "modal/train/benchmark_general_understanding.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_fixed_source_groups_and_languages(self):
        rows = benchmark.build_cases(ROOT)
        self.assertEqual(len(rows), 74)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual(rows, benchmark.build_cases(ROOT))
        self.assertTrue(any(r["source"] == "ghana_nlp_speech" for r in rows))
        self.assertTrue(all(r.get("source_split") != "train" for r in rows))
        self.assertEqual(sum(r["category"] == "locked_medical" for r in rows), 32)

    def test_input_does_not_contain_hidden_reference_fields(self):
        rows = benchmark.build_cases(ROOT)
        inputs = [{k: row[k] for k in ("id", "messages", "tools") if k in row} for row in rows]
        self.assertTrue(all(set(r) <= {"id", "messages", "tools"} for r in inputs))
        for row in rows:
            if row.get("reference_answer"):
                self.assertNotIn(row["reference_answer"], str(row["messages"]))


if __name__ == "__main__":
    unittest.main()
