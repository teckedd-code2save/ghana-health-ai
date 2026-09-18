import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("stronger_comparison", Path(__file__).resolve().parents[1] / "modal/train/benchmark_stronger_response.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StrongerComparisonTests(unittest.TestCase):
    def test_no_reference_answers_or_duplicate_cases_sent(self):
        row = {"id": "case-a", "messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello"}]}
        module.validate_cases([row])
        for cases in ([], [row, row], [{**row, "reference_answer": "secret"}], [{}]):
            with self.assertRaises(ValueError):
                module.validate_cases(cases)
        with self.assertRaises(ValueError):
            module.validate_cases([{**row, "messages": row["messages"][:-1]}])


if __name__ == "__main__":
    unittest.main()
