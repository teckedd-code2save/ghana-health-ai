import copy
import unittest
from evaluate_semantic_pairs import make_cases, score


class SemanticPairsTests(unittest.TestCase):
    def cases(self):
        rows = [{"premise": f"Source {i}", "hypothesis": f"Claim {i}", "label": str(i % 3)} for i in range(450)]
        return make_cases(rows, copy.deepcopy(rows))

    def test_parallel_choices_and_source_labels(self):
        cases = self.cases()
        self.assertEqual(len(cases), 900)
        for tw, en in zip(cases[::2], cases[1::2], strict=True):
            self.assertEqual(tw["expected"], en["expected"])
            self.assertEqual(tw["options"], en["options"])
            self.assertIn(tw["source_row"]["premise"], tw["messages"][0]["content"])

    def test_strict_complete_paired_scoring(self):
        cases = self.cases()
        results = [{"id": row["id"], "variant": variant, "prediction": row["expected"], "limit_reached": False}
                   for variant in ("base", "adapter") for row in cases]
        report = score(cases, results)
        self.assertEqual(report["groups"]["adapter:tw"]["correct"], 450)
        results[-1]["prediction"] = "A or B"
        report = score(cases, results)
        self.assertEqual(report["groups"]["adapter:en"]["invalid"], 1)
        self.assertEqual(report["paired_changes"]["en"]["regressed"], 1)
        with self.assertRaises(ValueError):
            score(cases, results[:-1])
        with self.assertRaises(ValueError):
            score(cases, results[:-1] + [results[-2]])


if __name__ == "__main__":
    unittest.main()
