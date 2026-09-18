import copy
import unittest

from summarize_balanced_afrique import conversation_report


class ConversationSummaryTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i in range(6):
            first = {"id": str(i), "turn": 0, "messages": [{"role": "user", "content": "First question"}],
                     "prediction": "Actual first answer", "limit_reached": False}
            second = {"id": str(i), "turn": 1, "messages": [*first["messages"],
                {"role": "assistant", "content": first["prediction"]}, {"role": "user", "content": "Follow up"}],
                "prediction": "Actual second answer", "limit_reached": False}
            rows.extend((first, second))
        return rows

    def test_actual_history_is_checked_without_semantic_score(self):
        result = conversation_report(self.fixture())
        self.assertEqual(result["turns"], 12)
        self.assertEqual(result["count"], 6)
        self.assertNotIn("accuracy", result)

    def test_missing_duplicate_and_substituted_history_rejected(self):
        original = self.fixture()
        for change in (lambda r: r.pop(), lambda r: r.append(r[0]),
                       lambda r: r[1]["messages"][1].update(content="A canned answer")):
            rows = copy.deepcopy(original)
            change(rows)
            with self.assertRaises(ValueError):
                conversation_report(rows)


if __name__ == "__main__":
    unittest.main()
