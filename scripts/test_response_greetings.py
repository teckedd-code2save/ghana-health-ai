import copy
import json
import unittest

from check_response_candidate import ROOT, greeting_cases


class GreetingDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / "data/response-adaptation/greeting-regressions.v1.json").read_text())

    def test_only_inputs_reach_model_and_source_is_unchanged(self):
        self.fixture["cases"][0]["expected_answer"] = "Do not send this target"
        self.fixture["cases"][0]["messages"][0]["reviewer_note"] = "Do not send this note"
        before = copy.deepcopy(self.fixture)
        cases = greeting_cases(self.fixture)
        self.assertEqual(self.fixture, before)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0]["messages"], [{"role": "user", "content": "wo ho te s3n"}])
        for case in cases:
            self.assertEqual(set(case), {"id", "messages"})
            self.assertTrue(all(set(message) == {"role", "content"} for message in case["messages"]))
        self.assertEqual(len(cases[-1]["messages"]), 3)

    def test_missing_duplicate_and_invalid_histories_fail_closed(self):
        for change in (
            lambda fixture: fixture["cases"].pop(),
            lambda fixture: fixture["cases"][1].update(id=fixture["cases"][0]["id"]),
            lambda fixture: fixture["cases"][0].update(messages=[]),
            lambda fixture: fixture["cases"][0]["messages"][0].update(role="system"),
            lambda fixture: fixture["cases"][0]["messages"][0].update(content=" "),
            lambda fixture: fixture["cases"][0]["messages"].append({"role": "assistant", "content": "A"}),
        ):
            fixture = copy.deepcopy(self.fixture)
            change(fixture)
            with self.assertRaises(ValueError):
                greeting_cases(fixture)


if __name__ == "__main__":
    unittest.main()
