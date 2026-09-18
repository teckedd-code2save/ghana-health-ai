import copy
import unittest

from audit_response_coverage import coverage


class ResponseCoverageTests(unittest.TestCase):
    def fixture(self):
        def row(task, language, multi=False):
            history = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Earlier reply"}] if multi else []
            return {"task": task, "language": language, "messages": history + [
                {"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
        rows = [row("general_conversation", "en", True), row("health_response", "tw"),
                row("auxiliary_translation", "tw"), row("native_intent_entities", "tw")]
        keys = ["train:" + item["task"] + ":" + item["language"] for item in rows]
        return rows, {"counts": dict.fromkeys(keys, 1), "supervised_tokens": dict.fromkeys(keys, 100),
                      "rejected": [], "truncated_targets": 0}

    def test_auxiliary_twi_and_english_multiturn_do_not_inflate_twi_response_coverage(self):
        rows, tokens = self.fixture()
        before = copy.deepcopy(rows)
        result = coverage(rows, tokens)
        self.assertEqual(rows, before)
        self.assertEqual(result["twi_response_task_rows"], 1)
        self.assertEqual(result["twi_response_fraction_of_all_target_tokens"], 0.25)
        self.assertEqual(result["health_fraction_of_twi_response_tokens"], 1)
        self.assertEqual(result["twi_final_target_multiturn_rows"], 0)
        self.assertTrue(result["structural_gaps"])

    def test_twi_multiturn_is_counted_when_present(self):
        rows, tokens = self.fixture()
        rows[1]["messages"] = rows[0]["messages"]
        result = coverage(rows, tokens)
        self.assertEqual(result["twi_final_target_multiturn_rows"], 1)
        self.assertFalse(result["structural_gaps"])

    def test_changed_counts_missing_tokens_and_truncation_are_not_reported_as_coverage(self):
        for change in (
            lambda data: data["counts"].update({"train:health_response:tw": 2}),
            lambda data: data["supervised_tokens"].pop("train:health_response:tw"),
            lambda data: data["supervised_tokens"].update({"train:health_response:tw": -1}),
            lambda data: data.update(truncated_targets=1),
            lambda data: data.update(rejected=["missing"]),
        ):
            rows, tokens = self.fixture()
            change(tokens)
            with self.assertRaises(ValueError):
                coverage(rows, tokens)


if __name__ == "__main__":
    unittest.main()
