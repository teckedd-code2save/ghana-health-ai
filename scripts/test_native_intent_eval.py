import json
import unittest
from evaluate_native_intents import evaluation_inputs, score


class NativeIntentEvalTests(unittest.TestCase):
    def test_prompt_defines_schema_from_train_not_heldout_annotations(self):
        train = [{"messages": [{"content": json.dumps({"intent": "alarm", "entities": [{"type": "TIME", "text": "7am"}]})}]}]
        cases = [{"id": "a", "messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "wake me"},
                  {"role": "assistant", "content": "PRIVATE_REFERENCE"}]}]
        result = evaluation_inputs(cases, train, "schema-v2")
        self.assertEqual(len(result[0]["messages"]), 2)
        self.assertNotIn("PRIVATE_REFERENCE", str(result))
        self.assertNotIn("7am", str(result))
        self.assertIn("TIME", str(result))
        self.assertIn('"type"', str(result))
        self.assertEqual(cases[0]["messages"][0]["content"], "old")

    def test_intent_semantics_are_separate_from_entity_format(self):
        target = {"intent": "alarm", "entities": []}
        cases = [{"id": "a", "messages": [{"content": json.dumps(target)}]}]
        row = {"id": "a", "prediction": '{"intent":"alarm","entities":{}}', "limit_reached": False}
        result = score(cases, [{**row, "variant": v} for v in ("base", "adapter")])["scores"]["base"]
        self.assertEqual(result["intent_accuracy"], 0)
        self.assertEqual(result["intent_label_accuracy_independent_of_entity_format"], 1)
        row["limit_reached"] = True
        result = score(cases, [{**row, "variant": v} for v in ("base", "adapter")])["scores"]["base"]
        self.assertEqual(result["intent_label_accuracy_independent_of_entity_format"], 0)

    def test_strict_score_includes_missing_and_spurious_entities(self):
        expected = {"intent": "alarm", "entities": [{"type": "TIME", "text": "7am"}]}
        cases = [{"id": "a", "messages": [{"content": json.dumps(expected)}]}]
        base = {"id": "a", "variant": "base", "prediction": json.dumps(expected), "limit_reached": False}
        adapted = {**base, "variant": "adapter", "prediction": json.dumps({"intent": "alarm", "entities": [{"type": "TIME", "text": "8am"}]})}
        report = score(cases, [base, adapted])["scores"]
        self.assertEqual(report["base"]["entity_micro_f1"], 1)
        self.assertEqual(report["adapter"]["intent_accuracy"], 1)
        self.assertEqual(report["adapter"]["entity_micro_f1"], 0)
        adapted["prediction"] = "```json\n" + adapted["prediction"] + "\n```"
        fenced = score(cases, [base, adapted])["scores"]["adapter"]
        self.assertEqual(fenced["unparseable_or_incomplete"], 0)
        self.assertEqual(fenced["strict_format_invalid"], 1)
        self.assertEqual(fenced["intent_accuracy"], 1)
        with self.assertRaises(ValueError):
            score(cases, [base])


if __name__ == "__main__":
    unittest.main()
