import unittest
from prepare_native_intents import convert, entities


class NativeIntentTests(unittest.TestCase):
    def test_unicode_source_offsets_and_target_preserved(self):
        row = {"text": "Ma me dɔn no dum berɛ abɔ anɔpa nɔnkron", "intent": "alarm",
               "example_id": "train-00000000", "spans": [{"start_byte": 26, "limit_byte": 39, "label": "TIME"}],
               "target": "TIME: anɔpa nɔnkron"}
        self.assertEqual(entities(row), [{"type": "TIME", "text": "anɔpa nɔnkron"}])
        value = convert(row, "tw", "train", ["alarm"])
        self.assertEqual(value["source_evidence"], row)
        self.assertEqual(value["messages"][1]["content"], row["text"])
        other = convert({**row, "intent": "change_alarm"}, "tw", "train", ["alarm", "change_alarm"])
        self.assertNotEqual(value["id"], other["id"])
        row["spans"][0]["start_byte"] -= 1
        with self.assertRaises(ValueError):
            entities(row)

    def test_no_invented_missing_entities(self):
        row = {"text": "Set an alarm", "intent": "alarm", "example_id": "train-1", "spans": [], "target": ""}
        self.assertEqual(entities(row), [])
        row["target"] = "TIME: 7am"
        with self.assertRaises(ValueError):
            entities(row)


if __name__ == "__main__":
    unittest.main()
