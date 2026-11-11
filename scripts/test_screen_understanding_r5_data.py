import unittest
from screen_understanding_r5_data import inspect_messages, validate_decision


def pair(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class ScreenTests(unittest.TestCase):
    def test_unanchored_date(self):
        result = inspect_messages(pair("How many days until Christmas?", "Today's date is February 5, 2023."))
        self.assertIn("unanchored_current_information", {r["code"] for r in result})

    def test_historical_date_is_not_current(self):
        self.assertEqual(inspect_messages(pair("What happened in 2023?", "A festival was held in 2023.")), [])

    def test_missing_apostrophe_does_not_hide_current_claim(self):
        result = inspect_messages(pair("What about this years competitions?", "The next competition is in 2023."))
        self.assertEqual(result[0]["code"], "unanchored_current_information")

    def test_explicit_clock_anchor(self):
        self.assertEqual(inspect_messages(pair("Assume today is 5 February 2023. What date is today?",
                                              "Today's date is 5 February 2023.")), [])

    def test_identity_is_review_not_blanket_brand_filter(self):
        self.assertEqual(inspect_messages(pair("What is OpenAI?", "OpenAI develops software.")), [])
        self.assertEqual(inspect_messages(pair("Who are you?", "I am an AI developed by OpenAI."))[0]["code"],
                         "inherited_provider_identity")

    def test_bad_unicode_and_roles(self):
        self.assertEqual(inspect_messages(pair("Hello", "bad\ufffd"))[0]["severity"], "invalid")
        self.assertEqual(inspect_messages([{"role": "assistant", "content": "Hello"}])[0]["severity"], "invalid")

    def test_findings_cannot_be_silently_approved(self):
        record = {"messages": pair("How are you?", "I'm sorry to hear that.")}
        decision = {"record_sha256": "x", "reviewer_type": "agent", "human_reviewed": False,
                    "reviewer": "test", "reason": "source checked", "decision": "retain"}
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            validate_decision(record, decision, lambda _: "x")

    def test_partial_repair_review_is_insufficient(self):
        record = {"messages": pair("Hello", "Hello")}
        decision = {"record_sha256": "x", "reviewer_type": "agent", "human_reviewed": False,
                    "reviewer": "test", "reason": "source checked", "decision": "repair",
                    "messages": pair("Hello", "Hello there"), "per_turn_review": [{"index": 0}]}
        with self.assertRaisesRegex(ValueError, "per-turn"):
            validate_decision(record, decision, lambda _: "x")


if __name__ == "__main__":
    unittest.main()
