import unittest
from audit_understanding_r4_data import normalized, sentence_repeats


class AuditTests(unittest.TestCase):
    def test_repeated_sentence(self):
        text = "This is the same long sentence. This is the same long sentence."
        self.assertEqual(sentence_repeats(text), {"this is the same long sentence": 2})

    def test_short_acknowledgments_are_not_loop_flags(self):
        self.assertEqual(sentence_repeats("Yes. Yes."), {})

    def test_punctuation_is_not_spelling_normalization(self):
        self.assertEqual(normalized(" Me ho yɛ! "), "me ho yɛ")
        self.assertNotEqual(normalized("yɛ"), normalized("y3"))

    def test_shared_lead_in_is_not_duplicate_sentence(self):
        self.assertEqual(sentence_repeats("First check the door is closed. First check the window is open."), {})


if __name__ == "__main__":
    unittest.main()
