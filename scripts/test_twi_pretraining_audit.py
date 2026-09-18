import unittest

from audit_twi_pretraining import Sampler, normalized, pair_flags, text_flags


LIMITS = {"minimum_words_per_candidate": 5, "max_characters_per_candidate": 20000,
          "repeated_eight_word_fraction": 0.3}


class AuditTests(unittest.TestCase):
    def test_alignment_warning_does_not_discard_twi(self):
        twi = "Mewɔ cedi 20 wɔ me kotoku mu."
        self.assertIn("numeric_alignment_review", pair_flags("I have 200 cedis in my pocket.", twi))
        self.assertEqual(text_flags(twi, LIMITS), [])
        self.assertEqual(twi, "Mewɔ cedi 20 wɔ me kotoku mu.")

    def test_no_twi_marker_requirement(self):
        self.assertEqual(text_flags("Me ne Kofi tena ha daa.", LIMITS), [])

    def test_unicode_dedup_is_canonical_without_changing_input(self):
        self.assertEqual(normalized("Ɛyɛ, ɔpɛ!"), "ɛyɛ ɔpɛ")
        self.assertEqual(normalized("a\u0301"), normalized("á"))

    def test_repetition_and_markup_not_hidden(self):
        self.assertIn("within_text_repetition", text_flags("Me ne Kofi ne Ama tena fie ha. " * 10, LIMITS))
        self.assertIn("encoding_or_markup_review", text_flags("Some source <|im_end|> control tokens here.", LIMITS))
        self.assertIn("short_fragment", text_flags("Hello", LIMITS))

    def test_stratified_samples_are_deterministic(self):
        rows = [{"id": str(i), "text": str(i)} for i in range(20)]
        a, b = Sampler(3), Sampler(3)
        for row in rows:
            a.add(("source", "style", "candidate"), row)
        for row in reversed(rows):
            b.add(("source", "style", "candidate"), row)
        self.assertEqual(a.rows(), b.rows())
        self.assertEqual(len(a.rows()), 3)


if __name__ == "__main__":
    unittest.main()
