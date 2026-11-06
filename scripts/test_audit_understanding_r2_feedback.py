import unittest

from audit_understanding_r2_feedback import cross_turn_flags, fixed_history_state


class FeedbackTests(unittest.TestCase):
    def test_short_sentence_repeated_across_turns(self):
        messages = [{"role": "user", "content": "Help?"},
                    {"role": "assistant", "content": "Sorry to hear that."},
                    {"role": "user", "content": "What next?"},
                    {"role": "assistant", "content": "Sorry to hear that."},
                    {"role": "user", "content": "Please answer."}]
        flags = cross_turn_flags(messages, "Sorry to hear that.")
        self.assertEqual(flags["consecutive_identical_assistant_replies"], 3)
        self.assertTrue(flags["same_as_previous_assistant"])
        self.assertIsNone(flags["semantic_or_safety_score"])

    def test_different_answers_do_not_prove_semantics(self):
        flags = cross_turn_flags([{"role": "assistant", "content": "Yes."}], "No.")
        self.assertFalse(flags["review_required"])
        self.assertIsNone(flags["semantic_or_safety_score"])

    def test_quotes_from_user_do_not_count_as_previous_reply(self):
        self.assertFalse(cross_turn_flags([{"role": "user", "content": "Hello"}], "Hello")["review_required"])

    def test_nonconsecutive_and_whitespace(self):
        flags = cross_turn_flags([{"role": "assistant", "content": "A  B"},
                                  {"role": "assistant", "content": "Different"}], "a b")
        self.assertEqual(flags["matches_previous_assistant_indices"], [0])
        self.assertEqual(flags["consecutive_identical_assistant_replies"], 1)

    def test_empty_is_not_repetition(self):
        self.assertFalse(cross_turn_flags([{"role": "assistant", "content": ""}], "")["review_required"])

    def test_fixed_history_is_exact_and_identified(self):
        messages = [{"role": "user", "content": "A\n"}, {"role": "assistant", "content": "B"},
                    {"role": "user", "content": "C"}]
        state = fixed_history_state(messages, "r2_meaning", {"fixture": True})
        self.assertEqual(state["turns"][0]["question"], "A\n")
        self.assertEqual(state["turns"][0]["origin"], "fixed_observed_history")
        self.assertEqual(state["turns"][0]["identity"], {"fixture": True})

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            fixed_history_state([{"role": "assistant", "content": "C"}], "base", {})


if __name__ == "__main__":
    unittest.main()
