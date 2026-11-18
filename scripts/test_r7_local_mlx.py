import unittest
from r7_local_mlx import fresh, history, visible


class Histories(unittest.TestCase):
    def test_variants_never_share_assistant_replies(self):
        state = fresh()
        state["rounds"] = [{"question": "hello", "sides": {
            "r7": {"answer": "R7 answer", "complete": True},
            "r6": {"answer": "R6 answer", "complete": True}}}]
        self.assertEqual(history(state, "r7")[-1]["content"], "R7 answer")
        self.assertEqual(history(state, "r6")[-1]["content"], "R6 answer")

    def test_partial_answers_remain_visible_but_never_enter_context(self):
        state = fresh()
        state["rounds"] = [{"question": "hello", "sides": {
            "r7": {"answer": "partial", "complete": False, "finish_reason": "repetition_guard"},
            "r6": {"answer": "complete", "complete": True}}}]
        self.assertEqual(history(state, "r7"), [])
        self.assertEqual(len(history(state, "r6")), 2)
        self.assertEqual(visible(state, "r7")[-1]["content"], "partial")

    def test_pending_turn_or_empty_eos_does_not_poison_followup(self):
        state = fresh()
        state["rounds"] = [{"question": "hello", "sides": {}},
                           {"question": "again", "sides": {"r7": {"complete": True, "answer": " "}}}]
        self.assertEqual(history(state, "r7"), [])
        self.assertEqual(history(state, "r6"), [])

    def test_new_chats_are_independent(self):
        first, second = fresh(), fresh()
        self.assertNotEqual(first["id"], second["id"])
        first["rounds"].append({"question": "old", "sides": {}})
        self.assertEqual(second["rounds"], [])


if __name__ == "__main__":
    unittest.main()
