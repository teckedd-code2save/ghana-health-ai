import copy
import unittest
from augment_understanding_r4_spelling import augment


class SpellingTests(unittest.TestCase):
    def test_only_user_epsilon_changes(self):
        row = {"id": "a", "source_id": "s", "group_id": "g", "original_ref": {}, "review": {},
               "messages": [{"role": "user", "content": "Ɛyɛ 3?"}, {"role": "assistant", "content": "Ɛyɛ."}]}
        original = copy.deepcopy(row)
        result = augment(row, {})
        self.assertEqual(row, original)
        self.assertEqual(result["messages"][0]["content"], "Ɛy3 3?")
        self.assertEqual(result["messages"][1], row["messages"][1])
        self.assertEqual(result["source_id"], row["source_id"])
        self.assertEqual(result["group_id"], row["group_id"])
        self.assertEqual(result["augmentation"]["edits"][0]["character_positions"], [2])

    def test_no_change_has_no_duplicate(self):
        self.assertIsNone(augment({"messages": [{"role": "user", "content": "Hello 3"}]}, {}))


if __name__ == "__main__":
    unittest.main()
