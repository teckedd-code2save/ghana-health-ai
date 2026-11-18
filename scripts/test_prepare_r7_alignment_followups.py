from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_r7_alignment_followups as followups


def source(identity, en, tw):
    return {"id": identity, "source_id": identity, "split": "train", "training_eligible": True,
        "messages": [{"role": "user", "content": en}, {"role": "assistant", "content": tw}],
        "metadata": {"target_language": "tw", "record_type": "sentence", "group_id": identity,
            "alignment_pair": {"english": en, "twi": tw, "generated_target": False}}, "original_ref": {"source": identity}}


class FollowupTests(unittest.TestCase):
    def test_binds_code_and_rejects_changed_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "tokenizer.json"
            asset.write_text("original", encoding="utf-8")
            expected = followups.data.sha(asset)
            with patch.object(followups, "ASSET_HASHES", {asset.name: expected}):
                pins = {}
                result = followups.bind_transformation(directory, pins)
                self.assertEqual(result["tokenizer_assets"], {asset.name: expected})
                self.assertIn("prepare_r7_alignment_followups.py", result["code_hashes"])
                self.assertEqual(pins[str(asset.resolve())], expected)
                asset.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    followups.bind_transformation(directory, {})

    def test_only_original_final_reference_is_supervised(self):
        first, second = source("one", "first original", "first target"), source("two", "second original", "second target")
        snapshot = deepcopy([first, second])
        for kind in followups.KINDS:
            row = followups.derive(first, second, kind, {"sha": "proof"})
            self.assertEqual(row["messages"][-1]["content"], second["messages"][-1]["content"])
            self.assertEqual(row["metadata"]["training_assistant_indices"], [4])
            self.assertFalse(row["metadata"]["generated_twi_target"])
            self.assertFalse(row["metadata"]["natural_conversation"])
            self.assertIn("one", row["metadata"]["group_aliases"])
            self.assertIn("two", row["metadata"]["group_aliases"])
        self.assertEqual([first, second], snapshot)

    def test_pairing_is_disjoint_and_deterministic(self):
        rows = [source(str(i), "source " + str(i), "target " + str(i)) for i in range(20)]
        first = list(followups.pairs(rows))
        self.assertEqual(first, list(followups.pairs(list(reversed(rows)))))
        self.assertEqual(len({current["id"] for _, current, _ in first}), len(rows))
        for i in range(0, len(first), 2):
            self.assertEqual(first[i][0], first[i + 1][1])
            self.assertEqual(first[i][1], first[i + 1][0])

    def test_wrong_language_context_is_not_a_new_twi_label(self):
        row = followups.derive(source("one", "first English", "first Twi"), source("two", "next English", "next Twi"),
                               "wrong_language_repair", {"sha": "proof"})
        self.assertEqual(row["messages"][2]["content"], "next English")
        self.assertTrue(row["metadata"]["counterfactual_wrong_language_context"])

    def test_reject_non_train_or_generated_reference(self):
        one, two = source("a", "first source", "first target"), source("b", "other source", "other target")
        for key, value in (("split", "test"), ("training_eligible", False)):
            changed = {**two, key: value}
            with self.assertRaises(ValueError):
                followups.derive(one, changed, "new_request", {})
        two["metadata"]["alignment_pair"]["generated_target"] = True
        with self.assertRaises(ValueError):
            followups.derive(one, two, "new_request", {})


if __name__ == "__main__":
    unittest.main()
