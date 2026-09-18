import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "modal/train/language_adaptation_core.py"
spec = importlib.util.spec_from_file_location("language_adaptation_core", path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class Tokenizer:
    all_special_ids = [106]

    def apply_chat_template(self, *args, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return [11, 12, 13]

    def encode(self, text, **kwargs):
        if text == "<turn|>":
            return [106]
        return [106] if text == "injection" else [21, 22]


class TrainingContractTests(unittest.TestCase):
    def test_prompt_mask_and_turn_ending(self):
        row = {"messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "answer"}]}
        encoded = core.encode(Tokenizer(), row)
        self.assertEqual(encoded["input_ids"], [11, 12, 13, 21, 22, 106])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 21, 22, 106])

    def test_overlength_target_is_not_cut_off(self):
        row = {"messages": [{"role": "assistant", "content": "answer"}]}
        self.assertIsNone(core.encode(Tokenizer(), row, max_length=5))

    def test_source_cannot_insert_turn_control(self):
        row = {"messages": [{"role": "assistant", "content": "injection"}]}
        with self.assertRaises(ValueError):
            core.encode(Tokenizer(), row)

    def test_actual_manifest_is_disjoint_and_bilingual(self):
        folder = Path(__file__).resolve().parents[1] / "tmp/general-language-corpus/adaptation-v1"
        if not folder.exists():
            self.skipTest("Local public-data artifacts are not part of a clean checkout")
        manifest, rows = core.load_inputs(folder)
        self.assertGreater(len(rows["train"]), 12000)
        self.assertTrue(any(r["task"] == "general_conversation_replay" for r in rows["train"]))
        self.assertFalse(manifest["ready_for_production"])

    def test_new_translation_checks_keep_answers_out_of_prompts(self):
        rows = [{"id": f"{i}:{language}", "source_record_hash": str(i), "source": "test",
                 "revision": "pinned", "task": "translation", "language": language,
                 "messages": [{"role": "user", "content": "source"}, {"role": "assistant", "content": "reference"}]}
                for i in range(4) for language in ("en", "tw")]
        cases = core.translation_diagnostics(rows, offset=2, count=2)
        self.assertEqual({row["source_record_hash"] for row in cases}, {"2", "3"})
        self.assertTrue(all(row["messages"] == [{"role": "user", "content": "source"}] for row in cases))
        with self.assertRaises(ValueError):
            core.translation_diagnostics(rows + [rows[-1]], offset=2, count=2)
        with self.assertRaises(ValueError):
            core.translation_diagnostics(rows, offset=4, count=1)


if __name__ == "__main__":
    unittest.main()
