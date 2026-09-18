import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_language_adaptation import OverlapIndex, canonical, content_for_overlap, group_pairs, protected_texts, quality_reason


class LanguageAdaptationTests(unittest.TestCase):
    def test_development_greetings_are_protected_without_generic_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "data/response-adaptation/greeting-regressions.v1.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"rubric": "not a source", "cases": [{"messages": [
                {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "wo ho te s3n"}]}]}))
            with patch("build_language_adaptation.ROOT", root), patch("build_language_adaptation.read_jsonl", return_value=[]):
                self.assertEqual(protected_texts(), ["wo ho te s3n"])

    def test_normalization_preserves_twi(self):
        self.assertEqual(canonical("Ɛyɛ, ɔpɛ!"), "ɛyɛ ɔpɛ")

    def test_shared_text_transitively_groups_variants(self):
        rows = [
            {"id": "a", "en": "hello", "tw": "akwaaba", "block": "a"},
            {"id": "b", "en": "Hello!", "tw": "maakye", "block": "b"},
            {"id": "c", "en": "morning", "tw": "maakye", "block": "c"},
        ]
        self.assertEqual(len({r["group"] for r in group_pairs(rows)}), 1)
        self.assertEqual(len({r["split"] for r in rows}), 1)

    def test_source_blocks_stay_together(self):
        rows = [{"id": str(i), "en": str(i), "tw": "x" + str(i), "block": "same"} for i in range(3)]
        self.assertEqual(len({r["group"] for r in group_pairs(rows)}), 1)

    def test_quarantine_without_rewriting(self):
        self.assertEqual(quality_reason("buy 2 kg", "tɔ 20 kg"), "numeric_alignment_review")
        self.assertEqual(quality_reason("same", "same"), "identical_languages")
        self.assertEqual(quality_reason("hello", ""), "empty")
        self.assertIsNone(quality_reason("hello", "akwaaba"))

    def test_overlap_finds_embedded_protected_passage(self):
        index = OverlapIndex(["one two three four five six seven eight nine", "short text"])
        self.assertTrue(index.contains("prefix one two three four five six seven eight suffix"))
        self.assertTrue(index.contains("Short text!"))
        self.assertFalse(index.contains("some other text"))

    def test_shared_task_instruction_does_not_reject_all_translations(self):
        def row(source, target):
            return {"task": "translation", "messages": [{"content": "Translate from Twi to English. Return only the translation.\n\n" + source}, {"content": target}]}
        index = OverlapIndex(content_for_overlap(row("akwaaba", "welcome")))
        self.assertFalse(any(index.contains(t) for t in content_for_overlap(row("maakye", "good morning"))))


if __name__ == "__main__":
    unittest.main()
