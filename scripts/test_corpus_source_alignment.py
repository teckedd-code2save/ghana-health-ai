import unittest
from prepare_source_alignment import examples


def source(kind="sentence"):
    return {"source_id": "fixture:1", "source_hash": "a" * 64, "group_id": "shared", "split": "validation",
        "record_type": kind, "origin": "source_preserved", "topic": "unclassified", "license": "fixture",
        "source_file": "fixture.jsonl", "source_file_sha256": "b" * 64, "source_revision": "pinned",
        "twi": "Fixture source text.", "english": "I am a teacher.", "reference_preserved": True}


class SourceAlignmentTests(unittest.TestCase):
    def test_both_directions_preserve_exact_sources_and_split(self):
        row = source(); pairs = list(examples(row))
        self.assertEqual(len(pairs), 2)
        self.assertEqual([p[1]["messages"][-1]["content"] for p in pairs], [row["twi"], row["english"]])
        self.assertTrue(all(p[1]["split"] == "validation" and p[1]["group_id"] == "shared" for p in pairs))
        self.assertTrue(all(p[1]["loss_policy"] == "assistant_only" and not p[1]["generated_target"] for p in pairs))

    def test_author_identity_is_quoted_translation_not_assistant_biography(self):
        _, row = list(examples(source()))[0]
        self.assertIn("do not answer as yourself", row["messages"][0]["content"])
        self.assertEqual(row["messages"][1]["content"], '{"source_text": "I am a teacher."}')

    def test_lexicon_and_question_views_are_separate(self):
        self.assertEqual(list(examples(source("dictionary_entry")))[0][0], "lexicon")
        view, row = list(examples(source("grounded_qa")))[0]
        self.assertEqual(view, "questions")
        self.assertIn("Translate the source question", row["messages"][0]["content"])

    def test_no_protected_or_generated_reference_promotion(self):
        row = source(); row["split"] = "protected"
        with self.assertRaises(ValueError): list(examples(row))
        row = source(); row["reference_preserved"] = False
        with self.assertRaises(ValueError): list(examples(row))


if __name__ == "__main__": unittest.main()
