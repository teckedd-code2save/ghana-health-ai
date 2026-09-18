import unittest

from export_public_speech_text import SOURCES, project_language, split_for


class SpeechProjectionTests(unittest.TestCase):
    def test_akan_is_not_silently_labelled_as_verified_twi(self):
        self.assertEqual(project_language("waxal_akan_full"), "ak")
        self.assertEqual(project_language("ghana_nlp_speech_full"), "tw")
        with self.assertRaises(KeyError):
            project_language("unknown")

    def test_waxal_only_labelled_akan_asr(self):
        source = SOURCES[0]
        for split in ("train", "validation", "test"):
            self.assertEqual(split_for(source, f"data/ASR/aka/aka-{split}-00000.parquet"), split)
        for filename in ("data/ASR/aka/aka-unlabeled-00000.parquet",
                         "data/TTS/twi/twi-train-00000.parquet",
                         "data/ASR/ewe/ewe-train-00000.parquet", "README.md"):
            self.assertIsNone(split_for(source, filename))

    def test_gn_source_and_column_allowlist(self):
        self.assertEqual(split_for(SOURCES[1], "data/train-00000-of-00003.parquet"), "train")
        self.assertIsNone(split_for(SOURCES[1], "data/test-00000.parquet"))
        for source in SOURCES:
            self.assertNotIn("audio", source["columns"])
            self.assertIn(source["text_column"], source["columns"])


if __name__ == "__main__":
    unittest.main()
