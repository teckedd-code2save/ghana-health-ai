import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from voice_listening import sample, save_tail_preference, save_voice_observation, save_voice_review, voice_choices


class VoiceListeningTests(unittest.TestCase):
    def fixture(self, folder):
        wav = b"only a test fixture, not a real recording"
        name = "listen-question-stable-twi-6.wav"
        row = {"id": "question", "voice": "stable-twi-6", "text": "test phrase", "revision": "pinned",
               "listening_file": name, "listening_sha256": hashlib.sha256(wav).hexdigest()}
        (folder / name).write_bytes(wav)
        (folder / "listening.json").write_text(json.dumps({"results": [row]}))
        return row, wav

    def test_verified_audio_and_review_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row, wav = self.fixture(folder)
            self.assertEqual(sample("question", "stable-twi-6", folder), (row, wav))
            record = save_voice_review("question", "stable-twi-6", "Unsure", 3, "Automated test only",
                                       path=folder / "test.sqlite3", folder=folder)
            self.assertEqual(record["sample"], row)
            self.assertFalse(record["training_eligible"])

    def test_rejects_unknown_or_tampered_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row, _ = self.fixture(folder)
            with self.assertRaises(ValueError):
                sample("../../reference", "stable-twi-6", folder)
            (folder / row["listening_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                sample("question", "stable-twi-6", folder)

    def test_observation_preserves_words_without_inventing_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            args = ("question", "stable-twi-6", "Closer voice, muffled ending. Test only.")
            first = save_voice_observation(*args, path=folder / "db", folder=folder)
            repeated = save_voice_observation(*args, path=folder / "db", folder=folder)
            self.assertEqual(first, repeated)
            self.assertEqual(first["note"], args[2])
            self.assertIsNone(first["pronunciation"])
            self.assertIsNone(first["naturalness"])
            self.assertFalse(first["training_eligible"])

    def test_streaming_sample_has_distinct_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original, wav = self.fixture(folder)
            row = {**original, "voice": "cosy-twi-streaming", "synthesis_voice": "cosy-twi-owner-reference",
                   "listening_file": "listen-stream-fp16-question-cosy-twi-owner-reference.wav"}
            (folder / row["listening_file"]).write_bytes(wav)
            (folder / "cosy-streaming-listening.json").write_text(json.dumps({"results": [row]}))
            self.assertEqual(sample("question", "cosy-twi-streaming", folder), (row, wav))
            self.assertEqual(sample("question", "stable-twi-6", folder)[0], original)

    def test_rejects_path_escape_and_unrated_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row, _ = self.fixture(folder)
            with self.assertRaises(ValueError):
                save_voice_review("question", "stable-twi-6", None, None, "", path=folder / "db", folder=folder)
            row["listening_file"] = "speaker-1-reference.wav"
            (folder / "listening.json").write_text(json.dumps({"results": [row]}))
            with self.assertRaisesRegex(ValueError, "Unexpected listening file"):
                sample("question", "stable-twi-6", folder)

    def tail_fixture(self, folder):
        original, _ = self.fixture(folder)
        rows = []
        for variant in ("fp32-stream", "fp16-whole"):
            wav = variant.encode()
            row = {**original, "voice": "cosy-twi-owner-reference", "variant": variant,
                   "listening_file": f"listen-tail-{variant}-question.wav",
                   "listening_sha256": hashlib.sha256(wav).hexdigest()}
            (folder / row["listening_file"]).write_bytes(wav)
            rows.append(row)
        (folder / "cosy-tail-listening.json").write_text(json.dumps({"results": rows}))
        wav = b"original streaming fixture"
        stream = {**original, "voice": "cosy-twi-streaming",
                  "listening_file": "listen-stream-fp16-question-cosy-twi-owner-reference.wav",
                  "listening_sha256": hashlib.sha256(wav).hexdigest()}
        (folder / stream["listening_file"]).write_bytes(wav)
        (folder / "cosy-streaming-listening.json").write_text(json.dumps({"results": [stream]}))
        return rows

    def test_tail_variants_remain_distinct_and_phrase_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertNotIn("cosy-twi-fp32-stream", dict((value, label) for label, value in voice_choices("question", folder)))
            rows = self.tail_fixture(folder)
            row, _ = sample("question", "cosy-twi-fp32-stream", folder)
            self.assertEqual(row["voice"], "cosy-twi-fp32-stream")
            self.assertEqual(row["synthesis_voice"], "cosy-twi-owner-reference")
            self.assertEqual(row["listening_sha256"], rows[0]["listening_sha256"])
            self.assertIn(("CosyVoice A", "cosy-twi-fp32-stream"), voice_choices("question", folder))
            self.assertNotIn(("CosyVoice A", "cosy-twi-fp32-stream"), voice_choices("shopping", folder))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                sample("shopping", "cosy-twi-fp32-stream", folder)

    def test_preference_is_idempotent_and_not_a_quality_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.tail_fixture(folder)
            first = save_tail_preference("A", path=folder / "db", folder=folder)
            self.assertEqual(first, save_tail_preference("A", path=folder / "db", folder=folder))
            self.assertEqual(first["owner_response"], "A")
            self.assertEqual(first["selected"]["variant"], "fp32-stream")
            self.assertEqual(set(first["options"]), {"A", "B", "original"})
            self.assertIsNone(first["pronunciation"])
            self.assertIsNone(first["naturalness"])
            self.assertFalse(first["training_eligible"])
            with self.assertRaises(ValueError):
                save_tail_preference("C", path=folder / "db", folder=folder)
            selected = folder / first["selected"]["listening_file"]
            selected.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                save_tail_preference("A", path=folder / "db", folder=folder)

    def test_preference_rejects_mismatched_or_ambiguous_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = self.tail_fixture(folder)
            rows[1]["text"] = "Different sentence"
            manifest = folder / "cosy-tail-listening.json"
            manifest.write_text(json.dumps({"results": rows}))
            with self.assertRaisesRegex(ValueError, "same sentence"):
                save_tail_preference("A", path=folder / "db", folder=folder)
            manifest.write_text(json.dumps({"results": rows + [rows[0]]}))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                sample("question", "cosy-twi-fp32-stream", folder)


if __name__ == "__main__":
    unittest.main()
