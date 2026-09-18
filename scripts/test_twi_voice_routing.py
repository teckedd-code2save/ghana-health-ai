import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("stable_tts", Path(__file__).resolve().parents[1] / "modal/stable_twi_tts_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)
fallback_spec = importlib.util.spec_from_file_location("fallback_tts", Path(__file__).resolve().parents[1] / "modal/tts_service.py")
fallback = importlib.util.module_from_spec(fallback_spec)
fallback_spec.loader.exec_module(fallback)


class VoiceRoutingTests(unittest.TestCase):
    def test_ascii_twi_does_not_switch_voice(self):
        for text in ("Akwaaba, wo ho te sen?", "Mema wo akye, mepɛ sɛ meka asɛm.", "Mepɛ tomato kilogram abien."):
            self.assertEqual(service._language_mode(text, "tw"), "twi")

    def test_only_explicit_english_mode_or_spans_switch_voice(self):
        self.assertEqual(service._language_mode("Hello there", "en"), "mixed")
        self.assertEqual(service._language_mode("Mepɛ [computer science]", "tw"), "mixed")
        self.assertEqual(service._language_mode("Mepɛ [computer", "tw"), "twi")

    def test_long_speech_preserves_every_word_and_ending(self):
        text = "Akwaaba. Wo ho te sɛn? " * 40 + "This final sentence must remain."
        clean = service._clean_text(text)
        self.assertGreater(len(clean), 500)
        chunks = service._chunk_text(clean)
        self.assertEqual(" ".join(chunks), clean)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertTrue(chunks[-1].endswith("This final sentence must remain."))

    def test_english_span_stays_together(self):
        text = "word " * 58 + "[computer science]kasa"
        chunks = service._chunk_text(service._clean_text(text))
        self.assertTrue(any("[computer science]kasa" in c for c in chunks))
        self.assertEqual(" ".join(chunks), service._clean_text(text))

    def test_oversize_is_explicit_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "text_too_long"):
            service._clean_text("a " * 2000)
        with self.assertRaisesRegex(ValueError, "speech_span_too_long"):
            service._chunk_text("a" * 301)

    def test_fallback_also_preserves_long_reply_and_currency(self):
        text = "Akwaaba. " * 80 + "Final words."
        clean = fallback._prepare_text(text, "en")
        self.assertTrue(clean.endswith("Final words."))
        self.assertEqual(" ".join(fallback._chunk_text(clean)), clean)
        self.assertIn("Ghana cedis", fallback._prepare_text("GHS 25", "en"))
        self.assertNotIn("Health Service", fallback._prepare_text("GHS 25", "en"))


if __name__ == "__main__":
    unittest.main()
