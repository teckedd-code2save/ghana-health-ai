import unittest

from summarize_language_checkpoint import repetition_count, summarize


class CheckpointSummaryTests(unittest.TestCase):
    def events(self):
        return [{"type": "provenance", "comparison_scope": "Paired frozen base and adapter"},
                *[{"type": "prediction", "variant": variant, "id": "fixture",
                   "messages": [{"role": "user", "content": "Test"}],
                   "prediction": "A complete answer.", "limit_reached": False}
                  for variant in ("base", "adapter")]]

    def test_reports_flags_not_semantic_scores(self):
        events = self.events()
        events[-1].update(prediction="one two three four five six seven eight " * 8, limit_reached=True)
        report = summarize(events)
        self.assertEqual(report["cases_per_variant"], 1)
        self.assertEqual(report["variants"]["adapter"]["repetition_flag_ids"], ["fixture"])
        self.assertEqual(report["variants"]["base"]["repetition_flag_ids"], [])
        self.assertNotIn("accuracy", report)

    def test_incomplete_duplicates_and_changed_prompts_rejected(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[-1])),
                       lambda rows: rows[-1].update(messages=[])):
            rows = self.events()
            change(rows)
            with self.assertRaises(ValueError):
                summarize(rows)

    def test_unicode_and_short_text(self):
        self.assertEqual(repetition_count(""), 0)
        self.assertEqual(repetition_count("Ɛyɛ dɛn?"), 0)
        self.assertGreaterEqual(repetition_count("ɛyɛ sɛ ɛbɛyɛ yɛn ho asɛm wɔ hɔ " * 5), 5)


if __name__ == "__main__":
    unittest.main()
