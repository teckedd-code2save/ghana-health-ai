import copy
import unittest

from calibrate_dialogue_translation import digest, requests, structural_flags, validate_report


class DialogueTranslationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "public:a", "messages": [
            {"role": "user", "content": "Find a book."}, {"role": "assistant", "content": "Which book?"},
            {"role": "user", "content": "I forgot."}, {"role": "assistant", "content": "Describe it."}]}]
        self.demos = [{"en": "English " + str(i), "tw": "Target " + str(i)} for i in range(3)]

    def test_context_and_isolated_do_not_change_originals(self):
        before = copy.deepcopy(self.rows)
        result = requests(self.rows, "gemma", self.demos)
        self.assertEqual(len(result), 8)
        self.assertEqual(self.rows, before)
        self.assertNotIn("Which book?", result[0]["messages"][-1]["content"])
        self.assertIn("Which book?", result[4]["messages"][-1]["content"])
        native = requests(self.rows, "afrique", self.demos)
        self.assertEqual(len(native), 4)
        self.assertTrue(native[0]["prompt"].endswith("English: Find a book.\nTwi:"))

    def test_bad_role_duplicate_and_unexpected_fields_rejected(self):
        for change in (lambda r: r.append(r[0]), lambda r: r[0].update(rubric="secret"),
                       lambda r: r[0]["messages"][0].update(role="system"),
                       lambda r: r[0]["messages"][0].update(content="")):
            rows = copy.deepcopy(self.rows)
            change(rows)
            with self.assertRaises(ValueError):
                requests(rows, "gemma", self.demos)

    def test_structural_flags_do_not_certify_meaning(self):
        self.assertIn("numeric_surface_mismatch_review", structural_flags("2 kg", "3 kg", "stop"))
        self.assertIn("incomplete_generation", structural_flags("Hi", "Hello", "length"))
        self.assertIn("repetition", structural_flags("Hi", "a b c d e f g h " * 5, "stop"))
        self.assertEqual(structural_flags("A short source", "A different meaning", "stop"), [])

    def test_results_require_exact_complete_input_and_turn_identity(self):
        from dialogue_translation_core import MODELS
        req = requests(self.rows, "gemma", self.demos)
        receipt = {"run_id": "run", "variant": "gemma", "inputs_sha256": digest(self.rows), "demonstrations_sha256": digest(self.demos)}
        report = {**receipt, "model": MODELS["gemma"][0], "revision": MODELS["gemma"][1],
                  "requests_sha256": digest(req), "results": [{k: r[k] for k in ("id", "source_id", "turn", "method")} for r in req]}
        validate_report(report, receipt, self.rows, self.demos)
        for change in (lambda r: r.update(revision="wrong"), lambda r: r["results"].pop(),
                       lambda r: r["results"][0].update(turn=3), lambda r: r.update(requests_sha256="wrong")):
            altered = copy.deepcopy(report)
            change(altered)
            with self.assertRaises(ValueError):
                validate_report(altered, receipt, self.rows, self.demos)

    def test_preemption_restores_matching_partial_without_repeating_work(self):
        import json
        import tempfile
        from pathlib import Path
        from dialogue_translation_core import restore_progress
        req = requests(self.rows, "gemma", self.demos)
        expected = {"run_id": "run", "variant": "gemma", "model": "model", "revision": "revision",
            "inputs_sha256": digest(self.rows), "inputs": self.rows,
            "demonstrations_sha256": digest(self.demos), "demonstrations": self.demos,
            "requests_sha256": digest(req), "requests": req, "runtime": "pinned", "thinking": False,
            "decoding": {"temperature": 0}, "results": [], "started_at_unix": 123}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(restore_progress(folder, expected), expected)
            partial = {**expected, "results": [{k: req[0][k] for k in ("id", "source_id", "turn", "method")}]}
            (folder / "partial.json").write_text(json.dumps(partial))
            restored = restore_progress(folder, expected)
            self.assertEqual(restored["results"], partial["results"])
            self.assertEqual(restored["resume_count"], 1)
            with self.assertRaises(ValueError):
                restore_progress(folder, {**expected, "decoding": {"temperature": 1}})
            partial["inputs"] = []
            (folder / "partial.json").write_text(json.dumps(partial))
            with self.assertRaises(ValueError):
                restore_progress(folder, expected)


if __name__ == "__main__":
    unittest.main()
