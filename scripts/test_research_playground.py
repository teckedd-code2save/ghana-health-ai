import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from research_playground import browser_secret, candidate_variants, display, existing_note, history_messages, restored, response_candidate, save_review, save_turn

spec = importlib.util.spec_from_file_location("pilot_service", Path(__file__).resolve().parents[1] / "modal/research_pilot_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class PlaygroundTests(unittest.TestCase):
    def test_response_candidate_is_explicit_and_checksum_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selection.json"
            self.assertIsNone(response_candidate(path))
            path.write_text(json.dumps({"enabled": False}))
            self.assertIsNone(response_candidate(path))
            selection = {"enabled": True, "run_id": "response_v2_20260910T093637Z", "checkpoint": "adapter", "adapter_sha256": "a" * 64}
            path.write_text(json.dumps(selection))
            self.assertEqual(response_candidate(path)["adapter_sha256"], "a" * 64)
            self.assertEqual(candidate_variants(response_candidate(path)), ["response_v2", "response_base"])
            path.write_text(json.dumps({**selection, "run_id": "afrique_v3_20260910T103220Z"}))
            self.assertEqual(candidate_variants(response_candidate(path)), ["afrique_v3", "afrique_base"])
            path.write_text(json.dumps({**selection, "run_id": "afrique_v6_20260910T160000Z"}))
            self.assertEqual(candidate_variants(response_candidate(path)), ["afrique_v6", "afrique9_base"])
            self.assertEqual(candidate_variants(None), [])
            path.write_text(json.dumps({**selection, "checkpoint": "../other"}))
            with self.assertRaises(ValueError):
                response_candidate(path)

    def test_rejects_invalid_remote_input_before_loading(self):
        for messages, variant in [([], "pilot"), ([{"role": "system", "content": "x"}], "pilot"),
                                  ([{"role": "user", "content": "x"}], "other"),
                                  ([{"role": "user", "content": " "}], "base"),
                                  ([{"role": "user", "content": "x" * 8001}], "base")]:
            with self.assertRaises(ValueError):
                service.validate_messages(messages, variant)

    def test_actual_history_without_ui_labels_or_failed_answers(self):
        turns = [{"id": "one", "question": "Hello", "answer": "Hi", "variant": "pilot", "status": "complete"},
                 {"id": "two", "question": "bad", "answer": "partial", "variant": "base", "status": "failed"}]
        messages = history_messages(turns, "How are you?")
        self.assertEqual([m["content"] for m in messages], ["Hello", "Hi", "How are you?"])
        service.validate_messages(messages, "pilot")
        self.assertIn("Pilot v1", display(turns)[1]["content"])
        self.assertNotIn("Pilot v1", json.dumps(messages))

    def test_refresh_preserves_answer_and_marks_interrupted_turn(self):
        turns = [{"id": "one", "question": "hi", "answer": "hello", "variant": "pilot", "status": "streaming"}]
        result = restored(turns)
        self.assertEqual(result[0]["answer"], "hello")
        self.assertEqual(result[0]["status"], "stopped")
        self.assertEqual(restored([{"variant": "invented"}]), [])

    def test_reviews_require_real_record_and_preserve_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reviews.sqlite3"
            secret = browser_secret(path)
            self.assertEqual(secret, browser_secret(path))
            save_turn({"id": "test", "provenance": {"variant": "pilot", "adapter_sha256": service.SHA256}}, path)
            save_review("test", "Incorrect", "Automated test; not a human label", path)
            self.assertEqual(existing_note("test", path), "Automated test; not a human label")
            with self.assertRaises(ValueError):
                save_review("nonexistent", "Useful", "", path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("SELECT rating FROM reviews").fetchone()[0], "Incorrect")
                self.assertEqual(json.loads(db.execute("SELECT record FROM turns").fetchone()[0])["provenance"]["adapter_sha256"], service.SHA256)


if __name__ == "__main__":
    unittest.main()
