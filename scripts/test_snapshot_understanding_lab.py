import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from snapshot_understanding_lab import snapshot


class SnapshotTests(unittest.TestCase):
    def test_preserves_evidence_without_browser_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "lab.sqlite3"
            state = {"id": "session", "loop_guard": False, "rounds": [
                {"id": "round", "question": "question", "sides": {
                    "base": {"status": "incomplete", "verified": True,
                             "turn": {"answer": "cut \ufffd", "messages": []}, "error": None}}}]}
            with sqlite3.connect(store) as db:
                db.execute("CREATE TABLE r4_sessions(id TEXT,record TEXT)")
                db.execute("CREATE TABLE r4_reviews(id INTEGER,note TEXT)")
                db.execute("CREATE TABLE r4_meta(key TEXT,value TEXT)")
                db.execute("INSERT INTO r4_meta VALUES('browser_secret','do-not-copy')")
                db.execute("INSERT INTO r4_sessions VALUES(?,?)", ("session", json.dumps(state)))
            before = store.read_bytes()
            out = root / "out"
            result = snapshot(store, out)
            self.assertEqual(result["reply_slots"], 1)
            self.assertEqual(len(result["invalid_unicode_outputs"]), 1)
            self.assertEqual(json.loads((out / "sessions.json").read_text()), [state])
            self.assertEqual(store.read_bytes(), before)
            self.assertFalse(any("do-not-copy" in f.read_text() for f in out.iterdir()))
            with self.assertRaises(ValueError):
                snapshot(store, out)


if __name__ == "__main__":
    unittest.main()
