import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r3_evaluation as e


class EvaluationTests(unittest.TestCase):
    def test_freezes_verified_length_limited_comparison_and_preserves_old_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            old.mkdir()
            messages = [{"role": "user", "content": "Example test question"}]
            e.save(old / "cases.json", [{"id": "original", "messages": messages}])
            e.save(old / "language-heldout.json", [{"id": "held", "text": "heldout"}])
            hashes = {p.name: e.sha(p) for p in old.iterdir()}
            e.save(old / "manifest.json", {"artifacts": hashes})
            state = {"rounds": [{"id": "round1", "status": "incomplete", "verified": True,
                "shared_messages": messages, "messages_sha256": e.digest(messages), "sides": {}},
                {"id": "round2", "status": "error", "verified": False}]}
            store = root / "store.sqlite3"
            with sqlite3.connect(store) as db:
                db.execute("CREATE TABLE comparison_sessions (id TEXT, record TEXT)")
                db.execute("INSERT INTO comparison_sessions VALUES (?,?)", ("s1", json.dumps(state)))
            rollout = root / "rollouts.json"
            e.save(rollout, [{"id": "rollout1", "language": "tw", "user_turns": ["first", "followup"], "rubric": ["Keep meaning"]}])
            with patch.object(e, "R2_EVALUATION_SHA", e.sha(old / "manifest.json")):
                result = e.prepare(old, rollout, store, ["s1"], root / "out")
            self.assertEqual(result["counts"]["fixed_prompts"], 2)
            self.assertEqual(result["counts"]["owner_observed_prompts"], 1)
            self.assertEqual(result["kind"], "development_not_blind_final")
            self.assertEqual((root / "out/language-heldout.json").read_bytes(), (old / "language-heldout.json").read_bytes())
            self.assertEqual(e.read(root / "out/FROZEN.json")["manifest_sha256"], e.sha(root / "out/manifest.json"))
            self.assertEqual(e.sha(old / "cases.json"), hashes["cases.json"])


if __name__ == "__main__":
    unittest.main()
