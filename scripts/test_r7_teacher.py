import importlib.util
from pathlib import Path
import tempfile
import unittest

from r7_teacher_controls import controls

spec = importlib.util.spec_from_file_location("teacher_contract", Path(__file__).resolve().parents[1] / "modal/train/r7_teacher.py")
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": r["id"], "messages": r["messages"]} for r in controls()]

    def test_controls_do_not_send_gold(self):
        self.assertEqual(len(self.rows), 22)
        self.assertTrue(all(r["training_eligible"] is False for r in controls()))
        self.assertTrue(teacher.validate("r7-test-v1", "gemma31", self.rows))
        with self.assertRaises(ValueError):
            teacher.validate("r7-test-v1", "gemma31", controls())

    def test_identity_binds_source_and_profile(self):
        first = teacher.validate("r7-test-v1", "gemma31", self.rows)
        self.assertNotEqual(first, teacher.validate("r7-test-v1", "qwen35", self.rows))
        self.rows[0]["messages"][-1]["content"] += " changed"
        self.assertNotEqual(first, teacher.validate("r7-test-v1", "gemma31", self.rows))

    def test_duplicate_and_empty_requests_rejected(self):
        for rows in [[], self.rows + self.rows, self.rows * 4]:
            with self.assertRaises(ValueError):
                teacher.validate("r7-test-v1", "gemma31", rows)

    def test_incomplete_history_rejected(self):
        for messages in [[{"role": "assistant", "content": "fake"}],
                         [{"role": "user", "content": "one"}, {"role": "user", "content": "two"},
                          {"role": "user", "content": "three"}]]:
            with self.assertRaises(ValueError):
                teacher.validate("r7-test-v1", "gemma31", [{"id": "bad", "messages": messages}])

    def test_atomic_save(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run" / "progress.json"
            teacher.save(path, {"one": 1})
            teacher.save(path, {"one": 2})
            self.assertEqual(json.loads(path.read_text()), {"one": 2})
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_path_and_profile_validation(self):
        for run, profile in [("../bad", "gemma31"), ("good-run", "missing")]:
            with self.assertRaises(ValueError):
                teacher.validate(run, profile, self.rows)


if __name__ == "__main__":
    unittest.main()
