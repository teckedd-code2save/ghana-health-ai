"""Offline tests; no GPU, provider calls, or generated training labels."""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal/train"))
from medical_pilot_core import check_pilot_files, encode_row, parsed_interpretation


class Tokenizer:
    eos_token_id = 4

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3, 4]

    def __call__(self, text, **kwargs):
        return {"input_ids": [3]}


class CoreTests(unittest.TestCase):
    def test_remote_relocated_import(self):
        import modal

        source = Path(__file__).resolve().parents[1] / "modal/train/train_medical_response_pilot.py"
        module = types.ModuleType("medical_pilot_remote_import_test")
        module.__file__ = "/root/train_medical_response_pilot.py"
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(modal, "is_local", return_value=False):
            exec(compile(source.read_text(), module.__file__, "exec"), module.__dict__)
        self.assertEqual(module.ROOT, Path("/root"))

    def test_prompt_mask_and_no_target_truncation(self):
        row = {"id": "fixture", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        self.assertEqual(encode_row(Tokenizer(), row, 4)["labels"], [-100, -100, 3, 4])
        with self.assertRaises(ValueError):
            encode_row(Tokenizer(), row, 3)

    def test_schema_fails_closed(self):
        self.assertIsNone(parsed_interpretation("[]"))
        self.assertIsNone(parsed_interpretation('{"reply":"hello"}'))
        self.assertIsNone(parsed_interpretation("Here is JSON: {}"))

    def test_actual_export(self):
        root = Path(__file__).resolve().parents[1] / "tmp/medical-response-pilot/v1"
        manifest = json.loads((root / "manifest.json").read_text())
        train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
        holdout = [json.loads(line) for line in (root / "holdout.jsonl").read_text().splitlines()]
        check_pilot_files(manifest, train, holdout)
        corrupted = json.loads(json.dumps(holdout))
        corrupted[0]["provenance"]["group_id"] = train[0]["provenance"]["group_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_pilot_files(manifest, train, corrupted)


if __name__ == "__main__":
    unittest.main()
