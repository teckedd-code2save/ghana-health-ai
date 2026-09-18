import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("corpus_checkpoint", Path(__file__).resolve().parents[1] / "modal/train/corpus_checkpoint.py")
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class CheckpointTests(unittest.TestCase):
    def test_stages_and_checks_original_blob_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "source"; source.mkdir()
            value = b"test checkpoint, not model weights"
            blob = root / hashlib.sha256(value).hexdigest(); blob.write_bytes(value)
            (source / "one.safetensors").symlink_to(blob)
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "one.safetensors"}}))
            result = checkpoint.stage_checkpoint(source, root / "staged", "pinned")
            self.assertEqual(result["bytes"], len(value))
            self.assertEqual((root / "staged/one.safetensors").read_bytes(), value)
            with self.assertRaises(ValueError): checkpoint.stage_checkpoint(source, root / "staged", "pinned")
            blob.write_bytes(b"tampered")
            with self.assertRaises(ValueError): checkpoint.stage_checkpoint(source, root / "bad", "pinned")
            self.assertFalse((root / "bad").exists())

    def test_rejects_index_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../one.safetensors"}}))
            with self.assertRaises(ValueError): checkpoint.stage_checkpoint(root, root / "out", "pinned")


if __name__ == "__main__": unittest.main()
