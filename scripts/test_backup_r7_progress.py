import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import backup_r7_progress as backup


class BackupTests(unittest.TestCase):
    def test_streamed_archive_matches_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "R7-STATUS.md").write_text("A private status")
            data = root / "tmp/r7-20261117"
            data.mkdir(parents=True)
            (data / "result.json").write_text('{"value":42}')
            runtime = data / "baseline-launch-py311"
            runtime.mkdir()
            (runtime / "package.json").write_text('{"public_runtime":"reinstallable"}')
            with patch.object(backup, "ROOT", root):
                path, checksum, manifest = backup.build(data / "backups")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            self.assertFalse(any("baseline-launch-py311" in name for name in manifest["files"]))
            with tarfile.open(path) as archive:
                self.assertEqual(json.load(archive.extractfile("BACKUP_MANIFEST.json")), manifest)
                for name, expected in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), expected)
                    self.assertEqual(archive.getmember(name).mode, 0o600)

    def test_credentials_across_read_boundaries_rejected(self):
        reader = backup.CheckedReader(io.BytesIO(b"prefix sk-proj-" + b"x" * 32))
        reader.read(12)
        with self.assertRaises(ValueError):
            reader.read(40)

    def test_symlinks_rejected_and_partial_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside.md").write_text("outside")
            (root / "R7-STATUS.md").symlink_to(root / "outside.md")
            with patch.object(backup, "ROOT", root), self.assertRaises(ValueError):
                backup.build(root / "backups")
            self.assertFalse((root / "backups/snapshot.tmp").exists())


if __name__ == "__main__":
    unittest.main()
