import json
from pathlib import Path
import tempfile
import unittest

from checkpoint_understanding_r4 import freeze, verify


class SnapshotTests(unittest.TestCase):
    def test_round_trip_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "record.json"
            source.write_text('{"source_id":"one"}')
            out = root / "snapshot"
            result = freeze(["research=" + str(source)], out)
            self.assertEqual(result["files"], 1)
            self.assertEqual(len(verify(out)["files"]), 1)
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["files"]["research/record.json"]["sha256"] = "0" * 64
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify(out)

    def test_user_store_and_link_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chat.sqlite3"
            source.write_bytes(b"SQLite format 3")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                freeze(["research=" + str(source)], root / "snapshot")
            link = root / "linked.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "linked"):
                freeze(["research=" + str(link)], root / "snapshot2")


if __name__ == "__main__":
    unittest.main()
