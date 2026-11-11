"""Offline checks for bounded result collection; no Modal calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from run_understanding_r4_evaluation import download_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def volume(self, entries, values):
        return SimpleNamespace(
            iterdir=lambda *a, **k: [SimpleNamespace(path=p, type=SimpleNamespace(name="FILE")) for p in entries],
            read_file=lambda p: iter([values[p]]),
        )

    def test_complete_bytes_and_nested_paths(self):
        values = {"/run/one.json": b'{"a":1}', "/run/records/two.json": b'{"b":2}'}
        download_snapshot(self.volume(list(values), values), "/run", self.out)
        for name, content in values.items():
            self.assertEqual((self.out / name.removeprefix("/run/")).read_bytes(), content)

    def test_rejects_unexpected_prefix_before_downloading(self):
        with self.assertRaises(ValueError):
            download_snapshot(self.volume(["/different/one.json"], {}), "/run", self.out)
        self.assertFalse(list(self.out.rglob("*")))

    def test_rejects_duplicate_paths(self):
        with self.assertRaises(ValueError):
            download_snapshot(self.volume(["/run/one.json", "run/one.json"], {}), "/run", self.out)

    def test_never_overwrites_existing_results(self):
        (self.out / "one.json").write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            download_snapshot(self.volume(["/run/one.json"], {"/run/one.json": b"changed"}), "/run", self.out)
        self.assertEqual((self.out / "one.json").read_bytes(), b"original")

    def test_propagates_network_errors(self):
        def unavailable(_):
            raise ConnectionError("fixture")
        volume = self.volume(["/run/one.json"], {})
        volume.read_file = unavailable
        with self.assertRaises(ConnectionError):
            download_snapshot(volume, "/run", self.out)

    def test_concurrency_is_bounded(self):
        for workers in (0, 9):
            with self.assertRaises(ValueError):
                download_snapshot(self.volume([], {}), "/run", self.out, workers=workers)


if __name__ == "__main__":
    unittest.main()
