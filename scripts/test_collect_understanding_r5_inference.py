"""Offline R5 volume readback tests; the fake volume cannot launch or mutate."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import collect_understanding_r5_inference as collector
import prepare_understanding_r5_inference as builder
import r5_inference as r5
from test_prepare_understanding_r5_inference import fixture, HELPERS


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent, self.parent_sha, self.readback, self.run = fixture(self.root)
        self.receipt = r5.read(self.root / "receipt.json")
        self.terminal = r5.read(self.root / "terminal.json")
        self.identifier = self.receipt["run_id"]
        self.calls = []
        outer = self

        class Volume:
            def read_file(self, path):
                outer.calls.append(path)
                if not path.startswith(outer.receipt["remote_directory"] + "/"):
                    raise AssertionError("Read outside approved run")
                data = (outer.readback / path.lstrip("/")).read_bytes()
                for index in range(0, len(data), 37):
                    yield data[index:index + 37]
        self.volume = Volume()

    def collect(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return collector.collect(self.volume, self.root / "receipt.json", self.root / "terminal.json",
                                     self.root / "output", **options)

    def test_generic_completed_receipt_collects_and_prepares_without_new_hash_pin(self):
        original = (self.root / "terminal.json").read_bytes()
        result = self.collect(include_pass1=True)
        snapshot = self.root / "output"
        self.assertFalse(result["gpu_called"])
        self.assertFalse(result["remote_mutated"])
        self.assertEqual(result["adapter_weights_sha256"],
                         self.terminal["adapters"][0]["files"]["adapter_model.safetensors"])
        self.assertEqual((snapshot / "terminal.json").read_bytes(), original)
        report = r5.read(snapshot / "READBACK.json")
        self.assertEqual(report["network_operations"], ["volume.read_file"])
        self.assertFalse(report["checkpoint_state_binary_collected"])
        self.assertTrue(any("responses-pass1-adapter/adapter_model" in path for path in self.calls))
        self.assertFalse(any(path.endswith("/state.pt") or path.endswith("/rows.jsonl") for path in self.calls))
        built = builder.prepare(self.parent, self.parent_sha, snapshot / "receipt.json", snapshot / "terminal.json",
                                snapshot, self.root / "bundle", helpers_root=HELPERS, include_pass1=True)
        self.assertIn("r5_pass1", built["variants"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.collect()

    def test_canonical_paths_cannot_escape_final_run(self):
        physical = self.terminal["physical_directory"]
        expected = "/artifacts" + self.receipt["remote_directory"] + "/responses-adapter/adapter_model.safetensors"
        self.assertEqual(collector.canonical(physical + "/responses-adapter/adapter_model.safetensors",
                                            self.identifier, physical), expected)
        self.assertEqual(collector.canonical(expected, self.identifier, physical), expected)
        for value in (physical + "/../elsewhere", "/artifacts/another-run/file", "relative", physical):
            with self.assertRaises(ValueError):
                collector.canonical(value, self.identifier, physical)

    def test_optional_approved_sha_rejects_before_weights_read(self):
        with self.assertRaisesRegex(ValueError, "approved SHA"):
            self.collect(expected_weights="0" * 64)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse(any(path.endswith("adapter_model.safetensors") for path in self.calls))

    def test_unfinished_or_wrong_run_rejected_before_any_volume_read(self):
        with self.assertRaises(ValueError):
            self.collect(identifier="understanding_19990101T000000Z")
        self.terminal.update(state="paused_budget", training_completed=False)
        r5.base.save(self.root / "terminal.json", self.terminal)
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.calls, [])

    def test_truncated_weights_leave_no_completed_snapshot(self):
        original = self.volume.read_file

        def truncated(path):
            if path.endswith("adapter_model.safetensors"):
                yield b"partial"
            else:
                yield from original(path)
        self.volume.read_file = truncated
        before = (self.root / "terminal.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.collect()
        self.assertFalse((self.root / "output").exists())
        self.assertEqual((self.root / "terminal.json").read_bytes(), before)

    def test_status_change_during_readback_is_not_sealed(self):
        original = self.volume.read_file
        reads = 0

        def changed(path):
            nonlocal reads
            if path.endswith("/status.json"):
                reads += 1
                if reads > 1:
                    yield b'{"state":"changed"}'
                    return
            yield from original(path)
        self.volume.read_file = changed
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.collect()
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
