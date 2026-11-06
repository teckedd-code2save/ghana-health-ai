"""Offline durability tests using a fake Modal volume; no upload, GPU or ledger."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import checkpoint_understanding_r3 as c
from test_run_understanding_r3 import fixture, reseal, write


class FakeVolume:
    def __init__(self):
        self.files, self.puts, self.reads = {}, [], []
        self.truncate = False

    def read_file(self, name):
        self.reads.append(name)
        if name not in self.files:
            raise FileNotFoundError(name)
        raw = self.files[name]
        if self.truncate:
            raw = raw[:-1]
        for i in range(0, len(raw), 97):
            yield raw[i:i + 97]

    @contextmanager
    def batch_upload(self, force):
        if force is not False:
            raise AssertionError("Never overwrite")
        yield self

    def put_file(self, path, destination):
        if destination in self.files:
            raise FileExistsError(destination)
        self.puts.append(destination)
        self.files[destination] = path.read_bytes()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.code, self.source = self.root / "code", self.root / "source"
        self.code.mkdir(); self.source.mkdir()
        for name in c.CODE:
            path = self.code / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# research fixture\n")
        frozen = self.source / "scripts/frozen.py"
        frozen.parent.mkdir()
        frozen.write_text("# pinned research dependency\n")
        for context in (patch.object(c, "ROOT", self.code),
                        patch.object(c.runner, "FROZEN", {"scripts/frozen.py": c.sha(frozen)})):
            context.start()
            self.addCleanup(context.stop)
        self.pack = fixture(self.root / "pack")
        self.out = self.root / "snapshots/prelaunch-v1"
        self.volume = FakeVolume()
        modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda name, create_if_missing: self.volume),
                                exception=SimpleNamespace(NotFoundError=FileNotFoundError))
        context = patch.dict(sys.modules, {"modal": modal})
        context.start()
        self.addCleanup(context.stop)

    def freeze(self, **kwargs):
        return c.freeze(self.pack, self.out, "prelaunch-v1", self.source, **kwargs)

    def registered(self):
        folder = self.root / "registered"
        contents = {"README.md": "# Registered synthetic examples",
                    "requests/" + "a" * 64 + ".json": '{"model":"test","input":"synthetic research"}',
                    "responses/" + "a" * 64 + ".json": '{"output":"synthetic answer"}',
                    "versions.jsonl": '{"id":"synthetic-1","training_eligible":false}\n',
                    "observed-feedback-proofs.json": '{"private":"owner feedback must stay local"}',
                    "owner-reviews.json": '{"private":"freeform owner reviews"}',
                    "budget.sqlite3": "canonical spend store must never be read",
                    "email-attachment.json": '{"private":"attachment"}'}
        for name, raw in contents.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw)
        write(folder / "manifest.json", {"dataset_id": "synthetic-test", "dataset_version": 1,
            "schema": "r3-conversation-repair-dataset-v1",
            "files": {name: {"sha256": c.sha(folder / name), "bytes": (folder / name).stat().st_size} for name in contents}})
        return folder

    def test_prelaunch_requires_no_submission_and_preserves_exact_bytes(self):
        value = self.freeze()
        self.assertEqual(value["phase"], "pre_results")
        self.assertFalse(value["gpu_launched_by_checkpoint_tool"])
        self.assertNotIn("gpu_launched", value)
        self.assertEqual(c.verify(self.out), value)
        self.assertFalse(value["ledger_copied"])
        self.assertFalse(self.volume.puts)
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.out / "checkpoint.tar.gz").stat().st_mode & 0o777, 0o600)
        with tarfile.open(self.out / "checkpoint.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), set(value["files"]) | {"manifest.json"})
            for name, proof in value["files"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), proof["sha256"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.freeze()

    def test_pre_results_accepts_submitted_training_without_claiming_no_launch(self):
        run = self.root / "run"
        write(run / "launch.json", {"state": "submitted", "mode": "train", "call_id": "fc-fixture",
              "run_id": "understanding_20261106T062502Z", "data_sha256": c.sha(self.pack / "manifest.json")})
        value = self.freeze(runs=[run], phase="prelaunch")
        self.assertEqual(value["phase"], "pre_results")
        self.assertEqual(value["provenance"]["run_receipts"][0]["state"], "submitted")
        self.assertNotIn("gpu_launched", value)
        self.assertFalse(value["gpu_launched_by_checkpoint_tool"])

    def test_registered_synthetic_only_and_excluded_files_never_read(self):
        folder = self.registered()
        original = Path.read_bytes
        def guarded(path):
            if path.parent == folder and c.DENIED.search(path.name):
                raise AssertionError("Disallowed content opened")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            value = self.freeze(registered=[folder])
        self.assertIn("registered/0/requests/" + "a" * 64 + ".json", value["files"])
        self.assertIn("registered/0/responses/" + "a" * 64 + ".json", value["files"])
        self.assertIn("registered/0/README.md", value["files"])
        self.assertEqual(len(value["omitted"]), 4)
        self.assertFalse(any(c.DENIED.search(name) for name in value["files"]))

    def test_unregistered_or_changed_synthetic_inputs_rejected(self):
        folder = self.registered()
        (folder / ("requests/" + "a" * 64 + ".json")).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Registered artifact changed"):
            self.freeze(registered=[folder])
        self.assertFalse(self.out.exists())

    def test_pack_cannot_smuggle_denied_or_unallowlisted_bound_files(self):
        for name in ("observed-feedback-proofs.json", "owner-reviews.json", "email-attachment.json", "random-user-log.json"):
            write(self.pack / name, {"private": "not eligible"})
            reseal(self.pack)
            with self.assertRaises(ValueError):
                self.freeze()
            (self.pack / name).unlink()
        self.assertFalse(self.out.exists())

    def test_held_content_and_cases_are_allowed_research_inputs(self):
        for name in ("held-content.jsonl", "cases.jsonl", "run-selection.json"):
            write(self.pack / name, {"id": "research-development-case"})
        reseal(self.pack)
        value = self.freeze()
        self.assertIn("pack/cases.jsonl", value["files"])
        self.assertIn("pack/held-content.jsonl", value["files"])
        self.assertIn("pack/run-selection.json", value["files"])

    def test_structured_parent_admission_and_handoff_bind_registered_dataset(self):
        folder = self.registered()
        checksum = c.sha(folder / "manifest.json")
        write(self.code / c.REGISTRATION[0], {"manifest_sha256": checksum, "dataset_id": "synthetic-test"})
        write(self.code / c.REGISTRATION[1], {"dataset_manifest_sha256": checksum, "kind": "agent_review", "human_reviewed": False,
            "records": [{"id": "synthetic-1", "decision": "admit", "record_sha256": "a" * 64, "note": "source checked"}]})
        value = self.freeze(registered=[folder])
        self.assertIn("registration/" + c.REGISTRATION[1], value["files"])
        self.assertIn("code/scripts/admit_understanding_r3_repairs.py", value["files"])
        self.assertIn("code/scripts/test_admit_understanding_r3_repairs.py", value["files"])
        self.assertIn("code/scripts/prepare_understanding_r3_evaluation.py", value["files"])

    def test_wrong_dataset_parent_review_is_rejected(self):
        folder = self.registered()
        write(self.code / c.REGISTRATION[1], {"dataset_manifest_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "unselected dataset"):
            self.freeze(registered=[folder])

    def test_only_explicit_receipts_and_postrun_requires_training(self):
        run = self.root / "run"
        write(run / "train.call.json", {"state": "submitted", "mode": "train", "run_id": "understanding_20261106T100000Z",
              "call_id": "fc-fixture", "data_sha256": c.sha(self.pack / "manifest.json")})
        write(run / "train-progress-01/status.json", {"state": "paused_budget", "mode": "train",
              "training_completed": False, "data_sha256": c.sha(self.pack / "manifest.json")})
        for name in ("audition.sqlite3", "budget.sqlite3", "owner-reviews.json", "observed-feedback-proofs.json", "unknown.json"):
            write(run / name, {"must": "remain local"})
        with self.assertRaisesRegex(ValueError, "training receipt"):
            self.freeze(phase="postrun")
        value = self.freeze(runs=[run], phase="postrun")
        self.assertIn("receipts/0/train.call.json", value["files"])
        self.assertIn("receipts/0/train-progress-01/status.json", value["files"])
        self.assertEqual(len([n for n in value["files"] if n.startswith("receipts/")]), 2)
        paused = next(r for r in value["provenance"]["run_receipts"] if r.get("state") == "paused_budget")
        self.assertFalse(paused["training_completed"])

    def test_receipt_must_belong_to_selected_pack(self):
        run = self.root / "run"
        write(run / "train.call.json", {"mode": "train", "data_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "another pack"):
            self.freeze(runs=[run])

    def test_execution_includes_bound_code_and_pack_but_not_arbitrary_neighbors(self):
        folder = self.root / "execution"
        shutil.copytree(self.pack, folder / "inputs")
        execution = {"mode": "smoke", "data_sha256": c.sha(self.pack / "manifest.json"),
                     "code_hashes": {}, "reused_code_hashes": {}}
        for group, names, field in (("r3", c.runner.CODE, "code_hashes"),
                                   ("reused", c.runner.REUSED, "reused_code_hashes"),
                                   ("frozen", c.runner.FROZEN, None)):
            for name in names:
                path = folder / group / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# sealed package fixture")
                if field:
                    execution[field][name] = c.sha(path)
        write(folder / "execution.json", execution)
        write(folder / "audition.sqlite3", {"not": "selected"})
        with patch.object(c.runner, "verify_execution", return_value=execution) as validate:
            value = self.freeze(executions=[folder])
        validate.assert_called_once_with(folder)
        self.assertIn("executions/0/inputs/source-manifests.json", value["files"])
        self.assertIn("executions/0/r3/modal/train/train_understanding_r3.py", value["files"])
        self.assertFalse(any("sqlite" in name for name in value["files"]))

    def test_real_code_allowlist_passes_credential_screen(self):
        source_root = Path(c.__file__).resolve().parents[1]
        expected = {}
        # Public checkouts omit internal notes; retain their synthetic fixtures.
        for name in c.CODE:
            source = source_root / name
            if source.suffix == ".py" and source.is_file():
                shutil.copyfile(source, self.code / name)
                expected["code/" + name] = c.sha(source)
        value = self.freeze()
        for name, checksum in expected.items():
            self.assertEqual(value["files"][name]["sha256"], checksum)
        self.assertIn("code/scripts/checkpoint_understanding_r3.py", value["files"])
        self.assertIn("code/docs/TWI_UNDERSTANDING_R3_20261106.md", value["files"])

    def test_source_mutation_during_freeze_is_not_published(self):
        target = self.code / "tmp/r3-runner-contract.md"
        original = Path.read_bytes
        def changed(path):
            raw = original(path)
            if path == target:
                path.write_bytes(raw + b"changed")
            return raw
        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(ValueError, "changed during freeze"):
                self.freeze()
        self.assertFalse(self.out.exists())

    def test_selected_secrets_database_magic_and_links_rejected(self):
        selected = self.code / "docs/TWI_UNDERSTANDING_R3_20261106.md"
        for raw in (b"-----BEGIN " + b"PRIVATE KEY-----", b"sk-proj-" + b"X" * 35, b"SQLite format 3\x00payload"):
            selected.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "credential/database"):
                self.freeze()
        selected.unlink()
        selected.symlink_to(self.code / "tmp/r3-runner-contract.md")
        with self.assertRaises(ValueError):
            self.freeze()
        self.assertFalse(self.out.exists())

    def test_structured_credential_field_rejected(self):
        folder = self.registered()
        path = folder / ("requests/" + "a" * 64 + ".json")
        write(path, {"api_key": "an-otherwise-unrecognized-secret"})
        manifest = c.read(folder / "manifest.json")
        manifest["files"][path.relative_to(folder).as_posix()] = {"sha256": c.sha(path), "bytes": path.stat().st_size}
        write(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Credential field"):
            self.freeze(registered=[folder])

    def test_upload_needs_authorization_before_reading_or_importing(self):
        with self.assertRaisesRegex(ValueError, "authorize-upload"):
            c.remote(self.root / "missing", "test", upload=True)
        self.assertFalse(self.volume.puts)

    def test_full_readback_idempotent_upload_and_new_version(self):
        self.freeze()
        first = c.remote(self.out, "test", upload=True, authorize=True)
        self.assertEqual(first["prefix"], c.PREFIX + "/prelaunch-v1")
        self.assertTrue(first["full_remote_readback_verified"])
        self.assertEqual(len(self.volume.puts), 2)
        c.remote(self.out, "test", upload=True, authorize=True)
        c.remote(self.out, "test")
        self.assertEqual(len(self.volume.puts), 2)
        newer = self.root / "snapshots/prelaunch-v2"
        c.freeze(self.pack, newer, "prelaunch-v2", self.source)
        c.remote(newer, "test", upload=True, authorize=True)
        self.assertEqual(len(self.volume.puts), 4)

    def test_conflicting_remote_version_never_partially_uploads(self):
        self.freeze()
        self.volume.files[c.PREFIX + "/prelaunch-v1/checkpoint.tar.gz"] = b"different snapshot"
        with self.assertRaisesRegex(ValueError, "Immutable remote"):
            c.remote(self.out, "test", upload=True, authorize=True)
        self.assertFalse(self.volume.puts)
        self.assertFalse((self.out / "storage.json").exists())

    def test_partial_upload_can_resume_without_overwriting(self):
        self.freeze()
        self.volume.files[c.PREFIX + "/prelaunch-v1/manifest.json"] = (self.out / "manifest.json").read_bytes()
        c.remote(self.out, "test", upload=True, authorize=True)
        self.assertEqual(self.volume.puts, [c.PREFIX + "/prelaunch-v1/checkpoint.tar.gz"])

    def test_existing_storage_receipt_cannot_silently_switch_profile(self):
        self.freeze()
        c.remote(self.out, "test", upload=True, authorize=True)
        with self.assertRaisesRegex(ValueError, "storage receipt differs"):
            c.remote(self.out, "different-profile", upload=True, authorize=True)
        self.assertEqual(len(self.volume.puts), 2)

    def test_missing_and_truncated_readback_fail(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "missing"):
            c.remote(self.out, "test")
        c.remote(self.out, "test", upload=True, authorize=True)
        self.volume.truncate = True
        with self.assertRaisesRegex(ValueError, "differs"):
            c.remote(self.out, "test")

    def test_upload_has_no_success_receipt_without_complete_readback(self):
        self.freeze()
        original = self.volume.put_file
        def put(path, destination):
            original(path, destination)
            if destination.endswith("checkpoint.tar.gz"):
                self.volume.truncate = True
        with patch.object(self.volume, "put_file", side_effect=put):
            with self.assertRaisesRegex(ValueError, "Complete remote checksum"):
                c.remote(self.out, "test", upload=True, authorize=True)
        self.assertFalse((self.out / "storage.json").exists())

    def test_archive_tampering_rejected(self):
        self.freeze()
        with (self.out / "checkpoint.tar.gz").open("wb") as stream:
            stream.write(b"not an archive")
        with self.assertRaises(tarfile.ReadError):
            c.verify(self.out)

    def test_forged_archive_cannot_expand_allowlist(self):
        value = self.freeze()
        name, raw = "code/scripts/unapproved_user_export.py", b"# not allowlisted"
        original = {}
        with tarfile.open(self.out / "checkpoint.tar.gz") as archive:
            original = {m.name: archive.extractfile(m).read() for m in archive if m.name != "manifest.json"}
        value["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        write(self.out / "manifest.json", value)
        with tarfile.open(self.out / "checkpoint.tar.gz", "w:gz") as archive:
            original.update({"manifest.json": (self.out / "manifest.json").read_bytes(), name: raw})
            for path, data in original.items():
                item = tarfile.TarInfo(path)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            c.verify(self.out)


if __name__ == "__main__":
    unittest.main()
