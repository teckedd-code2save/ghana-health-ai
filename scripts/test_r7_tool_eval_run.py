"""Offline packaging/leak/launch-gate tests; no Modal resources or inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_r7_tool_candidates as candidates
import r7_tool_eval as ev
import r7_tool_eval_run as run
from test_r7_tool_eval import IDENTITY, ROOT, TOKENIZER, ScriptedRuntime, native_call
from check_afrique_training_contract import load_tokenizer


class SealTests(unittest.TestCase):
    def test_seal_rejects_wrong_pin_extra_file_and_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ev.save(folder / "item.json", {"actual": True})
            checksum = run.seal(folder, {"version": "test"})
            self.assertEqual(ev.verify_seal(folder, checksum)["version"], "test")
            with self.assertRaises(ValueError):
                ev.verify_seal(folder, "0" * 64)
            ev.save(folder / "extra.json", {})
            with self.assertRaisesRegex(ValueError, "Unbound"):
                ev.verify_seal(folder, checksum)
            (folder / "extra.json").unlink()
            ev.save(folder / "item.json", {"actual": False})
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                ev.verify_seal(folder, checksum)

    def test_seal_rejects_links(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ev.save(folder / "item.json", {})
            checksum = run.seal(folder, {})
            (folder / "alias").symlink_to(folder / "item.json")
            with self.assertRaisesRegex(ValueError, "Linked"):
                ev.verify_seal(folder, checksum)

    def test_source_import_does_not_write_bytecode_into_sealed_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "test_module.py"
            source.write_text("value = 7\n")
            module = ev.import_file("_tool_eval_test_import", source)
            self.assertEqual(module.value, 7)
            self.assertFalse((folder / "__pycache__").exists())

    def test_launch_needs_explicit_authorization_before_resource_contact(self):
        with patch.object(run, "verify_execution", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "authorize-gpu"):
                run.launch("missing", "0" * 64, "missing-receipt", "test", "main")

    def test_launch_rejects_wrong_python_before_modal_import(self):
        with patch.object(run, "verify_execution", return_value=({}, {}, {}, [])), patch.object(run.sys, "version_info", (3, 13)):
            with self.assertRaisesRegex(ValueError, "Python 3.11"):
                run.launch("missing", "0" * 64, "receipt", "test", "main", authorize_gpu=True)


@unittest.skipUnless(TOKENIZER.is_dir(), "Pinned native tokenizer unavailable")
class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source, cls.evaluation = cls.root / "source", cls.root / "eval"
        candidates.prepare(cls.source)
        cls.prepared = run.prepare(cls.source, TOKENIZER, cls.evaluation)
        cls.checksum = cls.prepared["manifest_sha256"]
        cls.manifest, cls.plan, cls.refs = run.verify_evaluation(cls.evaluation, cls.checksum)
        cls.tokenizer = load_tokenizer(TOKENIZER)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_bounded_all_dev_inventory_and_pinned_protocol(self):
        self.assertEqual(self.prepared["cases"], 72)
        self.assertEqual(self.prepared["maximum_generations"], 144)
        self.assertEqual(self.prepared["model_calls"], 0)
        self.assertEqual(self.manifest["protocol"]["decoding"], ev.DECODING)
        self.assertEqual(self.manifest["protocol"]["tools_sha256"], ev.digest(ev.world.tools()))
        self.assertFalse(self.manifest["training_eligible"])
        self.assertEqual(self.manifest["source_manifest_sha256"], ev.SOURCE_SHA)

    def test_protected_manifest_and_existing_source_are_unchanged(self):
        self.assertEqual(candidates.verify_package(self.source)["manifest_sha256"], ev.SOURCE_SHA)
        source = ROOT / "tmp/r7-20261117/tool-candidates-v1"
        if source.exists():
            self.assertEqual(candidates.verify_package(source)["manifest_sha256"], ev.SOURCE_SHA)

    def test_preparation_refuses_overwrite(self):
        with self.assertRaisesRegex(ValueError, "fresh evaluation"):
            run.prepare(self.source, TOKENIZER, self.evaluation)

    def test_measured_prompt_ids_are_actual_native_ids(self):
        measurements = ev.read(self.evaluation / "token-measurements.json")
        for case, measured in zip(self.plan["cases"], measurements, strict=True):
            ids = ev.native_prompt(self.tokenizer, case["messages"], self.plan["tools"])
            self.assertEqual(measured, {"id": case["id"], "prompt_ids_sha256": ev.digest(ids), "prompt_tokens": len(ids)})
            self.assertLessEqual(len(ids) + 256, 4096)

    def test_no_training_ids_admitted_as_dev(self):
        source = self.root / "mutated-source"
        source.mkdir()
        for name in run.SOURCE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.source / name).read_bytes())
        rows = ev.read_rows(source / "candidates/dev.jsonl")
        rows[0]["split"] = "train"
        (source / "candidates/dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "Non-dev"):
            run.derive_cases(source)

    def test_frozen_code_change_blocks_reuse(self):
        original = ev.sha
        with patch.object(ev, "sha", wraps=ev.sha) as checksum:
            checksum.side_effect = lambda path: "0" * 64 if Path(path) == Path(run.__file__).parent / "r7_tool_eval.py" else original(path)
            with self.assertRaisesRegex(ValueError, "frozen"):
                run.verify_code(self.evaluation)

    def audit_fixture(self, record):
        folder = self.root / ("audit-" + ev.digest(record)[:8])
        folder.mkdir()
        manifest = {"version": "understanding-adaptation-r7"}
        ev.save(folder / "source-manifests.json", [{"manifest_path": "dataset-manifest.json"}])
        ev.save(folder / "dataset-manifest.json", {"records_path": "records.jsonl"})
        ev.save(folder / "manifest.json", manifest)
        for name in ("source-witnesses.jsonl", "records.jsonl"):
            (folder / name).write_text(json.dumps(record) + "\n")
        import run_understanding_r7 as training
        with patch.object(training, "verify_pack", return_value=(manifest, [], [])):
            return run.audit_training(folder, ev.sha(folder / "manifest.json"), self.evaluation)

    def test_train_leak_same_source_group_reworded_content_rejected(self):
        row = {"id": "new-id", "source_id": self.refs[0]["source_group_id"],
               "messages": [{"role": "user", "content": "Different words"}, {"role": "assistant", "content": "Different answer"}]}
        with self.assertRaisesRegex(ValueError, "leaked"):
            self.audit_fixture(row)

    def test_train_leak_alias_metadata_rejected(self):
        row = {"id": "alias-id", "source_id": "unrelated", "metadata": {"group_id": self.refs[0]["source_group_id"]},
               "messages": [{"role": "user", "content": "Different"}, {"role": "assistant", "content": "Unrelated"}]}
        with self.assertRaisesRegex(ValueError, "leaked"):
            self.audit_fixture(row)

    def test_train_leak_identical_history_under_new_id_rejected(self):
        row = deepcopy(ev.read_rows(self.source / "candidates/dev.jsonl")[0])
        row.update(id="unrelated-id", source_id="unrelated-source", source_group_id="unrelated-group")
        row["metadata"] = {}
        row["original_ref"] = {}
        with self.assertRaisesRegex(ValueError, "leaked"):
            self.audit_fixture(row)

    def test_nonoverlapping_training_record_clears(self):
        report = self.audit_fixture({"id": "clear", "source_id": "independent", "text": "Entirely unrelated text concerning geology."})
        self.assertTrue(report["overlap_free"])
        self.assertEqual(report["scanned"]["records.jsonl"]["records"], 1)

    def make_progress(self):
        case, ref = self.plan["cases"][0], self.refs[0]
        actual = ref["execution"]
        runtime = ScriptedRuntime(self.tokenizer, [native_call(actual["call"]["name"], actual["call"]["arguments"]),
                                                   candidates.assistant_response(actual)])
        record = ev.run_case(case, runtime, self.plan["tools"], self.tokenizer, IDENTITY)
        binding = {"identity": IDENTITY, "plan_sha256": ev.digest(self.plan), "evaluation_sha256": self.checksum}
        progress = {"binding": binding, "state": "running", "results": [record], "inflight": None}
        progress["snapshot_sha256"] = ev.digest(progress)
        return binding, progress

    def test_partial_report_does_not_claim_complete_or_unseen_success(self):
        binding, progress = self.make_progress()
        execution = {"evaluation": str(self.evaluation)}
        with patch.object(run, "verify_execution", return_value=(execution, binding, self.plan, self.refs)):
            report = run.score("fixture-package", "0" * 64, progress)
            self.assertEqual(report["summary"]["pending_cases"], 71)
            self.assertEqual(report["summary"]["totals"]["expected_outcome"], 1)
            self.assertEqual(report["summary"]["totals"]["task_success"], 1)

    def test_score_rejects_forged_execution_even_with_rehashed_snapshot(self):
        binding, progress = self.make_progress()
        progress["results"][0]["execution"]["result_sha256"] = "0" * 64
        progress["snapshot_sha256"] = ev.digest({k: v for k, v in progress.items() if k != "snapshot_sha256"})
        with patch.object(run, "verify_execution", return_value=({"evaluation": str(self.evaluation)}, binding, self.plan, self.refs)):
            with self.assertRaisesRegex(ValueError, "replay differs"):
                run.score("fixture-package", "0" * 64, progress)

    def test_compare_replays_not_trusts_verified_flag(self):
        path = self.root / "forged-report.json"
        ev.save(path, {"verified": True, "scores": [], "evidence": {
            "package": "fixture", "execution_sha256": "0" * 64, "progress": {}, "reviews": []}})
        with patch.object(run, "score", return_value={"verified": True, "scores": ["actual replay differs"]}):
            with self.assertRaisesRegex(ValueError, "replayed actual outputs"):
                run.replay_report(path)

    def test_actual_bundle_verification_no_model_load_or_gpu(self):
        bundle = ROOT / "tmp/r6-demo/bundle-v1"
        if not bundle.exists():
            self.skipTest("Pinned R6 bundle unavailable")
        with patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            result = run.verify_inference(bundle, "529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a", "r6_response")
        self.assertTrue(result["r6_training"]["training_completed"])


if __name__ == "__main__":
    unittest.main()
