"""Offline R3 pack/execution tests. No Modal submission, Torch, or model download."""
import ast
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_understanding_r3 as r

SOURCE = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def reseal(pack):
    manifest = r.read(pack / "manifest.json")
    manifest["artifacts"] = {p.relative_to(pack).as_posix(): r.sha(p) for p in pack.rglob("*")
                             if p.is_file() and p.relative_to(pack).as_posix() not in {"manifest.json", "PREPARED.json"}}
    write(pack / "manifest.json", manifest)
    write(pack / "PREPARED.json", {"manifest_sha256": r.sha(pack / "manifest.json")})


def bind_source(pack):
    path = "datasets/test-v1/manifest.json"
    manifest = r.read(pack / path)
    manifest["records_sha256"] = r.sha(pack / manifest["records_path"])
    write(pack / path, manifest)
    write(pack / "source-manifests.json", [{"dataset_id": manifest["dataset_id"], "dataset_version": manifest["dataset_version"],
                                           "manifest_path": path, "manifest_sha256": r.sha(pack / path)}])


def fixture(pack):
    pack.mkdir()
    messages = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Followup"}, {"role": "assistant", "content": "Answer"}]
    rows, witnesses, records = [], [], []
    for i, collection in enumerate(("responses", "english_retention", "language", "meaning")):
        language = collection == "language"
        row = {"id": "row-" + str(i), "source_id": "source-" + str(i), "collection": collection,
               "split": "train", "objective": "causal_lm" if language else "final_assistant",
               "input_ids": [7, 8, 248046], "labels": [7, 8, 248046] if language else [-100, 8, 248046],
               "attention_mask": [1, 1, 1], "sequence_tokens": 3, "target_tokens": 3 if language else 2}
        witness = {k: row[k] for k in ("id", "source_id", "collection")}
        witness.update(metadata={"source_preserved": True}, original_ref={"row": i})
        witness.update({"text": "Retained Twi"} if language else {"messages": messages})
        record = {"id": row["id"], "source_id": row["source_id"], "split": "train", "training_eligible": True,
                  **({"text": witness["text"]} if language else {"messages": messages})}
        ref = {"dataset_id": "test", "dataset_version": "v1", "record_id": record["id"],
               "source_id": record["source_id"], "record_sha256": r.digest(record)}
        row["dataset_ref"] = witness["dataset_ref"] = ref
        rows.append(row); witnesses.append(witness)
        records.append(record)
    write(pack / "datasets/test-v1/manifest.json", {"dataset_id": "test", "dataset_version": "v1",
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "records_path": "datasets/test-v1/records.jsonl", "record_count": len(records)})
    (pack / "datasets/test-v1/records.jsonl").write_text("".join(json.dumps(v) + "\n" for v in records))
    bind_source(pack)
    for name, values in (("rows.jsonl", rows), ("source-witnesses.jsonl", witnesses)):
        (pack / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json"):
        write(pack / "tokenizer" / name, {})
    write(pack / "stages.json", [{"name": "responses", "schedule": [0, 1, 2, 3, 0, 1, 2, 3], "snapshot_after_updates": [2]}])
    write(pack / "evaluation-manifest.json", {"version": "test-evaluation"})
    write(pack / "overlap-audit.json", {"overlap_free": True})
    write(pack / "manifest.json", {"version": r.VERSION, "model": r.MODEL, "revision": r.REVISION,
        "initialization": r.INITIALIZATION, "private_research_only": True, "production_eligible": False,
        "optimization": {"learning_rate": 2e-5, "grad_accum": 2, "warmup_fraction": .05, "checkpoint_every": 1,
                         "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False}})
    reseal(pack)
    return pack


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.pack = fixture(self.root / "pack")

    def mutate(self, name, mutate):
        value = r.read(self.pack / name)
        mutate(value)
        write(self.pack / name, value)
        reseal(self.pack)

    def prepare(self, mode="smoke", smoke_directory=None, name="execution"):
        hashes = {p.name: r.sha(p) for p in (self.pack / "tokenizer").iterdir()}
        with patch.dict(sys.modules, {"check_afrique_training_contract": SimpleNamespace(ASSET_HASHES=hashes)}):
            return r.prepare(self.pack, self.root / name, mode, SOURCE, smoke_directory)

    def test_valid_pack_recomputes_continuous_schedule(self):
        manifest, rows, stages = r.verify_pack(self.pack)
        stage = stages[0]
        self.assertEqual(stage["optimizer_updates"], 4)
        self.assertEqual(stage["effective_target_tokens"], 16)
        self.assertEqual(stage["sequence_tokens"], 24)
        self.assertEqual(stage["maximum_row_exposures"], 2)
        self.assertEqual(stage["snapshot_after_updates"], [2])
        self.assertEqual(stage["collection_target_tokens"], {c: 4 for c in ("responses", "english_retention", "language", "meaning")})
        self.assertEqual(sum(stage["collection_target_fractions"].values()), 1)
        self.assertEqual(manifest["initialization"]["adapter_weights_sha256"], r.INITIAL_SHA)

    def test_seal_and_artifact_tampering_rejected(self):
        (self.pack / "rows.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            r.verify_pack(self.pack)
        reseal(self.pack)
        with self.assertRaises((ValueError, KeyError)):
            r.verify_pack(self.pack)

    def test_linked_and_unbound_artifacts_rejected(self):
        (self.pack / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            r.verify_pack(self.pack)
        (self.pack / "extra.json").unlink()
        (self.pack / "extra.json").symlink_to(self.pack / "evaluation-manifest.json")
        with self.assertRaisesRegex(ValueError, "Linked"):
            r.verify_pack(self.pack)

    def test_wrong_initializer_optimizer_recipe_and_overlap_rejected(self):
        original = r.read(self.pack / "manifest.json")
        mutations = [lambda v: v["initialization"].update(adapter_weights_sha256="f" * 64),
                     lambda v: v["initialization"].update(new_optimizer=False),
                     lambda v: v["optimization"].update(learning_rate=.001),
                     lambda v: v["optimization"].update(truncate=True),
                     lambda v: v["optimization"].update(grad_accum=True)]
        for change in mutations:
            write(self.pack / "manifest.json", original)
            self.mutate("manifest.json", change)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)
        write(self.pack / "manifest.json", original)
        self.mutate("overlap-audit.json", lambda v: v.update(overlap_free=False))
        with self.assertRaisesRegex(ValueError, "overlap"):
            r.verify_pack(self.pack)

    def test_arbitrary_collections_and_primary_are_supported(self):
        for name in ("rows.jsonl", "source-witnesses.jsonl"):
            rows = r.json_rows(self.pack / name)
            rows[0]["collection"], rows[1]["collection"] = "repaired_dialogue", "retained_english"
            (self.pack / name).write_text("".join(json.dumps(v) + "\n" for v in rows))
        self.mutate("stages.json", lambda v: v[0].update(primary_collections=["repaired_dialogue"]))
        r.verify_pack(self.pack)

    def test_bad_schedule_or_snapshot_never_silently_changes(self):
        original = r.read(self.pack / "stages.json")
        for change in ({"schedule": [0, 99]}, {"schedule": [0] * 5 + [1]},
                       {"snapshot_after_updates": [4]}, {"snapshot_after_updates": [2, 1]},
                       {"optimizer_updates": 99}, {"schedule": [1, 2, 3]}):
            write(self.pack / "stages.json", original)
            self.mutate("stages.json", lambda v: v[0].update(change))
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)

    def test_masked_history_and_matching_source_witness_required(self):
        rows = r.json_rows(self.pack / "rows.jsonl")
        for change in ({"labels": [7, 8, 248046], "target_tokens": 3}, {"source_id": "wrong"}, {"split": "validation"}):
            changed = copy.deepcopy(rows)
            changed[0].update(change)
            (self.pack / "rows.jsonl").write_text("".join(json.dumps(v) + "\n" for v in changed))
            reseal(self.pack)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)

    def test_snapshot_must_be_versioned_admitted_and_record_eligible(self):
        path = "datasets/test-v1/manifest.json"
        original = r.read(self.pack / path)
        for change in ({"dataset_version": ""}, {"admission_status": "pending"}, {"record_count": 90000}):
            write(self.pack / path, {**original, **change})
            bind_source(self.pack)
            reseal(self.pack)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)
        write(self.pack / path, original)
        records = r.json_rows(self.pack / "datasets/test-v1/records.jsonl")
        records[0]["training_eligible"] = False
        (self.pack / "datasets/test-v1/records.jsonl").write_text("".join(json.dumps(v) + "\n" for v in records))
        bind_source(self.pack)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "not admitted"):
            r.verify_pack(self.pack)

    def test_every_packed_row_requires_membership_even_when_unscheduled(self):
        self.mutate("stages.json", lambda v: v[0].update(schedule=[0, 1, 2, 0, 1, 2]))
        rows = r.json_rows(self.pack / "rows.jsonl")
        del rows[3]["dataset_ref"]
        (self.pack / "rows.jsonl").write_text("".join(json.dumps(v) + "\n" for v in rows))
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "dataset reference"):
            r.verify_pack(self.pack)

    def test_dataset_reference_record_hash_and_history_are_bound(self):
        rows = r.json_rows(self.pack / "rows.jsonl")
        witnesses = r.json_rows(self.pack / "source-witnesses.jsonl")
        for change in ({"record_id": "missing"}, {"dataset_version": "v2"},
                       {"source_id": "other"}, {"record_sha256": "0" * 64}):
            for name, original in (("rows.jsonl", rows), ("source-witnesses.jsonl", witnesses)):
                altered = copy.deepcopy(original)
                altered[0]["dataset_ref"].update(change)
                (self.pack / name).write_text("".join(json.dumps(v) + "\n" for v in altered))
            reseal(self.pack)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)
        (self.pack / "rows.jsonl").write_text("".join(json.dumps(v) + "\n" for v in rows))
        witnesses[0]["messages"][1]["content"] = "Changed prior answer"
        (self.pack / "source-witnesses.jsonl").write_text("".join(json.dumps(v) + "\n" for v in witnesses))
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "admitted exact prefix"):
            r.verify_pack(self.pack)

    def test_changed_source_record_requires_new_bound_reference(self):
        records = r.json_rows(self.pack / "datasets/test-v1/records.jsonl")
        records[0]["messages"] += [{"role": "user", "content": "Next"}, {"role": "assistant", "content": "Next answer"}]
        (self.pack / "datasets/test-v1/records.jsonl").write_text("".join(json.dumps(v) + "\n" for v in records))
        bind_source(self.pack)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "record/source identity"):
            r.verify_pack(self.pack)
        for name in ("rows.jsonl", "source-witnesses.jsonl"):
            rows = r.json_rows(self.pack / name)
            rows[0]["dataset_ref"]["record_sha256"] = r.digest(records[0])
            (self.pack / name).write_text("".join(json.dumps(v) + "\n" for v in rows))
        reseal(self.pack)
        r.verify_pack(self.pack)

    def receipt(self, updates, mode="train", alter=None):
        _, rows, stages = r.verify_pack(self.pack)
        stage = stages[0]
        order = stage["schedule"][:updates * 2]
        position = {"completed_updates": updates, "completed_microbatches": len(order), "microbatches_in_update": 0,
                    "exposure": {"row_visits": dict(Counter(str(i) for i in order)),
                                 "input_tokens": sum(len(rows[i]["input_ids"]) for i in order),
                                 "effective_target_tokens": sum(sum(t != -100 for t in rows[i]["labels"][1:]) for i in order)}}
        if alter:
            alter(position)
        return r.consumption_receipt(rows, stage, 2, position, snapshots=stage["admitted_dataset_snapshots"],
                                     mode=mode, execution_sha="a" * 64, data_sha="b" * 64,
                                     checkpoint={"state_sha256": "c" * 64} if updates else None)

    def test_consumption_excludes_inventory_and_reports_actual_class_tokens(self):
        zero, partial, complete = self.receipt(0), self.receipt(3), self.receipt(4)
        self.assertEqual(zero["consumed"]["row_exposures"], 0)
        self.assertEqual(zero["inventory"]["packed_rows"], 4)
        self.assertEqual(partial["consumed"]["row_exposures"], 6)
        self.assertEqual(partial["consumed"]["unique_rows"], 4)
        self.assertEqual(partial["consumed"]["input_tokens"], 18)
        self.assertEqual(partial["consumed"]["effective_target_tokens"], 12)
        self.assertEqual(partial["consumed"]["by_class"]["responses"]["row_exposures"], 2)
        self.assertEqual(partial["consumed"]["by_class"]["language"]["effective_target_tokens"], 2)
        self.assertEqual(len(partial["rows"]), 4)
        self.assertFalse(partial["training_complete"])
        self.assertEqual(complete["consumed"]["row_exposures"], 8)
        self.assertTrue(complete["training_complete"])
        self.assertEqual(complete["consumed"]["by_dataset"][0]["effective_target_tokens"], 16)

    def test_resume_consumption_is_cumulative_not_double_counted_and_smoke_is_separate(self):
        paused, resumed = self.receipt(2), self.receipt(3)
        self.assertEqual(resumed["consumed"]["row_exposures"], 6)
        self.assertNotEqual(resumed["consumed"]["row_exposures"], paused["consumed"]["row_exposures"] + 6)
        smoke = self.receipt(4, mode="smoke")
        self.assertTrue(smoke["schedule_complete"])
        self.assertFalse(smoke["training_complete"])
        self.assertEqual(smoke["mode"], "smoke")
        for alter in (lambda p: p["exposure"].update(row_visits={"0": 6}),
                      lambda p: p["exposure"].update(effective_target_tokens=9999),
                      lambda p: p.update(microbatches_in_update=1),
                      lambda p: p.update(completed_updates=2)):
            with self.assertRaises(ValueError):
                self.receipt(3, alter=alter)

    def test_worker_receipt_reads_committed_state_not_later_exception_metrics(self):
        _, rows, stages = r.verify_pack(self.pack)
        destination = self.root / "remote-run"
        output = destination / "responses"
        checkpoint = output / "attempt-test/update-00000003"
        checkpoint.mkdir(parents=True)
        write(checkpoint / "manifest.json", {"identity": {"run_sha256": "d" * 64, "data_sha256": "b" * 64}})
        write(checkpoint.parent / "report.json", {"checkpoint": str(checkpoint), "checkpoint_completed_updates": 3,
            "run_sha256": "a" * 64, "data_sha256": "b" * 64,
            "position": {"completed_updates": 4, "completed_microbatches": 8}})
        position = {"completed_updates": 3, "completed_microbatches": 6, "microbatches_in_update": 0,
                    "exposure": {"row_visits": {"0": 2, "1": 2, "2": 1, "3": 1},
                                 "input_tokens": 18, "effective_target_tokens": 12}}
        loader = Mock(return_value={"position": position})
        commit = Mock()
        namespace = {"Path": Path, "read": r.read, "save": r.save, "require": r.require,
            "r2": SimpleNamespace(checkpoint_proof=lambda *args: {"state_sha256": "c" * 64}),
            "_read_checkpoint": loader, "execution_sha": "a" * 64, "contract": {"data_sha256": "b" * 64},
            "consumption_receipt": r.consumption_receipt, "rows": rows, "stage": stages[0],
            "mode": "train", "identifier": "test-run", "destination": destination,
            "artifacts": SimpleNamespace(commit=commit)}
        tree = ast.parse((r.ROOT / "modal/train/train_understanding_r3.py").read_text())
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        callback = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_commit")
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "worker-checkpoint-commit", "exec"), namespace)
        publish = namespace["checkpoint_commit"](output, stages[0]["schedule"], 2)
        publish()
        receipt = r.read(destination / "consumption.json")
        self.assertEqual(receipt["completed_updates"], 3)
        self.assertEqual(receipt["consumed"]["row_exposures"], 6)
        self.assertEqual(receipt["consumed"]["unique_dataset_records"], 4)
        self.assertFalse(receipt["training_complete"])
        publish()
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(commit.call_count, 2)

    def test_immutable_execution_binds_all_code_and_pack(self):
        value = self.prepare()
        package = self.root / "execution"
        self.assertEqual(r.verify_execution(package), value)
        self.assertEqual(value["hard_seconds"], 900)
        with self.assertRaisesRegex(ValueError, "new execution"):
            self.prepare()
        (package / "r3/modal/train/train_understanding_r3.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Packaged code changed"):
            r.verify_execution(package)

    def test_train_requires_matching_real_weight_smoke(self):
        with self.assertRaisesRegex(ValueError, "smoke required"):
            self.prepare("train")
        contract = self.prepare()
        smoke = self.root / "evidence"
        write(smoke / "execution.json", contract)
        run = "understanding_20261106T060000Z"
        write(smoke / "status.json", {"state": "completed", "mode": "smoke", "run_id": run,
            "execution_sha256": r.sha(smoke / "execution.json"), "data_sha256": contract["data_sha256"],
            "remote_directory": "/artifacts/" + r.VERSION + "/" + run})
        write(smoke / "smoke.json", {"passed": True, "real_weight_backward": True,
            "full_state_recovery": {"equivalent": True}, "resume_parameter_and_buffer_bitwise_equal": True,
            "initial_adapter_values_exact": True, "native_encoding": {"all_rows_verified": 4}})
        trained = self.prepare("train", smoke, "training")
        self.assertEqual(trained["hard_seconds"], 10800)
        self.mutate("manifest.json", lambda v: v["optimization"].update(learning_rate=1e-5))
        with self.assertRaisesRegex(ValueError, "binding differs|differs from smoke"):
            self.prepare("train", smoke, "changed")

    def test_launch_requires_authorization_before_any_import(self):
        with self.assertRaisesRegex(ValueError, "authorize-gpu"):
            r.launch(self.root / "missing", self.root / "receipt.json", "test", False)
        self.assertFalse((self.root / "receipt.json").exists())

    def test_submission_receipt_survives_pre_submit_failure(self):
        self.prepare()
        receipt = self.root / "receipt.json"
        def fail(name, path):
            self.assertEqual(r.read(receipt)["state"], "submitting")
            raise RuntimeError("simulated import failure")
        with patch.dict(sys.modules, {"modal": SimpleNamespace()}), patch.object(importlib.util, "spec_from_file_location", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "import failure"):
                r.launch(self.root / "execution", receipt, "test", True)
            with self.assertRaisesRegex(ValueError, "Submission exists"):
                r.launch(self.root / "execution", receipt, "test", True)
        self.assertNotIn("call_id", r.read(receipt))

    def test_new_root_cannot_write_into_r2(self):
        self.assertEqual(r.mounted_root(self.root), self.root / r.VERSION)
        (self.root / r.VERSION).rmdir()
        (self.root / r.VERSION).symlink_to(self.root)
        with self.assertRaises(ValueError):
            r.mounted_root(self.root)

    def test_native_encoding_receives_unmodified_complete_histories(self):
        _, rows, _ = r.verify_pack(self.pack)
        witnesses = r.json_rows(self.pack / "source-witnesses.jsonl")
        seen = []
        def encode(tokenizer, witness, limit):
            self.assertEqual(limit, 4096)
            self.assertEqual(witness, witnesses[int(witness["id"].split("-")[-1])])
            self.assertEqual([m["role"] for m in witness["messages"]], ["user", "assistant", "user", "assistant"])
            seen.append(witness["id"])
            return rows[int(witness["id"].split("-")[-1])]
        modules = {"check_afrique_training_contract": SimpleNamespace(load_tokenizer=lambda _: object(), encode_complete=encode),
                   "prepare_understanding_r2": SimpleNamespace(encode_language=lambda tokenizer, text: rows[2])}
        with patch.dict(sys.modules, modules):
            self.assertEqual(r.verify_native_rows(self.pack, rows)["all_rows_verified"], 4)
            changed = copy.deepcopy(rows)
            changed[0]["input_ids"][0] = 99
            with self.assertRaisesRegex(ValueError, "Native encoding differs"):
                r.verify_native_rows(self.pack, changed)
        self.assertEqual(seen[:3], ["row-0", "row-1", "row-3"])

    def test_initializer_requires_all_source_hashes_and_completed_run(self):
        source = self.root / r.SOURCE_RUN
        for name in r.SOURCE_FILES:
            write(source / name, {"state": "completed", "training_completed": True})
        hashes = {name: r.sha(source / name) for name in r.SOURCE_FILES}
        with patch.object(r, "SOURCE_FILES", hashes):
            self.assertEqual(r.verify_initialization(self.root), source / "meaning-adapter")
            write(source / "status.json", {"state": "running", "training_completed": False})
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                r.verify_initialization(self.root)
            hashes["status.json"] = r.sha(source / "status.json")
            with self.assertRaisesRegex(ValueError, "not complete"):
                r.verify_initialization(self.root)

    def test_worker_can_be_imported_without_submission(self):
        self.prepare()
        package = self.root / "execution"
        spec = importlib.util.spec_from_file_location("r3_worker_import_test", package / "r3/modal/train/train_understanding_r3.py")
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, GHA_R3_PACKAGE=str(package)), patch.dict(sys.modules, {spec.name: worker}):
            spec.loader.exec_module(worker)
            self.assertEqual(worker.CONTRACT["mode"], "smoke")
            self.assertEqual(worker.app.name, "ghana-understanding-adaptation-r3")

    def test_resume_requires_exact_identity_and_graceful_terminal_pause(self):
        identifier = "understanding_20261106T060000Z"
        claim = {"call_id": "fc-test", "execution_sha256": "a" * 64}
        outcome = {"state": "paused_budget", "mode": "train", "training_completed": False,
                   "run_id": identifier, "execution_sha256": "a" * 64, "data_sha256": "b" * 64}
        self.assertEqual(r.r2.terminal_pause(claim, outcome, identifier, "a" * 64, "b" * 64)["call_id"], "fc-test")
        for change in ({"state": "running"}, {"state": "failed"}, {"training_completed": True}, {"data_sha256": "c" * 64}):
            with self.assertRaises(ValueError):
                r.r2.terminal_pause(claim, {**outcome, **change}, identifier, "a" * 64, "b" * 64)


class WorkerTests(unittest.TestCase):
    def test_worker_bounds_packaging_and_trainable_initializer(self):
        source = (r.ROOT / "modal/train/train_understanding_r3.py").read_text()
        tree = ast.parse(source)
        for name, timeout in (("smoke", 900), ("train", 10800)):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
            for key, value in {"timeout": timeout, "retries": 0, "min_containers": 0,
                               "max_containers": 1, "single_use_containers": True, "include_source": False}.items():
                self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertIn("/r3-package/r3/modal/train:", source)
        self.assertIn('image.add_local_dir(str(PACKAGE), "/r3-package")', source)
        self.assertIn('PeftModel.from_pretrained(base, str(initializer)', source)
        self.assertIn("is_trainable=True", source)
        self.assertNotIn("get_peft_model(base", source)
        self.assertIn("stop_after_updates=boundary", source)
        self.assertIn('resume = Path(result["checkpoint"])', source)
        self.assertIn("r2.inherit_prefix", source)
        self.assertIn("assert_checkpoint_equivalent(resume, restored)", source)

    def test_full_state_helpers_remain_byte_pinned(self):
        r.r2.activate(SOURCE)


if __name__ == "__main__":
    unittest.main()
