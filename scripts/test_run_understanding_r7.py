"""Offline sealed R7 pack, provenance, bounded launch and receipt tests."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_understanding_r7 as r
from test_r7_encoding import ASSETS, SOURCE, tool_row


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


def reseal(pack, bind_overlap=True):
    if bind_overlap:
        audit = r.read(pack / "overlap-audit.json")
        for name, key in (("held-content.jsonl", "held_content_sha256"),
                          ("source-witnesses.jsonl", "source_witnesses_sha256"),
                          ("evaluation-manifest.json", "evaluation_manifest_sha256")):
            audit[key] = r.sha(pack / name)
        write(pack / "overlap-audit.json", audit)
    manifest = r.read(pack / "manifest.json")
    manifest["artifacts"] = {p.relative_to(pack).as_posix(): r.sha(p) for p in pack.rglob("*")
                             if p.is_file() and p.name not in {"PREPARED.json"}
                             and p.relative_to(pack).as_posix() != "manifest.json"}
    write(pack / "manifest.json", manifest)
    write(pack / "PREPARED.json", {"manifest_sha256": r.sha(pack / "manifest.json")})


def bind_source(pack):
    path = "datasets/test/manifest.json"
    manifest = r.read(pack / path)
    manifest["records_sha256"] = r.sha(pack / manifest["records_path"])
    write(pack / path, manifest)
    write(pack / "source-manifests.json", [{"dataset_id": "test", "dataset_version": "v7",
        "manifest_path": path, "manifest_sha256": r.sha(pack / path)}])


def fixture(pack):
    pack.mkdir()
    rows, witnesses, records = [], [], []
    messages = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Followup"}, {"role": "assistant", "content": "Answer"}]
    for i, collection in enumerate(("responses", "tools", "health", "commerce", "meaning", "language", "english_retention", "tools")):
        language = collection == "language"
        row = {"id": "row-" + str(i), "source_id": "source-" + str(i), "collection": collection,
               "split": "train", "objective": "causal_lm" if language else "final_assistant",
               "input_ids": [7, 8, 248046], "labels": [7, 8, 248046] if language else [-100, 8, 248046],
               "attention_mask": [1, 1, 1], "sequence_tokens": 3, "target_tokens": 3 if language else 2}
        payload = {"text": "Retained text"} if language else {"messages": copy.deepcopy(messages)}
        if collection == "tools":
            payload = {k: v for k, v in tool_row(response=i == 7).items() if k != "id"}
        witness = {k: row[k] for k in ("id", "source_id", "collection")}
        witness.update(version=r.WITNESS_VERSION, metadata={"group_id": "train-group-" + str(i)},
                       original_ref={"row": i}, **payload)
        record = {"id": row["id"], "source_id": row["source_id"], "split": "train", "training_eligible": True,
                  "metadata": witness["metadata"], "original_ref": witness["original_ref"], **payload}
        reference = {"dataset_id": "test", "dataset_version": "v7", "record_id": row["id"],
                     "source_id": row["source_id"], "record_sha256": r.digest(record)}
        row["dataset_ref"] = witness["dataset_ref"] = reference
        rows.append(row); witnesses.append(witness); records.append(record)
    write(pack / "datasets/test/manifest.json", {"dataset_id": "test", "dataset_version": "v7",
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "records_path": "datasets/test/records.jsonl", "record_count": len(records)})
    write_rows(pack / "datasets/test/records.jsonl", records)
    bind_source(pack)
    write_rows(pack / "rows.jsonl", rows)
    write_rows(pack / "source-witnesses.jsonl", witnesses)
    for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json"):
        write(pack / "tokenizer" / name, {})
    write(pack / "stages.json", [{"name": "responses", "schedule": list(range(8)) * 2,
        "snapshot_after_updates": [4], "primary_collections": ["responses", "tools", "health", "commerce"]}])
    write(pack / "evaluation-manifest.json", {"version": "test-heldout-v1"})
    write_rows(pack / "held-content.jsonl", [{"id": "held-1", "source_id": "held-source", "text": "A separate heldout passage"}])
    write(pack / "overlap-audit.json", {"overlap_free": True, "protected_units": 1})
    write(pack / "manifest.json", {"version": r.VERSION, "model": r.MODEL, "revision": r.REVISION,
        "initialization": r.INITIALIZATION, "private_research_only": True, "production_eligible": False,
        "optimization": {"learning_rate": 1e-5, "grad_accum": 2, "warmup_fraction": .05, "checkpoint_every": 1,
                         "seed": 42, "precision": "bf16_base_fp32_adapters", "packing": False, "truncate": False}})
    reseal(pack)
    return pack


class PackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.pack = fixture(self.root / "pack")

    def prepare(self):
        import check_afrique_training_contract as contract
        hashes = {p.name: r.sha(p) for p in (self.pack / "tokenizer").iterdir()}
        with patch.object(contract, "ASSET_HASHES", hashes), patch.object(r, "verify_native_rows") as native:
            value = r.prepare(self.pack, self.root / "execution", "smoke", SOURCE)
            native.assert_called_once()
            return value

    def test_exact_finite_token_exposure_accounting(self):
        _, rows, stages = r.verify_pack(self.pack)
        stage = stages[0]
        self.assertEqual(stage["optimizer_updates"], 8)
        self.assertEqual(stage["effective_target_tokens"], 32)
        self.assertEqual(stage["sequence_tokens"], 48)
        self.assertEqual(stage["row_exposures"], {str(i): 2 for i in range(8)})
        self.assertEqual(stage["schedule_sha256"], r.digest(stage["schedule"]))
        self.assertEqual(stage["collection_target_tokens"]["tools"], 8)
        self.assertEqual(stage["collection_input_tokens"]["tools"], 12)
        self.assertEqual(sum(stage["collection_target_fractions"].values()), 1)
        self.assertEqual({x["collection"] for x in rows}, r.COLLECTIONS)

    def test_verify_is_read_only(self):
        before = {str(p): p.read_bytes() for p in self.pack.rglob("*") if p.is_file()}
        r.verify_pack(self.pack)
        after = {str(p): p.read_bytes() for p in self.pack.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_seal_artifact_and_symlink_tampering(self):
        path = self.pack / "rows.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b"{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            r.verify_pack(self.pack)
        path.write_bytes(original)
        extra = self.pack / "unbound"
        extra.write_text("unbound")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            r.verify_pack(self.pack)
        extra.unlink()
        extra.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "Linked"):
            r.verify_pack(self.pack)

    def test_exact_messages_tools_and_metadata_binding(self):
        original = r.json_rows(self.pack / "source-witnesses.jsonl")
        changes = [lambda w: w[0].update(messages=w[0]["messages"][:2]),
                   lambda w: w[1]["tools"][0]["function"].update(description="Changed schema"),
                   lambda w: w[0]["metadata"].update(group_id="erased"),
                   lambda w: w[0].update(original_ref={"row": 42}),
                   lambda w: w[0].update(version="r6-source-witness-v1")]
        for change in changes:
            witnesses = copy.deepcopy(original)
            change(witnesses)
            write_rows(self.pack / "source-witnesses.jsonl", witnesses)
            reseal(self.pack)
            with self.assertRaisesRegex(ValueError, "admitted record|provenance differs|Unversioned"):
                r.verify_pack(self.pack)

    def test_source_record_hash_not_bypassed_by_resealing_pack(self):
        records = r.json_rows(self.pack / "datasets/test/records.jsonl")
        records[0]["messages"][-1]["content"] = "Changed record"
        write_rows(self.pack / "datasets/test/records.jsonl", records)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "Unbound dataset records"):
            r.verify_pack(self.pack)
        bind_source(self.pack)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "record/source identity differs"):
            r.verify_pack(self.pack)

    def test_schema_presence_is_bound_even_for_empty_tools(self):
        witnesses = r.json_rows(self.pack / "source-witnesses.jsonl")
        witnesses[0]["tools"] = []
        write_rows(self.pack / "source-witnesses.jsonl", witnesses)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "Tool schemas differ"):
            r.verify_pack(self.pack)

    def test_no_overlap_free_boolean_bypass(self):
        write(self.pack / "overlap-audit.json", {"overlap_free": True})
        reseal(self.pack, bind_overlap=False)
        with self.assertRaisesRegex(ValueError, "Unbound overlap"):
            r.verify_pack(self.pack)

    def test_empty_protection_rejected_even_with_matching_hashes(self):
        write_rows(self.pack / "held-content.jsonl", [])
        write(self.pack / "overlap-audit.json", {"overlap_free": True, "protected_units": 0})
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "Empty/uncleared"):
            r.verify_pack(self.pack)

    def test_actual_exact_prefix_and_source_group_overlap_rejected(self):
        for held in ({"id": "held", "text": "Earlier question\nEarlier answer"},
                     {"id": "held", "text": "Different content", "metadata": {"family_id": "train-group-0"}},
                     {"id": "held", "text": "Different content", "source_id": "source-0"}):
            write_rows(self.pack / "held-content.jsonl", [held])
            reseal(self.pack)
            with self.assertRaisesRegex(ValueError, "overlap"):
                r.verify_pack(self.pack)

    def test_invalid_masks_eos_and_histories_rejected(self):
        original = r.json_rows(self.pack / "rows.jsonl")
        for ids, labels in (([7, 8, 248044], [-100, 8, 248044]), ([7, 8, 248046], [-100, 8, -100]),
                            ([7, 248046, 248046], [-100, 248046, 248046]), ([7, 8, 248046], [7, 8, 248046])):
            rows = copy.deepcopy(original)
            rows[0].update(input_ids=ids, labels=labels, target_tokens=sum(t != -100 for t in labels))
            write_rows(self.pack / "rows.jsonl", rows)
            reseal(self.pack)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)

    def test_bounds_and_fabricated_accounting_rejected(self):
        original = r.read(self.pack / "stages.json")
        for changes in ({"schedule": [0] * 50001}, {"schedule": [0] * 5 + [1]},
                        {"schedule": [True, 1]}, {"effective_target_tokens": 99},
                        {"schedule_sha256": "0" * 64}, {"collection_input_tokens": {}},
                        {"primary_collections": ["invented"]}, {"snapshot_after_updates": [999]}):
            write(self.pack / "stages.json", [{**original[0], **changes}])
            reseal(self.pack)
            with self.assertRaises(ValueError):
                r.verify_pack(self.pack)
        self.assertEqual(r.MAX_SCHEDULE, 50000)

    def test_prepare_immutable_and_frozen_versions_unchanged(self):
        import run_understanding_r6 as r6
        before = copy.deepcopy({k: getattr(r6, k) for k in ("VERSION", "SOURCE_RUN", "INITIALIZATION", "BOUNDS", "CODE")})
        contract = self.prepare()
        self.assertEqual(before, {k: getattr(r6, k) for k in before})
        self.assertEqual(contract["initialization"]["adapter_weights_sha256"], r.INITIAL_SHA)
        self.assertEqual(contract, r.verify_execution(self.root / "execution"))
        self.assertEqual(contract["maximum_gpu_workers"], 1)
        with self.assertRaisesRegex(ValueError, "new execution"):
            self.prepare()
        (self.root / "execution/r7/scripts/r7_encoding.py").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Packaged code changed"):
            r.verify_execution(self.root / "execution")

    def test_train_requires_completed_smoke(self):
        import check_afrique_training_contract as contract
        hashes = {p.name: r.sha(p) for p in (self.pack / "tokenizer").iterdir()}
        with patch.object(contract, "ASSET_HASHES", hashes), patch.object(r, "verify_native_rows"):
            with self.assertRaisesRegex(ValueError, "smoke required"):
                r.prepare(self.pack, self.root / "execution", "train", SOURCE)

    def test_no_launch_without_authorization_or_duplicate_receipt(self):
        with self.assertRaisesRegex(ValueError, "authorize-gpu"):
            r.launch(self.root, self.root / "receipt", "offline", False)
        self.prepare()
        write(self.root / "receipt", {})
        with self.assertRaisesRegex(ValueError, "Submission exists"):
            r.launch(self.root / "execution", self.root / "receipt", "offline", True)

    def test_detached_launch_receipt_uses_r7_only_with_mocked_modal(self):
        self.prepare()
        spawn, run = Mock(return_value=SimpleNamespace(object_id="fc-offline")), Mock(return_value=nullcontext())
        worker = SimpleNamespace(app=SimpleNamespace(run=run), image=SimpleNamespace(object_id="im-offline"),
                                 smoke=SimpleNamespace(spawn=spawn))
        spec = SimpleNamespace(name="train_understanding_r7", loader=Mock())
        with patch.dict(sys.modules, {"modal": SimpleNamespace(enable_output=lambda: nullcontext())}), \
                patch.dict(os.environ), patch.object(importlib.util, "spec_from_file_location", return_value=spec) as locate, \
                patch.object(importlib.util, "module_from_spec", return_value=worker):
            value = r.launch(self.root / "execution", self.root / "receipt", "offline", True)
            self.assertEqual(os.environ["GHA_R7_PACKAGE"], str(self.root / "execution"))
            self.assertEqual(locate.call_args.args[1], self.root / "execution/r7/modal/train/train_understanding_r7.py")
        run.assert_called_once_with(detach=True)
        self.assertEqual(value, r.read(self.root / "receipt"))
        self.assertEqual(value["call_id"], "fc-offline")
        spawn.assert_called_once_with(value["run_id"], value["execution_sha256"], None)

    def test_partial_consumption_is_not_training_completion(self):
        _, rows, stages = r.verify_pack(self.pack)
        position = {"completed_microbatches": 4, "completed_updates": 2, "microbatches_in_update": 0,
                    "exposure": {"row_visits": {str(i): 1 for i in range(4)}, "input_tokens": 12,
                                 "effective_target_tokens": 8}}
        value = r.consumption_receipt(rows, stages[0], 2, position, snapshots=[], mode="train",
                                      execution_sha="a" * 64, data_sha="b" * 64, checkpoint={"proof": "test"})
        self.assertFalse(value["training_complete"])
        self.assertFalse(value["schedule_complete"])
        self.assertEqual(value["consumed"]["row_exposures"], 4)
        self.assertEqual(value["planned"]["input_tokens"], 48)
        position["exposure"]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "accounting differs"):
            r.consumption_receipt(rows, stages[0], 2, position, snapshots=[], mode="train",
                                  execution_sha="a" * 64, data_sha="b" * 64, checkpoint={})

    def test_resume_requires_terminal_paused_call_with_same_identity(self):
        contract = {**self.prepare(), "mode": "train"}
        identifier = "understanding_20261117T120000Z"
        checksum = r.sha(self.root / "execution/execution.json")
        claim = {"call_id": "fc-previous", "execution_sha256": checksum}
        base = {"state": "paused_budget", "mode": "train", "training_completed": False,
                "run_id": identifier, "execution_sha256": checksum, "data_sha256": contract["data_sha256"]}
        get = Mock()
        lookup = Mock(return_value=SimpleNamespace(get=get))
        dictionary = Mock(return_value={identifier: claim})
        modal = SimpleNamespace(Dict=SimpleNamespace(from_name=dictionary), FunctionCall=SimpleNamespace(from_id=lookup))
        for change in ({"state": "running"}, {"state": "failed"}, {"training_completed": True}, {"data_sha256": "0" * 64}):
            get.return_value = {**base, **change}
            with patch.dict(sys.modules, {"modal": modal}), patch.dict(os.environ), \
                    patch.object(r, "verify_execution", return_value=contract):
                with self.assertRaisesRegex(ValueError, "confirmed graceful pause"):
                    r.launch(self.root / "execution", self.root / "receipt", "offline", True, identifier)
            self.assertFalse((self.root / "receipt").exists())
            lookup.assert_called_with("fc-previous")
            get.assert_called_with(timeout=0)

    def test_initializer_rejects_each_changed_source_hash(self):
        for name, expected in r.SOURCE_FILES.items():
            def checksum(path):
                relative = Path(path).relative_to(self.root / r.SOURCE_RUN).as_posix()
                return "0" * 64 if relative == name else r.SOURCE_FILES[relative]
            with self.subTest(name=name), patch.object(r, "safe_file", side_effect=lambda root, name: root / name), \
                    patch.object(r, "sha", side_effect=checksum):
                with self.assertRaisesRegex(ValueError, "initializer/provenance changed"):
                    r.verify_initialization(self.root)

    def test_native_reencoding_passes_exact_witness_and_detects_modified_tokens(self):
        import check_afrique_training_contract as contract
        import prepare_understanding_r2 as replay
        import r7_encoding
        _, rows, _ = r.verify_pack(self.pack)
        witnesses = r.json_rows(self.pack / "source-witnesses.jsonl")
        def encode(tokenizer, witness, length):
            self.assertEqual(length, 4096)
            index = int(witness["id"].split("-")[-1])
            self.assertEqual(witness, witnesses[index])
            return rows[index]
        with patch.object(contract, "load_tokenizer", return_value=object()), \
                patch.object(r7_encoding, "encode_complete", side_effect=encode), \
                patch.object(replay, "encode_language", return_value=rows[5]):
            evidence = r.verify_native_rows(self.pack, rows)
            self.assertEqual(evidence["all_rows_verified"], 8)
            self.assertEqual(evidence["masking_eos"]["tool_schema_rows"], 2)
            self.assertEqual(evidence["masking_eos"]["tool_call_targets"], 1)
            self.assertEqual(evidence["masking_eos"]["tool_response_context_rows"], 1)
            modified = copy.deepcopy(rows)
            modified[1]["input_ids"][0] += 1
            with self.assertRaisesRegex(ValueError, "Native encoding differs"):
                r.verify_native_rows(self.pack, modified)


class NativePackTests(unittest.TestCase):
    def test_real_tokenizer_verify_prepare_and_sealed_execution_without_remote_calls(self):
        if not (ASSETS / "tokenizer.json").is_file():
            self.skipTest("Pinned local tokenizer unavailable")
        try:
            from check_afrique_training_contract import load_tokenizer
            from prepare_understanding_r2 import encode_language
            from r7_encoding import encode_complete
            tokenizer = load_tokenizer(ASSETS)
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pack = fixture(root / "pack")
            rows = r.json_rows(pack / "rows.jsonl")
            witnesses = r.json_rows(pack / "source-witnesses.jsonl")
            for row, witness in zip(rows, witnesses, strict=True):
                encoded = encode_language(tokenizer, witness["text"]) if row["objective"] == "causal_lm" else encode_complete(tokenizer, witness, 4096)
                row.update(encoded, sequence_tokens=len(encoded["input_ids"]),
                           target_tokens=sum(t != -100 for t in encoded["labels"]))
            write_rows(pack / "rows.jsonl", rows)
            for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json"):
                shutil.copyfile(ASSETS / name, pack / "tokenizer" / name)
            reseal(pack)
            manifest_sha = r.sha(pack / "manifest.json")
            _, verified, _ = r.verify_pack(pack)
            evidence = r.verify_native_rows(pack, verified)
            self.assertEqual(evidence["all_rows_verified"], 8)
            self.assertEqual(evidence["masking_eos"]["supervised_chat_eos"], 7)
            self.assertEqual(evidence["masking_eos"]["language_final_eos"], 1)
            with patch.dict(sys.modules, {"modal": None}):
                contract = r.prepare(pack, root / "execution", "smoke", SOURCE)
                self.assertEqual(contract, r.verify_execution(root / "execution"))
            self.assertEqual(manifest_sha, r.sha(pack / "manifest.json"))
            self.assertEqual(rows, r.json_rows(root / "execution/inputs/rows.jsonl"))


class WorkerTests(unittest.TestCase):
    def test_worker_bounds_and_reused_full_state_core(self):
        source = (r.ROOT / "modal/train/train_understanding_r7.py").read_text()
        tree = ast.parse(source)
        for name, timeout in (("smoke", 900), ("train", 28800)):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
            for key, value in {"gpu": "H100", "max_containers": 1, "min_containers": 0, "retries": 0,
                               "single_use_containers": True, "include_source": False, "timeout": timeout}.items():
                self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertEqual(r.BOUNDS["train"], (28200, 28800))
        for text in ("from understanding_training_core import train_schedule, assert_checkpoint_equivalent",
                     "from understanding_recovery import select_recovery", "r2.inherit_prefix", "stop_after_updates=boundary",
                     'resume = Path(result["checkpoint"])', 'return status("paused_budget"',
                     "PeftModel.from_pretrained(base, str(initializer)", '"r6_optimizer_loaded": False',
                     "Tool smoke requires call targets and tool-response histories"):
            self.assertIn(text, source)
        self.assertNotIn("run_understanding_r6", source)

    def test_r6_measured_completion_hashes_and_adapter_receipt(self):
        completed = SOURCE / "tmp/understanding-r6-20261113/train-collected"
        if not completed.is_dir():
            self.skipTest("Original local R6 evidence unavailable")
        self.assertEqual(r.SOURCE_RUN, "understanding-adaptation-r6/understanding_20261113T152209Z")
        for name in ("status.json", "execution.json", "adapters.json", "consumption.json"):
            self.assertEqual(r.sha(completed / name), r.SOURCE_FILES[name])
        status = r.read(completed / "status.json")
        self.assertEqual(status["state"], "completed")
        self.assertTrue(status["training_completed"])
        self.assertTrue(r.read(completed / "consumption.json")["training_complete"])
        final = next(a for a in r.read(completed / "adapters.json") if a["directory"].endswith("/responses-adapter"))
        self.assertTrue(final["checkpoint_adapter_tensors_verified"])
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            self.assertEqual(final["files"][name], r.SOURCE_FILES["responses-adapter/" + name])
        self.assertEqual(final["files"]["adapter_config.json"], r.INITIALIZATION["adapter_config_sha256"])

    def test_frozen_helpers_remain_byte_pinned(self):
        r.r2.activate(SOURCE)
        for name, checksum in r.FROZEN.items():
            self.assertEqual(r.sha(r.ROOT / name), checksum)


if __name__ == "__main__":
    unittest.main()
