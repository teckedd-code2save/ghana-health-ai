"""Fixture-only R5 builder tests; tokenization and final verification are mocked.

These tests exercise preparation boundaries, not model quality or real native
encoding. Every input file is constructed in a temporary directory.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import random
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import prepare_understanding_r5 as builder


def encoded(identity, target_tokens=2):
    token = int(builder.data.digest(identity)[:6], 16) + 100
    targets = [7] * (target_tokens - 1) + [248046]
    return {"input_ids": [token] + targets, "labels": [-100] + targets,
            "attention_mask": [1] * (target_tokens + 1)}


def schedule_rows(primary=13, replay=20, new_conversations=8):
    rows = []
    other_origins = ("inherited_r4", "existing_instruction", "new_reviewed_translation",
                     "new_spelling_augmentation", "inherited_r4")
    for collection, count in (("responses", primary), ("english_retention", replay),
                              ("meaning", replay), ("language", replay)):
        for index in range(count):
            natural = collection == "responses" and index < new_conversations
            origin = ("new_reviewed_translation" if natural else
                      other_origins[(index - new_conversations) % len(other_origins)]
                      if collection == "responses" else "synthetic_replay")
            rows.append({"id": collection + "-" + str(index), "collection": collection,
                         "new_source_conversation": natural, "preparation_origin": origin,
                         **encoded(collection + str(index))})
    return rows


def reseal(root, marker="PREPARED.json", extra=None):
    root.mkdir(parents=True, exist_ok=True)
    manifest = builder.data.read(root / "manifest.json") if (root / "manifest.json").exists() else {}
    manifest.update(extra or {})
    manifest["artifacts"] = {path.relative_to(root).as_posix(): builder.data.sha(path)
                             for path in root.rglob("*") if path.is_file()
                             and path.name not in {"manifest.json", marker}}
    builder.data.save(root / "manifest.json", manifest)
    builder.data.save(root / marker, {"manifest_sha256": builder.data.sha(root / "manifest.json")})


class ScheduleTests(unittest.TestCase):
    def assert_schedule_policy(self, rows, stage):
        visits = Counter(stage["schedule"])
        for index, row in enumerate(rows):
            if row["collection"] == "responses":
                self.assertEqual(visits[index], 2 if row["new_source_conversation"] else 1)
            else:
                self.assertLessEqual(visits[index], 1)
        boundary = stage["passes"][0]["end_sequence"]
        self.assertEqual(boundary % 16, 0)
        self.assertEqual(stage["snapshot_after_updates"], [boundary // 16])
        self.assertEqual(stage["passes"][1]["end_sequence"], len(stage["schedule"]))
        for epoch, batch in enumerate((stage["schedule"][:boundary], stage["schedule"][boundary:])):
            expected_primary = {i for i, row in enumerate(rows) if row["collection"] == "responses"
                                and (epoch == 0 or row["new_source_conversation"])}
            actual_primary = [i for i in batch if rows[i]["collection"] == "responses"]
            self.assertEqual(set(actual_primary), expected_primary)
            self.assertEqual(len(actual_primary), len(expected_primary))
            self.assertEqual({rows[index]["collection"] for index in batch},
                             {"responses", "english_retention", "meaning", "language"})
            target = sum(builder.effective_tokens(rows[i]) for i in expected_primary)
            total = sum(builder.effective_tokens(rows[i]) for i in batch)
            self.assertEqual(stage["passes"][epoch]["target_tokens"], total)
            self.assertEqual(stage["passes"][epoch]["sequence_count"], len(batch))
            fractions = {collection: sum(builder.effective_tokens(rows[i]) for i in batch
                                        if rows[i]["collection"] == collection) / total
                         for collection in ("responses", "english_retention", "meaning", "language")}
            self.assertAlmostEqual(sum(fractions.values()), 1.0)
            for collection, ratio in builder.previous.RATIOS.items():
                amount = sum(builder.effective_tokens(rows[index]) for index in batch
                             if rows[index]["collection"] == collection)
                self.assertGreater(amount, 0)
                self.assertLessEqual(amount, int(target * ratio))

    def test_all_responses_once_only_new_conversations_twice_with_finite_replay(self):
        rows = schedule_rows()
        stage = builder.schedule(rows)
        self.assertEqual(stage, builder.schedule(deepcopy(rows)))
        self.assert_schedule_policy(rows, stage)
        self.assertEqual(stage["response_exposure_policy"], {
            "new_source_backed_conversation": 2, "inherited_and_project_authored": 1, "spelling_variant": 1})

    def test_token_fractions_use_weighted_scheduled_tokens_not_history_counts(self):
        rows = schedule_rows()
        for index, row in enumerate(rows):
            if row["new_source_conversation"]:
                row.update(encoded(row["id"], target_tokens=5 + 2 * index))
        stage = builder.schedule(rows)
        self.assert_schedule_policy(rows, stage)
        new_tokens = sum(builder.effective_tokens(row) for row in rows if row["new_source_conversation"])
        scheduled_new_tokens = sum(builder.effective_tokens(rows[i]) for i in stage["schedule"]
                                   if rows[i]["new_source_conversation"])
        self.assertEqual(scheduled_new_tokens, 2 * new_tokens)
        total_tokens = sum(builder.effective_tokens(rows[i]) for i in stage["schedule"])
        self.assertNotAlmostEqual(scheduled_new_tokens / total_tokens,
                                  sum(rows[i]["new_source_conversation"] for i in stage["schedule"]) / len(stage["schedule"]))

    def test_project_repairs_without_new_source_conversations_cannot_form_pass_two(self):
        with self.assertRaisesRegex(ValueError, "conversation"):
            builder.schedule(schedule_rows(new_conversations=0))

    def test_truthy_non_boolean_conversation_flag_does_not_qualify(self):
        rows = schedule_rows()
        for row in rows:
            row["new_source_conversation"] = 1
        with self.assertRaisesRegex(ValueError, "conversation"):
            builder.schedule(rows)

    def test_schedule_does_not_modify_rows_or_global_random_state(self):
        rows = schedule_rows()
        before, state = deepcopy(rows), random.getstate()
        builder.schedule(rows)
        self.assertEqual(rows, before)
        self.assertEqual(random.getstate(), state)

    def test_missing_primary_or_replay_fails(self):
        for missing in ("responses", "english_retention", "meaning", "language"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                builder.schedule([row for row in schedule_rows() if row["collection"] != missing])

    def test_invalid_target_mask_fails(self):
        rows = schedule_rows()
        rows[0]["labels"] = [-100] * len(rows[0]["input_ids"])
        with self.assertRaises(ValueError):
            builder.schedule(rows)

    def test_unalignable_boundary_does_not_drop_primary_to_succeed(self):
        with self.assertRaisesRegex(ValueError, "align"):
            builder.schedule(schedule_rows(primary=16, replay=20))


class BuilderFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.r4, self.r2, self.instruction, self.admission, self.fresh, self.out = [
            self.root / name for name in ("r4", "r2", "instruction", "admission", "fresh", "out")]
        for path in (self.r4, self.r2, self.instruction, self.admission, self.fresh):
            path.mkdir()
        (self.r4 / "tokenizer").mkdir()
        for name in ("tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json"):
            builder.data.save(self.r4 / "tokenizer" / name, {"synthetic_fixture": True})
        builder.data.save(self.r4 / "source-manifests.json", [])
        builder.data.save(self.r4 / "evaluation-manifest.json", {"synthetic_fixture": True})
        builder.data.write_rows(self.r4 / "held-content.jsonl", [{"id": "old", "text": "Older protected synthetic fixture."}])
        builder.data.write_rows(self.fresh / "held-content.jsonl", [{"id": "fresh", "text": "Fresh protected synthetic fixture."}])
        builder.data.write_rows(self.instruction / "fixture.train.jsonl", [])
        self.records = []
        for index in range(12):
            identity = "new-" + str(index).zfill(2)
            messages = [{"role": "user", "content": "Fixture prompt " + identity + "."},
                        {"role": "assistant", "content": "Fixture reply " + identity + "."}]
            if index == 11:
                messages += [{"role": "user", "content": "Fixture follow-up " + identity + "."},
                             {"role": "assistant", "content": "Fixture continuation " + identity + "."}]
            self.records.append({"id": identity, "source_id": "source-" + identity,
                "group_id": "group-" + identity, "messages": messages,
                "english_messages": deepcopy(messages), "original_english_messages": deepcopy(messages),
                "split": "train", "training_eligible": True,
                "origin": "source_grounded_agent_translated_and_separately_reviewed",
                "original_ref": {"synthetic_fixture": identity},
                "review": {"status": "agent_checked", "findings": []}})
        replay_rows, replay_witnesses = [], []
        for collection in ("meaning", "language"):
            for index in range(20):
                identity = collection + "-" + str(index)
                replay_rows.append({"id": identity, "source_id": identity, "collection": collection,
                    "split": "train", "objective": "final_assistant" if collection == "meaning" else "causal_lm",
                    **encoded(identity)})
                replay_witnesses.append({"id": identity, "source_id": identity, "collection": collection,
                    "text": "Synthetic replay fixture " + identity + ".",
                    "metadata": {"origin": "source_preserved", "group_id": "group-" + identity},
                    "original_ref": {"synthetic_fixture": identity}})
        builder.data.write_rows(self.r2 / "rows.jsonl", replay_rows)
        builder.data.write_rows(self.r2 / "source-witnesses.jsonl", replay_witnesses)
        reseal(self.r4)
        reseal(self.r2)
        reseal(self.instruction, "SEALED.json")
        reseal(self.fresh, "FROZEN.json", {"version": "synthetic-fresh-v1"})
        self.write_admission()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(builder, "R4_SHA", builder.data.sha(self.r4 / "manifest.json")))
        self.stack.enter_context(patch.object(builder.previous, "R2_SHA", builder.data.sha(self.r2 / "manifest.json")))
        self.loader = self.stack.enter_context(patch.object(builder, "load_tokenizer", return_value=object()))
        self.encoder = self.stack.enter_context(patch.object(builder, "encode_complete", side_effect=self.fake_encode))
        self.runtime = SimpleNamespace(INITIALIZATION={"synthetic_fixture": True}, VERSION="synthetic-r5",
            verify_native_rows=Mock(side_effect=lambda path, rows: {"all_rows_verified": len(rows)}),
            verify_pack=Mock(return_value=({}, [], [])))
        self.stack.enter_context(patch.dict(sys.modules, {"run_understanding_r5": self.runtime,
                                                        "torch": None, "transformers": None, "modal": None}))
        self.stack.enter_context(patch.object(socket, "create_connection", side_effect=AssertionError("No network")))

    def fake_encode(self, tokenizer, witness, limit):
        self.assertEqual(limit, 4096)
        return encoded(witness["id"])

    def write_admission(self):
        builder.data.write_rows(self.admission / "conversations.jsonl", self.records)
        reseal(self.admission, "READY.json", {"admission_status": "admitted", "private_research_only": True,
            "production_eligible": False, "fresh_suite_sha256": builder.data.sha(self.fresh / "manifest.json")})

    def prepare(self):
        return builder.prepare(self.r4, self.r2, self.instruction, [self.admission],
                               self.fresh, self.out, "synthetic-v1")


class BuilderBoundaryTests(BuilderFixture, unittest.TestCase):
    def test_valid_fixture_preserves_source_hashes_and_complete_prefixes(self):
        before = deepcopy(self.records)
        result = self.prepare()
        self.assertFalse(result["selection"]["training_executed"])
        self.assertEqual(self.records, before)
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        snapshots = builder.data.read(self.out / "source-manifests.json")
        records_by_ref = {}
        for ref in snapshots:
            manifest = builder.data.read(self.out / ref["manifest_path"])
            for record in builder.previous.read_rows(self.out / manifest["records_path"]):
                records_by_ref[(manifest["catalog_dataset_id"], record["id"])] = record
        for row in rows:
            ref = row["dataset_ref"]
            self.assertEqual(ref["record_sha256"], builder.data.digest(records_by_ref[(row["collection"], ref["record_id"])]))
        lengths = [len(call.args[1]["messages"]) for call in self.encoder.call_args_list]
        self.assertIn(4, lengths)
        self.runtime.verify_native_rows.assert_called_once()
        self.runtime.verify_pack.assert_called_once()

    def test_native_encoding_failure_must_abort_not_seal_reduced_data(self):
        def encode(tokenizer, witness, limit):
            if witness["id"] == "new-00:assistant-0":
                raise ValueError("Synthetic native template failure")
            return self.fake_encode(tokenizer, witness, limit)
        self.encoder.side_effect = encode
        with self.assertRaisesRegex(ValueError, "native|encoding|template"):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())

    def test_failed_final_verification_must_not_leave_ready_marker(self):
        self.runtime.verify_pack.side_effect = ValueError("Synthetic final verification failure")
        with self.assertRaisesRegex(ValueError, "verification"):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())

    def test_unbound_instruction_file_rejected_before_scheduling(self):
        record = {**deepcopy(self.records[0]), "id": "unbound-fixture",
                  "quality_status": "scoped_agent_checked_research", "target_origin": "project_authored",
                  "provenance": {"synthetic_fixture": True}, "admission_sha256": "a" * 64}
        builder.data.write_rows(self.instruction / "unlisted.train.jsonl", [record])
        with patch.object(builder, "schedule", side_effect=AssertionError("Unbound input reached scheduling")):
            with self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(self.out.exists())

    def test_input_mutation_during_encoding_rejected_before_ready(self):
        changed = False
        def encode(*args):
            nonlocal changed
            if not changed:
                (self.instruction / "fixture.train.jsonl").write_text("\n", encoding="utf-8")
                changed = True
            return self.fake_encode(*args)
        self.encoder.side_effect = encode
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())

    def test_original_full_context_protects_all_connected_source_views(self):
        original = [{"role": "user", "content": "Synthetic original request uses its own distinct words."},
                    {"role": "assistant", "content": "The original reference response has different words here."}]
        self.records[0]["original_english_messages"] = original
        self.records[1]["family_id"] = self.records[0]["family_id"] = "shared-fixture-family"
        builder.data.write_rows(self.fresh / "held-content.jsonl", [
            {"id": "original-context", "text": "\n".join(message["content"] for message in original)}])
        reseal(self.fresh, "FROZEN.json")
        self.write_admission()
        # This test isolates the holdout boundary from first-pass alignment.
        def keep_rows(rows):
            return {"name": "responses", "schedule": list(range(len(rows))), "primary_collections": ["responses"]}
        with patch.object(builder, "schedule", side_effect=keep_rows):
            self.prepare()
        output = builder.previous.read_rows(self.out / "rows.jsonl")
        ids = {row["dataset_ref"]["record_id"] for row in output}
        self.assertFalse({"new-00", "new-01", "new-00:english", "new-01:english"} & ids)

    def resolution_fixture(self, bound=True):
        record = self.records[0]
        record["review"].update(
            source_review="Independent synthetic review.",
            screen_resolutions=[{"code": "synthetic_review", "turn": 1, "reason": "Fixture-specific resolution."}],
            decision_sha256="a" * 64,
            screened_messages_sha256=builder.data.digest(record["messages"]) if bound else "0" * 64,
            screened_english_messages_sha256=builder.data.digest(record["english_messages"]) if bound else "0" * 64)
        record["original_ref"]["review_decision_sha256"] = "a" * 64
        self.write_admission()
        original = builder.inspect_messages
        def screen(messages):
            return ([{"code": "synthetic_review", "turn": 1, "severity": "review"}]
                    if messages == record["messages"] else original(messages))
        return screen

    def test_bound_decision_specific_resolution_is_honored(self):
        screen = self.resolution_fixture()
        with patch.object(builder, "inspect_messages", side_effect=screen):
            self.prepare()
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        self.assertIn("new-00", {row["dataset_ref"]["record_id"] for row in rows})

    def test_resolution_bound_to_other_text_cannot_override_finding(self):
        screen = self.resolution_fixture(bound=False)
        with patch.object(builder, "inspect_messages", side_effect=screen):
            self.prepare()
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        self.assertNotIn("new-00", {row["dataset_ref"]["record_id"] for row in rows})

    def test_resolution_cannot_override_invalid_history(self):
        original_screen = self.resolution_fixture()
        def invalid(messages):
            return [{**finding, "severity": "invalid"} for finding in original_screen(messages)]
        with patch.object(builder, "inspect_messages", side_effect=invalid):
            self.prepare()
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        self.assertNotIn("new-00", {row["dataset_ref"]["record_id"] for row in rows})

    def test_mixed_source_types_keep_exact_exposures_after_remapping_and_report_actual_tokens(self):
        inherited = self.records.pop(0)
        inherited["origin"] = "agent_translated_source_with_synthetic_user_spelling"
        instruction = self.records.pop(0)
        instruction.update(target_origin="project_authored", quality_status="scoped_agent_checked_research",
                           provenance={"synthetic_fixture": True}, admission_sha256="a" * 64)
        self.records[0]["origin"] = "project_authored_agent_translated_and_separately_reviewed"
        self.write_admission()
        builder.data.write_rows(self.instruction / "fixture.train.jsonl", [instruction])
        reseal(self.instruction, "SEALED.json")
        builder.data.write_rows(self.r4 / "inherited.jsonl", [inherited])
        builder.data.save(self.r4 / "inherited-manifest.json", {
            "catalog_dataset_id": "responses", "records_path": "inherited.jsonl"})
        builder.data.save(self.r4 / "source-manifests.json", [{
            "manifest_path": "inherited-manifest.json",
            "manifest_sha256": builder.data.sha(self.r4 / "inherited-manifest.json")}])
        reseal(self.r4)
        self.stack.enter_context(patch.object(builder, "R4_SHA", builder.data.sha(self.r4 / "manifest.json")))
        result = self.prepare()
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        stage = builder.data.read(self.out / "stages.json")[0]
        visits = Counter(stage["schedule"])
        new_ids = {record["id"] for record in self.records if record["origin"].startswith("source_grounded_")}
        tokens = Counter()
        for index, row in enumerate(rows):
            natural = row["collection"] == "responses" and row["dataset_ref"]["record_id"] in new_ids
            self.assertEqual(row["new_source_conversation"], natural)
            if row["collection"] == "responses":
                self.assertEqual(visits[index], 2 if natural else 1)
            else:
                self.assertLessEqual(visits[index], 1)
            tokens[row["preparation_origin"]] += visits[index] * builder.effective_tokens(row)
        self.assertEqual(dict(tokens), result["selection"]["target_tokens_by_preparation_origin"])
        scheduled_total = sum(tokens.values())
        natural_tokens = sum(visits[i] * builder.effective_tokens(row) for i, row in enumerate(rows)
                             if row["new_source_conversation"])
        self.assertLess(natural_tokens, tokens["new_reviewed_translation"])
        fraction = natural_tokens / scheduled_total
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 1)
        self.assertFalse(result["selection"]["training_executed"])
        second = stage["schedule"][stage["passes"][0]["end_sequence"]:]
        self.assertTrue(all(rows[i]["new_source_conversation"] for i in second if rows[i]["collection"] == "responses"))
        self.assertFalse(result["selection"].get("automatic_promotion", False))


class SpellingAugmentationTests(BuilderFixture, unittest.TestCase):
    def eligible(self, record):
        return int(builder.data.digest(record["source_id"])[:8], 16) % 3 == 0

    def chosen_parent(self):
        return next(record for record in self.records if self.eligible(record))

    def add_symbols(self):
        for record in self.records:
            for message in record["messages"]:
                message["content"] += (" Synthetic q\u025b Q\u0190 3." if message["role"] == "user"
                                       else " Synthetic answer marker \u025b.")
        self.write_admission()

    def prepare_spelling(self):
        # Isolate augmentation from the independently tested replay budget policy.
        def keep_rows(rows):
            consolidation = [index for index, row in enumerate(rows) if row.get("new_source_conversation") is True]
            return {"name": "responses", "schedule": list(range(len(rows))) + consolidation,
                    "primary_collections": ["responses"]}
        with patch.object(builder, "schedule", side_effect=keep_rows):
            return self.prepare()

    def output_records(self):
        return builder.previous.read_rows(self.out / "datasets/responses/records.jsonl")

    def children(self):
        return [record for record in self.output_records() if record.get("augmentation")]

    def test_hash_bucket_membership_is_order_independent_and_separate_origin(self):
        self.add_symbols()
        before = deepcopy(self.records)
        expected = {record["id"] + ":ascii-input-v1" for record in before if self.eligible(record)}
        first = self.prepare_spelling()
        self.assertTrue(expected)
        self.assertEqual({row["id"] for row in self.children()}, expected)
        self.assertEqual(self.records, before)
        self.assertEqual(first["selection"]["candidate_histories"]["new_spelling_augmentation"], len(expected))
        self.assertEqual(first["selection"]["spelling_augmentation_new_independent_answers"], 0)
        self.assertEqual(first["selection"]["new_selected_records"], len(before))
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        child_rows = [row for row in rows if ":ascii-input-v1:" in row["id"]]
        self.assertTrue(child_rows)
        self.assertTrue(all(row["preparation_origin"] == "new_spelling_augmentation" for row in child_rows))
        self.assertTrue(all(row["new_source_conversation"] is False for row in child_rows))
        stage = builder.data.read(self.out / "stages.json")[0]
        visits = Counter(stage["schedule"])
        self.assertTrue(all(visits[index] == 1 for index, row in enumerate(rows)
                            if row["preparation_origin"] == "new_spelling_augmentation"))
        self.records.reverse()
        self.write_admission()
        self.out = self.root / "reordered-output"
        self.prepare_spelling()
        self.assertEqual({row["id"] for row in self.children()}, expected)

    def test_only_lowercase_epsilon_in_user_text_changes_and_provenance_survives(self):
        self.add_symbols()
        chosen = self.chosen_parent()
        chosen["messages"].insert(0, {"role": "system", "content": "Synthetic system marker \u025b."})
        chosen.update(family_id="synthetic-family", tree_id="synthetic-tree",
                      original_group_id="synthetic-ancestor", source_ids=["synthetic-source-ancestor"])
        self.write_admission()
        parents = {record["id"]: deepcopy(record) for record in self.records}
        self.prepare_spelling()
        for child in self.children():
            parent = parents[child["original_ref"]["parent_record_id"]]
            self.assertEqual(child["source_id"], parent["source_id"])
            self.assertEqual(child["group_id"], parent["group_id"])
            self.assertEqual(builder.previous.source_keys(child), builder.previous.source_keys(parent))
            self.assertEqual(child["original_ref"]["parent_record_sha256"], builder.data.digest(parent))
            self.assertEqual(child["original_ref"]["parent_admission_sha256"],
                             builder.data.sha(self.admission / "manifest.json"))
            self.assertEqual(child["english_messages"], parent["english_messages"])
            self.assertEqual(child["original_english_messages"], parent["original_english_messages"])
            self.assertFalse(child["augmentation"]["assistant_targets_changed"])
            self.assertFalse(child["augmentation"]["independent_source"])
            self.assertFalse(child["augmentation"]["native_texting_naturalness_reviewed"])
            self.assertEqual(child["split"], parent["split"])
            for original, derived in zip(parent["messages"], child["messages"], strict=True):
                expected = original["content"].replace("\u025b", "3") if original["role"] == "user" else original["content"]
                self.assertEqual(derived, {**original, "content": expected})
            for edit in child["augmentation"]["edits"]:
                original = parent["messages"][edit["message_index"]]["content"]
                derived = child["messages"][edit["message_index"]]["content"]
                restored = list(derived)
                for position in edit["character_positions"]:
                    restored[position] = "\u025b"
                self.assertEqual("".join(restored), original)
                self.assertEqual(edit["original_sha256"], builder.data.digest(original))
                self.assertEqual(edit["derived_sha256"], builder.data.digest(derived))

    def test_no_eligible_user_character_creates_no_augmentation(self):
        for record in self.records:
            record["messages"][1]["content"] += " Assistant-only marker \u025b."
        self.write_admission()
        result = self.prepare_spelling()
        self.assertEqual(self.children(), [])
        self.assertEqual(result["selection"]["candidate_histories"].get("new_spelling_augmentation", 0), 0)

    def check_excluded_forms(self, examples):
        selected = self.chosen_parent()
        for index, (label, text) in enumerate(examples):
            with self.subTest(form=label):
                selected["messages"][0]["content"] = text
                self.write_admission()
                self.out = self.root / ("excluded-" + str(index))
                self.prepare_spelling()
                self.assertEqual(len(self.children()), 0)

    def test_supported_literal_and_url_markers_are_excluded(self):
        self.check_excluded_forms([
            ("straight_double", 'Synthetic literal "q\u025b".'),
            ("backtick", "Synthetic literal `q\u025b`."),
            ("open_angle", "Synthetic literal <q\u025b."),
            ("close_angle", "Synthetic literal q\u025b>."),
            ("https", "Synthetic link https://fixture.invalid/q\u025b"),
            ("http", "Synthetic link http://fixture.invalid/q\u025b"),
        ])

    def test_other_quoted_and_url_forms_are_excluded(self):
        self.check_excluded_forms([
            ("straight_single", "Synthetic literal 'q\u025b'."),
            ("smart_double", "Synthetic literal \u201cq\u025b\u201d."),
            ("smart_single", "Synthetic literal \u2018q\u025b\u2019."),
            ("uppercase_scheme", "Synthetic link HTTPS://fixture.invalid/q\u025b"),
        ])

    def test_protected_later_user_turn_excludes_the_whole_history(self):
        selected = self.chosen_parent()
        selected["messages"][0]["content"] = "Synthetic q\u025b without literal markers."
        selected["messages"].extend([
            {"role": "user", "content": 'Later synthetic literal "q\u025b".'},
            {"role": "assistant", "content": "Later synthetic response."}])
        self.write_admission()
        self.prepare_spelling()
        self.assertEqual(self.children(), [])

    def test_augmented_text_hit_blocks_parent_and_other_connected_views(self):
        selected = self.chosen_parent()
        selected["messages"][0]["content"] = "Synthetic fixture q\u025b is isolated here."
        selected["family_id"] = "held-synthetic-family"
        sibling = next(record for record in self.records if record is not selected)
        sibling["family_id"] = selected["family_id"]
        builder.data.write_rows(self.fresh / "held-content.jsonl", [{"id": "synthetic-child-hit",
            "text": selected["messages"][0]["content"].replace("\u025b", "3")}])
        reseal(self.fresh, "FROZEN.json")
        self.write_admission()
        self.prepare_spelling()
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        excluded_sources = {selected["source_id"], sibling["source_id"]}
        self.assertFalse(excluded_sources & {row["source_id"] for row in rows})

    def test_preexisting_spelling_rows_are_retained_without_reaugmentation(self):
        old = deepcopy(self.records[0])
        old.update(id="old-fixture:ascii-input-v1", source_id="old-fixture-source",
                   group_id="old-fixture-group", origin="agent_translated_source_with_synthetic_user_spelling")
        old["messages"][0]["content"] = "Inherited synthetic spelling q3."
        old["messages"][1]["content"] = "Inherited synthetic answer \u025b."
        builder.data.write_rows(self.r4 / "inherited.jsonl", [old])
        builder.data.save(self.r4 / "inherited-manifest.json", {
            "catalog_dataset_id": "responses", "records_path": "inherited.jsonl"})
        builder.data.save(self.r4 / "source-manifests.json", [{
            "manifest_path": "inherited-manifest.json",
            "manifest_sha256": builder.data.sha(self.r4 / "inherited-manifest.json")}])
        reseal(self.r4)
        self.stack.enter_context(patch.object(builder, "R4_SHA", builder.data.sha(self.r4 / "manifest.json")))
        self.prepare_spelling()
        snapshots = {row["id"]: row for row in self.output_records()}
        self.assertEqual(snapshots[old["id"]], old)
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        inherited = [row for row in rows if row["source_id"] == old["source_id"]]
        self.assertTrue(inherited)
        self.assertTrue(all(row["preparation_origin"] == "inherited_r4" for row in inherited))
        self.assertFalse(any(row["id"].count(":ascii-input-v1") > 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
