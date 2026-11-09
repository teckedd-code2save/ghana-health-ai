"""Offline R5 integrity regressions using only temporary synthetic sources."""
from copy import deepcopy
import re
import unittest
from unittest.mock import patch

import check_afrique_training_contract as contract
import prepare_understanding_r5 as builder
from test_prepare_understanding_r5 import BuilderFixture, reseal


class SyntheticTokenizer:
    """Character codec for exercising the real encoder, not native token equivalence."""
    eos_token_id = contract.EOS
    all_special_ids = contract_special_ids = {contract.EOS}
    model_max_length = 4096

    def encode(self, text, **kwargs):
        return [contract.EOS if value == "<eos>" else ord(value)
                for value in re.findall(r"<eos>|.", text, re.DOTALL)]

    def decode(self, tokens, **kwargs):
        return "".join("<eos>" if token == contract.EOS else chr(token) for token in tokens)

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join("[" + message["role"] + "]\n" + message["content"].strip() + "<eos>\n"
                       for message in messages)
        return text + ("[assistant]\n" if add_generation_prompt else "")


class NativePrefixTests(unittest.TestCase):
    def test_system_and_prior_assistants_are_masked_for_every_complete_prefix(self):
        tokenizer = SyntheticTokenizer()
        messages = [{"role": "system", "content": "Synthetic system."},
                    {"role": "user", "content": "First synthetic request."},
                    {"role": "assistant", "content": "First synthetic answer."},
                    {"role": "user", "content": "Second synthetic request."},
                    {"role": "assistant", "content": "Second synthetic answer."}]
        prefixes = list(builder.previous.prefixes({"messages": messages}))
        self.assertEqual([len(prefix) for prefix in prefixes], [3, 5])
        for prefix in prefixes:
            encoded = contract.encode_complete(tokenizer, {"id": "fixture", "messages": prefix}, 4096)
            prompt = tokenizer.encode(tokenizer.apply_chat_template(prefix[:-1], add_generation_prompt=True))
            labels = encoded["labels"]
            self.assertEqual(labels[:len(prompt)], [-100] * len(prompt))
            self.assertEqual(encoded["input_ids"][:len(prompt)], prompt)
            self.assertEqual(tokenizer.decode(labels[len(prompt):-1]), prefix[-1]["content"])
            self.assertEqual(labels[-1], contract.EOS)
            self.assertEqual(labels.count(contract.EOS), 1)

    def test_real_encoder_distinguishes_overlength_and_invalid_unicode(self):
        row = {"id": "fixture", "messages": [
            {"role": "user", "content": "Synthetic input."},
            {"role": "assistant", "content": "Synthetic reply."}]}
        with self.assertRaisesRegex(ValueError, "^overlength_no_truncation: fixture$"):
            contract.encode_complete(SyntheticTokenizer(), row, 10)
        row["messages"][1]["content"] += "\ufffd"
        with self.assertRaisesRegex(ValueError, "Invalid Unicode"):
            contract.encode_complete(SyntheticTokenizer(), row, 4096)


class ProtectionTests(unittest.TestCase):
    def record(self, identity, **fields):
        return {"id": identity, "source_id": "source-" + identity, "messages": [
            {"role": "user", "content": "Synthetic question " + identity + "."},
            {"role": "assistant", "content": "Synthetic answer " + identity + "."}], **fields}

    def blocked(self, records, text):
        return builder.previous.protected_groups(list(builder.protection_views(records)),
                                                builder.HoldoutIndex([("synthetic-held", text)]))

    def test_transitive_ancestry_across_top_level_metadata_and_source_metadata(self):
        a = self.record("a", family_id="shared-family", metadata={"tree_id": "nested-tree"})
        b = self.record("b", family_id="shared-family", metadata={"group_id": "other-group"})
        c = self.record("c", source_metadata={"tree_id": "nested-tree", "source_ids": ["ancestor"]})
        d = self.record("d", source_ids=["ancestor"])
        records = [a, b, c, d, self.record("unrelated")]
        before = deepcopy(records)
        self.assertEqual(set(self.blocked(records, a["messages"][0]["content"])), {"a", "b", "c", "d"})
        self.assertEqual(records, before)

    def test_each_full_history_and_assistant_prefix_is_scanned_separately(self):
        for field in ("messages", "english_messages", "original_english_messages"):
            for length in (2, 4):
                with self.subTest(field=field, length=length):
                    record = self.record("a")
                    history = [{"role": "user", "content": "Unique earlier request."},
                               {"role": "assistant", "content": "Distinct earlier answer."},
                               {"role": "user", "content": "Later conversation request."},
                               {"role": "assistant", "content": "Later conversation answer."}]
                    record[field] = history
                    text = "\n".join(message["content"] for message in history[:length])
                    self.assertEqual(set(self.blocked([record], text)), {"a"})


class BuilderIntegrityTests(BuilderFixture, unittest.TestCase):
    def keep_rows(self, rows):
        return {"name": "responses", "schedule": list(range(len(rows))), "primary_collections": ["responses"]}

    def test_only_explicit_overlength_is_excluded_without_truncation(self):
        def encode(tokenizer, witness, limit):
            if witness["id"] == "new-11:assistant-1":
                raise ValueError("overlength_no_truncation: " + witness["id"])
            return self.fake_encode(tokenizer, witness, limit)
        self.encoder.side_effect = encode
        self.prepare()
        excluded = builder.previous.read_rows(self.out / "excluded.jsonl")
        self.assertIn({"id": "new-11:assistant-1", "reason": "overlength_no_truncation",
                       "detail": "overlength_no_truncation: new-11:assistant-1"}, excluded)
        rows = builder.previous.read_rows(self.out / "rows.jsonl")
        self.assertIn("new-11:assistant-0", {row["id"] for row in rows})
        self.assertNotIn("new-11:assistant-1", {row["id"] for row in rows})

    def test_other_value_errors_cannot_masquerade_as_length_exclusions(self):
        for message in ("Invalid Unicode in message", "Invalid assistant-only mask",
                        "Token boundary mismatch", "overlength_no_truncation: another-record"):
            with self.subTest(message=message):
                self.encoder.side_effect = ValueError(message)
                with self.assertRaises(ValueError) as raised:
                    self.prepare()
                self.assertEqual(str(raised.exception), message)
                self.assertFalse(self.out.exists())

    def test_final_verification_interruption_revokes_marker(self):
        self.runtime.verify_pack.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())
        self.assertTrue((self.out / "manifest.json").exists())

    def test_mutation_during_final_verification_revokes_marker(self):
        def verify(path):
            builder.data.write_rows(self.instruction / "fixture.train.jsonl", [{"changed": True}])
            return {}, [], []
        self.runtime.verify_pack.side_effect = verify
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())

    def test_late_unbound_instruction_file_rejected(self):
        def verify(path):
            builder.data.write_rows(self.instruction / "late.train.jsonl", [])
            return {}, [], []
        self.runtime.verify_pack.side_effect = verify
        with self.assertRaisesRegex(ValueError, "Unbound"):
            self.prepare()
        self.assertFalse((self.out / "PREPARED.json").exists())

    def test_resealed_input_cannot_replace_initial_snapshot(self):
        def encode(*args):
            builder.data.write_rows(self.instruction / "fixture.train.jsonl", [{"changed": True}])
            reseal(self.instruction, "SEALED.json")
            return self.fake_encode(*args)
        self.encoder.side_effect = encode
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_admission_conversations_must_be_manifest_bound(self):
        manifest = builder.data.read(self.admission / "manifest.json")
        del manifest["artifacts"]["conversations.jsonl"]
        builder.data.save(self.admission / "manifest.json", manifest)
        builder.data.save(self.admission / "READY.json", {
            "manifest_sha256": builder.data.sha(self.admission / "manifest.json")})
        with self.assertRaisesRegex(ValueError, "manifest-bound"):
            self.prepare()
        self.encoder.assert_not_called()

    def test_original_prefix_hit_excludes_entire_connected_tree(self):
        self.records[0]["original_english_messages"] = deepcopy(self.records[-1]["messages"])
        self.records[0]["metadata"] = {"tree_id": "shared-tree"}
        self.records[1]["tree_id"] = "shared-tree"
        text = "\n".join(m["content"] for m in self.records[0]["original_english_messages"][:2])
        builder.data.write_rows(self.fresh / "held-content.jsonl", [{"id": "fixture-prefix", "text": text}])
        reseal(self.fresh, "FROZEN.json")
        self.write_admission()
        with patch.object(builder, "schedule", side_effect=self.keep_rows):
            self.prepare()
        ids = {r["dataset_ref"]["record_id"] for r in builder.previous.read_rows(self.out / "rows.jsonl")}
        self.assertFalse(ids & {"new-00", "new-01", "new-00:english", "new-01:english"})

    def bound_record(self):
        record = deepcopy(self.records[0])
        record["review"].update(decision_sha256="a" * 64,
            screened_messages_sha256=builder.data.digest(record["messages"]),
            screened_english_messages_sha256=builder.data.digest(record["english_messages"]),
            screen_resolutions=[{"code": "fixture", "turn": 1, "reason": "Independent fixture review."}])
        record["original_ref"]["review_decision_sha256"] = "a" * 64
        return record

    def test_resolution_requires_matching_decision_and_nonblank_reason(self):
        record = self.bound_record()
        self.assertEqual(builder.review_resolutions(record), {("fixture", 1)})
        record["original_ref"]["review_decision_sha256"] = "b" * 64
        self.assertEqual(builder.review_resolutions(record), set())
        record = self.bound_record()
        record["review"]["screen_resolutions"][0]["reason"] = "  "
        self.assertEqual(builder.review_resolutions(record), set())

    def test_english_and_spelling_views_cannot_borrow_twi_text_approval(self):
        record = self.bound_record()
        english = {**record, "origin": "source_english_separately_reviewed_for_r5"}
        english["review"]["screened_english_messages_sha256"] = "b" * 64
        self.assertEqual(builder.review_resolutions(english), set())
        record = self.bound_record()
        record["messages"][0]["content"] += " Generated spelling."
        record["origin"] = "agent_translated_source_with_synthetic_user_spelling"
        self.assertEqual(builder.review_resolutions(record), set())

    def test_synthetic_admission_to_pack_dryrun(self):
        import admit_understanding_r5 as admission
        from test_admit_understanding_r5 import bundle
        from test_select_understanding_r5_sources import source
        sources, proposals, decisions = bundle([source("dryrun-" + str(i)) for i in range(12)])
        source_file, proposal_file, review_file = [self.root / name for name in
                                                  ("sources.jsonl", "proposals.json", "review.json")]
        builder.data.write_rows(source_file, sources)
        builder.data.save(proposal_file, {"agent": "fixture-translator", "human_reviewed": False,
            "training_eligible": False, "source_file": str(source_file),
            "source_file_sha256": builder.data.sha(source_file), "records": proposals})
        builder.data.save(review_file, {"kind": "agent_review", "reviewer": "fixture-reviewer",
            "human_reviewed": False, "proposal_sha256": builder.data.sha(proposal_file), "records": decisions})
        self.admission = self.root / "checked-admission"
        result = admission.admit([(proposal_file, review_file)], self.r4, self.fresh, self.admission)
        self.assertEqual(result["admitted_histories"], 12)
        prepared = self.prepare()
        self.assertEqual(prepared["selection"]["new_selected_records"], 12)
        self.assertFalse(prepared["selection"]["training_executed"])
        self.assertTrue((self.out / "PREPARED.json").is_file())


if __name__ == "__main__":
    unittest.main()
