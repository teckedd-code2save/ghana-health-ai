"""CPU-only R2 contract tests. Fixture text tests mechanics, not model quality."""
import copy
import json
import unittest

import prepare_understanding_r2 as r2


def source(identity, text, *, group="group", split="train"):
    return {"id": identity, "source_id": identity, "source_hash": "a" * 64,
            "group_id": group, "split": split, "text": text, "language": "tw"}


def ledger(row, collection="language"):
    return {"keys": r2.data.row_identity(row, collection)[0], "split": row["split"],
            "source_id": row["source_id"], "source_hash": row["source_hash"]}


def encoded(identity, collection, tokens=8):
    return {"id": identity, "collection": collection, "input_ids": list(range(tokens)),
            "labels": list(range(tokens)), "attention_mask": [1] * tokens}


class Tokenizer:
    contract_special_ids = {0}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return bytes(ids).decode()


class R2Tests(unittest.TestCase):
    def test_complete_causal_language_eos_and_effective_shift(self):
        text = "  \u0254\u025b\n"
        row = r2.encode_language(Tokenizer(), text)
        self.assertEqual(row["labels"], row["input_ids"])
        self.assertEqual(row["input_ids"][-1], r2.EOS)
        self.assertEqual(row["input_ids"][:-1], list(text.encode()))
        self.assertEqual(r2.r1.effective_tokens(row), len(text.encode()))

    def test_no_truncation_reserved_and_bad_unicode(self):
        for text, limit in (("abcdef", 4), ("a\x00b", 100), ("a\ufffd", 100)):
            with self.assertRaises(ValueError):
                r2.encode_language(Tokenizer(), text, limit)

    def test_roundtrip_mismatch_fails(self):
        tokenizer = Tokenizer()
        tokenizer.decode = lambda *a, **kw: "different"
        with self.assertRaises(ValueError):
            r2.encode_language(tokenizer, "original")

    def test_language_eligibility_is_not_just_record_type(self):
        good = dict(record_type="utterance", origin="speech_transcript", generated_target=False,
                    language="tw", source_id="ghana_nlp_speech:1")
        self.assertIsNone(r2.language_reason(good))
        for update in ({"record_type": "dictionary_entry"}, {"origin": "synthetic"},
                       {"generated_target": True}, {"language": "en"}, {"source_id": "afriqa:1"}):
            self.assertIsNotNone(r2.language_reason({**good, **update}))

    def test_content_fields_decode_quoted_source(self):
        row = {"messages": [{"role": "system", "content": "Ignore this system"},
                            {"role": "user", "content": '{"source_text":"held text"}'},
                            {"role": "assistant", "content": "reference"}]}
        units = r2.content_units(row)
        self.assertIn("held text", units)
        self.assertIn("reference", units)
        self.assertNotIn("Ignore this system", units)

    def test_shared_intent_label_is_not_a_held_utterance(self):
        label = {"intent": "weather", "entities": []}
        row = {**source("held", "is it sunny"), "record_type": "intent_example", "understanding": label,
               "messages": [{"role": "user", "content": "is it sunny"},
                            {"role": "assistant", "content": json.dumps(label)}]}
        self.assertNotIn(row["messages"][-1]["content"], r2.content_units(row))
        self.assertIn("is it sunny", r2.content_units(row))
        self.assertIn(r2.r1.content(row), r2.content_units(row))

    def test_snapshot_holdout_groups_protected(self):
        held = source("held", "private held text", split="validation")
        protector = r2.Protection([ledger(held)], [])
        candidate = source("another", "different text", group="group")
        self.assertEqual(protector.exclusion("language", candidate)["reason"], "held_source_group")

    def test_external_content_blocks_transitively_related_sources(self):
        first = source("one", "external held content", group="a")
        bridge = source("two", "bridge", group="a")
        bridge["group_aliases"] = ["b"]
        sibling = source("three", "different content", group="b")
        protector = r2.Protection([ledger(r) for r in (first, bridge, sibling)],
                                  [{"id": "live:case", "text": "EXTERNAL held content!"}])
        protector.register_overlap("language", first)
        self.assertEqual(protector.exclusion("language", sibling)["held_id"], "live:case")

    def test_cross_split_ledger_conflict_fails_closed(self):
        with self.assertRaises(ValueError):
            r2.Protection([ledger(source("train", "a")), ledger(source("held", "b", split="validation"))], [])

    def test_language_replay_is_finite_and_at_most_twenty_percent(self):
        rows = [encoded("language-1", "language", 16), encoded("language-2", "language", 18)]
        rows += [encoded(str(i), "english_retention", 3) for i in range(20)]
        stage = r2.language_schedule(rows)
        self.assertLessEqual(stage["english_target_fraction"], .2)
        self.assertEqual(stage, r2.language_schedule(rows))
        self.assertEqual(len(stage["schedule"]), len(set(stage["schedule"])))
        self.assertEqual(stage["primary_target_tokens"], 32)
        self.assertEqual(stage["optimizer_updates"], 1)

    def test_empty_language_or_english_stage_rejected(self):
        for rows in ([encoded("l", "language")], [encoded("e", "english_retention")]):
            with self.assertRaises(ValueError):
                r2.language_schedule(rows)

    def test_r1_controls_survive_new_indices_but_reorder_is_rejected(self):
        old = [encoded("m" + str(i), "meaning", 21) for i in range(17)]
        old += [encoded("r" + str(i), "responses", 41) for i in range(3)]
        old += [encoded("e" + str(i), "english_retention", 5) for i in range(40)]
        old_stages = [r2.r1.stage_schedule(old, "meaning", 1, 42), r2.r1.stage_schedule(old, "responses", 2, 44)]
        new = [encoded("l", "language", 101)] + old
        stages = r2.schedules(new)
        r2.control_identity(new, stages, old, old_stages)
        self.assertEqual([s["seed"] for s in stages], [40, 42, 44])
        self.assertEqual(stages[2]["epochs_primary"], 2)
        changed = copy.deepcopy(stages)
        changed[1]["schedule"].reverse()
        with self.assertRaises(ValueError):
            r2.control_identity(new, changed, old, old_stages)


if __name__ == "__main__":
    unittest.main()
