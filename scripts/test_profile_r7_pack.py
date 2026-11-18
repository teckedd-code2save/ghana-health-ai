"""CPU-only accounting tests with small sealed packs and no tokenizer download."""
import copy
from pathlib import Path
import tempfile
import unittest

import profile_r7_pack as profile


def example(identity, source, collection="meaning", metadata=None, text="original target"):
    metadata = {"group_id": "group:" + source, "group_aliases": [source],
                "r7_view": {"parent_record_id": "parent:" + identity}, **(metadata or {})}
    witness = {"id": identity, "source_id": source, "collection": collection,
               "dataset_ref": {"record_id": identity}, "metadata": metadata,
               "original_ref": {"r7_upstream_admission": {"release_manifest_sha256": "release",
                  "dataset_id": "source-" + collection, "dataset_version": "fixture"}}}
    if collection == "language":
        witness["text"] = text
        ids, labels = [2, 3], [2, 3]
    else:
        witness["messages"] = [{"role": "user", "content": "source question"}, {"role": "assistant", "content": text}]
        ids, labels = [1, 2, 3], [-100, 2, 3]
    row = {k: witness[k] for k in ("id", "source_id", "collection", "dataset_ref")}
    row.update(objective="causal_lm" if collection == "language" else "final_assistant",
               input_ids=ids, labels=labels, attention_mask=[1] * len(ids), sequence_tokens=len(ids), target_tokens=2)
    return row, witness


def fixture(root):
    pack = root / "pack"
    pack.mkdir()
    examples = [
        example("v0", "a", metadata={"task": "source_preserved_bilingual_alignment", "target_language": "tw"}),
        example("v1", "a", metadata={"origin": "programmatically_augmented_original_parallel_sentences", "target_language": "tw"}),
        example("v2", "a", "language", {"prior_version": "r6", "language": "tw"}),
        example("v3", "c", "responses", {"prior_version": "r5"}),
        example("v4", "d", metadata={"record_type": "intent_example", "language": "tw"}, text='{"intent":"label"}'),
    ]
    examples[1][1]["original_ref"].update(current={"inventory_candidate_id": "parallel:a"}, previous={"inventory_candidate_id": "parallel:b"})
    examples[1][0]["input_ids"] = [9, 2, 3]
    examples[4][0].update(input_ids=[7, 4, 5], labels=[-100, 4, 5])
    profile.save_rows(pack / "rows.jsonl", [r for r, _ in examples])
    profile.save_rows(pack / "source-witnesses.jsonl", [w for _, w in examples])
    profile.save(pack / "stages.json", [{"name": "fixture", "schedule": [0, 1, 2, 3, 4, 0]}])
    profile.save_rows(pack / "duplicates.jsonl", [{"id": "removed", "retained_id": "v0", "collection": "meaning"}])
    profile.save_rows(pack / "exclusions.jsonl", [])
    profile.save(pack / "report.json", {"source_records": 6, "derived_views_considered": 6, "retained_views": 5,
                 "duplicate_views": 1, "excluded_views": 0, "views_by_collection": {"meaning": 3, "responses": 1, "language": 1}})
    reseal(pack)
    return pack


def reseal(pack):
    profile.save(pack / "manifest.json", {"artifacts": {p.name: profile.sha(p) for p in pack.iterdir()
                  if p.name not in {"manifest.json", "PREPARED.json"}}})
    profile.save(pack / "PREPARED.json", {"manifest_sha256": profile.sha(pack / "manifest.json")})


class ProfileTests(unittest.TestCase):
    def test_alias_components_merge_transitively_without_claiming_independence(self):
        groups = profile.Families()
        groups.connect(["c", "d"])
        groups.connect(["a", "b"])
        groups.connect(["b", "c"])
        self.assertEqual({groups.root(k) for k in "abcd"}, {"a"})

    def test_language_uses_labels_not_id_or_collection_and_separates_structured_targets(self):
        _, witness = example("tw-en", "english-looking-id")
        self.assertEqual(profile.language(witness)[1], "unknown")
        witness["metadata"].update(record_type="intent_example", language="tw")
        self.assertEqual(profile.language(witness)[:2], ("tw", "structured_intent_label"))
        witness["messages"][-1]["tool_calls"] = [{"id": "call", "function": {"name": "lookup", "arguments": {}}}]
        self.assertEqual(profile.language(witness)[1], "structured_tool_call")

    def test_exact_payloads_preserve_whitespace_and_match_text_and_chat(self):
        _, chat = example("a", "source")
        _, text = example("b", "source", "language")
        self.assertEqual(profile.target_payload(chat), profile.target_payload(text))
        text["text"] += " "
        self.assertNotEqual(profile.digest(profile.target_payload(chat)), profile.digest(profile.target_payload(text)))

    def test_profile_counts_schedule_reuse_context_aliases_and_crossclass_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = fixture(root)
            before = {p.name: profile.sha(p) for p in pack.iterdir()}
            result = profile.prepare(pack, root / "profile")
            total = result["total"]
            self.assertEqual(total["retained_views"], 5)
            self.assertEqual(total["unique_upstream_admitted_records"], 5)
            self.assertEqual(total["unique_target_source_id_values"], 3)
            self.assertEqual(total["unique_source_id_values_including_context_refs"], 4)
            self.assertEqual(total["connected_declared_alias_groups"], 3)
            self.assertEqual(total["scheduled_presentations"], 6)
            self.assertEqual(total["tokens_scheduled"]["input_tokens"], 17)
            self.assertEqual(total["tokens_scheduled"]["effective_target_tokens"], 11)
            self.assertEqual(result["duplicates"]["same_class_removed"], 1)
            self.assertEqual(result["duplicates"]["residual_crossclass_groups"], 1)
            self.assertEqual(result["duplicates"]["residual_same_class_groups"], 0)
            self.assertEqual(result["target_reuse"]["cross_method_same_source_target_groups"], 1)
            self.assertTrue(profile.verify(root / "profile")["verified"])
            self.assertEqual(before, {p.name: profile.sha(p) for p in pack.iterdir()})
            with self.assertRaisesRegex(ValueError, "New profile directory"):
                profile.prepare(pack, root / "profile")

    def test_context_only_turns_do_not_inflate_supervised_target_exposure(self):
        row, witness = example("a", "source")
        witness["messages"].insert(1, {"role": "assistant", "content": "context only"})
        description = profile.describe(row, witness, 2)
        self.assertEqual(description["effective_target_tokens"], 2)
        self.assertEqual(description["target_payload_sha256"], profile.digest({"text": "original target"}))

    def test_rejects_unsealed_modified_and_inside_pack_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = fixture(root)
            with self.assertRaisesRegex(ValueError, "outside sealed pack"):
                profile.prepare(pack, pack / "profile")
            (pack / "rows.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                profile.prepare(pack, root / "profile")

    def test_rejects_out_of_range_schedule_and_mismatched_witness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = fixture(root)
            profile.save(pack / "stages.json", [{"schedule": [5]}])
            reseal(pack)
            with self.assertRaisesRegex(ValueError, "outside pack"):
                profile.prepare(pack, root / "profile")
        row, witness = example("a", "source")
        witness = copy.deepcopy(witness)
        witness["source_id"] = "different"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            profile.describe(row, witness, 1)


if __name__ == "__main__":
    unittest.main()
