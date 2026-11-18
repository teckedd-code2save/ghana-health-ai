from copy import deepcopy
import unittest

import prepare_understanding_r7 as builder


class PackBuilderTests(unittest.TestCase):
    def source(self):
        return {"id": "dialogue-one", "source_id": "source-one", "collection": "responses", "split": "train",
            "training_eligible": True, "metadata": {"group_id": "tree-one"},
            "original_ref": {"sha256": "source-proof"}, "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First source answer"},
                {"role": "user", "content": "A follow-up"},
                {"role": "assistant", "content": "Second source answer"}]}

    def test_exact_prefixes_are_registered_without_source_mutation(self):
        source = self.source()
        original = deepcopy(source)
        ref = {"dataset_id": "responses", "dataset_version": "v1", "record_sha256": builder.run.digest(source)}
        views = list(builder.prefix_views(source, {}, ref))
        self.assertEqual(source, original)
        self.assertEqual([len(v["messages"]) for v in views], [2, 4])
        self.assertNotEqual(views[0]["id"], views[1]["id"])
        self.assertTrue(all(v["source_id"] == source["source_id"] for v in views))
        self.assertEqual(views[-1]["messages"], source["messages"])
        self.assertEqual(views[0]["original_ref"]["r7_upstream_admission"], ref)

    def test_group_protection_applies_to_both_directions(self):
        first = self.source()
        second = deepcopy(first)
        second.update(id="different-task-view", source_id="source-alias")
        second["messages"] = [{"role": "user", "content": "Different source view"},
                              {"role": "assistant", "content": "Different target"}]
        held = [{"id": "held", "text": "Not a shared literal", "metadata": {"group_id": "tree-one"}}]
        self.assertEqual(set(builder.protected_records([first, second], held)), {first["id"], second["id"]})

    def test_context_only_assistant_is_never_a_supervised_target(self):
        source = self.source()
        source["metadata"]["training_assistant_indices"] = [3]
        views = list(builder.prefix_views(source, {}, {"dataset_id": "x", "dataset_version": "v1", "record_sha256": "x"}))
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["messages"], source["messages"])
        self.assertEqual(views[0]["metadata"]["r7_view"]["assistant_message_index"], 3)
        for invalid in ([], [0], [3, 1], [1, 1], [True]):
            source["metadata"]["training_assistant_indices"] = invalid
            with self.assertRaises(ValueError):
                list(builder.prefix_views(source, {}, {"dataset_id": "x", "dataset_version": "v1", "record_sha256": "x"}))

    def test_full_history_content_protection(self):
        source = self.source()
        held = [{"id": "held", "text": "First question\nFirst source answer"}]
        self.assertIn(source["id"], builder.protected_records([source], held))

    def test_schedule_covers_all_views_once(self):
        rows = [{"id": str(i)} for i in range(57)]
        first = builder.one_pass_schedule(rows)
        self.assertEqual(sorted(first["schedule"]), list(range(57)))
        self.assertEqual(first, builder.one_pass_schedule(rows))
        for rows in ([], [None] * (builder.run.MAX_SCHEDULE + 1)):
            with self.assertRaises(ValueError):
                builder.one_pass_schedule(rows)

    def test_missing_provenance_or_incomplete_history_rejected(self):
        for mutation in (lambda r: r.pop("metadata"), lambda r: r["messages"].pop()):
            source = self.source()
            mutation(source)
            with self.assertRaises(ValueError):
                list(builder.prefix_views(source, {}, {"dataset_id": "x", "dataset_version": "v1", "record_sha256": "x"}))


if __name__ == "__main__":
    unittest.main()
