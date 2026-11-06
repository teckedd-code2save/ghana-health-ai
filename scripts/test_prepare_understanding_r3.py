import unittest
import tempfile
from pathlib import Path
import prepare_understanding_r3 as p
from run_understanding_r3 import admitted_sources


def row(identity, collection, tokens):
    return {"id": identity, "collection": collection,
            "input_ids": [1] + [2] * tokens, "labels": [-100] + [2] * tokens}


class PackTests(unittest.TestCase):
    def test_each_assistant_has_its_complete_prefix(self):
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(6)]
        self.assertEqual(list(p.prefixes({"messages": messages})), [messages[:2], messages[:4], messages])

    def test_system_preserved(self):
        messages = [{"role": "system", "content": "context"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self.assertEqual(list(p.prefixes({"messages": messages})), [messages])

    def test_reject_incomplete_history(self):
        with self.assertRaises(ValueError):
            list(p.prefixes({"messages": [{"role": "user", "content": "q"}]}))

    def test_schedule_is_repeatable_and_retention_without_replacement(self):
        rows = [row("response", "responses", 600)]
        for category in p.RATIOS:
            rows += [row(category + str(i), category, 50) for i in range(20)]
        a = p.replay_schedule(rows)
        self.assertEqual(a, p.replay_schedule(rows))
        self.assertEqual(a["schedule"].count(0), 2)
        self.assertTrue(all(a["schedule"].count(i) <= 1 for i in range(1, len(rows))))
        self.assertEqual(a["target_tokens_by_collection"], {"responses": 1200, "english_retention": 400, "meaning": 200, "language": 200})

    def test_group_closure_excludes_related_versions(self):
        a = {"id": "a", "source_id": "sourceA", "group_id": "tree", "text": "A protected source sentence."}
        b = {"id": "b", "source_id": "sourceB", "group_id": "tree", "text": "Different continuation."}
        c = {"id": "c", "source_id": "sourceC", "group_id": "other", "text": "Separate source."}
        held = p.HoldoutIndex([("held", a["text"])])
        self.assertEqual(set(p.protected_groups([a, b, c], held)), {"a", "b"})

    def test_replay_linked_source_also_excluded(self):
        new = {"id": "new", "source_id": "history", "source_ids": ["turn1"], "messages": [{"role": "user", "content": "Protected prompt"}]}
        old = {"id": "old", "source_id": "turn1", "metadata": {"group_id": "oldgroup"}, "text": "Different translation"}
        self.assertEqual(set(p.protected_groups([new, old], p.HoldoutIndex([("h", "Protected prompt")]))), {"new", "old"})

    def test_meaning_embedded_source_is_checked(self):
        record = {"messages": [{"role": "system", "content": "not a heldout"},
                                {"role": "user", "content": '{"source_text":"Original Twi"}'}]}
        self.assertIn("Original Twi", p.fields(record))
        self.assertNotIn("not a heldout", p.fields(record))

    def test_pass_snapshot_is_aligned_without_losing_response_rows(self):
        rows = [row("response" + str(i), "responses", 600) for i in range(9)]
        for category in p.RATIOS:
            rows += [row(category + str(i), category, 50) for i in range(100)]
        stage = p.replay_schedule(rows)
        self.assertEqual(stage["passes"][0]["end_sequence"] % 16, 0)
        self.assertEqual(stage["snapshot_after_updates"], [stage["passes"][0]["end_sequence"] // 16])
        for i in range(9):
            self.assertEqual(stage["schedule"].count(i), 2)
        for i in range(9, len(rows)):
            self.assertLessEqual(stage["schedule"].count(i), 1)

    def test_registered_record_and_prefix_are_verified_by_trainer(self):
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "Twi " + str(i)} for i in range(4)]
        source = {"id": "chat1", "source_id": "original1", "split": "train", "training_eligible": True,
                  "messages": messages, "origin": "source_grounded_ai_translated_and_agent_corrected"}
        ref = p.dataset_reference("responses", "20261106-r3-v1", source)
        rows = [{"id": "chat1:assistant0", "collection": "responses", "source_id": "original1",
                 "objective": "final_assistant", "dataset_ref": ref}]
        witnesses = [{**rows[0], "messages": messages[:2]}]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            references = p.register_datasets(folder, rows, witnesses, {("responses", "chat1"): source}, "20261106-r3-v1")
            artifacts = {f.relative_to(folder).as_posix(): p.data.sha(f) for f in folder.rglob("*") if f.is_file()}
            found = admitted_sources(folder, rows, witnesses, artifacts)
            self.assertEqual(found[0]["admitted_record_count"], 1)
            self.assertEqual(found[0]["dataset_id"], "gha-responses")
            witnesses[0]["messages"] = [{"role": "user", "content": "Changed"}, messages[1]]
            with self.assertRaisesRegex(ValueError, "exact prefix"):
                admitted_sources(folder, rows, witnesses, artifacts)
            witnesses[0]["messages"] = messages[:2]
            rows[0]["dataset_ref"] = {**ref, "record_sha256": "0" * 64}
            witnesses[0]["dataset_ref"] = rows[0]["dataset_ref"]
            with self.assertRaisesRegex(ValueError, "identity differs"):
                admitted_sources(folder, rows, witnesses, artifacts)

    def test_unadmitted_source_cannot_get_dataset_reference(self):
        source = {"id": "unreviewed", "source_id": "source", "split": "train", "training_eligible": False}
        with self.assertRaisesRegex(ValueError, "Unadmitted"):
            p.dataset_reference("responses", "v1", source)


if __name__ == "__main__":
    unittest.main()
