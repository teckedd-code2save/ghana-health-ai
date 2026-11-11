"""Synthetic English-only integrity tests; no real corpora or model access."""
from copy import deepcopy
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import select_understanding_r5_sources as selection


def source(identity, group=None, **metadata):
    messages = [{"role": "user", "content": "Synthetic request " + identity + "."},
                {"role": "assistant", "content": "Synthetic response " + identity + "."}]
    return {"id": identity, "source_id": "source-" + identity,
            "collection": "english_retention", "messages": messages,
            "metadata": {"group_id": group or "group-" + identity, "split": "train",
                         "origin": "source_preserved", "source_hash": selection.data.digest(messages),
                         **metadata},
            "original_ref": {"fixture": identity, "split": "train"}}


def seal(root, rows, seal_name="PREPARED.json", omit=()):
    root.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        selection.data.write_rows(root / name, values)
    manifest = {"version": "synthetic-fixture-v1", "artifacts": {
        name: selection.data.sha(root / name) for name in rows if name not in omit}}
    selection.data.save(root / "manifest.json", manifest)
    selection.data.save(root / seal_name, {"manifest_sha256": selection.data.sha(root / "manifest.json")})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.r2, self.r4, self.out = [self.root / name for name in ("r2", "r4", "out")]
        self.candidates = self.root / "candidates.jsonl"
        self.sources = [source("alpha"), source("beta"), source("gamma")]
        self.held = [{"id": "held", "text": "Separate synthetic exclusion sentence."}]
        self.write_inputs()
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("No network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def write_inputs(self, proposed=None, omit_r2=(), omit_r4=()):
        seal(self.r2, {"source-witnesses.jsonl": self.sources}, omit=omit_r2)
        seal(self.r4, {"held-content.jsonl": self.held}, omit=omit_r4)
        selection.data.write_rows(self.candidates, [
            {"id": row["id"], "record_sha256": selection.data.digest(row)}
            for row in (self.sources if proposed is None else proposed)])

    def run_select(self, **kwargs):
        return selection.select(self.candidates, self.r2, self.r4, self.out, **kwargs)

    def chosen(self):
        return selection.previous.read_rows(self.out / "sources.jsonl")

    def test_selection_is_deterministic_non_admitting_and_preserves_sources(self):
        before = {p: selection.data.sha(p) for p in self.root.rglob("*") if p.is_file()}
        manifest = self.run_select(limit=2, shards=2)
        self.assertEqual(manifest["selected"], 2)
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual([r["source_record"]["id"] for r in self.chosen()], ["alpha", "beta"])
        self.assertEqual([r["index"] for r in self.chosen()], [0, 1])
        self.assertTrue(all(not r["training_eligible"] for r in self.chosen()))
        shards = [selection.previous.read_rows(self.out / ("source-shard-" + str(i) + ".jsonl"))
                  for i in range(2)]
        self.assertEqual(sorted(r["index"] for shard in shards for r in shard), [0, 1])
        self.assertEqual(before, {p: selection.data.sha(p) for p in before})

    def test_unbound_required_artifacts_rejected(self):
        for pack in ("r2", "r4"):
            with self.subTest(pack=pack):
                self.write_inputs(omit_r2=("source-witnesses.jsonl",) if pack == "r2" else (),
                                  omit_r4=("held-content.jsonl",) if pack == "r4" else ())
                with self.assertRaises(ValueError):
                    self.run_select()
                self.assertFalse(self.out.exists())

    def test_empty_protection_is_not_clearance(self):
        self.held = []
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.run_select()
        self.assertFalse(self.out.exists())

    def test_duplicate_witness_ids_rejected(self):
        self.sources.append(deepcopy(self.sources[0]))
        self.write_inputs(proposed=self.sources[:1])
        with self.assertRaises(ValueError):
            self.run_select()

    def test_duplicate_candidate_ids_rejected(self):
        self.write_inputs(proposed=[self.sources[0], self.sources[0]])
        with self.assertRaises(ValueError):
            self.run_select()

    def test_nontrain_source_cannot_be_selected(self):
        self.sources[0]["metadata"]["split"] = "test"
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.run_select()

    def test_protected_sibling_outside_candidate_list_still_blocks_group(self):
        self.sources[1]["metadata"]["group_id"] = self.sources[0]["metadata"]["group_id"]
        self.held[0]["text"] = self.sources[0]["messages"][0]["content"]
        self.write_inputs(proposed=self.sources[1:])
        self.run_select()
        self.assertEqual([r["source_record"]["id"] for r in self.chosen()], ["gamma"])

    def test_screened_out_source_still_protects_sibling(self):
        self.sources[1]["metadata"]["group_id"] = self.sources[0]["metadata"]["group_id"]
        self.held[0]["text"] = self.sources[0]["messages"][0]["content"]
        self.write_inputs()
        original = selection.inspect_messages
        def screen(messages):
            return ([{"code": "synthetic_review", "turn": 1, "severity": "review"}]
                    if messages == self.sources[0]["messages"] else original(messages))
        with patch.object(selection, "inspect_messages", side_effect=screen):
            self.run_select()
        self.assertEqual([r["source_record"]["id"] for r in self.chosen()], ["gamma"])

    def test_connected_ancestry_not_just_direct_group_controls_dedup(self):
        self.sources[0]["metadata"]["family_id"] = "shared-family"
        self.sources[1]["metadata"]["family_id"] = "shared-family"
        self.write_inputs()
        report = self.run_select()
        self.assertEqual(report["selected"], 2)
        self.assertEqual([r["source_record"]["id"] for r in self.chosen()], ["alpha", "gamma"])

    def test_changed_inputs_during_screening_abort_before_output(self):
        original = selection.inspect_messages
        changed = False
        def screen(messages):
            nonlocal changed
            if not changed:
                self.held.append({"id": "late", "text": "New synthetic protected material."})
                selection.data.write_rows(self.r4 / "held-content.jsonl", self.held)
                changed = True
            return original(messages)
        with patch.object(selection, "inspect_messages", side_effect=screen):
            with self.assertRaises(ValueError):
                self.run_select()
        self.assertFalse(self.out.exists())

    def test_bool_limits_not_accepted_as_integers(self):
        with self.assertRaises(ValueError):
            self.run_select(limit=True)

    def test_linked_candidate_file_rejected(self):
        target = self.root / "real-candidates.jsonl"
        self.candidates.rename(target)
        self.candidates.symlink_to(target)
        with self.assertRaises(ValueError):
            self.run_select()

    def test_mutation_while_publishing_does_not_publish_a_manifest(self):
        original = selection.data.write_rows
        def write(path, rows):
            original(path, rows)
            if path == self.out / "sources.jsonl":
                original(self.candidates, [])
        with patch.object(selection.data, "write_rows", side_effect=write):
            with self.assertRaises(ValueError):
                self.run_select()
        self.assertFalse((self.out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
