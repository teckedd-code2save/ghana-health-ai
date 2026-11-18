"""Offline source inventory tests using synthetic local fixtures, never APIs."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import prepare_r7_source_inventory as inventory


def node(identity, parent=None, tree="root", role="prompter", text="Explain a useful concept.", split="train", **extra):
    return {"message_id": identity, "parent_id": parent, "message_tree_id": tree, "role": role, "text": text,
            "lang": "en", "rank": 0 if role == "assistant" else None, "review_count": 3, "review_result": True,
            "deleted": False, "synthetic": False, "labels": {"name": ["pii"], "value": [0.0], "count": [3]},
            "tree_state": "ready_for_export", "source_split": split,
            "source_ref": {"file": "fixture.parquet", "file_sha256": "a" * 64, "row_index": 0}, **extra}


def conversation():
    return [node("root"), node("a1", "root", role="assistant", text="Here is a complete explanation of that useful concept."),
            node("u2", "a1", text="Actually, give two contrasting examples instead."),
            node("a2", "u2", role="assistant", text="- First: distinct example one\n- Second: distinct example two")]


def families(rows):
    result = inventory.Families()
    for row in rows:
        result.connect([row["message_id"], row["message_tree_id"]])
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def rows_file(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def pack_fixture(folder, record, held=False):
    folder.mkdir()
    rows_file(folder / "datasets/responses/records.jsonl", [record])
    write(folder / "datasets/responses/manifest.json", {"record_count": 1,
        "records_path": "datasets/responses/records.jsonl", "records_sha256": inventory.sha(folder / "datasets/responses/records.jsonl")})
    write(folder / "source-manifests.json", [{"manifest_path": "datasets/responses/manifest.json",
        "manifest_sha256": inventory.sha(folder / "datasets/responses/manifest.json")}])
    rows_file(folder / "source-witnesses.jsonl", [{**record, "collection": "responses", "metadata": {"origin": "fixture"}}])
    rows_file(folder / "rows.jsonl", [{"id": record["id"], "collection": "responses", "input_ids": [1, 2], "labels": [-100, 2]}])
    write(folder / "stages.json", [{"schedule": [0, 0]}])
    write(folder / "evaluation-manifest.json", {"version": "fixture"})
    write(folder / "overlap-audit.json", {"overlap_free": True, "protected_units": int(held)})
    if held:
        rows_file(folder / "held-content.jsonl", [{"id": "reserved", "text": "Never shared heldout material"}])
    write(folder / "manifest.json", {"artifacts": {p.relative_to(folder).as_posix(): inventory.sha(p)
                                                    for p in folder.rglob("*") if p.is_file()}})
    write(folder / "PREPARED.json", {"manifest_sha256": inventory.sha(folder / "manifest.json")})


class HistoryTests(unittest.TestCase):
    def test_complete_parent_chain_preserved_without_substituting_siblings(self):
        rows = conversation()
        selected, ledger = inventory.oasst_candidates(rows, [], families(rows), {}, set(), 1000)
        final = next(r for r in selected if r["id"] == "oasst:a2")
        self.assertEqual(final["path_message_ids"], [r["message_id"] for r in rows])
        self.assertEqual([m["content"] for m in final["messages"]], [r["text"] for r in rows])
        self.assertEqual([a["parent_id"] for a in final["ancestry"]], [None, "root", "a1", "u2"])
        self.assertEqual(final["coverage_bucket"], "correction_cue")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(r["status"] == "eligible" and not r["training_eligible"] for r in selected))

    def test_invalid_ancestry_excluded(self):
        for mutation, expected in ((lambda r: r[3].update(parent_id="missing"), "missing_parent"),
                                   (lambda r: r[2].update(parent_id="a2"), "cycle"),
                                   (lambda r: r[1].update(message_tree_id="another"), "cross_tree_parent"),
                                   (lambda r: r[1].update(source_split="validation"), "cross_split_parent"),
                                   (lambda r: r[2].update(role="assistant"), "invalid_role_sequence")):
            with self.subTest(expected=expected):
                rows = conversation()
                mutation(rows)
                _, ledger = inventory.oasst_candidates(rows, [], families(rows), {}, set(), 1000)
                final = next(r for r in ledger if r["id"] == "oasst:a2")
                self.assertEqual(final["status"], "excluded")
                self.assertIn(expected, final["reasons"])

    def test_rank_filter_does_not_rewrite_context(self):
        rows = conversation()
        rows[1]["rank"] = 1
        selected, ledger = inventory.oasst_candidates(rows, [], families(rows), {}, set(), 1000)
        self.assertFalse(selected)
        self.assertTrue(all("history_not_all_rank_zero" in row["reasons"] for row in ledger))

    def test_validation_tree_and_connected_branch_excluded(self):
        rows = conversation()
        related = [node("other", tree="other"), node("other_a", "other", tree="other", role="assistant")]
        family = families(rows + related)
        family.connect(["root", "other"])
        blocked = {family.root("root"): {"kind": "heldout"}}
        selected, ledger = inventory.oasst_candidates(rows + related, [], family, blocked, set(), 1000)
        self.assertFalse(selected)
        self.assertTrue(all("protected_source_family" in r["reasons"] for r in ledger))
        for row in rows:
            row["source_split"] = "validation"
        selected, ledger = inventory.oasst_candidates([], rows, families(rows), {}, set(), 1000)
        self.assertFalse(selected)
        self.assertTrue(all("original_validation_split" in r["reasons"] for r in ledger))

    def test_pii_labels_contact_patterns_and_mixed_language_never_exported(self):
        for changes, reason in (({"labels": {"name": ["pii"], "value": [1.0]}}, "pii_annotation_positive_or_missing"),
                                ({"text": "Contact person@example.test"}, "possible_email"),
                                ({"text": "Call +233 24 123 4567"}, "possible_phone"),
                                ({"lang": "tw"}, "history_not_entirely_original_english"),
                                ({"review_result": False}, "upstream_review_not_positive")):
            rows = conversation()
            rows[0].update(changes)
            selected, ledger = inventory.oasst_candidates(rows, [], families(rows), {}, set(), 1000)
            self.assertFalse(selected)
            self.assertTrue(all(reason in r["reasons"] for r in ledger))

    def test_queue_does_not_arbitrarily_take_first_twelve(self):
        values = [{"id": str(i), "review_flags": [{"code": "correction_cue"}] if i >= 100 else []} for i in range(200)]
        selected, deferred = inventory.queue_order(copy.deepcopy(values), 100)
        reverse, _ = inventory.queue_order(list(reversed(copy.deepcopy(values))), 100)
        self.assertEqual(selected, reverse)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len(deferred), 100)
        self.assertEqual(sum(r["coverage_bucket"] == "correction_cue" for r in selected), 50)

    def test_pattern_flags_are_literal_not_semantic_admission(self):
        messages = [{"role": "user", "content": "List category examples."},
                    {"role": "assistant", "content": "- A: same item\n- B: same item\n- B: same item"},
                    {"role": "user", "content": "Actually, distinguish them."},
                    {"role": "assistant", "content": "I'm sorry to hear that."}]
        flags = inventory.history_flags(messages)
        codes = {f["code"] for f in flags}
        self.assertTrue({"same_literal_example_across_category_labels", "repeated_list_items", "correction_cue",
                         "followup_turn", "sympathy_only_exact_phrase", "short_target_le8_words"} <= codes)
        report = inventory.pattern_summary([{"messages": messages}] * 2)
        self.assertEqual(report["assistant_occurrences"], 4)
        self.assertEqual(report["distinct_assistant_targets_exact"], 2)
        self.assertFalse(report["pattern_flags_are_semantic_certification"])

    def test_source_faithful_past_search_claim_still_requires_repair_review(self):
        messages = [{"role": "user", "content": "What are the common species?"},
                    {"role": "assistant", "content": "The sources I could find online varied in their answers."}]
        flags = inventory.history_flags(messages)
        finding = next(f for f in flags if f["code"] == "past_search_claim_without_tool_evidence")
        self.assertTrue(finding["repair_review_required"])
        self.assertFalse(finding["tool_execution_evidence_present"])

    def test_occurrences_do_not_become_category_coverage(self):
        summary = inventory.pattern_summary([{"messages": [{"role": "user", "content": "Mention a word"},
            {"role": "assistant", "content": "birds birds fish fish dogs"}]}])
        self.assertNotIn("category_coverage", summary)
        self.assertFalse(summary["pattern_flags_are_semantic_certification"])

    def test_replay_reports_prompt_changes_separately_from_missing_targets(self):
        old = {"source_id": "source", "messages": [{"role": "system", "content": "Old translation instruction"},
               {"role": "user", "content": "Original source"}, {"role": "assistant", "content": "Unchanged answer"}]}
        current = [{"collection": "meaning", "source_id": "source", "messages": [
            {"role": "user", "content": "New translation wrapper"}, {"role": "assistant", "content": "Unchanged answer"}]}]
        result = inventory.compare_replay(old, "meaning", inventory.replay_indexes(current))
        self.assertEqual(result["missing_exact_history_views"], 1)
        self.assertEqual(result["assistant_target_occurrences_missing_for_same_source"], 0)
        self.assertTrue(result["source_present_in_r6"])
        current[0]["messages"][-1]["content"] = "Changed answer"
        result = inventory.compare_replay(old, "meaning", inventory.replay_indexes(current))
        self.assertEqual(result["assistant_target_occurrences_missing_for_same_source"], 1)

    def test_replay_and_freshness_follow_family_not_raw_id_spelling(self):
        graph = inventory.Families()
        graph.connect(["old-leaf", "new-tree"])
        old = {"source_id": "old-leaf", "messages": [{"role": "user", "content": "Question"},
                                                      {"role": "assistant", "content": "Same answer"}]}
        current = [{**old, "source_id": "new-tree", "collection": "responses"}]
        result = inventory.compare_replay(old, "responses", inventory.replay_indexes(current, graph), graph)
        self.assertTrue(result["source_present_in_r6"])
        self.assertFalse(result["raw_source_id_present_in_r6"])
        self.assertEqual(result["assistant_target_occurrences_missing_for_same_source"], 0)
        rows = conversation()
        graph = families(rows)
        graph.connect(["old-leaf", "root"])
        selected, _ = inventory.oasst_candidates(rows, [], graph, {}, {graph.root("old-leaf")}, 1000)
        self.assertTrue(all(r["already_in_r6_source_family"] for r in selected))

    def test_raw_assistant_counts_do_not_repeat_ancestor_targets(self):
        result = inventory.raw_assistant_patterns(conversation())
        self.assertEqual(result["assistant_occurrences"], 2)
        self.assertEqual(result["distinct_assistant_targets_exact"], 2)
        self.assertEqual(result["english_followup_user_nodes"], 1)
        self.assertEqual(result["english_followup_correction_cue_nodes"], 1)


class SourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_inputs_detect_mutation_duplicate_json_and_symlinks(self):
        path = self.root / "source.json"
        write(path, {"a": 1})
        inputs = inventory.Inputs()
        inputs.read(path)
        write(path, {"a": 2})
        with self.assertRaisesRegex(ValueError, "changed"):
            inputs.recheck()
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "linked"):
            inventory.Inputs().bind(link)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inventory.strict_json('{"a":1,"a":2}')

    def test_pack_hashes_and_origin_missingness_measured(self):
        record = {"id": "sample", "source_id": "source", "messages": [{"role": "user", "content": "Question"},
                  {"role": "assistant", "content": "Answer"}]}
        pack_fixture(self.root / "pack", record, held=True)
        result = inventory.load_pack(self.root / "pack", inventory.Inputs())
        stats = result["collections"]["responses"]
        self.assertEqual(stats["record_origins"], {"missing": 1})
        self.assertEqual(stats["witness_origins"], {"fixture": 1})
        self.assertEqual(stats["scheduled_exposures"], 2)
        self.assertEqual(stats["scheduled_effective_target_tokens"], 2)
        (self.root / "pack/source-witnesses.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            inventory.load_pack(self.root / "pack", inventory.Inputs())

    def test_health_suffix_collision_is_not_a_fabricated_alignment(self):
        rows = [{"id": "ID_TR_Eng_Gha_ABC", "question": "Question one", "answer": "Answer one", "language_country": "Eng_Gha",
                 "source_split": "train", "source_ref": {}},
                {"id": "ID_VL_Aka_Gha_ABC", "question": "Different question", "answer": "Different answer", "language_country": "Aka_Gha",
                 "source_split": "validation", "source_ref": {}}]
        index = inventory.HoldoutIndex([("held", "Unrelated held-out text")])
        selected, ledger, grouping = inventory.health_candidates(rows, index, 1000)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["status"], "pending")
        self.assertIn("cross_language_family_alignment_unverified", selected[0]["reasons"])
        self.assertEqual(grouping["suffix_cross_split_collisions"]["train:validation"], 1)
        self.assertEqual(grouping["cross_language_pairs_verified"], 0)
        rows[1]["question"] = "Question one"
        selected, ledger, _ = inventory.health_candidates(rows, index, 1000)
        self.assertFalse(selected)
        self.assertIn("normalized_question_family_crosses_holdout", ledger[0]["reasons"])

    def test_original_parquet_projection_never_loads_user_id(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = node("root")
        raw = {k: row[k] for k in inventory.OASST_COLUMNS}
        raw["user_id"] = "PRIVATE-CONTRIBUTOR-IDENTIFIER"
        path = self.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist([raw]), path)
        result = inventory.Inputs().parquet(path, inventory.OASST_COLUMNS, "train")
        encoded = json.dumps(result)
        self.assertNotIn("user_id", encoded)
        self.assertNotIn("PRIVATE-CONTRIBUTOR-IDENTIFIER", encoded)
        self.assertEqual(result[0]["source_ref"]["row_index"], 0)

    def test_complete_inventory_is_immutable_and_never_admitted(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        train = conversation()
        val = [node("val", tree="val", split="validation", text="A wholly different heldout question"),
               node("val_a", "val", tree="val", role="assistant", split="validation", text="A reserved answer")]
        for name, values in (("train", train), ("validation", val)):
            folder = self.root / name
            folder.mkdir()
            raw = [{k: row[k] for k in inventory.OASST_COLUMNS} | {"user_id": "not-for-export"} for row in values]
            pq.write_table(pa.Table.from_pylist(raw), folder / "part.parquet")
        health = self.root / "health"
        health.mkdir()
        for split, question, answer in (("train", "What is a useful health education practice", "Offer general educational information"),
                                         ("validation", "Reserved validation question", "Reserved health answer"),
                                         ("test_holdout", "Reserved test question", "")):
            pq.write_table(pa.Table.from_pylist([{"id": "ID_" + split, "question": question, "answer": answer,
                                                  "language_country": "Eng_Gha"}]), health / (split + ".parquet"))
        old = {"id": "old", "source_id": "old-source", "messages": [{"role": "user", "content": "Prior distinct request"},
                                                                            {"role": "assistant", "content": "Prior useful answer"}]}
        current = {"id": "current", "source_id": "oasst:root", "tree_id": "root", "split": "train", "origin": "fixture",
                   "messages": [{"role": "user" if n["role"] == "prompter" else "assistant", "content": n["text"]} for n in train]}
        pack_fixture(self.root / "r5", old, held=True)
        pack_fixture(self.root / "r6", current)
        rows_file(self.root / "histories.jsonl", [current])
        sources = {str(p): inventory.sha(p) for p in self.root.rglob("*") if p.is_file()}
        args = (self.root / "r6", self.root / "r5", self.root / "histories.jsonl", self.root / "train",
                self.root / "validation", health, self.root / "out")
        summary = inventory.prepare(*args)
        self.assertEqual(summary["oasst_queue"]["histories"], 2)
        self.assertEqual(summary["health_queue"]["rows"], 1)
        self.assertFalse(summary["training_eligible"])
        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(summary["gpu_runs"], 0)
        self.assertEqual(summary["synthetic_twi_created"], 0)
        for name, checksum in sources.items():
            self.assertEqual(inventory.sha(name), checksum)
        inventory.verify(self.root / "out")
        for file in (self.root / "out").glob("*.jsonl"):
            self.assertNotIn("not-for-export", file.read_text())
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            inventory.prepare(*args)
        (self.root / "out/oasst-candidates.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            inventory.verify(self.root / "out")

    def test_parallel_original_refs_types_and_missing_splits(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import corpus_record_types as types
        corpus, release = self.root / "corpus", self.root / "release"
        ghana, community = corpus / "ghana_nlp_parallel", corpus / "community_parallel"
        ghana.mkdir(parents=True); community.mkdir(); release.mkdir()
        csv_path = ghana / "public.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
            writer.writeheader()
            writer.writerow({"id": "1", "text": "This is an original example sentence.", "label": "Original reference one."})
            writer.writerow({"id": "2", "text": "This was held out previously.", "label": "Original held reference."})
        parquet_path = community / "train.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"eng": "word entry", "twi": "Original lexical entry", "count": 1, "source_files": "02_english-twi-dictionary"},
            {"eng": "Unassigned original source sentence for review.", "twi": "Original reference two.", "count": 1,
             "source_files": "youversion_bible_v1861"}]), parquet_path)
        accounts = [{"source_id": "ghana_nlp_parallel:1", "split": "train"},
                    {"source_id": "ghana_nlp_parallel:2", "split": "validation"},
                    {"source_id": "community_parallel:0", "split": "train"}]
        rows_file(release / "source-accounting.jsonl", accounts)
        rows_file(release / "type-assessments.jsonl", [])
        write(release / "manifest.json", {"record_type_policy": types.POLICY,
             "artifacts": {p.name: inventory.sha(p) for p in release.iterdir() if p.is_file()}})
        write(release / "READY.json", {"manifest_sha256": inventory.sha(release / "manifest.json")})
        before = {str(p): inventory.sha(p) for p in corpus.rglob("*") if p.is_file()}
        values, ref = inventory.parallel_sources(corpus, release, inventory.Inputs())
        self.assertEqual(len(values), 4)
        self.assertEqual(values[0]["english"], "This is an original example sentence.")
        self.assertEqual(values[0]["source_ref"]["csv_record"], 2)
        self.assertEqual(values[2]["type_assessment"]["effective_type"], "dictionary_entry")
        self.assertEqual(values[3]["source_split"], "unassigned")
        groups = inventory.Families()
        for row in values:
            groups.connect(inventory.identity_keys(row))
        candidates, ledger = inventory.parallel_candidates(values, groups, {}, inventory.HoldoutIndex([]), set())
        self.assertEqual({r["routing"] for r in candidates}, {"sentences", "lexicon"})
        self.assertTrue(all(not r["generated_target"] and not r["text_exported"] and not r["training_eligible"] for r in candidates))
        self.assertEqual(ledger[1]["status"], "excluded")
        self.assertEqual(ledger[3]["status"], "pending")
        self.assertIn("no_inherited_split_assignment", ledger[3]["reasons"])
        for path, checksum in before.items():
            self.assertEqual(inventory.sha(path), checksum)

    def test_parallel_long_prose_not_silently_routed_as_sentence(self):
        import corpus_record_types as types
        text = "One original sentence. Another original sentence. A third original sentence. A fourth original sentence."
        assessment = types.classify(text, "Original reference.", "ghana_nlp_parallel", "professional_parallel", "sentence", "compiled_parallel_text")
        row = {"source_id": "ghana_nlp_parallel:1", "family": "ghana_nlp_parallel", "genre": "professional_parallel",
               "english": text, "twi": "Original reference.", "type_assessment": assessment,
               "inherited_assessment": assessment, "inherited_account": {}, "source_split": "train", "lineage": [],
               "source_ref": {"file_sha256": "a" * 64}, "document_like_structure": True}
        candidates, ledger = inventory.parallel_candidates([row], inventory.Families(), {}, inventory.HoldoutIndex([]), set())
        self.assertFalse(candidates)
        self.assertEqual(ledger[0]["routing"], "documents_review")
        self.assertIn("document_like_structure_requires_separate_review", ledger[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
