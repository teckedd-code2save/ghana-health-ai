"""CPU-only fixtures for the R7 target auditor; not translation or training data."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import audit_r7_response_targets as audit


def fixture(target="The blue chair stays here. The blue chair stays here.",
            english="The blue chair stays here."):
    messages = [{"role": "user", "content": "Offline structural test."}, {"role": "assistant", "content": target}]
    source = [{"role": "user", "content": "Offline structural test."}, {"role": "assistant", "content": english}]
    record = {"id": "fixture", "source_id": "oasst:fixture-tree", "tree_id": "fixture-tree", "group_id": "fixture-group",
              "messages": messages, "english_messages": source, "final_history_sha256": audit.digest(messages),
              "training_eligible": True, "origin": "synthetic_unit_test_only"}
    flags = [{**flag, "id": record["id"], "source_id": record["source_id"],
              "source_ref": {"record_sha256": audit.digest(record), "line": 1}}
             for flag in audit.inventory.history_flags(messages)]
    joined = {record["tree_id"]: {"messages": source, "original_ref": {"fixture_only": True}}}
    return record, flags, joined


def approval(record):
    target, english = (record[key][1]["content"] for key in ("messages", "english_messages"))
    pair = audit.adjacent_duplicates(target)[0]
    proof = {"review_method": "agent_source_alignment", "unchanged_meaning": True,
             "target_sha256": audit.digest(target), "english_sha256": audit.digest(english),
             "kept_span": pair["kept_span"], "duplicate_span": pair["duplicate_span"],
             "english_span": [0, len(english)], "reason": "Synthetic identical-language structural fixture; source unit occurs once."}
    review = {"id": record["id"], "record_sha256": audit.digest(record), "turn": 1,
              "target_sha256": audit.digest(target), "english_sha256": audit.digest(english),
              "review_id": "fixture-review", "decision": "approve_exact_adjacent_deletion"}
    proposal = {"id": record["id"], "record_sha256": audit.digest(record), "turn": 1,
                "review_id": review["review_id"], "proof": proof}
    return review, proposal


class ResponseAuditTests(unittest.TestCase):
    def test_whole_input_accounted_without_editing_or_admission(self):
        record, flags, joined = fixture()
        before = copy.deepcopy((record, flags, joined))
        result = audit.audit_records([record], flags, joined)
        self.assertEqual((record, flags, joined), before)
        self.assertEqual(result["records"][0]["original_record"], record)
        self.assertFalse(result["records"][0]["training_eligible"])
        self.assertTrue(result["records"][0]["original_record"]["training_eligible"])
        self.assertEqual(result["candidate_corrections"], [])
        report = audit.summary(result)
        self.assertEqual(report["records_accounted"], 1)
        self.assertEqual(report["assistant_turns_inspected"], 1)
        self.assertEqual(report["inventory_flags_preserved"], len(flags))
        self.assertEqual(report["agent_reviewed_records"], 0)

    def test_no_pattern_is_not_semantic_clearance(self):
        record, flags, joined = fixture("A single unique response remains unchanged.")
        result = audit.audit_records([record], flags, joined)
        self.assertEqual(result["records"][0]["automatic_status"], "no_selected_pattern_not_semantic_clearance")
        self.assertFalse(result["records"][0]["native_certified"])

    def test_c_family_normalization_collision_preserved(self):
        record, flags, joined = fixture("1. C\n2. C++\n3. C#", "1. C\n2. C++\n3. C#")
        result = audit.audit_records([record], flags, joined)
        finding = next(f for f in result["records"][0]["findings"]
                       if f["inventory_flag"]["code"] == "repeated_list_items")
        self.assertEqual(finding["classification"], "normalization_collision_review")
        self.assertEqual(finding["target_matches"][0]["distinct_exact_forms"], 3)
        self.assertEqual(result["candidate_corrections"], [])

    def test_fenced_unfenced_inline_and_unclosed_code_protected(self):
        for text in ["```python\nvalue = function(a, b, c, d)\nvalue = function(a, b, c, d)\n```",
                     "function f(a, b, c, d) {\nreturn f(a, b, c, d);\n}",
                     "Use `cache.put(request, response)` here.", "```python\na = 1"]:
            with self.subTest(text=text):
                self.assertTrue(audit.protected_spans(text))
        text = "```python\nfirst second third fourth fifth sixth seventh eighth\n" * 1
        text += "first second third fourth fifth sixth seventh eighth\n```"
        record, flags, joined = fixture(text, text)
        result = audit.audit_records([record], flags, joined)
        self.assertTrue(any(f["classification"] == "protected_code_repetition"
                            for f in result["records"][0]["findings"]))

    def test_eight_gram_offsets_reproduce_exact_original_text(self):
        text = "one two three four five six seven eight. " * 3
        evidence = audit.pattern_evidence(text)
        group = next(g for g in evidence["repeated_8gram_at_least_three_times"]
                     if g["normalized_unit"] == "one two three four five six seven eight")
        self.assertEqual(len(group["occurrences"]), 3)
        for match in group["occurrences"]:
            self.assertEqual(match["text"], text[match["start"]:match["end"]])

    def test_cross_label_reuse_is_not_category_error(self):
        text = "- Hardware: keyboard, screen\n- Input devices: keyboard, mouse"
        record, flags, joined = fixture(text, text)
        result = audit.audit_records([record], flags, joined)
        finding = next(f for f in result["records"][0]["findings"]
                       if f["inventory_flag"]["code"] == "same_literal_example_across_category_labels")
        self.assertEqual(finding["classification"], "cross_label_shared_literal_review")
        self.assertFalse(finding["semantic_equivalence_certified"])
        self.assertEqual({s["label"] for s in finding["target_matches"][0]["occurrences"]}, {"Hardware", "Input devices"})

    def test_lyrics_quotation_and_template_not_silently_removed(self):
        text = "Chorus:\nWe return to the same place.\nWe return to the same place."
        record, flags, joined = fixture(text, text)
        result = audit.audit_records([record], flags, joined)
        self.assertEqual(result["records"][0]["findings"][0]["classification"], "creative_repetition_review")
        self.assertEqual(result["candidate_corrections"], [])
        for text in ['"This is a repeated quotation." "This is a repeated quotation."',
                     "[Project Name], [Project Description], [Technologies Used]\n" * 2]:
            self.assertTrue(audit.protected_spans(text))

    def test_nonadjacent_duplicates_are_review_only(self):
        text = "The blue chair stays here. Another thing is described here. The blue chair stays here."
        record, flags, joined = fixture(text)
        result = audit.audit_records([record], flags, joined)
        self.assertTrue(result["review_queue"])
        self.assertEqual(audit.adjacent_duplicates(text), [])
        self.assertEqual(result["candidate_corrections"], [])

    def test_exact_adjacent_deletion_requires_bound_alignment(self):
        record, _, _ = fixture()
        _, proposal = approval(record)
        target, english = record["messages"][1]["content"], record["english_messages"][1]["content"]
        edit = audit.safe_adjacent_deletion(target, english, proposal["proof"])
        self.assertEqual(target[:edit["start"]] + target[edit["end"]:], english)
        self.assertEqual(edit["replacement"], "")
        for changed in [{}, {**proposal["proof"], "unchanged_meaning": False},
                        {**proposal["proof"], "target_sha256": "0" * 64},
                        {**proposal["proof"], "duplicate_span": [0, 3]}]:
            with self.assertRaises(ValueError):
                audit.safe_adjacent_deletion(target, english, changed)

    def test_code_quote_and_repeated_source_block_even_approved_deletion(self):
        for target, english in [
            ("```\nThe blue chair stays here. The blue chair stays here.\n```", "The blue chair stays here."),
            ('"The blue chair stays here. The blue chair stays here."', "The blue chair stays here."),
            ("The blue chair stays here. The blue chair stays here.", "The blue chair stays here. The blue chair stays here."),
        ]:
            record, _, _ = fixture(target, english)
            base, _, _ = fixture()
            _, proposal = approval(base)
            proposal["proof"].update(target_sha256=audit.digest(target), english_sha256=audit.digest(english))
            if target == english:
                proposal["proof"]["english_span"] = [0, len("The blue chair stays here.")]
            with self.assertRaises(ValueError):
                audit.safe_adjacent_deletion(target, english, proposal["proof"])

    def test_adjacent_bullet_removal_does_not_merge_separate_sections(self):
        target = "- The blue chair stays here\n- The blue chair stays here"
        english = "- The blue chair stays here"
        pair = next(p for p in audit.adjacent_duplicates(target) if p["kind"] == "list_item")
        proof = {"review_method": "agent_source_alignment", "unchanged_meaning": True,
                 "kind": "list_item",
                 "target_sha256": audit.digest(target), "english_sha256": audit.digest(english),
                 "kept_span": pair["kept_span"], "duplicate_span": pair["duplicate_span"],
                 "english_span": [0, len(english)], "reason": "Synthetic identical-language list fixture."}
        edit = audit.safe_adjacent_deletion(target, english, proof)
        self.assertEqual(target[:edit["start"]] + target[edit["end"]:], english)
        separate = "Section one:\n- The blue chair stays here\nSection two:\n- The blue chair stays here"
        self.assertEqual(audit.adjacent_duplicates(separate), [])

    def test_changed_records_registered_hash_bound_and_pending(self):
        record, flags, joined = fixture()
        review, proposal = approval(record)
        with self.assertRaisesRegex(ValueError, "registered explicit approval"):
            audit.audit_records([record], flags, joined, deletion_proofs=[proposal])
        result = audit.audit_records([record], flags, joined, agent_reviews=[review], deletion_proofs=[proposal])
        candidate = result["candidate_corrections"][0]
        changed = candidate["candidate_record"]
        self.assertEqual(changed["messages"][0], record["messages"][0])
        self.assertEqual(changed["english_messages"], record["english_messages"])
        self.assertEqual(changed["source_id"], record["source_id"])
        self.assertFalse(changed["training_eligible"])
        self.assertEqual(changed["final_history_sha256"], audit.digest(changed["messages"]))
        self.assertEqual(candidate["candidate_record_sha256"], audit.digest(changed))
        self.assertEqual(result["correction_registry"][0]["candidate_sha256"], audit.digest(candidate))
        self.assertEqual(record["messages"][1]["content"].count("The blue chair"), 2)

    def test_two_claim_views_one_source_turn(self):
        text = "The sources I could find online disagreed."
        record, flags, joined = fixture(text, text)
        flag = next(f for f in flags if f["code"] == "past_search_claim_without_tool_evidence")
        flags.append({**flag, "view": "preserved_english_source"})
        result = audit.audit_records([record], flags, joined)
        summary = audit.summary(result)
        self.assertEqual(summary["past_search_flag_views"], 2)
        self.assertEqual(summary["past_search_unique_turns"], 1)
        self.assertEqual(summary["past_search_unique_records"], 1)
        self.assertEqual(result["candidate_corrections"], [])

    def test_additional_unsupported_capability_is_not_a_new_search_receipt(self):
        text = "My programming allows me to access any information on the internet."
        record, flags, joined = fixture(text, text)
        result = audit.audit_records([record], flags, joined)
        self.assertEqual(result["review_queue"][0]["finding"]["classification"], "unsupported_internet_capability_review")
        self.assertEqual(audit.summary(result)["past_search_flag_views"], 0)

    def test_agent_and_automatic_review_counts_remain_distinct(self):
        record, flags, joined = fixture()
        review, _ = approval(record)
        result = audit.audit_records([record], flags, joined, agent_reviews=[review])
        report = audit.summary(result)
        self.assertEqual(report["agent_reviewed_turns"], 1)
        self.assertEqual(report["agent_reviewed_records"], 1)
        self.assertEqual(report["automatically_semantically_certified_records"], 0)
        self.assertEqual(report["candidate_corrections"], 0)
        self.assertTrue(result["review_queue"][0]["finding"]["agent_reviewed"])

    def test_mismatched_flag_history_and_review_fail_closed(self):
        record, flags, joined = fixture()
        altered = copy.deepcopy(record)
        altered["messages"][1]["content"] += " Changed."
        with self.assertRaisesRegex(ValueError, "Flag source hash"):
            audit.audit_records([altered], flags, joined)
        with self.assertRaisesRegex(ValueError, "Original final-history"):
            audit.audit_records([altered], [], joined)
        review, _ = approval(record)
        review["english_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Agent reviewed turn"):
            audit.audit_records([record], flags, joined, agent_reviews=[review])
        bad = copy.deepcopy(flags)
        next(f for f in bad if f["code"] == "repeated_sentence_or_line")["duplicate_occurrences"] = 5
        with self.assertRaisesRegex(ValueError, "Sentence flag count"):
            audit.audit_records([record], bad, joined)

    def test_raw_original_english_join_checks_ids_tree_role_text_and_file(self):
        record, _, _ = fixture()
        queue = [{"tree_id": record["tree_id"], "messages": record["english_messages"],
                  "message_ids": ["u", "a"], "source_file": "source.parquet", "source_file_sha256": "pin"}]
        raw = {identity: {"message_id": identity, "message_tree_id": record["tree_id"], "role": role,
                          "lang": "en", "text": message["content"]}
               for identity, role, message in zip(["u", "a"], ["prompter", "assistant"], record["english_messages"])}
        ref = {"file": "/fixture/source.parquet", "sha256": "pin"}
        joined = audit.join_english([record], queue, raw, {"fixture": True}, ref)
        self.assertEqual(joined[record["tree_id"]]["messages"], record["english_messages"])
        for key, value in [("message_tree_id", "other"), ("text", "changed"), ("role", "prompter"), ("lang", "tw")]:
            changed = copy.deepcopy(raw)
            changed["a"][key] = value
            with self.assertRaisesRegex(ValueError, "Original English message"):
                audit.join_english([record], queue, changed, {}, ref)
        with self.assertRaisesRegex(ValueError, "file reference"):
            audit.join_english([record], queue, raw, {}, {**ref, "sha256": "wrong"})

    def test_candidate_only_artifacts_and_no_network(self):
        record, flags, joined = fixture()
        inputs = {"fixture_only": True}
        with patch.object(socket, "socket", side_effect=AssertionError("No network")):
            files = audit.build_package([record], flags, joined, inputs, agent_reviews=[])
        manifest = json.loads(files["manifest.json"])
        self.assertFalse(manifest["automatic_edits_enabled"])
        self.assertFalse(manifest["training_ready"])
        self.assertEqual(files["candidate-corrections.jsonl"], b"")
        self.assertEqual(files["correction-registry.jsonl"], b"")
        self.assertEqual(set(files), set(manifest["artifacts"]) | {"manifest.json", "SEALED.json"})
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(audit.hashlib.sha256(files[name]).hexdigest(), expected)

    def test_prepare_refuses_overwrite_and_verifier_detects_tampering(self):
        record, flags, joined = fixture()
        loaded = ([record], flags, joined, {"histories": {"sha256": "fixture"}})
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, "load_inputs", return_value=loaded), \
                patch.object(audit, "materialize_agent_reviews", return_value=[]):
            out = Path(directory) / "audit"
            report = audit.prepare(out)
            self.assertTrue(report["verified"])
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                audit.prepare(out)
            (out / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Audit replay differs"):
                audit.verify_package(out)

    def test_actual_agent_notes_cannot_bind_an_unseen_snapshot(self):
        with self.assertRaisesRegex(ValueError, "inspected original snapshot"):
            audit.materialize_agent_reviews([], "changed")


if __name__ == "__main__":
    unittest.main()
