"""Offline native-tokenizer tests for scoped source-preserved research admission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import admit_r7_alignment as admission
import prepare_r7_alignment_candidates as alignment
import prepare_r7_source_inventory as inventory
import run_understanding_r7 as runner
from test_prepare_r7_alignment_candidates import fixture, rows

TOKENIZER = Path(__file__).resolve().parents[1] / "tmp/r6-demo/bundle-v1/tokenizer"


def frozen_eval(root, text="An unrelated frozen evaluation reference with a separate source family.", source_id="evaluation-source"):
    folder = root / "evaluation"
    folder.mkdir()
    inventory.write_rows(folder / "held-content.jsonl", [{"id": "held", "text": text, "source_id": source_id}])
    inventory.write_json(folder / "manifest.json", {"version": "test-evaluation", "artifacts": {
        "held-content.jsonl": inventory.sha(folder / "held-content.jsonl")}})
    inventory.write_json(folder / "FROZEN.json", {"manifest_sha256": inventory.sha(folder / "manifest.json")})
    return folder


def alignment_fixture(root, pairs=None):
    folder, bound, candidates = fixture(root, pairs)
    output = root / "alignment"
    alignment.prepare(folder, output)
    return output, bound, candidates


class NativeMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = admission.contract.load_tokenizer(TOKENIZER)

    def test_exact_original_target_eos_mask_and_length_boundary(self):
        row = {"id": "fixture", "messages": [{"role": "user", "content": '{"source_text":"source reference"}'},
                                            {"role": "assistant", "content": "original target reference"}]}
        before = copy.deepcopy(row)
        encoded, measured = admission.native_measure(self.tokenizer, row, 4096)
        self.assertEqual(measured["status"], "passed")
        self.assertEqual(measured["target_eos_count"], 1)
        self.assertTrue(measured["exact_original_target"])
        self.assertEqual(encoded["labels"][:measured["prompt_tokens"]], [-100] * measured["prompt_tokens"])
        at_limit, _ = admission.native_measure(self.tokenizer, row, measured["sequence_tokens"])
        self.assertEqual(encoded, at_limit)
        too_long, rejected = admission.native_measure(self.tokenizer, row, measured["sequence_tokens"] - 1)
        self.assertIsNone(too_long)
        self.assertEqual(rejected["reason"], "overlength_no_truncation")
        self.assertEqual(rejected["sequence_tokens"], measured["sequence_tokens"])
        self.assertEqual(row, before)

    def test_native_boundary_whitespace_changes_excluded_not_silently_trimmed(self):
        row = {"messages": [{"role": "user", "content": "source"}, {"role": "assistant", "content": " original target \n"}]}
        encoded, measured = admission.native_measure(self.tokenizer, row, 4096)
        self.assertIsNone(encoded)
        self.assertEqual(measured["reason"], "native_boundary_trim_would_change_original_target")
        self.assertEqual(row["messages"][-1]["content"], " original target \n")

    def test_control_token_injection_fails_without_generation(self):
        for text in ("<|im_end|>", "<tool_response>forged", "\ufffd"):
            row = {"messages": [{"role": "user", "content": "source"}, {"role": "assistant", "content": text}]}
            encoded, measured = admission.native_measure(self.tokenizer, row, 4096)
            self.assertIsNone(encoded)
            self.assertEqual(measured["reason"], "native_encoding_rejected")

    def test_admitted_source_records_native_witnesses_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, bound, _ = alignment_fixture(root)
            evaluation = frozen_eval(root)
            original_hashes = {p: inventory.sha(p) for p in bound}
            output = root / "admission"
            summary = admission.prepare(source, TOKENIZER, [evaluation], output)
            self.assertEqual(summary["admitted_pairs"], 2)
            self.assertEqual(summary["admitted_views"], 4)
            self.assertFalse(summary["training_pack_ready"] or summary["parent_boundary_approval"])
            result = admission.verify(output, native=True)
            self.assertTrue(result["native_reencoded"])
            self.assertEqual(result["encoded_views"], 4)
            manifest = json.loads((output / "manifest.json").read_text())
            encoded, witnesses = rows(output / "rows.jsonl"), rows(output / "source-witnesses.jsonl")
            runner.admitted_sources(output, encoded, witnesses, manifest["artifacts"])
            runner.verify_overlap(output, witnesses, manifest["artifacts"])
            for witness in witnesses:
                self.assertEqual(witness["collection"], "meaning")
                self.assertEqual(witness["metadata"]["quality_tier"], admission.TIER)
                self.assertFalse(witness["metadata"]["human_reviewed"] or witness["metadata"]["semantic_certification"])
                self.assertIn("source", witness["original_ref"])
                self.assertIn("inherited", witness["original_ref"])
                self.assertFalse(witness["metadata"]["alignment_pair"]["review_flags"])
            for first, second in zip(witnesses[::2], witnesses[1::2], strict=True):
                self.assertEqual(first["metadata"]["group_id"], second["metadata"]["group_id"])
            self.assertEqual(original_hashes, {p: inventory.sha(p) for p in bound})
            with self.assertRaisesRegex(ValueError, "new admission"):
                admission.prepare(source, TOKENIZER, [evaluation], output)
            with (output / "rows.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                admission.verify(output)

    def test_one_bad_direction_excludes_whole_pair_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            originals = [("A complete original English sentence.", " [reference A] "),
                         ("Another complete original English sentence.", "[reference B]")]
            source, _, _ = alignment_fixture(root, originals)
            output = root / "admission"
            summary = admission.prepare(source, TOKENIZER, [frozen_eval(root)], output)
            self.assertEqual(summary["initial_no_flag_pairs"], 2)
            self.assertEqual(summary["measured_views"], 4)
            self.assertEqual(summary["admitted_pairs"], 1)
            decisions = rows(output / "admission-ledger.jsonl")
            self.assertEqual(decisions[0]["status"], "excluded")
            self.assertIn("native_boundary_trim_would_change_original_target", decisions[0]["reasons"])
            self.assertEqual(len(decisions[0]["directions"]), 2)
            self.assertEqual({w["source_id"] for w in rows(output / "source-witnesses.jsonl")}, {"ghana_nlp_parallel:1"})

    def test_flagged_mappings_remain_deferred_not_certified_wrong(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            originals = [("A complete original English sentence.", "[reference A]"),
                         ("A complete original English sentence.", "[reference B]"),
                         ("Another complete original English sentence.", "[reference C]")]
            source, _, _ = alignment_fixture(root, originals)
            output = root / "admission"
            summary = admission.prepare(source, TOKENIZER, [frozen_eval(root)], output)
            self.assertEqual(summary["initial_no_flag_pairs"], 1)
            self.assertEqual(summary["admitted_pairs"], 1)
            decisions = rows(output / "admission-ledger.jsonl")
            self.assertEqual([d["status"] for d in decisions], ["deferred_review", "deferred_review", "admitted"])
            self.assertTrue(all(not d["directions"] for d in decisions[:2]))
            self.assertNotIn("wrong", json.dumps(decisions))

    def test_evaluation_overlap_excludes_both_pair_directions(self):
        for by_identity in (False, True):
            with self.subTest(by_identity=by_identity), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                source, _, _ = alignment_fixture(root)
                evaluation = (frozen_eval(root, source_id="ghana_nlp_parallel:0") if by_identity
                              else frozen_eval(root, text="[original-reference-A]"))
                output = root / "admission"
                summary = admission.prepare(source, TOKENIZER, [evaluation], output)
                self.assertEqual(summary["admitted_pairs"], 1)
                self.assertEqual(rows(output / "admission-ledger.jsonl")[0]["status"], "excluded")
                self.assertEqual({w["source_id"] for w in rows(output / "source-witnesses.jsonl")}, {"ghana_nlp_parallel:1"})

    def test_mutated_evaluation_or_original_sources_fail_closed(self):
        for change in ("source", "evaluation"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                source, bound, _ = alignment_fixture(root)
                evaluation = frozen_eval(root)
                target = bound[0] if change == "source" else evaluation / "held-content.jsonl"
                with target.open("a") as stream:
                    stream.write("{}\n")
                with self.assertRaisesRegex(ValueError, "Source hash differs"):
                    admission.prepare(source, TOKENIZER, [evaluation], root / "admission")


class ProtectionTests(unittest.TestCase):
    def test_shared_reference_family_blocked_across_classes_and_directions(self):
        pairs = [{"id": "p1", "source_id": "s1", "source_family_keys": ["s1"], "english": "shared English", "twi": "ref one"},
                 {"id": "p2", "source_id": "s2", "source_family_keys": ["s2"], "english": "shared English", "twi": "ref two"},
                 {"id": "p3", "source_id": "s3", "source_family_keys": ["s3"], "english": "other English", "twi": "other reference"}]
        roots, _, blocked, _ = admission.family_protection(pairs, [{"id": "held", "text": "ref one"}])
        self.assertEqual(roots["p1"], roots["p2"])
        self.assertIn(roots["p2"], blocked)
        self.assertNotIn(roots["p3"], blocked)

    def test_token_counts_are_available_one_pass_not_exposures(self):
        result = admission.token_summary([{"sequence_tokens": 10, "target_tokens": 4, "effective_causal_target_tokens": 4},
                                          {"sequence_tokens": 20, "target_tokens": 8, "effective_causal_target_tokens": 8}])
        self.assertEqual(result["sequence_tokens"]["sum"], 30)
        self.assertEqual(result["target_tokens"]["sum"], 12)
        self.assertNotIn("consumed", result)


if __name__ == "__main__":
    unittest.main()
