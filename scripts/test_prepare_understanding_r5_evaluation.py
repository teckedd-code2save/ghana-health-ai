"""Synthetic schema/receipt tests only; never load the private fresh payload."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_understanding_r5_evaluation as adapter
import run_understanding_r5_evaluation as runner
import test_r5_fresh_evaluation as design
from test_run_understanding_r5_evaluation import prepared_fixture, Runtime, CORE, prompt_ids
from types import SimpleNamespace


def corpus_fixture():
    corpus = design.fixture()
    corpus.update(version=adapter.SOURCE_VERSION, status="private_proposal_not_sealed_final_test",
                  comparison={"variants": ["base", "r4", "r5"],
                              "decoding_proposal": {k: v for k, v in adapter.DECODING.items() if k != "enable_thinking"}})
    group = corpus["groups"][0]
    group["informal3"] = True
    for language in ("en", "tw"):
        group["inputs"][language]["messages"] = [
            {"role": "user", "content": language + " fixed \u025b"},
            {"role": "assistant", "content": "Authored \u025b context"},
            {"role": "user", "content": language + " final \u025b"}]
    group["reviews"][0]["facts"]["f"] = "REVIEW_ONLY_SENTINEL"
    rollout = copy.deepcopy(group)
    rollout.update(id="rollout", source_group_id="synthetic:rollout", mode="own_history_rollout", informal3=False)
    rollout["inputs"] = {lang: {"user_turns": [lang + " rollout " + str(i) for i in range(3)],
                                "output_languages": [lang, "mixed", lang]} for lang in ("en", "tw")}
    rollout["reviews"] = [{**copy.deepcopy(group["reviews"][0]), "slot": i} for i in (1, 2, 3)]
    corpus["groups"].append(rollout)
    return corpus


def sealed_fixture(root, corpus=None):
    root.mkdir(parents=True)
    source = root / "r5_fresh_evaluation.json"
    corpus = corpus or corpus_fixture()
    adapter.save(source, corpus)
    audit = {**design.summary(corpus, design.validate(corpus)), "corpus_file_sha256": adapter.sha(source),
             "validator_file_sha256": adapter.sha(Path(design.__file__)),
             "prior_inventory": {"complete_manifest_coverage": False, "missing_manifest_artifacts": ["synthetic-missing"]}}
    adapter.save(root / "audit.json", audit)
    seal = {"version": "r5-private-heldout-seal-v1", "private": True, "training_eligible": False,
            "execution_authorized": False, "native_certification": False, "clinical_certification": False,
            "model_calls": 0,
            "artifacts": {"r5_fresh_evaluation.json": adapter.sha(source), "audit.json": adapter.sha(root / "audit.json")},
            "validator": {"sha256": adapter.sha(Path(design.__file__))}, "inventory_sha256": audit["inventory_sha256"]}
    adapter.save(root / "HELDOUT_SEAL.json", seal)
    return source, root / "HELDOUT_SEAL.json"


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.seal = sealed_fixture(self.root / "design")

    def build(self):
        return adapter.prepare(self.source, adapter.sha(self.source), self.root / "suite",
                               seal=self.seal, seal_sha=adapter.sha(self.seal))

    def reseal(self):
        folder = self.root / "suite"
        manifest = adapter.read(folder / "manifest.json")
        manifest["artifacts"] = {name: adapter.sha(folder / name) for name in adapter.ARTIFACTS}
        adapter.save(folder / "manifest.json", manifest)
        checksum = adapter.sha(folder / "manifest.json")
        adapter.save(folder / "FROZEN.json", {"manifest_sha256": checksum})
        return checksum

    def test_lossless_schema_variants_reviews_and_unmodified_private_source(self):
        before = {p.name: p.read_bytes() for p in self.source.parent.iterdir()}
        result = self.build()
        manifest, cases, rollouts, held = adapter.load(self.root / "suite", result["evaluation_sha256"])
        self.assertEqual(manifest["variants"], ["base", "r4_response", "r5_response"])
        self.assertEqual(manifest["counts"], {"fixed_prompts": 3, "rollout_scenarios": 2,
                                             "rollout_user_turns": 6, "source_groups": 2, "language_rows": 0})
        self.assertEqual(held, [])
        parent = next(c for c in cases if c["id"] == "fixture-tw")
        child = next(c for c in cases if c["id"] == "fixture-tw-i3")
        self.assertEqual(child["messages"][1], parent["messages"][1])
        self.assertIn("3", child["messages"][0]["content"])
        self.assertEqual(child["source_group_id"], parent["source_group_id"])
        self.assertEqual(child["review_slots"], parent["review_slots"])
        self.assertEqual(rollouts[0]["output_languages"][1], "mixed")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.parent.iterdir()})
        self.assertFalse(manifest["execution_authorized"])
        self.assertEqual(manifest["kind"], adapter.KIND)
        self.assertFalse(adapter.read(self.root / "suite/design-audit.json")["prior_inventory"]["complete_manifest_coverage"])
        self.assertEqual((self.root / "suite").stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in (self.root / "suite").iterdir()))
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_changed_bytes_seal_validator_and_rewritten_mapping_fail_closed(self):
        self.build()
        cases = adapter.read(self.root / "suite/cases.json")
        cases[0]["messages"][0]["content"] = "Rewritten stimulus"
        adapter.save(self.root / "suite/cases.json", cases)
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            adapter.load(self.root / "suite", self.reseal())
        (self.root / "design/audit.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            adapter.prepare(self.source, adapter.sha(self.source), self.root / "other",
                            seal=self.seal, seal_sha=adapter.sha(self.seal))
        self.assertFalse((self.root / "other").exists())

    def test_wrong_source_hash_and_proposal_protocol_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            adapter.prepare(self.source, "0" * 64, self.root / "suite", seal=self.seal, seal_sha=adapter.sha(self.seal))
        original = corpus_fixture()
        for mutate in (lambda c: c.update(execution_authorized=True), lambda c: c.update(native_certification=True),
                       lambda c: c["comparison"]["decoding_proposal"].update(seed=42),
                       lambda c: c["comparison"].update(variants=["base", "r3", "r5"]),
                       lambda c: c["groups"][0]["reviews"].clear(),
                       lambda c: c["groups"][0]["inputs"]["tw"].update(output_languages=["unsupported"])):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                adapter.adapt(changed)

    def test_extra_artifacts_and_linked_payload_rejected(self):
        result = self.build()
        extra = self.root / "suite/feedback.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            adapter.load(self.root / "suite", result["evaluation_sha256"])
        extra.unlink()
        extra.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Linked"):
            adapter.load(self.root / "suite", result["evaluation_sha256"])

    def test_adapter_cannot_write_inside_sealed_design(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            adapter.prepare(self.source, adapter.sha(self.source), self.source.parent / "derived",
                            seal=self.seal, seal_sha=adapter.sha(self.seal))
        self.assertFalse((self.source.parent / "derived").exists())


class FreshExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source, seal = sealed_fixture(self.root / "design")
        self.suite = self.root / "suite"
        self.suite_sha = adapter.prepare(source, adapter.sha(source), self.suite,
                                         seal=seal, seal_sha=adapter.sha(seal))["evaluation_sha256"]
        self.bundle, self.bundle_sha, self.inputs, self.consumption = prepared_fixture(self.root)
        evidence = self.root / "audit/evidence/full-scope.json"
        runner.save(evidence, {"synthetic": True, "checks": "Synthetic full-scope clearance, not real admission"})
        self.audit = self.root / "audit/audit.json"
        runner.save(self.audit, {"version": "r5-evaluation-source-audit-v1",
            "training_manifest_sha256": runner.sha(self.inputs / "manifest.json"),
            "evaluation_manifest_sha256": self.suite_sha, "evaluation_source_sha256": runner.sha(source),
            "overlap_free": True, "checks": {"source_groups_disjoint": True, "exact_text_disjoint": True,
                                              "normalized_text_disjoint": True, "full_protection_coverage": True},
            "evidence": {"evidence/full-scope.json": runner.sha(evidence)}})

    def build(self, name="execution", **kwargs):
        context = {"ready": True, "evaluation_sha256": self.suite_sha, "context_tokens": 4096,
                   "reserved_generation_tokens": 256, "truncate": False}
        with patch.object(runner, "context_preflight", return_value=context):
            return runner.prepare(self.bundle, self.bundle_sha, self.suite, self.inputs, self.consumption,
                                  self.root / name, evaluation_sha=self.suite_sha, source_audit_path=self.audit, **kwargs)

    def test_fresh_dynamic_counts_scope_audits_and_both_policy_packages(self):
        original = runner.files(self.suite)
        raw = self.build()
        guarded = self.build("guarded", policy="guarded")
        self.assertNotEqual(raw["execution_sha256"], guarded["execution_sha256"])
        for name, result in (("execution", raw), ("guarded", guarded)):
            contract, _, _, cases, rollouts, held = runner.verify(self.root / name, result["execution_sha256"])
            self.assertEqual(contract["expected_slots"], 27)
            self.assertEqual(contract["kind"], adapter.KIND)
            self.assertEqual(contract["variants"], list(runner.VARIANTS))
            self.assertEqual(held, [])
            self.assertEqual(len(list(runner.slots(cases, rollouts, held))), 27)
            proof = runner.read(self.root / name / "provenance/summary.json")
            self.assertEqual(proof["source_audit"]["sha256"], runner.sha(self.audit))
            self.assertEqual(proof["overlap_audit_sha256"], runner.sha(self.inputs / "overlap-audit.json"))
            self.assertEqual(runner.files(self.root / name / "evaluation"), original)
        self.assertEqual(runner.files(self.suite), original)
        with self.assertRaisesRegex(ValueError, "language-loss"):
            self.build("unbound-loss", loss=True)

    def test_fresh_requires_complete_training_specific_audit_and_evidence(self):
        with self.assertRaisesRegex(ValueError, "source audit"):
            runner.prepare(self.bundle, self.bundle_sha, self.suite, self.inputs, self.consumption,
                           self.root / "missing-audit", evaluation_sha=self.suite_sha)
        original = runner.read(self.audit)
        for mutate in (lambda a: a.update(training_manifest_sha256="0" * 64),
                       lambda a: a.update(evaluation_manifest_sha256="0" * 64),
                       lambda a: a.update(evaluation_source_sha256="0" * 64),
                       lambda a: a["checks"].update(full_protection_coverage=False),
                       lambda a: a["checks"].update(source_groups_disjoint=1),
                       lambda a: a.update(evidence={})):
            audit = copy.deepcopy(original)
            mutate(audit)
            runner.save(self.audit, audit)
            with self.assertRaises(ValueError):
                self.build()
        runner.save(self.audit, original)
        (self.audit.parent / "evidence/full-scope.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.build()

    def test_execution_cannot_write_into_frozen_inputs_or_inference(self):
        for root in (self.bundle, self.inputs, self.suite):
            with self.assertRaisesRegex(ValueError, "outside immutable"):
                runner.prepare(self.bundle, self.bundle_sha, self.suite, self.inputs, self.consumption,
                               root / "execution", evaluation_sha=self.suite_sha, source_audit_path=self.audit)
            self.assertFalse((root / "execution").exists())

    def test_fresh_engine_never_injects_reviews_language_metadata_or_other_histories(self):
        _, cases, rollouts, held = adapter.load(self.suite, self.suite_sha)
        plan = list(runner.slots(cases, rollouts, held))
        journal = CORE.Journal(self.root / "results/journal", {"kind": adapter.KIND, "evaluation_sha256": self.suite_sha})
        runtime = Runtime()
        core = SimpleNamespace(EOS=CORE.EOS, completion=CORE.completion, prompt_ids=prompt_ids)
        scoring = SimpleNamespace(objective_score=lambda *a: None)
        result = runner.evaluate(runtime, core, scoring, plan, journal, 1, "synthetic-proof", 100, clock=lambda: 0)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(len(runtime.calls), 27)
        for messages, variant in runtime.calls:
            self.assertNotIn("REVIEW_ONLY_SENTINEL", json.dumps(messages))
            self.assertNotIn("output_languages", json.dumps(messages))
        for messages, variant in runtime.calls[9:]:
            self.assertTrue(all(m["content"] == variant + ":reply" for m in messages if m["role"] == "assistant"))
        identities = {v: runtime.identity(v) for v in runner.VARIANTS}
        summary = runner.report(self.root / "results", plan, journal, {"kind": adapter.KIND}, identities)
        self.assertEqual(summary["source_groups"], ["synthetic:fixture", "synthetic:rollout"])
        review = runner.read(self.root / "results/review.json")
        self.assertEqual(review["kind"], adapter.KIND)
        for row in review["rows"]:
            self.assertEqual(row["review_criteria"]["slot"], row["turn"] + 1)
            self.assertIn("REVIEW_ONLY_SENTINEL", row["review_criteria"]["facts"].values())
            if row["evaluation_type"] == "rollout" and row["turn"] == 1:
                self.assertEqual(row["output_language"], "mixed")
        runner.evaluate(runtime, core, scoring, plan, journal, 2, "synthetic-proof", 100, clock=lambda: 0)
        self.assertEqual(len(runtime.calls), 27)


if __name__ == "__main__":
    unittest.main()
