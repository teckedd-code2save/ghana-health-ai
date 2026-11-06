"""Offline synthetic fixtures test packaging, not Twi quality or model accuracy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import prepare_understanding_r2_evaluation as evaluation


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in values))


def manifest(folder, extra=None):
    value = {"artifacts": {p.relative_to(folder).as_posix(): evaluation.sha(p)
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "manifest.json"}, **(extra or {})}
    save(folder / "manifest.json", value)
    return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.r1, self.results, self.live, self.snapshot = [self.root / p for p in ("r1", "results", "live", "snapshot")]
        self.sources, self.training, self.bound = [self.root / p for p in ("sources", "training", "bound")]
        original = [{"id": f"case-{i}", "split": "challenge" if i < 20 else "development",
            "messages": [{"role": "user", "content": f"Original source question {i}"}], "objective": None,
            "human_review": {"status": "pending"}} for i in range(30)]
        save(self.r1 / "evaluation/cases.json", original)
        manifest(self.r1 / "evaluation", {"model": "fixture-base", "revision": "fixture-revision",
            "protocol": {"decoding": evaluation.DECODING, "history_mode": "fixed_complete_history", "tools": []}})
        prompts = [{"id": c["id"], "messages_sha256": evaluation.digest(c["messages"]), "prompt_ids": [i, 42]}
                   for i, c in enumerate(original)]
        save(self.r1 / "prompts.json", prompts)
        proof = {"adapters": {v: {"identity_sha256": evaluation.digest([v, "identity"]),
                 "weights": {"sha256": evaluation.digest([v, "weights"])}} for v in evaluation.R1_VARIANTS[1:]}}
        save(self.r1 / "training-proof.json", proof)
        save(self.r1 / "tokenizer/tokenizer.json", {"fixture": True})
        save(self.r1 / "tokenizer/chat_template.jinja", "fixture")
        execution = manifest(self.r1, {"decoding": evaluation.DECODING, "runtime": {"fixture": "1"},
            "weight": {"sha256": evaluation.digest("base")}})
        # The execution artifact list also binds the nested evaluation manifest.
        execution["artifacts"]["evaluation/manifest.json"] = evaluation.sha(self.r1 / "evaluation/manifest.json")
        save(self.r1 / "manifest.json", execution)
        runtime = {"environment_sha256": evaluation.digest(execution["runtime"]),
            "tokenizer_sha256": execution["artifacts"]["tokenizer/tokenizer.json"],
            "chat_template_sha256": execution["artifacts"]["tokenizer/chat_template.jinja"]}
        for variant in evaluation.R1_VARIANTS:
            save(self.results / (variant + ".json"), {"load_verified": True, "decoding": evaluation.DECODING,
                "runtime": runtime, "suite_manifest_sha256": evaluation.sha(self.r1 / "evaluation/manifest.json"),
                "model": {"id": "fixture-base", "revision": "fixture-revision", "weights_sha256": evaluation.digest("base"),
                    "adapter_sha256": None if variant == "base" else proof["adapters"][variant]["identity_sha256"]},
                "records": [{"case_id": p["id"], "messages_sha256": p["messages_sha256"],
                    "prompt_token_ids_sha256": evaluation.digest(p["prompt_ids"]), "output": "Recorded fixture output",
                    "generated": True, "finish_reason": "eos"} for p in prompts]})
        save(self.results / "results-manifest.json", {"execution_manifest_sha256": evaluation.sha(self.r1 / "manifest.json"),
            "artifacts": {v + ".json": evaluation.sha(self.results / (v + ".json")) for v in evaluation.R1_VARIANTS}})
        for variant in evaluation.R1_VARIANTS:
            for cid in sorted(evaluation.LIVE_IDS):
                if variant != "meaning-adapter" and cid not in {"reported-informal", "standard", "english-control"}:
                    continue
                messages = [{"role": "user", "content": "Canonical raw " + cid}]
                saved = {"spec": {"case": cid, "variant": variant, "profile": "raw", "messages": messages,
                            "execution_sha": evaluation.sha(self.r1 / "manifest.json")},
                    "result": {"variant": variant, "load_verified": True, "decoding": evaluation.DECODING,
                        "messages_sha256": evaluation.digest(messages), "output": "Recorded raw reply",
                        "identity": {"execution_manifest_sha256": evaluation.sha(self.r1 / "manifest.json"),
                            "model": "fixture-base", "revision": "fixture-revision", "variant": variant,
                            "base_weights_sha256": evaluation.digest("base"), "runtime_sha256": runtime["environment_sha256"],
                            "tokenizer_sha256": runtime["tokenizer_sha256"], "chat_template_sha256": runtime["chat_template_sha256"],
                            "adapter_weights_sha256": None if variant == "base" else proof["adapters"][variant]["weights"]["sha256"]}}}
                save(self.live / f"{variant}-{cid}.json", saved)
                prompted = copy.deepcopy(saved)
                prompted["spec"]["profile"] = "conversational-system-v1"
                prompted["spec"]["messages"].insert(0, {"role": "system", "content": "Must not enter raw evaluation"})
                save(self.live / f"{variant}-{cid}-prompted.json", prompted)
        release = self.root / "release"
        originals = []
        for i, (kind, origin, generated) in enumerate([
            ("sentence", "compiled_parallel_text", False), ("utterance", "speech_transcript", False),
            ("dictionary", "compiled_parallel_text", False), ("sentence", "llm_generated", True),
            ("sentence", "unknown", False),
        ]):
            originals.append({"id": f"held-{i}", "source_id": f"held-source-{i}", "source_hash": f"held-hash-{i}",
                "group_id": f"held-group-{i}", "split": "validation", "language": "tw", "record_type": kind,
                "origin": origin, "generated_target": generated, "text": f"Protected original passage {i}"})
        jsonl(release / "validation.jsonl", originals)
        save(self.snapshot / "batches.json", [{"batch_id": "language", "release_path": str(release),
            "release_manifest_sha256": "fixture-release", "files": {"validation.jsonl": evaluation.sha(release / "validation.jsonl")}}])
        jsonl(self.snapshot / "records.jsonl", [{"collection": "language", "batch_id": "language", "file": "validation.jsonl",
            "line": i + 1, "row_sha256": evaluation.digest(row), "id": row["id"], "source_id": row["source_id"], "split": "validation"}
            for i, row in enumerate(originals)])
        manifest(self.snapshot)

    def freeze(self):
        return evaluation.freeze_sources(self.r1, self.results, self.live, self.snapshot, self.sources)

    def train(self, witness=None):
        witness = witness or {"id": "train-1", "collection": "language", "source_id": "train-source", "text": "Unrelated training text", "metadata": {}}
        jsonl(self.training / "source-witnesses.jsonl", [witness])
        manifest(self.training, {"model": "fixture-base", "revision": "fixture-revision", "selected_rows": 1,
            "snapshot_manifest_sha256": evaluation.sha(self.snapshot / "manifest.json")})

    def test_freeze_preserves_original_cases_and_raw_once(self):
        before = evaluation.sha(self.r1 / "evaluation/cases.json")
        self.freeze()
        info, cases = evaluation.verify_sources(self.sources)
        self.assertEqual(len(cases), 36)
        self.assertEqual({c["split"] for c in cases}, {"development"})
        self.assertEqual(evaluation.sha(self.sources / "r1-original-cases.json"), before)
        self.assertEqual(evaluation.sha(self.r1 / "evaluation/cases.json"), before)
        self.assertEqual(info["counts"]["language_loss_rows"], 2)
        plan = evaluation.read(self.sources / "comparison-plan.json")
        self.assertEqual([r["id"] for r in plan["checkpoints"]], list(evaluation.CHECKPOINTS))
        self.assertEqual(plan["saved_reference_counts"], {"base": 33, "r1_meaning": 36, "r1_response": 33})
        self.assertEqual(len(plan["missing_saved_slots"]), 6)

    def test_loss_uses_existing_snapshot_and_no_dictionary_or_generated_rows(self):
        self.freeze()
        heldout = evaluation.read(self.sources / "language-heldout.json")
        self.assertEqual([r["id"] for r in heldout], ["held-0", "held-1"])
        loss = evaluation.read(self.sources / "language-loss.json")
        self.assertEqual(loss["existing_validation_rows"], 5)
        self.assertFalse(loss["training_eligible"])
        self.assertFalse(loss["expose_as_examples"])
        self.assertFalse(loss["chat_template"])
        self.assertIn("shift", loss["labels"])

    def test_bound_pack_verifies_offline_and_returns_manifest_cases(self):
        self.freeze(); self.train()
        evaluation.prepare(self.training, self.sources, self.bound)
        info, cases = evaluation.verify_pack(self.bound, self.training)
        self.assertEqual(info["bindings"]["training_manifest_sha256"], evaluation.sha(self.training / "manifest.json"))
        self.assertEqual(len(cases), 36)
        self.assertEqual(evaluation.verify_pack(self.bound), (info, cases))

    def test_sources_alone_cannot_launch(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "binding is pending"):
            evaluation.verify_pack(self.sources)

    def test_cannot_overwrite_frozen_pack(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.freeze()

    def test_tamper_is_detected(self):
        self.freeze(); self.train(); evaluation.prepare(self.training, self.sources, self.bound)
        save(self.bound / "cases.json", [])
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            evaluation.verify_pack(self.bound)

    def test_wrong_training_manifest_rejected(self):
        self.freeze(); self.train(); evaluation.prepare(self.training, self.sources, self.bound)
        value = evaluation.read(self.training / "manifest.json"); value["model"] = "different"
        save(self.training / "manifest.json", value)
        with self.assertRaisesRegex(ValueError, "Wrong R2"):
            evaluation.verify_pack(self.bound, self.training)

    def test_actual_language_overlap_is_reported_and_blocks_verification(self):
        self.freeze(); self.train({"id": "train", "language": {"text": "Protected original passage 0"}, "metadata": {}})
        info = evaluation.prepare(self.training, self.sources, self.bound)
        self.assertFalse(info["ready_for_development_evaluation"])
        hits = evaluation.read(self.bound / "overlap-audit.json")["matches"]
        self.assertTrue(any(h.get("training_field") == "$.language.text" for h in hits))
        with self.assertRaisesRegex(ValueError, "overlap blocks"):
            evaluation.verify_pack(self.bound)

    def test_embedded_english_and_twi_fields_are_scanned(self):
        self.freeze()
        _, cases = evaluation.verify_sources(self.sources)
        for name in ("english", "twi"):
            row = {"id": "train", "content": json.dumps({name: "ORIGINAL source question 0!"})}
            audit = evaluation.audit_overlap(cases, [], [(1, row)])
            self.assertFalse(audit["overlap_free"])
            self.assertTrue(any(h["training_field"] == "$.content.decoded." + name for h in audit["matches"]))

    def test_source_group_overlap_blocks_even_with_different_text(self):
        self.freeze(); self.train({"id": "train", "text": "Different phrasing", "metadata": {"group_id": "held-group-0"}})
        evaluation.prepare(self.training, self.sources, self.bound)
        self.assertTrue(any(h["method"] == "group_id" for h in evaluation.read(self.bound / "overlap-audit.json")["matches"]))

    def test_changed_saved_decoding_not_reused(self):
        path = self.results / "base.json"
        value = evaluation.read(path); value["decoding"]["max_new_tokens"] = 128; save(path, value)
        index = evaluation.read(self.results / "results-manifest.json"); index["artifacts"][path.name] = evaluation.sha(path)
        save(self.results / "results-manifest.json", index)
        with self.assertRaisesRegex(ValueError, "protocol/identity"):
            self.freeze()

    def test_duplicate_raw_reference_slot_rejected(self):
        path = self.live / "meaning-adapter-standard.json"
        save(self.live / "duplicate.json", evaluation.read(path))
        with self.assertRaisesRegex(ValueError, "Repeated reference slot"):
            self.freeze()

    def test_missing_canonical_raw_case_rejected(self):
        (self.live / "meaning-adapter-negative.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing canonical"):
            self.freeze()

    def test_changed_validation_row_cannot_enter_loss(self):
        path = self.root / "release/validation.jsonl"
        path.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Changed validation source"):
            self.freeze()

    def test_duplicate_json_keys_and_escaping_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            evaluation.parse('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
            evaluation.safe(self.root, "../outside")


if __name__ == "__main__":
    unittest.main()
