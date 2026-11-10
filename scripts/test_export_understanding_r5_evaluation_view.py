"""Synthetic saved reports only: no GPU, model, network, or production mutations."""
import copy
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import export_understanding_r5_evaluation_view as view
import run_understanding_r5_evaluation as r

UNSAFE = '</script><script>globalThis.injected=true</script><img src=x onerror=alert(1)>&\u2028\u2029'


def fixture(folder, policy="raw"):
    folder.mkdir()
    cases = [{"id": "fixed", "messages": [{"role": "user", "content": UNSAFE}],
              "tags": ["negation"], "source_group_id": "one-source", "objective": {"test": True}}]
    rollouts = [{"id": "roll", "user_turns": ["First", "Second"], "rubric": ["Keep own history"]}]
    held = [{"id": "held-twi", "text": "PRIVATE HELDOUT TEXT"},
            {"id": "held-english", "loss_cohort": r.retention.ENGLISH, "messages": [{"role": "assistant", "content": "PRIVATE ENGLISH"}]}]
    plan = list(r.slots(cases, rollouts, held, tuple(view.VARIANTS), policy))
    kind = r.KIND
    execution = {"kind": kind, "generation_policy": policy, "expected_slots": len(plan)}
    r.save(folder / "execution.json", execution)
    binding = {"kind": kind, "generation_policy": policy, "run_id": "fixture-" + policy,
               "execution_sha256": r.sha(folder / "execution.json"), "variants": list(view.VARIANTS), "inference_sha256": "a" * 64}
    r.save(folder / "journal/binding.json", binding)
    identities = {v: {"variant": v, "manifest_sha256": binding["inference_sha256"]} for v in view.VARIANTS}
    proof_path = folder / "runtime/attempt-1.json"
    r.save(proof_path, {"binding": binding, "identities": identities})
    records, histories, previous = {}, {}, {}
    for slot in plan:
        variant, key = slot["variant"], slot["key"]
        if slot["kind"] == "loss" and (variant != "base" or slot["case_id"] == "held-english"):
            continue
        if slot["kind"] == "fixed" and variant == "r5_response":
            continue
        group = (variant, slot["case_id"])
        request = r.retention.request(slot["row"]) if slot["kind"] == "loss" else {"messages":
            cases[0]["messages"] if slot["kind"] == "fixed" else histories.get(group, []) + [{"role": "user", "content": slot["user"]}]}
        record = {"id": key, "binding_sha256": r.digest(binding), "runtime_proof_sha256": r.sha(proof_path),
                  "request": request, "inputs_sha256": r.digest(request), "attempt": 1, "verified": True,
                  "parent_sha256": r.digest(previous[group]) if group in previous else None,
                  **{k: slot[k] for k in ("kind", "variant", "case_id", "turn", "policy")}}
        if slot["kind"] == "loss":
            record.update(state="measured", loss={"predicted_tokens": 3, "summed_nll": 6.0}, predicted_tokens=3)
        elif slot["kind"] == "fixed" and variant == "r4_response":
            record.update(state="error", verified=False, error_type="FixtureError", error="Retained failure")
        else:
            state = "incomplete" if variant == "base" and slot["turn"] == 1 else "complete"
            output = UNSAFE if slot["kind"] == "fixed" else variant + ": own answer " + str(slot["turn"])
            record.update(state=state, result={"output": output, "finish_reason": "eos" if state == "complete" else "max_new_tokens",
                "complete": state == "complete", "diagnostics": {"repetition": {"4_word_ngrams": {"repeated_positions": 0}}}},
                objective={"literal_match": True, "passed": True} if slot["kind"] == "fixed" else None)
            if slot["kind"] == "rollout":
                histories[group] = request["messages"] + [{"role": "assistant", "content": output}]
                previous[group] = record
        records[key] = record
        r.save(folder / "journal/records" / (r.digest(key) + ".json"), {"record": record, "sha256": r.digest(record)})
    key = next(slot["key"] for slot in plan if slot["kind"] == "loss" and slot["variant"] == "r4_response")
    r.save(folder / "journal/attempts/1" / (r.digest(key) + ".json"), {"id": key, "state": "started", "attempt": 1})
    r.save(folder / "collection.json", {"receipt": {k: binding[k] for k in ("run_id", "execution_sha256")}, "files": r.files(folder)})
    journal = SimpleNamespace(binding=binding, get=records.get, commit=lambda: None)
    r.report(folder, plan, journal, {"kind": kind}, identities)
    r.save(folder / "DERIVED-PARTIAL-REPORT.json", {"collection_sha256": r.sha(folder / "collection.json"),
        "execution_sha256": binding["execution_sha256"], "generation_requests": 0, "saved_records": len(records), "planned_slots": len(plan)})
    return folder


def rebind_report(folder, name):
    manifest = r.read(folder / "results-manifest.json")
    manifest["files"][name] = r.sha(folder / name)
    r.save(folder / "results-manifest.json", manifest)


class EvaluationViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.folder = fixture(self.root / "report")

    def load(self):
        return view.load_results("Development raw", self.folder)[0]

    def agent(self, name="agent.json", hash_key="source_file_sha256"):
        path = self.root / name
        row = r.read(self.folder / "review.json")["rows"][0]
        r.save(path, {"version": "r5-agent-semantic-review-v1", "reviewer_type": "agent_not_native_human_or_clinical",
            "source_file": str(self.folder / "review.json"), hash_key: r.sha(self.folder / "review.json"),
            "rows": [{**{k: row[k] for k in ("slot_id", "case_id", "variant", "turn")}, "meaning": "uncertain",
                "usefulness": "partial", "requested_language": "not_applicable", "unsupported_claims": None, "notes": UNSAFE}]})
        return path

    def phrase(self):
        path = self.root / "phrase.json"
        row = r.read(self.folder / "review.json")["rows"][0]
        r.save(path, {"version": "r5-user-phrase-review-v1", "event_id": "synthetic-event",
            "reviewer_type": "user", "review_scope": "quoted_phrase_only",
            "full_response_reviewed": False, "training_eligible": False, "original_output_modified": False,
            "question": "Is this phrase correct?", "user_answer": "Change true to false.",
            "original_quote": "globalThis.injected=true", "corrected_quote": "globalThis.injected=false",
            "occurrences": [{"source_file_sha256": r.sha(self.folder / "review.json"),
                             **{k: row[k] for k in ("slot_id", "case_id", "variant", "turn")}}]})
        return path

    def test_phrase_correction_preserves_original_output_and_whole_answer_ratings(self):
        data = self.load()
        view.attach_reviews([data], [self.agent()])
        original_rows, original_ratings = copy.deepcopy(data["rows"]), copy.deepcopy(data["agent"])
        sources, hashes = view.attach_phrase_reviews([data], [self.phrase()])
        self.assertEqual(data["rows"], original_rows)
        self.assertEqual(data["agent"], original_ratings)
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(hashes), 1)
        self.assertTrue(all(row["human_review"]["status"] == "pending" for row in data["rows"]))
        self.assertEqual(next(iter(data["phrase_reviews"].values()))[0]["review"]["corrected_quote"],
                         "globalThis.injected=false")

    def test_phrase_correction_rejects_wrong_output_binding_and_duplicate_event(self):
        path = self.phrase()
        original = r.read(path)
        for mutate in (lambda d: d["occurrences"][0].update(source_file_sha256="0" * 64),
                       lambda d: d["occurrences"][0].update(variant="r5_response"),
                       lambda d: d.update(original_quote="not in the output"),
                       lambda d: d["occurrences"].append(d["occurrences"][0]),
                       lambda d: d.update(occurrences=[])):
            value = copy.deepcopy(original)
            mutate(value)
            r.save(path, value)
            with self.assertRaises(ValueError):
                view.attach_phrase_reviews([self.load()], [path])
        r.save(path, original)
        duplicate = self.root / "duplicate-phrase.json"
        r.save(duplicate, original)
        with self.assertRaisesRegex(ValueError, "Invalid phrase review"):
            view.attach_phrase_reviews([self.load()], [path, duplicate])

    def test_phrase_review_cannot_claim_full_approval_or_training_admission(self):
        path = self.phrase()
        original = r.read(path)
        for key in ("full_response_reviewed", "training_eligible", "original_output_modified"):
            value = copy.deepcopy(original)
            value[key] = True
            r.save(path, value)
            with self.assertRaisesRegex(ValueError, "full-response approval"):
                view.attach_phrase_reviews([self.load()], [path])

    def test_phrase_correction_is_rendered_separately_and_html_escaped(self):
        path = self.phrase()
        value = r.read(path)
        value["user_answer"] = UNSAFE
        r.save(path, value)
        result = view.export([("raw", self.folder)], [self.agent()], self.root / "view", [path])
        html = Path(result["path"]).read_text()
        embedded = html.split('<script id="snapshot" type="application/json">', 1)[1].split('</script>', 1)[0]
        data = json.loads(embedded)
        self.assertNotIn("<", embedded)
        self.assertEqual(html.count("<script"), 2)
        self.assertEqual(data["results"][0]["rows"][0]["output"], UNSAFE)
        self.assertEqual(data["phrase_review_sources"][0]["review"]["user_answer"], UNSAFE)
        self.assertIn("User-confirmed wording correction", html)

    def test_errors_pending_incomplete_and_partial_retention_are_not_dropped(self):
        data = self.load()
        self.assertEqual(len(data["rows"]), 9)
        self.assertEqual(data["started_without_result"], 1)
        self.assertEqual(data["summary"]["finalized_slots"], 9)
        self.assertEqual(data["summary"]["expected_slots"], 15)
        self.assertEqual(data["benchmark"]["totals"]["r4_response"]["states"]["error"], 1)
        self.assertEqual(data["benchmark"]["totals"]["r5_response"]["states"]["pending"], 1)
        self.assertEqual(data["benchmark"]["totals"]["base"]["states"]["incomplete"], 1)
        self.assertEqual(data["benchmark"]["totals"]["base"]["source_group_count"], 1)
        self.assertNotIn("PRIVATE HELDOUT TEXT", json.dumps(data))
        self.assertNotIn("PRIVATE ENGLISH", json.dumps(data))
        histories = [row["messages"] for row in data["rows"] if row["evaluation_type"] == "rollout" and row["turn"] == 1]
        self.assertEqual(len({json.dumps(h) for h in histories}), 3)

    def test_optional_reviews_remain_separate_and_support_both_real_hash_fields(self):
        for key in ("source_file_sha256", "source_sha256"):
            data = self.load()
            view.attach_reviews([data], [self.agent(hash_key=key)])
            self.assertEqual(len(data["agent"]), 1)
            self.assertEqual(next(iter(data["agent"].values()))["rating"]["meaning"], "uncertain")
            self.assertTrue(all(row["human_review"]["status"] == "pending" for row in data["rows"]))
        self.assertEqual(self.load()["agent"], {})

    def test_wrong_hash_duplicate_assignment_or_human_claim_rejected(self):
        path = self.agent()
        original = r.read(path)
        for mutate in (lambda d: d.update(source_file_sha256="0" * 64),
                       lambda d: d.update(reviewer_type="native_human"),
                       lambda d: d["rows"].append(d["rows"][0]),
                       lambda d: d["rows"][0].update(variant="r5_response")):
            data = copy.deepcopy(original)
            mutate(data)
            r.save(path, data)
            with self.assertRaises(ValueError):
                view.attach_reviews([self.load()], [path])

    def test_changed_manifest_artifact_or_collection_is_rejected(self):
        (self.folder / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.load()

    def test_resealed_review_cannot_rewrite_raw_output_or_drop_error(self):
        original = r.read(self.folder / "review.json")
        for mutate in (lambda d: d["rows"][0].update(output="rewritten"), lambda d: d["rows"].pop(1)):
            value = copy.deepcopy(original)
            mutate(value)
            r.save(self.folder / "review.json", value)
            rebind_report(self.folder, "review.json")
            with self.assertRaises(ValueError):
                self.load()

    def test_rebound_bad_raw_envelope_fails_its_own_digest(self):
        path = next((self.folder / "journal/records").glob("*.json"))
        value = r.read(path)
        value["record"]["state"] = "indeterminate"
        r.save(path, value)
        name = path.relative_to(self.folder).as_posix()
        collection = r.read(self.folder / "collection.json")
        collection["files"][name] = r.sha(path)
        r.save(self.folder / "collection.json", collection)
        derived = r.read(self.folder / "DERIVED-PARTIAL-REPORT.json")
        derived["collection_sha256"] = r.sha(self.folder / "collection.json")
        r.save(self.folder / "DERIVED-PARTIAL-REPORT.json", derived)
        rebind_report(self.folder, name)
        rebind_report(self.folder, "collection.json")
        with self.assertRaisesRegex(ValueError, "Raw record hash"):
            self.load()

    def test_unsafe_manifest_path_fails_closed(self):
        manifest = r.read(self.folder / "results-manifest.json")
        manifest["files"]["../escape.json"] = "0" * 64
        r.save(self.folder / "results-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Unsafe relative"):
            self.load()

    def test_missing_raw_record_fails_closed(self):
        next((self.folder / "journal/records").glob("*.json")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_linked_artifact_fails_closed(self):
        path = self.folder / "review.json"
        path.rename(self.root / "linked.json")
        path.symlink_to(self.root / "linked.json")
        with self.assertRaisesRegex(ValueError, "Escaping artifact"):
            self.load()

    def test_resealed_mechanical_totals_cannot_hide_errors(self):
        summary = r.read(self.folder / "summary.json")
        benchmark = r.read(self.folder / "benchmark.json")
        benchmark["totals"]["r4_response"]["states"]["error"] = 0
        summary["benchmark"] = {k: v for k, v in benchmark.items() if k != "binding"}
        r.save(self.folder / "summary.json", summary)
        r.save(self.folder / "benchmark.json", benchmark)
        rebind_report(self.folder, "summary.json")
        rebind_report(self.folder, "benchmark.json")
        with self.assertRaisesRegex(ValueError, "Mechanical denominators"):
            self.load()

    def test_resealed_retention_scores_cannot_replace_measured_nll(self):
        summary = r.read(self.folder / "summary.json")
        summary["mechanical_metrics"]["base"]["language_loss"][r.retention.TWI]["token_weighted_mean_nll"] = 999
        r.save(self.folder / "summary.json", summary)
        rebind_report(self.folder, "summary.json")
        with self.assertRaisesRegex(ValueError, "Retention denominators"):
            self.load()

    def test_normal_collected_runner_report_without_derivation_sidecar(self):
        (self.folder / "DERIVED-PARTIAL-REPORT.json").unlink()
        (self.folder / "collection.json").unlink()
        manifest = r.read(self.folder / "results-manifest.json")
        del manifest["files"]["collection.json"]
        r.save(self.folder / "results-manifest.json", manifest)
        binding = manifest["binding"]
        r.save(self.folder / "collection.json", {"receipt": {k: binding[k] for k in ("run_id", "execution_sha256")},
                                                "files": r.files(self.folder)})
        self.assertIsNone(self.load()["proof"]["derived"])

    def test_multiple_policies_are_not_pooled_and_duplicate_snapshot_rejected(self):
        guarded = fixture(self.root / "guarded", "guarded")
        result = view.export([("raw", self.folder), ("guarded", guarded)], [], self.root / "view")
        self.assertEqual(len(result["reports"]), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            view.export([("raw", self.folder), ("copy", self.folder)], [], self.root / "duplicate")

    def test_safe_embedding_private_output_and_unchanged_inputs(self):
        agent = self.agent()
        _, hashes = view.load_results("raw", self.folder)
        result = view.export([("raw", self.folder)], [agent], self.root / "view")
        path = Path(result["path"])
        html = path.read_text()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        self.assertEqual(html.count("<script"), 2)
        embedded = html.split('<script id="snapshot" type="application/json">', 1)[1].split('</script>', 1)[0]
        data = json.loads(embedded)
        self.assertEqual(data["results"][0]["rows"][0]["output"], UNSAFE)
        self.assertNotIn("<", embedded)
        self.assertNotIn("\u2028", embedded)
        self.assertNotIn("innerHTML", view.SCRIPT)
        self.assertIn("connect-src 'none'", html)
        view.unchanged(hashes)
        with self.assertRaisesRegex(ValueError, "new output"):
            view.export([("raw", self.folder)], [], self.root / "view")

    def test_transferred_judgment_keeps_its_label_and_proof(self):
        path = self.agent()
        value = r.read(path)
        value["rows"][0].update(transferred_judgment=True, independent_new_review=False,
                                transfer_proof={"original_rating": {"file_sha256": "a" * 64}})
        r.save(path, value)
        result = view.export([("raw", self.folder)], [path], self.root / "view")
        html = Path(result["path"]).read_text()
        embedded = html.split('<script id="snapshot" type="application/json">', 1)[1].split('</script>', 1)[0]
        rating = next(iter(json.loads(embedded)["results"][0]["agent"].values()))["rating"]
        self.assertTrue(rating["transferred_judgment"])
        self.assertFalse(rating["independent_new_review"])
        self.assertEqual(rating["transfer_proof"], value["rows"][0]["transfer_proof"])
        self.assertIn("Agent judgment reused from an exact raw match", html)
        self.assertIn("Exact-match transfer evidence (not a new independent review)", html)

    def test_mutation_during_render_never_publishes(self):
        def render(data):
            (self.folder / "review.json").write_text("changed")
            return "unused"
        with patch.object(view, "render_html", side_effect=render):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                view.export([("raw", self.folder)], [], self.root / "view")
        self.assertFalse((self.root / "view").exists())


if __name__ == "__main__":
    unittest.main()
