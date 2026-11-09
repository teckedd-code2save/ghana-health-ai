"""Offline producer-shaped exports containing only small, artificial neutral text."""
from contextlib import redirect_stderr, redirect_stdout
import copy
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import render_understanding_r4_results as renderer
import run_understanding_r4_evaluation as producer


def seal(folder):
    manifest = producer.read(folder / "results-manifest.json")
    manifest["files"] = {name: checksum for name, checksum in producer.files(folder).items()
                         if name not in {"status.json", "results-manifest.json", "collection.json"}}
    producer.save(folder / "results-manifest.json", manifest)
    collection = {"kind": producer.KIND, "receipt": {
        key: manifest["binding"][key] for key in ("run_id", "execution_sha256")},
        "files": {name: checksum for name, checksum in producer.files(folder).items() if name != "collection.json"}}
    producer.save(folder / "collection.json", collection)


def fixture(folder, policy, variants=renderer.VARIANTS, *, pending=False, state="completed", malicious=False,
            loss=False, unfinished=False, failed=False):
    """Use report(), not a hand-maintained approximation of its export format."""
    folder.mkdir()
    case_id = 'case-<tag>"&' if malicious else "case-fixed"
    text = '\n<hello> & "quoted" </script><script>bad()</script>\n' if malicious else "Neutral input."
    cases = [{"id": case_id, "messages": [{"role": "user", "content": text}],
              "rubric": [text], "objective": {"test": True}, "cohort": "artificial"}]
    rollouts = [{"id": "case-rollout", "user_turns": ["First input.", "Second input."], "rubric": ["Neutral check."]}]
    held = [{"id": "loss-1", "text": "Neutral held-out text."}] if loss else []
    plan = list(producer.slots(cases, rollouts, held, variants, policy))
    binding = {"version": producer.VERSION, "kind": producer.KIND, "run_id": "synthetic-" + policy,
               "execution_sha256": producer.digest(policy), "inference_sha256": producer.digest("inference"),
               "evaluation_sha256": producer.digest("evaluation"), "variants": list(variants),
               "generation_policy": policy, "guard_policy": producer.GUARD_POLICY if policy == "guarded" else None}
    identities = {v: {"variant": v, "checkpoint_sha256": producer.digest(v)} for v in variants}
    records, history = {}, {}
    for slot in plan:
        variant, kind, turn = slot["variant"], slot["kind"], slot["turn"]
        if pending and kind == "rollout" and turn == 1 and variant == "r4_response":
            continue
        record = {key: slot[key] for key in ("kind", "variant", "case_id", "turn", "policy")}
        record.update(state="complete", verified=True)
        if kind == "loss":
            record.update(state="measured", loss={"predicted_tokens": 5, "summed_nll": 10.0})
        else:
            messages = cases[0]["messages"] if kind == "fixed" else history.get(variant, []) + [
                {"role": "user", "content": slot["user"]}]
            output = text if malicious and kind == "fixed" else f"{policy} {variant} {kind} reply {turn}."
            pause = policy == "guarded" and kind == "fixed" and variant == "base"
            truncated = unfinished and kind == "fixed" and variant == "r4_response"
            record.update(state="paused" if pause else "incomplete" if truncated else "complete", request={"messages": messages},
                          objective={"literal_match": not pause, "passed": not (pause or truncated)} if kind == "fixed" else None,
                          result={"output": output, "finish_reason": "cancelled" if pause else "time_limit" if truncated else "eos",
                                  "complete": not (pause or truncated),
                                  "diagnostics": {"repetition": {"4_word_ngrams": {"repeated_positions": int(pause)}}}})
            if failed and kind == "fixed" and variant == "r4_response":
                record.update(state="error", verified=False)
            if kind == "rollout":
                history[variant] = messages + [{"role": "assistant", "content": output}]
        records[slot["key"]] = record
        producer.save(folder / "journal/records" / (producer.digest(slot["key"]) + ".json"), record)
    journal = SimpleNamespace(binding=binding, get=records.get, commit=lambda: None)
    summary = producer.report(folder, plan, journal, {}, identities)
    producer.save(folder / "status.json", {**binding, "state": state,
                  **{key: summary[key] for key in ("expected_slots", "finalized_slots")}})
    seal(folder)
    return folder


class ReportParser(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.stack, self.outputs, self.groups = [], [], []
        self.scripts = 0
        self.active = None
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.groups.append(attrs.get("data-case"))
        if tag == "script":
            self.scripts += 1
        if tag == "pre" and attrs.get("class") == "output":
            context = {key: value for _, parent in self.stack for key, value in parent.items() if key.startswith("data-")}
            self.active = {"context": context, "text": ""}
        if tag not in {"meta", "input", "br"}:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == "pre" and self.active is not None:
            self.outputs.append(self.active)
            self.active = None
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()

    def handle_data(self, data):
        if self.active is not None:
            self.active["text"] += data


class RendererTests(unittest.TestCase):
    def setUp(self):
        private = renderer.ROOT / "tmp"
        private.mkdir(exist_ok=True, mode=0o700)
        self.temp = tempfile.TemporaryDirectory(prefix="private-r4-render-tests-", dir=private)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = fixture(self.root / "raw", "raw", renderer.VARIANTS + (renderer.OPTIONAL,), loss=True)
        self.guarded = fixture(self.root / "guarded", "guarded")

    def render(self, **kwargs):
        return renderer.render_report(renderer.load_export(self.raw, "raw"), renderer.load_export(self.guarded, "guarded"), **kwargs)

    def change(self, folder, name, callback):
        path = folder / name
        value = producer.read(path)
        callback(value)
        producer.save(path, value)
        seal(folder)

    def test_complete_exports_keep_raw_and_guarded_separate(self):
        document = self.render()
        self.assertIn("Completed runs; full reported slot coverage", document)
        self.assertNotIn("PARTIAL comparison", document)
        self.assertIn("10.0 / 5 = 2.0", document)
        self.assertIn("7.3891", document)
        self.assertIn('<pre class="prompt">Neutral input.</pre>', document)
        self.assertIn("0 / 0 (not evaluated)", document)
        raw = renderer.load_export(self.raw, "raw")
        guarded = renderer.load_export(self.guarded, "guarded")
        self.assertEqual(dict(renderer.diagnostic_rows(raw.metrics["base"], [r for r in raw.rows if r["variant"] == "base"]))[
            "Guard-paused replies / verified replies"], "0 / 3")
        self.assertEqual(dict(renderer.diagnostic_rows(guarded.metrics["base"], [r for r in guarded.rows if r["variant"] == "base"]))[
            "Guard-paused replies / verified replies"], "1 / 3")
        self.assertEqual(raw.variants, renderer.VARIANTS + (renderer.OPTIONAL,))
        self.assertEqual(guarded.variants, renderer.VARIANTS)

    def test_optional_raw_pass1_not_required(self):
        self.raw = fixture(self.root / "raw-three", "raw")
        self.assertNotIn("PARTIAL comparison", self.render())

    def test_invalid_checksums_rejected_for_every_snapshot_layer(self):
        for folder, name in ((self.raw, "review.json"), (self.guarded, "summary.json"),
                             (self.raw, "status.json"), (self.guarded, "results-manifest.json")):
            with self.subTest(name=name):
                path = folder / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    renderer.load_export(folder, "raw" if folder == self.raw else "guarded")
                path.write_bytes(original)

    def test_unbound_summary_rejected(self):
        manifest = producer.read(self.raw / "results-manifest.json")
        del manifest["files"]["summary.json"]
        producer.save(self.raw / "results-manifest.json", manifest)
        collection = producer.read(self.raw / "collection.json")
        collection["files"]["results-manifest.json"] = producer.sha(self.raw / "results-manifest.json")
        producer.save(self.raw / "collection.json", collection)
        with self.assertRaisesRegex(ValueError, "Unbound summary"):
            renderer.load_export(self.raw, "raw")

    def test_wrong_policy_and_cross_policy_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Policy mismatch"):
            renderer.load_export(self.raw, "guarded")
        self.change(self.raw, "review.json", lambda value: value["rows"][0].update(generation_policy="guarded"))
        with self.assertRaisesRegex(ValueError, "Review policy mismatch"):
            self.render()

    def test_missing_variant_requires_explicit_partial(self):
        self.guarded = fixture(self.root / "guarded-two", "guarded", renderer.VARIANTS[:2])
        with self.assertRaisesRegex(ValueError, "Missing required variant: r4_response"):
            self.render()
        document = self.render(allow_partial=True)
        self.assertIn("PARTIAL comparison", document)
        self.assertIn("Missing variant; not measured", document)
        self.assertIn("Missing variant or case/turn row; no output available.", document)

    def test_unfinished_rows_and_run_labeled_precisely(self):
        self.raw = fixture(self.root / "raw-pending", "raw", pending=True, state="paused_budget")
        with self.assertRaisesRegex(ValueError, "Partial exports"):
            self.render()
        document = self.render(allow_partial=True)
        self.assertIn("PARTIAL comparison", document)
        for expected in ("8 / 9 total slots finalized", "1 / 3 chat slots pending", "Run state: paused_budget",
                         "state: pending", "No output recorded."):
            self.assertIn(expected, document)

    def test_finished_run_keeps_truncated_reply_incomplete(self):
        self.raw = fixture(self.root / "raw-truncated", "raw", unfinished=True)
        document = self.render()
        self.assertIn("Completed runs; full reported slot coverage", document)
        self.assertIn("state: incomplete; verified: True; finish_reason: time_limit; complete: False", document)
        self.assertIn("2 / 3", document)
        self.assertIn("Run completion does not mean every reply completed with EOS", document)

    def test_completed_run_with_unverified_reply_reports_coverage_gap(self):
        self.raw = fixture(self.root / "raw-error", "raw", failed=True)
        with self.assertRaisesRegex(ValueError, "2 / 3 chat replies verified"):
            self.render()
        document = self.render(allow_partial=True)
        self.assertIn("Completed runs; coverage gaps", document)
        self.assertNotIn("PARTIAL comparison", document)
        self.assertIn("Run state: completed; finalized slots: 9 / 9", document)
        self.assertIn("state: error; verified: False", document)

    def test_verified_reply_cannot_claim_false_completion(self):
        self.change(self.raw, "review.json", lambda value: value["rows"][0].update(complete=False))
        with self.assertRaisesRegex(ValueError, "Reply completion state differs"):
            self.render()

    def test_missing_review_row_does_not_invent_a_denominator(self):
        self.change(self.raw, "review.json", lambda value: value["rows"].pop())
        with self.assertRaisesRegex(ValueError, "review rows present"):
            self.render()
        document = self.render(allow_partial=True)
        self.assertIn("0 / unavailable (review rows missing)", document)
        self.assertIn("case/turn groups missing", document)

    def test_completion_state_must_be_collection_bound(self):
        (self.raw / "status.json").unlink()
        seal(self.raw)
        with self.assertRaisesRegex(ValueError, "Completion state unavailable"):
            self.render()
        self.assertIn("Completion state unavailable", self.render(allow_partial=True))

    def test_status_counts_cannot_imply_complete_coverage(self):
        self.change(self.raw, "status.json", lambda value: value.update(finalized_slots=1))
        with self.assertRaisesRegex(ValueError, "Status finalized_slots: 1"):
            self.render()

    def test_html_escaping_preserves_exact_outputs_and_input_text(self):
        self.raw = fixture(self.root / "raw-escaped", "raw", malicious=True)
        self.guarded = fixture(self.root / "guarded-escaped", "guarded", malicious=True)
        document = self.render()
        parser = ReportParser(document)
        self.assertEqual(parser.scripts, 1)
        self.assertNotIn("<script>bad()", document)
        self.assertIn("&lt;hello&gt; &amp; &quot;quoted&quot;", document)
        self.assertIn('case-<tag>"&', parser.groups)
        output = next(item for item in parser.outputs if item["context"]["data-case"].startswith("case-<tag>"))
        self.assertEqual(output["text"], '\n<hello> & "quoted" </script><script>bad()</script>\n')

    def test_output_grouping_and_own_history(self):
        document = self.render()
        parser = ReportParser(document)
        self.assertEqual(parser.groups, ["case-fixed", "case-rollout"])
        self.assertEqual(len(parser.outputs), 21)
        for output in parser.outputs:
            context, text = output["context"], output["text"]
            self.assertTrue(text.startswith(context["data-policy"] + " "))
            if context["data-case"] == "case-rollout":
                self.assertTrue(text.endswith("reply " + context["data-turn"] + "."))
        self.assertEqual(document.count("Own-model history:"), 2)
        self.assertNotIn("<details open", document)
        self.assertIn('id="case-filter"', document)
        self.assertIn('id="policy-filter"', document)
        self.assertIn("default-src 'none'; script-src 'sha256-", document)

    def test_empty_output_is_distinct_from_missing_output(self):
        row = renderer.load_export(self.raw, "raw").rows[0]
        row["output"] = ""
        cell = renderer.output_cell(row)
        self.assertIn("Recorded output is an empty string", cell)
        self.assertNotIn("No output recorded", cell)

    def test_duplicate_slots_and_inconsistent_counts_rejected(self):
        self.change(self.raw, "review.json", lambda value: value["rows"].append(copy.deepcopy(value["rows"][0])))
        with self.assertRaisesRegex(ValueError, "Duplicate review slot"):
            self.render()
        self.change(self.raw, "review.json", lambda value: value["rows"].pop())
        self.change(self.raw, "summary.json", lambda value: value["mechanical_metrics"]["base"].update(chat_verified=99))
        with self.assertRaisesRegex(ValueError, "Verified count"):
            self.render()

    def test_cross_export_evaluation_binding_must_match(self):
        raw, guarded = renderer.load_export(self.raw, "raw"), renderer.load_export(self.guarded, "guarded")
        guarded.summary["binding"]["evaluation_sha256"] = producer.digest("different")
        with self.assertRaisesRegex(ValueError, "Incomparable exports"):
            renderer.render_report(raw, guarded)

    def test_paths_cannot_escape_or_follow_links(self):
        with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
            renderer.artifact(self.raw, "../guarded/summary.json")
        (self.raw / "linked.json").symlink_to(self.guarded / "summary.json")
        with self.assertRaisesRegex(ValueError, "Linked or escaping"):
            renderer.artifact(self.raw, "linked.json")

    def test_private_output_restrictions_permissions_and_cli(self):
        output = self.root / "comparison.html"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = renderer.main(["--raw", str(self.raw), "--guarded", str(self.guarded), "--out", str(output)])
        self.assertEqual(result, 0)
        self.assertIn("Exact outputs", output.read_text())
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "already exists"):
            renderer.write_private(output, "overwrite")
        with self.assertRaisesRegex(ValueError, "repository tmp"):
            renderer.write_private(renderer.ROOT / "not-private.html", "private")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = renderer.main(["--raw", str(self.guarded), "--guarded", str(self.raw), "--out", str(self.root / "bad.html")])
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "bad.html").exists())


if __name__ == "__main__":
    unittest.main()
