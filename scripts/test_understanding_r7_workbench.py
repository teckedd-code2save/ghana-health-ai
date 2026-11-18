"""Synthetic R7/R6 CPU-only tests. Frozen audition, fake transport; no server or GPU calls."""
import asyncio
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import r2_inference as foundation
import understanding_r7_workbench as ui

SHA = "a" * 64
MANIFEST = {"version": ui.VERSION, "model": foundation.MODEL, "revision": foundation.REVISION,
            "base_sha256": foundation.BASE_SHA, "decoding": foundation.DECODING,
            "private": True, "training": False,
            "actor": {"app_name": "ghana-understanding-r7-demo-private", "class_name": "UnderstandingR7Demo"},
            "variants": {"base": None, **{v: {"directory": "/artifacts/" + v,
                "files": {"adapter_model.safetensors": c * 64}} for v, c in (("r6_response", "b"), ("r7_response", "c"))}}}


class FakeTransport:
    def __init__(self, demo, *, guarded=True, changes=None):
        self.demo, self.guarded, self.changes = demo, guarded, changes or {}
        self.calls, self.cancelled = [], []

    async def cancel(self, call_id):
        self.cancelled.append(call_id)

    async def __call__(self, messages, variant, expected_sha, request_id):
        self.calls.append((variant, copy.deepcopy(messages)))
        change = self.changes.get(variant, {})
        if change == "invalid_unicode_pre_start":
            raise ValueError("Invalid Unicode in message")
        output = change.get("output", variant + " answer") if isinstance(change, dict) else variant + " answer"
        result = {"identity": self.demo.identity(MANIFEST, expected_sha, variant),
                  "messages_sha256": ui.digest(messages), "prompt_ids_sha256": ui.digest({"prompt": messages}),
                  "decoding": copy.deepcopy(MANIFEST["decoding"]), "output": output, "output_ids": [1, 2, 3],
                  "seconds": .1, "finish_reason": "eos", "complete": True}
        if self.guarded:
            result["loop_guard"] = {"version": self.demo.LOOP_POLICY, "trigger": None}
        start = {"type": "start", "protocol": self.demo.PROTOCOL, "request_id": request_id,
                 "call_id": "fc-" + variant, "identity": copy.deepcopy(result["identity"]),
                 "messages_sha256": ui.digest(messages)}
        if self.guarded:
            start["loop_guard"] = self.demo.LOOP_POLICY
        yield start
        if output:
            yield {"type": "delta", "request_id": request_id, "index": 0, "text": output}
        if change == "missing_final":
            return
        if change == "disconnect":
            raise ConnectionError("Transport disconnected")
        if change == "different_final_text":
            result["output"] = "Not the text that was streamed"
        if change == "wait":
            await asyncio.Event().wait()
        if isinstance(change, dict):
            result.update(change)
        if result["finish_reason"] == "max_new_tokens":
            result["output_ids"] = [1] * MANIFEST["decoding"]["max_new_tokens"]
        yield {"type": "final", "request_id": request_id, "result": result}



class ComparisonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name).resolve() / "r7.sqlite3"
        self.demo = ui.load_helpers()
        self.store = ui.Store(self.path, SHA)
        self.workbench = ui.Workbench(copy.deepcopy(MANIFEST),
            types.SimpleNamespace(validate_messages=foundation.validate_messages), SHA, self.store, self.demo)
        self.transport = FakeTransport(self.demo)

    async def collect(self, state=None, question="  Hello\n\u025b  ", **kwargs):
        return [s async for s in self.workbench.compare(state or self.workbench.fresh(), question,
            remote=self.transport, cancel=self.transport.cancel, **kwargs)]

    async def test_r7_streams_first_and_each_model_only_receives_its_own_history(self):
        state = self.workbench.fresh()
        for question in ("  First\n", "Follow up  ", "New topic"):
            prior = copy.deepcopy(state)
            snapshots = await self.collect(state, question)
            state = snapshots[-1]
            self.assertTrue(state["rounds"][-1]["verified"])
            for variant, messages in self.transport.calls[-2:]:
                expected = [m for item in prior["rounds"] for m in (
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": variant + " answer"})]
                self.assertEqual(messages, expected + [{"role": "user", "content": question}])
            partial = next(s for s in snapshots if
                (s["rounds"][-1]["sides"]["r7_response"]["turn"] or {}).get("answer"))
            self.assertEqual(partial["rounds"][-1]["sides"]["r6_response"]["status"], "queued")
        self.assertEqual([v for v, _ in self.transport.calls], ["r7_response", "r6_response"] * 3)
        self.assertEqual(set(state["model_sessions"]), set(ui.VARIANTS))

    async def test_refresh_preserves_text_mode_reviews_and_browser_secret(self):
        state = (await self.collect())[-1]
        self.store.review(state["id"], state["rounds"][0]["id"], "r7_response", "Retain verbatim")
        reopened = ui.Store(self.path, SHA)
        self.assertEqual(reopened.load(state["id"]), state)
        self.assertEqual(reopened.browser_secret, self.store.browser_secret)
        self.assertEqual(reopened.reviews(state["id"])[0]["note"], "Retain verbatim")
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    async def test_raw_and_guarded_use_same_decoding_and_exact_input(self):
        for guarded in (False, True):
            self.transport = FakeTransport(self.demo, guarded=guarded)
            state = self.workbench.fresh(loop_guard=guarded)
            final = (await self.collect(state))[-1]
            self.assertTrue(final["rounds"][-1]["verified"])
            for side in final["rounds"][-1]["sides"].values():
                self.assertEqual(side["turn"]["remote_result"]["decoding"], foundation.DECODING)
                self.assertIs(side["turn"].get("loop_guard", False), guarded)
            self.assertEqual(self.transport.calls[0][1], self.transport.calls[1][1])
        changed = copy.deepcopy(final)
        changed["loop_guard"] = not final["loop_guard"]
        with self.assertRaisesRegex(ValueError, "Policy changed"):
            self.store.save(changed)

    async def test_identity_decoding_history_and_stream_mismatches_fail_closed(self):
        for mutation in ({"identity": {}}, {"messages_sha256": "f" * 64}, {"decoding": {}},
                         {"prompt_ids_sha256": "f" * 64}, {"loop_guard": None},
                         "missing_final", "disconnect", "different_final_text"):
            with self.subTest(mutation=mutation):
                self.transport = FakeTransport(self.demo, changes={"r6_response": mutation})
                state = (await self.collect())[-1]
                self.assertFalse(state["rounds"][-1]["verified"])
                self.assertFalse(self.workbench.ready(state))
                count = len(self.transport.calls)
                with self.assertRaises(ValueError):
                    await self.collect(state)
                self.assertEqual(len(self.transport.calls), count)

    async def test_cross_history_mix_is_rejected_before_a_new_request(self):
        state = (await self.collect())[-1]
        state = (await self.collect(state, "Second"))[-1]
        item = state["rounds"][-1]
        item["input_messages"]["r7_response"] = copy.deepcopy(item["input_messages"]["r6_response"])
        self.store.save(state)
        with self.assertRaisesRegex(ValueError, "not independent"):
            await self.collect(state, "Third")
        self.assertEqual(len(self.transport.calls), 4)

    async def test_stop_during_r7_preserves_partial_and_never_submits_r6(self):
        self.transport = FakeTransport(self.demo, changes={"r7_response": "wait"})
        stop = threading.Event()
        async for state in self.workbench.compare(self.workbench.fresh(), "Hello", stop=stop,
                remote=self.transport, cancel=self.transport.cancel):
            if (state["rounds"][-1]["sides"]["r7_response"]["turn"] or {}).get("answer"):
                stop.set()
        self.assertEqual([v for v, _ in self.transport.calls], ["r7_response"])
        self.assertEqual(self.transport.cancelled, ["fc-r7_response"])
        self.assertEqual(state["rounds"][-1]["sides"]["r6_response"]["status"], "skipped")
        self.assertEqual(ui.transcript(state, "r7_response")[-1]["content"], "r7_response answer")

    async def test_explicit_recovery_preserves_evidence_and_only_excludes_bad_branch(self):
        self.transport = FakeTransport(self.demo, changes={"r6_response": "disconnect"})
        state = (await self.collect())[-1]
        original = copy.deepcopy(state)
        recovered = self.workbench.recover(state)
        self.assertEqual(recovered["rounds"], original["rounds"])
        self.assertEqual([e["variant"] for e in recovered["context_exclusions"]], ["r6_response"])
        self.transport.changes = {}
        final = (await self.collect(recovered, "Next"))[-1]
        self.assertTrue(final["rounds"][-1]["verified"])
        self.assertEqual(len(self.transport.calls[-2][1]), 3)
        self.assertEqual(self.transport.calls[-1][1], [{"role": "user", "content": "Next"}])

    async def test_ui_layout_order_report_and_controls(self):
        app = ui.build_app(self.workbench, remote=self.transport, cancel=self.transport.cancel,
                           benchmark_report={"status": "pending"})
        config = app.get_config_file()
        headings = [c["props"].get("value", "") for c in config["components"] if c["type"] == "markdown"]
        self.assertEqual([h for h in headings if h.startswith("### ")], ["### R7", "### R6"])
        self.assertEqual(len([c for c in config["components"] if c["type"] == "chatbot"]), 2)
        self.assertTrue(any(c["props"].get("label") == "Saved report" for c in config["components"]))
        self.assertIn("repeat(2,minmax(0,1fr))", ui.CSS)
        self.assertFalse(app.is_running)
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values()}
        prior = self.workbench.fresh()
        failed = [v async for v in callbacks["respond"]("invalid \ufffd", prior, "guarded")][-1]
        self.assertTrue(failed[7]["interactive"])
        self.assertTrue(failed[9]["interactive"])
        self.assertEqual(failed[0], prior)
        self.assertFalse(self.transport.calls)
        final = [v async for v in callbacks["respond"]("Hello", prior, "guarded")][-1]
        self.assertTrue(final[0]["rounds"][-1]["verified"])
        self.assertTrue(final[7]["interactive"])
        restored = callbacks["restore"](final[0]["id"])
        self.assertEqual(restored[0], final[0])

    def test_r6_store_is_not_opened_or_migrated(self):
        import understanding_r6_workbench as r6
        old_path = self.path.with_name("r6.sqlite3")
        old_store = r6.Store(old_path, SHA)
        old_state = r6.new_session(SHA)
        old_store.save(old_state)
        before = old_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "separate R7 store"):
            ui.Store(old_path, SHA)
        self.assertEqual(old_path.read_bytes(), before)
        self.assertEqual(old_store.load(old_state["id"]), old_state)
        self.assertEqual(r6.VARIANTS, ("r6_response", "r5_response"))

    def test_reports_can_arrive_later_without_blocking_chat(self):
        path = self.path.parent / "results"
        self.assertEqual(ui.load_report(path)["state"], "not_collected")
        path.mkdir()
        report = {"state": "completed", "benchmarks": {"afrimmlu": {"correct": 2, "total": 3}}}
        (path / "r7.json").write_text(json.dumps(report))
        self.assertEqual(ui.load_report(path)["r7"], report)
        self.assertEqual(ui.load_report(path)["r6"]["state"], "not_collected")
        (path / "r6.json").write_text("{partial")
        self.assertEqual(ui.load_report(path)["r6"]["state"], "unavailable")
        self.assertFalse(self.transport.calls)

    def test_report_refresh_updates_table_and_collapsed_evidence_without_inference(self):
        path = self.path.parent / "benchmarks"
        path.mkdir()
        remote = Mock(side_effect=AssertionError("Benchmark refresh must not invoke inference"))
        app = ui.build_app(self.workbench, remote=remote, benchmark_report=path)
        config = app.get_config_file()
        details = next(c for c in config["components"] if c["type"] == "accordion"
                       and c["props"].get("label") == "Full benchmark evidence")
        self.assertFalse(details["props"]["open"])
        refresh = next(f for f in app.fns.values() if f.fn.__name__ == "refresh_benchmarks")
        self.assertEqual(len(refresh.outputs), 2)
        initial, evidence = refresh.fn()
        self.assertIn("No task results collected", initial)
        report = {"state": "completed", "benchmarks": {"mmlu": {"correct": 3, "total": 4, "accuracy": .75}}}
        (path / "r7.json").write_text(json.dumps(report))
        table, evidence = refresh.fn()
        self.assertIn("3/4 (75.0%)", table)
        self.assertIn("Not collected", table)
        self.assertEqual(evidence["r7"], report)
        self.assertEqual(self.store.conversations(), [])
        remote.assert_not_called()

    def test_old_actor_and_extra_variants_rejected_base_remains_service_only(self):
        self.assertIn("base", MANIFEST["variants"])
        self.assertNotIn("base", ui.VARIANTS)
        for name in ("ghana-understanding-r4-demo-private", "ghana-understanding-r6-demo-private"):
            manifest = copy.deepcopy(MANIFEST)
            manifest["actor"]["app_name"] = name
            with self.assertRaises(ValueError):
                ui.manifest_contract(manifest)
        manifest = copy.deepcopy(MANIFEST)
        manifest["variants"]["r4_response"] = manifest["variants"]["r6_response"]
        with self.assertRaises(ValueError):
            ui.manifest_contract(manifest)


class BenchmarkTableTests(unittest.TestCase):
    def report(self):
        return {
            "r6": {"state": "completed", "benchmarks": {
                "sib200_twi": {"accuracy": .25, "total": 4, "correct": 1, "subjects": {"health": {}}},
                "afrimmlu_twi": {"total": 3, "correct": 1, "accuracy": 1 / 3}}},
            "base": {"state": "completed", "benchmarks": {"base_only": {"correct": 1, "total": 2}}},
            "r7": {"benchmarks": {
                "afrimmlu_twi": {"subjects": {"math": {}}, "correct": 2, "accuracy": 2 / 3, "total": 3},
                "sib200_twi": {"total": 4, "accuracy": 0, "correct": 0}}, "state": "completed"},
        }

    def test_named_fields_stable_task_order_percentages_and_signed_points(self):
        status, rows = ui.benchmark_rows(self.report())
        self.assertEqual(ui.BENCHMARK_HEADERS,
                         ("Task", "R7 correct/total (%)", "R6 correct/total (%)", "Difference (points)"))
        self.assertEqual(rows, [
            ["AfriMMLU (Twi)", "2/3 (66.7%)", "1/3 (33.3%)", "+33.3"],
            ["SIB-200 (Twi)", "0/4 (0.0%)", "1/4 (25.0%)", "-25.0"],
        ])
        self.assertIn("Unverified results", status)
        self.assertNotIn("base only", str(rows))
        self.assertNotIn("subjects", str(rows))

    def test_partial_tasks_and_pending_counts_never_appear_complete(self):
        report = self.report()
        report["r6"]["state"] = "running"
        report["r6"]["benchmarks"]["mmlu"] = {"correct": 10, "total": 10}
        status, rows = ui.benchmark_rows(report)
        self.assertIn("R6: Running", status)
        self.assertEqual(rows[0][2:], ["Running", "-"])
        self.assertEqual(rows[-1], ["MMLU", "Not available", "Running", "-"])
        report.pop("r6")
        self.assertEqual(ui.benchmark_rows(report)[1][0][2:], ["Pending", "-"])
        self.assertNotIn("100.0%", ui.benchmark_view(report))

    def test_equal_scores_and_unmatched_totals(self):
        report = self.report()
        report["r7"]["benchmarks"]["sib200_twi"]["correct"] = 1
        report["r7"]["benchmarks"]["sib200_twi"]["accuracy"] = .25
        self.assertEqual(ui.benchmark_rows(report)[1][1][-1], "0.0")
        report["r7"]["benchmarks"]["sib200_twi"] = {"correct": 2, "total": 8, "accuracy": .25}
        self.assertEqual(ui.benchmark_rows(report)[1][1][-1], "Not matched")

    def test_invalid_counts_accuracy_and_zero_total_are_not_scores(self):
        for result in ({"correct": True, "total": 3}, {"correct": 2, "total": 0},
                       {"correct": -1, "total": 3}, {"correct": 4, "total": 3},
                       {"correct": 1, "total": 3, "accuracy": .9},
                       {"correct": 1, "total": 3, "accuracy": float("nan")},
                       {"correct": "1", "total": 3}, None):
            with self.subTest(result=result):
                self.assertIsNone(ui.benchmark_score(result))
        self.assertEqual(ui.benchmark_score({"correct": 2, "total": 3, "accuracy": .6667})[:2], (2, 3))

    def test_present_summary_requires_literal_verified_marker_and_all_completed_states(self):
        report = self.report()
        for summary in ({}, {"verified": False}, {"verified": "true"}, {"verified": True, "state": "pending"}, []):
            report["verified_summary"] = summary
            status, rows = ui.benchmark_rows(report)
            self.assertIn("Verification pending", status)
            self.assertEqual(rows[0][1:], ["Awaiting verification", "Awaiting verification", "-"])
        report["verified_summary"] = {"verified": True}
        self.assertTrue(ui.benchmark_rows(report)[0].startswith("Verified results"))
        for variant in ("base", "r6", "r7"):
            report[variant]["state"] = "running"
            self.assertTrue(ui.benchmark_rows(report)[0].startswith("Verification pending"))
            report[variant]["state"] = "completed"

    def test_stale_summary_cannot_certify_changed_aggregates(self):
        report = self.report()
        report["verified_summary"] = {"verified": True, "manifest_sha256": "a" * 64}
        self.assertTrue(ui.benchmark_rows(report)[0].startswith("Verification pending"))
        for variant in ("base", "r6", "r7"):
            report[variant]["manifest_sha256"] = "a" * 64
            report[variant]["benchmarks"] = {"mmlu": {"correct": 1, "total": 2}}
        report["verified_summary"]["tasks"] = {"mmlu": {
            variant: {"correct": 1, "total": 2} for variant in ("base", "r6", "r7")}}
        self.assertTrue(ui.benchmark_rows(report)[0].startswith("Verified results"))
        report["r7"]["benchmarks"]["mmlu"]["correct"] = 2
        self.assertTrue(ui.benchmark_rows(report)[0].startswith("Verification pending"))

    def test_summary_file_and_directory_loading_keep_all_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = self.report()
            for variant, data in report.items():
                (path / (variant + ".json")).write_text(json.dumps(data))
            summary = {"verified": True, "run_id": "test-run"}
            (path / "verified-summary.json").write_text(json.dumps(summary))
            report["verified_summary"] = summary
            self.assertEqual(ui.load_report(path), report)
            self.assertEqual(ui.load_report(path / "verified-summary.json"), report)
            (path / "verified-summary.json").write_text("{partial")
            self.assertTrue(ui.benchmark_rows(ui.load_report(path))[0].startswith("Verification pending"))

    def test_table_escapes_task_names_and_hides_hashes_and_base_scores(self):
        report = self.report()
        report["r7"]["benchmarks"]["<script>alert(1)</script>"] = {"correct": 1, "total": 1}
        report["r7"]["manifest_sha256"] = "b" * 64
        output = ui.benchmark_view(report)
        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn("<script>", output)
        self.assertNotIn("b" * 64, output)
        self.assertNotIn("base only", output)
        self.assertNotIn("1/2 (50.0%)", output)
        self.assertEqual(output.count('scope="col"'), 4)
        self.assertIn('aria-label="R7 versus R6 benchmark results"', output)


if __name__ == "__main__":
    unittest.main()
