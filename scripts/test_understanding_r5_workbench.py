"""Synthetic R5 CPU-only tests. Frozen audition, fake transport; no server or GPU calls."""
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
import understanding_r5_workbench as ui

SHA = "a" * 64
MANIFEST = {"version": ui.VERSION, "model": foundation.MODEL, "revision": foundation.REVISION,
            "base_sha256": foundation.BASE_SHA, "decoding": foundation.DECODING,
            "private": True, "training": False,
            "actor": {"app_name": "ghana-understanding-r5-demo-private", "class_name": "UnderstandingR5Demo"},
            "variants": {"base": None, **{v: {"directory": "/artifacts/" + v,
                "files": {"adapter_model.safetensors": c * 64}} for v, c in (("r4_response", "b"), ("r5_response", "c"))}}}


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
        self.path = Path(self.tmp.name).resolve() / "r5.sqlite3"
        self.demo = ui.load_helpers()
        self.validator = Mock(side_effect=foundation.validate_messages)
        self.module = types.SimpleNamespace(validate_messages=self.validator)
        self.store = ui.Store(self.path, SHA)
        self.workbench = ui.Workbench(copy.deepcopy(MANIFEST), self.module, SHA, self.store, self.demo)
        self.transport = FakeTransport(self.demo)

    def stream(self, state=None, question="  Hello\n\u025b  ", **kwargs):
        return self.workbench.compare(state or self.workbench.fresh(), question, remote=self.transport,
                                      cancel=self.transport.cancel, **kwargs)

    async def collect(self, **kwargs):
        return [s async for s in self.stream(**kwargs)]

    async def test_same_question_and_each_models_own_history_across_three_rounds(self):
        state = self.workbench.fresh()
        for question in ("  First\n", "Follow up  ", "New topic"):
            prior = copy.deepcopy(state)
            state = (await self.collect(state=state, question=question))[-1]
            self.assertTrue(state["rounds"][-1]["verified"])
            self.assertEqual(state["rounds"][-1]["status"], "complete")
            for v, actual in self.transport.calls[-3:]:
                expected = []
                for item in prior["rounds"]:
                    expected += [{"role": "user", "content": item["question"]},
                                 {"role": "assistant", "content": v + " answer"}]
                self.assertEqual(actual, expected + [{"role": "user", "content": question}])
                self.assertEqual(state["rounds"][-1]["input_messages"][v], actual)
        self.assertEqual([v for v, _ in self.transport.calls], list(ui.VARIANTS) * 3)
        self.assertEqual(self.transport.calls[0][1], self.transport.calls[1][1])
        self.assertEqual(self.transport.calls[1][1], self.transport.calls[2][1])
        self.assertGreater(self.validator.call_count, 9)
        self.assertNotIn("r5_response", self.demo.LABELS)

    async def test_snapshots_are_saved_before_submission_and_first_text_is_observed(self):
        stream = self.stream()
        first = await anext(stream)
        self.assertFalse(self.transport.calls)
        self.assertEqual(self.store.load(first["id"]), first)
        async for state in stream:
            self.assertEqual(self.store.load(state["id"]), state)
        for side in state["rounds"][-1]["sides"].values():
            self.assertGreaterEqual(side["ttft_seconds"], 0)
            self.assertGreaterEqual(side["wall_seconds"], side["ttft_seconds"])
            self.assertTrue(side["turn"]["loop_guard"])
            self.assertEqual(side["turn"]["remote_result"]["loop_guard"]["version"], self.demo.LOOP_POLICY)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        proof = ui.evidence(self.workbench, state)
        self.assertEqual(proof["input_messages_by_model"], state["rounds"][-1]["input_messages"])
        self.assertIn("not a fair latency benchmark", proof["timing_scope"])

    async def test_refresh_restores_exact_outputs_mode_reviews_and_browser_secret(self):
        state = (await self.collect())[-1]
        round_id = state["rounds"][0]["id"]
        for rating in ("base", "r4_response", "r5_response", "tie", "neither", "unsure"):
            self.store.review(state["id"], round_id, rating, "Human assessment")
        reopened = ui.Store(self.path, SHA)
        other = ui.Workbench(MANIFEST, self.module, SHA, reopened, self.demo)
        self.assertEqual(other.restore(state["id"]), state)
        self.assertEqual(reopened.browser_secret, self.store.browser_secret)
        self.assertEqual(len(reopened.reviews(state["id"])), 6)
        self.assertEqual(reopened.conversations()[0][1], state["id"])
        self.assertEqual(len(self.transport.calls), 3)
        with self.assertRaisesRegex(ValueError, "finished round"):
            reopened.review(state["id"], "foreign-round", "base", "")

    async def test_raw_mode_gets_new_session_and_never_changes_guarded_history(self):
        guarded = (await self.collect())[-1]
        raw = self.workbench.fresh(loop_guard=False)
        self.assertNotEqual(raw["id"], guarded["id"])
        self.assertEqual(raw["rounds"], [])
        self.transport = FakeTransport(self.demo, guarded=False)
        raw = (await self.collect(state=raw))[-1]
        for side in raw["rounds"][0]["sides"].values():
            self.assertFalse(side["turn"].get("loop_guard", False))
            self.assertNotIn("loop_guard", side["turn"]["remote_result"])
        self.assertEqual(self.store.load(guarded["id"]), guarded)
        changed = copy.deepcopy(guarded)
        changed["loop_guard"] = False
        with self.assertRaisesRegex(ValueError, "Policy changed"):
            self.store.save(changed)
        fresh = self.workbench.fresh()
        self.assertTrue(all(ui.transcript(fresh, v) == [] for v in ui.VARIANTS))

    async def test_routing_passes_identical_guard_to_all_variants(self):
        for enabled in (True, False):
            transport = FakeTransport(self.demo, guarded=enabled)
            with patch.object(self.demo, "routed_events", return_value=transport) as route:
                async for state in self.workbench.compare(self.workbench.fresh(loop_guard=enabled), "Hello",
                                                          cancel=transport.cancel):
                    pass
            route.assert_called_once_with(self.workbench.manifest, loop_guard=enabled)
            self.assertEqual([v for v, _ in transport.calls], list(ui.VARIANTS))
            self.assertTrue(state["rounds"][-1]["verified"])

    async def test_transport_routes_only_to_the_private_r5_actor(self):
        for guarded in (False, True):
            transport = FakeTransport(self.demo, guarded=guarded)
            stream = Mock(side_effect=transport)
            unused = Mock(side_effect=AssertionError("Wrong response mode"))
            actor = types.SimpleNamespace(**{name: types.SimpleNamespace(remote_gen=types.SimpleNamespace(aio=fn))
                for name, fn in (("stream_guarded", stream if guarded else unused),
                                 ("stream", unused if guarded else stream))})
            lookup = Mock(return_value=Mock(return_value=actor))
            with patch.dict(sys.modules, {"modal": types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=lookup))}):
                async for state in self.workbench.compare(self.workbench.fresh(loop_guard=guarded),
                                                          "Synthetic route test", cancel=transport.cancel):
                    pass
            self.assertEqual(lookup.call_count, 3)
            self.assertTrue(all(call.args == ("ghana-understanding-r5-demo-private", "UnderstandingR5Demo")
                                for call in lookup.call_args_list))
            self.assertEqual([v for v, _ in transport.calls], ["base", "r4_response", "r5_response"])
            self.assertTrue(state["rounds"][-1]["verified"])
            unused.assert_not_called()

    async def test_guard_missing_on_any_variant_rejects_evidence(self):
        for variant in ui.VARIANTS:
            self.transport = FakeTransport(self.demo, changes={variant: {"loop_guard": None}})
            state = (await self.collect())[-1]
            self.assertFalse(state["rounds"][-1]["verified"])
            self.assertEqual(state["rounds"][-1]["status"], "error")
            self.assertFalse(self.workbench.ready(state))

    async def test_bad_identity_history_decoding_prompt_and_final_output_are_rejected(self):
        for mutation in ({"identity": {}}, {"messages_sha256": "f" * 64}, {"decoding": {}},
                         {"prompt_ids_sha256": "f" * 64}, {"output_ids": []}, {"complete": False},
                         "missing_final", "disconnect", "different_final_text"):
            self.transport = FakeTransport(self.demo, changes={"r5_response": mutation})
            state = (await self.collect())[-1]
            self.assertFalse(self.workbench.ready(state), mutation)
            if mutation != {"complete": False}:
                self.assertFalse(state["rounds"][-1]["verified"], mutation)
            count = len(self.transport.calls)
            with self.assertRaisesRegex(ValueError, "Unfinished|Native prompt hashes differ"):
                await self.collect(state=state, question="Cannot continue")
            self.assertEqual(len(self.transport.calls), count)

    async def test_saved_complete_reply_evidence_is_revalidated(self):
        state = (await self.collect())[-1]
        state["rounds"][0]["sides"]["base"]["turn"]["remote_result"]["identity"] = {}
        self.store.save(state)
        before = len(self.transport.calls)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            await self.collect(state=state)
        self.assertEqual(len(self.transport.calls), before)

    async def test_cross_model_history_contamination_rejected_before_calls(self):
        state = (await self.collect())[-1]
        state = (await self.collect(state=state, question="Second"))[-1]
        item = state["rounds"][-1]
        item["input_messages"]["base"] = copy.deepcopy(item["input_messages"]["r5_response"])
        self.store.save(state)
        before = len(self.transport.calls)
        with self.assertRaisesRegex(ValueError, "not independent"):
            await self.collect(state=state, question="Third")
        self.assertEqual(len(self.transport.calls), before)

    async def test_distinct_histories_permit_distinct_native_prompt_hashes(self):
        state = (await self.collect())[-1]
        self.transport = FakeTransport(self.demo, changes={"r5_response": {"prompt_ids_sha256": "f" * 64}})
        state = (await self.collect(state=state, question="Next"))[-1]
        self.assertTrue(state["rounds"][-1]["verified"])
        self.assertTrue(self.workbench.ready(state))

    async def test_token_limit_and_loop_pause_can_follow_up_without_retry(self):
        output = "One two three four five six seven eight nine. " * 4
        changes = [{"complete": False, "finish_reason": "max_new_tokens"},
                   {"complete": False, "finish_reason": "time_limit"},
                   {"output": output, "complete": False, "finish_reason": "cancelled",
                    "loop_guard": {"version": self.demo.LOOP_POLICY, "trigger": self.demo.repeated_passage(output)}}]
        for change in changes:
            self.transport = FakeTransport(self.demo, changes={"r5_response": change})
            state = (await self.collect())[-1]
            original = copy.deepcopy(state["rounds"][0])
            self.assertTrue(self.workbench.ready(state))
            self.assertNotEqual(original["status"], "complete")
            state = (await self.collect(state=state, question="Please continue"))[-1]
            self.assertEqual(state["rounds"][0], original)
            self.assertEqual(len(self.transport.calls), 6)
            self.assertEqual(self.transport.calls[-1][1][-2]["content"], change.get("output", "r5_response answer"))

    async def test_empty_prompt_has_no_calls_and_empty_reply_has_no_fake_timing(self):
        for question in ("", " \n ", None):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                await self.collect(question=question)
        self.assertFalse(self.transport.calls)
        self.transport = FakeTransport(self.demo, changes={"base": {"output": "", "complete": False}})
        state = (await self.collect())[-1]
        side = state["rounds"][-1]["sides"]["base"]
        self.assertTrue(side["verified"])
        self.assertIsNone(side["ttft_seconds"])
        self.assertIn("Empty reply", ui.side_status(side))
        self.assertNotIn("First text", ui.side_status(side))
        self.assertFalse(self.workbench.ready(state))

    async def test_unexpected_stop_and_unguarded_cancellation_block_followups(self):
        for reason in ("unexpected_stop", "cancelled"):
            self.transport = FakeTransport(self.demo, changes={"r5_response": {"complete": False, "finish_reason": reason}})
            state = (await self.collect())[-1]
            self.assertFalse(self.workbench.ready(state))
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                await self.collect(state=state)

    async def test_same_histories_in_two_panels_still_require_matching_prompt_proof(self):
        self.transport = FakeTransport(self.demo, changes={"base": {"output": "shared answer"},
                                                          "r4_response": {"output": "shared answer"}})
        state = (await self.collect())[-1]
        self.transport = FakeTransport(self.demo, changes={"r4_response": {"prompt_ids_sha256": "f" * 64}})
        state = (await self.collect(state=state, question="Second"))[-1]
        self.assertEqual(state["rounds"][-1]["status"], "error")
        self.assertFalse(state["rounds"][-1]["verified"])

    async def test_stop_before_submission_skips_all_three(self):
        stop = threading.Event()
        stop.set()
        state = (await self.collect(stop=stop))[-1]
        self.assertFalse(self.transport.calls)
        self.assertEqual([s["status"] for s in state["rounds"][0]["sides"].values()], ["skipped"] * 3)
        self.assertFalse(self.workbench.ready(state))

    async def test_stop_during_stream_cancels_and_skips_remaining_models(self):
        self.transport = FakeTransport(self.demo, changes={"base": "wait"})
        stop = threading.Event()
        async for state in self.stream(stop=stop):
            turn = state["rounds"][-1]["sides"]["base"]["turn"]
            if turn and turn["answer"]:
                stop.set()
        self.assertEqual([v for v, _ in self.transport.calls], ["base"])
        self.assertEqual(self.transport.cancelled, ["fc-base"])
        self.assertEqual(state["rounds"][0]["status"], "cancelled")
        self.assertEqual(state["rounds"][0]["sides"]["r5_response"]["status"], "skipped")
        self.assertEqual(ui.transcript(state, "base")[-1]["content"], "base answer")

    async def test_stop_between_variants_prevents_later_submissions(self):
        stop = threading.Event()
        async for state in self.stream(stop=stop):
            if state["rounds"][-1]["sides"]["r4_response"]["status"] == "complete":
                stop.set()
        self.assertEqual([v for v, _ in self.transport.calls], ["base", "r4_response"])
        self.assertEqual(state["rounds"][-1]["sides"]["r5_response"]["status"], "skipped")

    async def test_disconnect_refresh_and_stale_state_never_retry(self):
        stream = self.stream()
        pending = await anext(stream)
        await stream.aclose()
        restored = self.workbench.restore(pending["id"])
        self.assertEqual(restored["rounds"][0]["status"], "cancelled")
        for state in (pending, restored):
            with self.assertRaises(ValueError):
                await self.collect(state=state)
        self.assertFalse(self.transport.calls)

    async def test_disconnect_during_stream_preserves_partial_and_cancels_remote(self):
        self.transport = FakeTransport(self.demo, changes={"base": "wait"})
        stream = self.stream()
        async for state in stream:
            if (state["rounds"][-1]["sides"]["base"]["turn"] or {}).get("answer"):
                break
        await stream.aclose()
        restored = self.workbench.restore(state["id"])
        self.assertFalse(self.workbench.ready(restored))
        self.assertEqual(self.transport.cancelled, ["fc-base"])
        self.assertEqual(ui.transcript(restored, "base")[-1]["content"], "base answer")

    async def test_crashed_pending_record_remains_blocked_after_refresh(self):
        stream = self.stream()
        state = await anext(stream)
        self.assertFalse(self.workbench.ready(self.workbench.restore(state["id"])))
        self.assertEqual(len(self.transport.calls), 0)
        await stream.aclose()

    async def test_dropped_model_history_fails_before_remote_submission(self):
        state = (await self.collect())[-1]
        before = len(self.transport.calls)

        def broken_audition(session, *args, **kwargs):
            session["turns"] = []
            return self.demo.audition(session, *args, **kwargs)

        state = (await self.collect(state=state, question="Follow up", audition_fn=broken_audition))[-1]
        self.assertEqual(len(self.transport.calls), before)
        self.assertEqual(state["rounds"][-1]["status"], "error")

    async def test_unicode_preflight_preserves_input_and_never_submits_or_cancels(self):
        state = self.workbench.fresh()
        for char in ("\ufffd", "\ud800", "\udfff", "\x00", "\x7f", "\ue000"):
            with self.subTest(char=ascii(char)):
                with self.assertRaisesRegex(ValueError, "Invalid Unicode"):
                    await self.collect(state=state, question="Hello " + char)
                self.assertEqual(self.store.load(state["id"]), state)
        self.assertFalse(self.transport.calls)
        self.assertFalse(self.transport.cancelled)
        state = (await self.collect(state=state, question="\u025b\u0254\n\tHello\r"))[-1]
        self.assertTrue(self.workbench.ready(state))

    async def test_invalid_output_requires_explicit_branch_only_recovery(self):
        self.transport.changes = {"r4_response": {"output": "Synthetic reply \u025by\ufffd",
            "complete": False, "finish_reason": "max_new_tokens"}}
        state = (await self.collect())[-1]
        original = copy.deepcopy(state)
        self.assertTrue(state["rounds"][0]["verified"])
        self.assertFalse(self.workbench.ready(state))
        self.assertEqual(self.workbench.recovery_plan(state)[0]["variant"], "r4_response")
        self.assertEqual(state, original)
        with self.assertRaisesRegex(ValueError, "exclude unusable"):
            await self.collect(state=state)
        recovered = self.workbench.recover(state)
        self.assertEqual(recovered["rounds"], original["rounds"])
        self.assertTrue(self.workbench.ready(recovered))
        self.assertEqual(len(self.transport.calls), 3)
        self.transport.changes = {}
        next_state = (await self.collect(state=recovered, question="Next topic"))[-1]
        self.assertEqual([len(m) for _, m in self.transport.calls[-3:]], [3, 1, 3])
        self.assertTrue(next_state["rounds"][-1]["verified"])
        self.assertEqual(next_state["rounds"][-1]["context_exclusions"], recovered["context_exclusions"])
        self.assertEqual(ui.transcript(next_state, "r4_response")[1]["content"], "Synthetic reply \u025by\ufffd")
        self.assertIn("Excluded from future context: turn 1", ui.context_status(next_state, "r4_response", []))

    async def test_synthetic_sixth_round_unicode_failure_recovers_without_rewriting_evidence(self):
        self.transport = FakeTransport(self.demo, guarded=False)
        state = self.workbench.fresh(loop_guard=False)
        for question in ("Synthetic question one", "Synthetic question two",
                         "Synthetic question three", "Synthetic question four"):
            state = (await self.collect(state=state, question=question))[-1]
        self.transport.changes = {"r4_response": {"output": "Synthetic reply \u025by\ufffd",
            "complete": False, "finish_reason": "max_new_tokens"}}
        state = (await self.collect(state=state, question="Synthetic question five"))[-1]
        # Reproduce the old transport-only validation path, never relax production validation.
        histories = {v: self.workbench.context_messages(state["rounds"], v, []) for v in ui.VARIANTS}
        self.transport.changes = {"r4_response": "invalid_unicode_pre_start"}
        with patch.object(self.workbench, "histories", return_value=histories), \
                patch.object(self.workbench, "validate_messages", side_effect=foundation.validate_messages):
            state = (await self.collect(state=state, question="Synthetic question six"))[-1]
        for item in state["rounds"]:
            item.pop("context_exclusions")
        self.store.save(state)
        self.store.review(state["id"], state["rounds"][4]["id"], "neither", "Keep this evidence")
        reviews = self.store.reviews(state["id"])
        with self.store.connection() as db:
            saved_turns = db.execute("SELECT * FROM turns ORDER BY rowid").fetchall()
        original = copy.deepcopy(state)
        failed = state["rounds"][5]["sides"]["r4_response"]["turn"]
        self.assertNotIn("call_id", failed)
        self.assertEqual(failed["cancellation"], "unconfirmed: no call identity received")
        plan = self.workbench.recovery_plan(state)
        self.assertEqual([(e["round_id"], e["variant"]) for e in plan],
                         [(state["rounds"][i]["id"], "r4_response") for i in (4, 5)])
        restored_view = self.callbacks()["restore"](state["id"])
        self.assert_controls_available(restored_view)
        self.assertTrue(restored_view[-1]["interactive"])
        self.assertIn("Remote cancellation not confirmed", restored_view[6])
        recovered = self.workbench.recover(self.workbench.restore(state["id"]))
        self.assertEqual(recovered["rounds"], original["rounds"])
        self.assertEqual(recovered["model_sessions"], original["model_sessions"])
        self.assertEqual(self.store.reviews(state["id"]), reviews)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT * FROM turns ORDER BY rowid").fetchall(), saved_turns)
        self.assertEqual(len(self.transport.calls), 18)
        self.assertFalse(self.transport.cancelled)
        self.transport.changes = {}
        for question in ("New question", "Another follow-up"):
            previous = copy.deepcopy(recovered)
            recovered = (await self.collect(state=recovered, question=question))[-1]
            self.assertTrue(self.workbench.ready(recovered))
            self.assertEqual(recovered["rounds"][:6], original["rounds"])
            for variant, messages in self.transport.calls[-3:]:
                expected = self.workbench.context_messages(previous["rounds"], variant, plan)
                self.assertEqual(messages, expected + [{"role": "user", "content": question}])
        proof = ui.evidence(self.workbench, recovered)
        self.assertEqual(proof["context_exclusions"], plan)
        self.assertEqual(proof["round_context_exclusions"], plan)
        self.assertFalse(recovered["loop_guard"])
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT * FROM turns ORDER BY rowid LIMIT ?", (len(saved_turns),)).fetchall(), saved_turns)

    async def test_exclusions_are_append_only_scoped_and_audited(self):
        self.transport.changes = {"r4_response": "disconnect"}
        state = (await self.collect())[-1]
        recovered = self.workbench.recover(state)
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.workbench.recover(state)
        with self.assertRaisesRegex(ValueError, "No unusable"):
            self.workbench.recover(recovered)
        reverted = copy.deepcopy(recovered)
        reverted["context_exclusions"] = []
        with self.assertRaisesRegex(ValueError, "append-only"):
            self.store.save(reverted)
        for field, value in (("variant", "base"), ("round_id", "foreign-round"), ("reason", "changed")):
            tampered = copy.deepcopy(recovered)
            tampered["context_exclusions"][0][field] = value
            self.assertFalse(self.workbench.ready(tampered))
        self.transport.changes = {"base": "disconnect"}
        state = (await self.collect(state=recovered, question="Second"))[-1]
        updated = self.workbench.recover(state)
        self.assertEqual(updated["context_exclusions"][:1], recovered["context_exclusions"])
        self.assertEqual(updated["context_exclusions"][1]["variant"], "base")
        self.assertTrue(self.workbench.ready(updated))
        self.transport.changes = {}
        state = (await self.collect(state=updated, question="Third"))[-1]
        state["rounds"][-1]["context_exclusions"] = []
        self.assertFalse(self.workbench.ready(state))

    async def test_recovery_cannot_bypass_conflicting_prompt_evidence(self):
        self.transport.changes = {"r4_response": {"prompt_ids_sha256": "f" * 64}}
        state = (await self.collect())[-1]
        with self.assertRaisesRegex(ValueError, "Native prompt hashes"):
            self.workbench.recover(state)
        self.assertEqual(self.store.load(state["id"]), state)

    async def test_concurrent_compare_and_recovery_never_duplicate_requests(self):
        prior = self.workbench.fresh()
        first = self.stream(state=prior)
        pending = await anext(first)
        with self.assertRaisesRegex(ValueError, "Unfinished"):
            self.workbench.recover(pending)
        with self.assertRaisesRegex(ValueError, "Stale"):
            await anext(self.stream(state=prior))
        await first.aclose()
        terminal = self.store.load(prior["id"])
        self.assertEqual(terminal["rounds"][-1]["status"], "cancelled")
        recovered = self.workbench.recover(terminal)
        self.assertEqual(len(recovered["context_exclusions"]), 3)
        self.assertTrue(self.workbench.ready(recovered))
        self.assertFalse(self.transport.calls)

    async def test_iterator_cleanup_failure_is_terminal_and_recoverable(self):
        class BrokenIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                raise RuntimeError("cleanup failure")

        state = (await self.collect(audition_fn=lambda *args, **kwargs: BrokenIterator()))[-1]
        self.assertEqual(state["rounds"][-1]["status"], "error")
        self.assertTrue(all("Stream cleanup failed" in side["error"] for side in state["rounds"][-1]["sides"].values()))
        self.assertTrue(self.workbench.ready(self.workbench.recover(state)))
        self.assertFalse(self.transport.calls)

    def callbacks(self):
        app = ui.build_app(self.workbench, remote=self.transport, cancel=self.transport.cancel)
        return {f.fn.__name__: f.fn for f in app.fns.values()}

    def assert_controls_available(self, view):
        self.assertTrue(view[9]["interactive"], "Question must not stay disabled")
        self.assertTrue(view[11]["interactive"], "New chat must not stay disabled")
        self.assertTrue(view[12]["interactive"], "Saved conversations must remain accessible")

    async def test_ui_preflight_and_terminal_errors_unlock_controls_and_recover(self):
        callbacks = self.callbacks()
        prior = self.workbench.fresh()
        for question in ("invalid \ufffd", "invalid \ud800"):
            failed = [v async for v in callbacks["respond"](question, prior, "guarded")][-1]
            self.assert_controls_available(failed)
            self.assertNotIn("value", failed[9])  # Leave the original client-side input untouched.
            self.assertEqual(failed[0], prior)
            self.assertIn("Invalid Unicode", failed[8])
            self.assertIn("This message was not submitted", failed[8])
            self.assertNotIn("Remote cancellation", str(failed))
            json.dumps(failed, ensure_ascii=False).encode("utf-8")
        self.assertFalse(self.transport.calls)
        self.transport.changes = {"r4_response": "disconnect"}
        failed = [v async for v in callbacks["respond"]("Hello", prior, "guarded")][-1]
        self.assert_controls_available(failed)
        self.assertTrue(failed[-1]["interactive"])
        self.assertIn("Needs context exclusion", failed[6])
        recovered = callbacks["recover"](failed[0])
        self.assert_controls_available(recovered)
        self.assertFalse(recovered[-1]["interactive"])
        self.assertIn("Excluded from future context", recovered[6])
        self.transport.changes = {}
        final = [v async for v in callbacks["respond"]("Next", recovered[0], "guarded")][-1]
        self.assert_controls_available(final)
        self.assertTrue(final[0]["rounds"][-1]["verified"])
        fresh = callbacks["new_chat"]("guarded")
        self.assert_controls_available(fresh)
        self.assertEqual(fresh[0]["rounds"], [])

    async def test_ui_duplicate_send_does_not_release_the_active_request(self):
        callbacks = self.callbacks()
        prior = self.workbench.fresh()
        first = callbacks["respond"]("First", prior, "guarded")
        pending = await anext(first)
        self.assertFalse(pending[9]["interactive"])
        duplicate = [v async for v in callbacks["respond"]("Duplicate", prior, "guarded")][-1]
        self.assertIn("already active", duplicate[8])
        self.assertFalse(duplicate[9]["interactive"])
        self.assertFalse(duplicate[11]["interactive"])
        recovery = callbacks["recover"](pending[0])
        self.assertIn("already active", recovery[8])
        callbacks["stop"](pending[0])
        final = [v async for v in first][-1]
        self.assert_controls_available(final)
        self.assertFalse(self.transport.calls)
        self.assertEqual(len(final[0]["rounds"]), 1)
        stale = [v async for v in callbacks["respond"]("Stale", prior, "guarded")][-1]
        self.assert_controls_available(stale)
        self.assertIn("Stale", stale[8])
        mismatched = [v async for v in callbacks["respond"]("Wrong mode", final[0], "raw")][-1]
        self.assert_controls_available(mismatched)
        self.assertIn("Mode changed", mismatched[8])

    async def test_ui_disconnect_and_render_exception_release_controls(self):
        callbacks = self.callbacks()
        prior = self.workbench.fresh()
        stream = callbacks["respond"]("First", prior, "guarded")
        await anext(stream)
        await stream.aclose()
        restored = callbacks["restore"](prior["id"])
        self.assert_controls_available(restored)
        self.assertEqual(restored[0]["rounds"][-1]["status"], "cancelled")
        with patch.object(self.store, "conversations", side_effect=[RuntimeError("render failure"), []]):
            result = [v async for v in callbacks["respond"]("Second", self.workbench.fresh(), "guarded")][-1]
        self.assert_controls_available(result)
        self.assertIn("render failure", result[8])
        self.assertFalse(self.transport.calls)

    async def test_ui_cleanup_exception_still_emits_unlocked_final_view(self):
        callbacks = self.callbacks()
        prior = self.workbench.fresh()

        class BrokenCompare:
            def __init__(self):
                self.first = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.first:
                    self.first = False
                    return prior
                raise RuntimeError("terminal failure")

            async def aclose(self):
                raise RuntimeError("cleanup failure")

        with patch.object(self.workbench, "compare", return_value=BrokenCompare()):
            views = [v async for v in callbacks["respond"]("Question", prior, "guarded")]
        self.assertFalse(views[0][9]["interactive"])
        self.assert_controls_available(views[-1])
        self.assertIn("cleanup failure", views[-1][8])
        self.assertFalse(self.transport.calls)

    async def test_ui_config_and_callbacks_without_server_or_remote_startup(self):
        remote = Mock(side_effect=AssertionError("No startup inference"))
        app = ui.build_app(self.workbench, remote=remote, cancel=self.transport.cancel)
        config = app.get_config_file()
        self.assertEqual(config["title"], "Twi R5 | Base, R4, R5")
        browser = next(c for c in config["components"] if c["type"] == "browserstate")
        self.assertTrue(browser["props"]["storage_key"].startswith("r5-workbench-"))
        labels = [c["props"]["value"] for c in config["components"] if c["type"] == "markdown"]
        self.assertEqual([label for label in labels if label.startswith("### ")], ["### Base", "### R4", "### R5"])
        self.assertEqual(len([c for c in config["components"] if c["type"] == "chatbot"]), 3)
        self.assertEqual(len([c for c in config["components"] if c["type"] == "browserstate"]), 1)
        checkpoints = [c["props"]["value"] for c in config["components"]
                       if "r5-checkpoint" in c["props"].get("elem_classes", [])]
        self.assertEqual(checkpoints, [
            "AfriqueQwen 9B, adapter off\n\n`" + MANIFEST["base_sha256"][:12] + "`",
            "R4 response stage\n\n`" + "b" * 12 + "`",
            "R5 continuation\n\n`" + "c" * 12 + "`"])
        self.assertFalse(app.is_running)
        self.assertIn('.r5-chat button[aria-label="Clear"] { display:none !important; }', ui.CSS)
        self.assertIn('grid-template-columns:repeat(3,minmax(0,1fr))', ui.CSS)
        self.assertIn('width:100% !important; max-width:1560px', ui.CSS)
        restore = next(f.fn for f in app.fns.values() if f.fn.__name__ == "restore")
        first = restore(None)
        second = restore(first[0]["id"])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2:5], ([], [], []))
        proof = ui.evidence(self.workbench, first[0])
        self.assertEqual(proof["identities"]["base"]["base_weights_sha256"], MANIFEST["base_sha256"])
        self.assertEqual(proof["identities"]["r4_response"]["adapter_weights_sha256"], "b" * 64)
        self.assertEqual(proof["identities"]["r5_response"]["adapter_weights_sha256"], "c" * 64)
        self.assertEqual(len(first), len(next(d for d in config["dependencies"] if any(e == "load" for _, e in d["targets"]))["outputs"]))
        remote.assert_not_called()
        self.assertNotIn("First text", ui.side_status(None))
        self.assertNotIn("<script>", ui.side_status({"status": "error", "verified": False, "error": "<script>bad</script>"}))


class LoaderAndStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def fixture_bundle(self):
        folder = self.root / "bundle"
        (folder / "code").mkdir(parents=True)
        runtime = b'''from pathlib import Path
import json
def validate_messages(messages):
    if not messages:
        raise ValueError("bound validator")
def load_bundle(folder, expected_sha):
    return json.loads((Path(folder) / "manifest.json").read_text())
class Runtime:
    def __init__(self, *args):
        raise AssertionError("No local Runtime construction")
'''
        sources = {"code/r5_inference.py": runtime, "code/r2_inference.py": b'BOUND_SIBLING = True\n'}
        manifest = {**copy.deepcopy(MANIFEST), "files": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}}
        for name, data in sources.items():
            (folder / name).write_bytes(data)
        data = json.dumps(manifest).encode()
        (folder / "manifest.json").write_bytes(data)
        return folder, hashlib.sha256(data).hexdigest(), manifest

    def test_r5_bundle_import_is_explicit_and_validator_bound_without_runtime(self):
        folder, checksum, manifest = self.fixture_bundle()
        before = list(sys.path)
        with patch.dict(sys.modules, {"r5_inference": types.SimpleNamespace(validate_messages=Mock())}):
            loaded, module = ui.bound_bundle(folder, checksum)
        self.assertEqual(loaded, manifest)
        self.assertEqual(module.__file__, str(folder / "code/r5_inference.py"))
        with self.assertRaisesRegex(ValueError, "bound validator"):
            module.validate_messages([])
        self.assertEqual(sys.path, before)
        self.assertFalse((folder / "code/__pycache__").exists())

    def test_manifest_code_sibling_and_path_hash_mismatches_rejected(self):
        folder, checksum, _ = self.fixture_bundle()
        with self.assertRaisesRegex(ValueError, "hash differs"):
            ui.bound_bundle(folder, "f" * 64)
        for name in ("r5_inference.py", "r2_inference.py"):
            path = folder / "code" / name
            original = path.read_bytes()
            path.write_text('raise AssertionError("Must not execute")\n')
            with self.assertRaisesRegex(ValueError, "hash differs"):
                ui.bound_bundle(folder, checksum)
            path.write_bytes(original)
        path = folder / "code/r2_inference.py"
        original = path.read_bytes()
        path.unlink()
        target = self.root / "elsewhere.py"
        target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Linked bundle artifact|Unsafe artifact path"):
            ui.bound_bundle(folder, checksum)

    def test_old_version_wrong_actor_and_missing_variant_rejected(self):
        for mutate in (lambda m: m.update(version="r4-comparison-inference-v1"),
                       lambda m: m["variants"].pop("r5_response"),
                       lambda m: m["actor"].update(class_name="UnderstandingR4Demo")):
            manifest = copy.deepcopy(MANIFEST)
            mutate(manifest)
            with self.assertRaises(ValueError):
                ui.manifest_contract(manifest)
        optional = copy.deepcopy(MANIFEST)
        optional["variants"]["r5_pass1"] = optional["variants"]["r5_response"]
        ui.manifest_contract(optional)

    def test_older_actors_and_r3_variant_are_rejected(self):
        for name in ("ghana-understanding-r2-demo-private", "ghana-understanding-r3-demo-private",
                     "ghana-understanding-r4-demo-private"):
            manifest = copy.deepcopy(MANIFEST)
            manifest["actor"]["app_name"] = name
            with self.assertRaisesRegex(ValueError, "separate R5 actor"):
                ui.manifest_contract(manifest)
        for variant in ("r3_response", "r4_pass1"):
            manifest = copy.deepcopy(MANIFEST)
            manifest["variants"][variant] = manifest["variants"]["r4_response"]
            with self.assertRaisesRegex(ValueError, "checkpoint variants"):
                ui.manifest_contract(manifest)

    def test_frozen_helpers_do_not_rebind_global_imports_or_labels(self):
        before = list(sys.path)
        poison = types.SimpleNamespace(VERSION="poison", repeated_passage=Mock(), validate_guard=Mock())
        with patch.dict(sys.modules, {"response_loop_guard": poison}):
            demo = ui.load_helpers()
            self.assertIs(sys.modules["response_loop_guard"], poison)
        self.assertEqual(demo.LOOP_POLICY, "repeat-block-v1")
        self.assertNotIn("r5_response", demo.LABELS)
        self.assertEqual(sys.path, before)
        for name in ui.HELPER_HASHES:
            (self.root / name).write_bytes((ui.DEFAULT_HELPER_CODE / name).read_bytes())
        (self.root / "response_loop_guard.py").write_text("raise AssertionError('changed helper')\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            ui.load_helpers(self.root)

    def test_r4_sqlite_and_different_bundle_rejected_without_modifying_bytes(self):
        import understanding_r4_workbench as r4_ui
        prior = self.root / "r4.sqlite3"
        old_store = r4_ui.Store(prior, SHA)
        old_session = r4_ui.new_session(SHA)
        old_store.save(old_session)
        original = prior.read_bytes()
        mode = prior.stat().st_mode
        files = sorted(p.name for p in self.root.iterdir())
        with self.assertRaisesRegex(ValueError, "separate R5 store"):
            ui.Store(prior, SHA)
        self.assertEqual(prior.read_bytes(), original)
        self.assertEqual(prior.stat().st_mode, mode)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), files)
        self.assertEqual(old_store.load(old_session["id"]), old_session)
        r5 = self.root / "r5.sqlite3"
        ui.Store(r5, SHA)
        original = r5.read_bytes()
        with self.assertRaisesRegex(ValueError, "another bundle"):
            ui.Store(r5, "f" * 64)
        self.assertEqual(r5.read_bytes(), original)

    def test_r4_import_and_state_are_not_rebound_by_r5(self):
        import understanding_r4_workbench as r4_ui
        snapshot = (r4_ui.VERSION, r4_ui.STORE_VERSION, r4_ui.VARIANTS, copy.deepcopy(r4_ui.LABELS))
        state = ui.new_session(SHA)
        store = ui.Store(self.root / "r5.sqlite3", SHA)
        store.save(state)
        self.assertEqual(state["execution_order"], ["base", "r4_response", "r5_response"])
        self.assertEqual(set(state["model_sessions"]), set(ui.VARIANTS))
        self.assertEqual(snapshot, (r4_ui.VERSION, r4_ui.STORE_VERSION, r4_ui.VARIANTS, r4_ui.LABELS))
        self.assertEqual(r4_ui.VARIANTS, ("base", "r3_response", "r4_response"))
        with store.connection() as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {"r5_meta", "r5_sessions", "r5_reviews"})

    def test_binding_preflight_does_not_hide_uncheckpointed_r5_sessions(self):
        path = self.root / "r5.sqlite3"
        store = ui.Store(path, SHA)
        state = ui.new_session(SHA)
        with store.connection() as held:
            held.execute("SELECT value FROM r5_meta WHERE key='binding'").fetchone()
            store.save(state)
            self.assertGreater(Path(str(path) + "-wal").stat().st_size, 0)
            reopened = ui.Store(path, SHA)
            self.assertEqual(reopened.load(state["id"]), state)
            self.assertEqual(reopened.browser_secret, store.browser_secret)

    def test_chat_defaults_to_isolated_r5_store_and_port_without_starting_server(self):
        app = Mock()
        app.queue.return_value = app
        module = types.SimpleNamespace(validate_messages=foundation.validate_messages)
        helpers = ui.load_helpers()
        bundle = self.root / "bundle"
        with patch.object(ui, "ROOT", self.root), patch.object(ui, "bound_bundle", return_value=(MANIFEST, module)), \
                patch.object(ui, "load_helpers", return_value=helpers), patch.object(ui, "build_app", return_value=app) as build:
            ui.main(["chat", "--bundle", str(bundle), "--expected-sha", SHA,
                     "--helper-code", str(ui.DEFAULT_HELPER_CODE)])
        expected = self.root / "tmp/understanding-r5-workbench" / SHA / "workbench.sqlite3"
        self.assertEqual(build.call_args.args[0].store.path, expected)
        self.assertTrue(expected.is_file())
        self.assertFalse((self.root / "tmp/understanding-r4-workbench").exists())
        options = app.launch.call_args.kwargs
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertEqual(options["server_port"], 7869)
        self.assertFalse(options["share"])
        self.assertFalse(options["mcp_server"])
        self.assertIn(str(expected.parent), options["blocked_paths"])

    def test_verify_cli_has_no_remote_or_store_creation_and_chat_requires_bundle(self):
        folder, checksum, _ = self.fixture_bundle()
        store = self.root / "not-created.sqlite3"
        output = io.StringIO()
        with redirect_stdout(output):
            ui.main(["verify", "--bundle", str(folder), "--expected-sha", checksum, "--store", str(store),
                     "--helper-code", str(ui.DEFAULT_HELPER_CODE)])
        proof = json.loads(output.getvalue())
        self.assertFalse(proof["remote_called"])
        self.assertFalse(proof["server_started"])
        self.assertEqual([v["variant"] for v in proof["variants"]], list(ui.VARIANTS))
        self.assertFalse(store.exists())
        with patch.object(ui, "build_app", side_effect=AssertionError("No server startup")) as build:
            with self.assertRaises((ValueError, FileNotFoundError)):
                ui.main(["chat", "--bundle", str(self.root / "missing"), "--expected-sha", SHA])
            build.assert_not_called()


class SealedBundleTests(unittest.TestCase):
    def test_real_r5_contract_and_bundled_helpers_verify_without_inference(self):
        import prepare_understanding_r5_inference as builder
        import r5_inference as r5
        from test_prepare_understanding_r5_inference import fixture, HELPERS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent, parent_sha, readback, _ = fixture(root)
            result = builder.prepare(parent, parent_sha, root / "receipt.json", root / "terminal.json",
                                     readback, root / "bundle", helpers_root=HELPERS, include_pass1=True)
            bundle, checksum = root / "bundle", result["manifest_sha256"]
            before = {p.relative_to(bundle): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in bundle.rglob("*") if p.is_file()}
            manifest, module = ui.bound_bundle(bundle, checksum)
            self.assertEqual(module.__file__, str(bundle / "code/r5_inference.py"))
            self.assertEqual(manifest["variants"]["r4_response"]["files"]["adapter_model.safetensors"], r5.R4_WEIGHTS)
            self.assertEqual(ui.VARIANTS, r5.VARIANTS)
            output = io.StringIO()
            with redirect_stdout(output), patch.object(module, "Runtime", side_effect=AssertionError("No GPU")), \
                    patch.object(ui, "build_app", side_effect=AssertionError("No server")), \
                    patch.object(ui, "Store", side_effect=AssertionError("No store creation")), \
                    patch.dict(os.environ, {"GHA_R4_HELPER_CODE": str(root / "wrong-r4-helper")}):
                ui.main(["verify", "--bundle", str(bundle), "--expected-sha", checksum])
            proof = json.loads(output.getvalue())
            self.assertFalse(proof["remote_called"])
            self.assertFalse(proof["server_started"])
            self.assertEqual(proof["actor"], {"app_name": "ghana-understanding-r5-demo-private", "class_name": "UnderstandingR5Demo"})
            self.assertEqual([identity["variant"] for identity in proof["variants"]], ["base", "r4_response", "r5_response"])
            self.assertIn("r5_pass1", manifest["variants"])
            self.assertNotIn("r5_pass1", [identity["variant"] for identity in proof["variants"]])
            self.assertEqual(Path(proof["store"]), ui.ROOT / "tmp/understanding-r5-workbench" / checksum / "workbench.sqlite3")
            after = {p.relative_to(bundle): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in bundle.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse(list(bundle.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
