"""CPU-only comparison tests; fake audition and fake transport, never GPU calls."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

import r2_comparison as comparison
import r2_inference as inference
import understanding_r2_demo as demo

SHA = "a" * 64
MANIFEST = {"model": inference.MODEL, "revision": inference.REVISION, "base_sha256": inference.BASE_SHA,
            "decoding": inference.DECODING, "variants": {v: None if v == "base" else
            {"files": {"adapter_model.safetensors": "b" * 64}} for v in inference.VARIANTS}}


def result(messages, variant, output=None):
    return {"identity": demo.identity(MANIFEST, SHA, variant), "messages_sha256": demo.digest(messages),
            "prompt_ids_sha256": "c" * 64, "decoding": copy.deepcopy(inference.DECODING),
            "output": output or variant + " answer", "output_ids": [1, 2, 3], "seconds": .1,
            "finish_reason": "eos", "complete": True}


class FakeAudition:
    def __init__(self, changes=None):
        self.calls, self.changes, self.controls = [], changes or {}, []

    async def __call__(self, state, question, variant, manifest, expected_sha, validate_messages, store,
                       *, remote, cancel, stop):
        messages = [{"role": "user", "content": question}]
        validate_messages(messages)
        turn = {"id": "turn-" + variant, "session_id": state["id"], "variant": variant,
                "identity": demo.identity(manifest, expected_sha, variant), "messages": messages,
                "question": question, "answer": "", "status": "pending", "finish_reason": "pending"}
        state["turns"].append(turn)
        yield copy.deepcopy(state)
        self.controls.append((stop, remote, cancel))
        if stop.is_set():
            turn.update(status="cancelled", finish_reason="cancelled")
            yield copy.deepcopy(state)
            return
        self.calls.append((variant, copy.deepcopy(messages)))
        turn["answer"] = variant + " "
        await asyncio.sleep(0)
        yield copy.deepcopy(state)
        if stop.is_set():
            turn.update(status="cancelled", finish_reason="cancelled")
        elif self.changes.get(variant) == "error":
            raise RuntimeError("simulated partial failure")
        elif self.changes.get(variant) == "missing_final":
            return
        else:
            value = result(messages, variant)
            value.update(self.changes.get(variant, {}))
            if value["finish_reason"] == "max_new_tokens":
                value["output_ids"] = [1] * inference.DECODING["max_new_tokens"]
            turn.update(answer=value["output"], remote_result=value,
                        status="complete" if value["complete"] else "incomplete", finish_reason=value["finish_reason"])
        yield copy.deepcopy(state)


class ComparisonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chat_store = Path(self.tmp.name) / "audition.sqlite3"
        self.store = comparison.comparison_store(self.chat_store)
        self.fake = FakeAudition()
        self.remote, self.cancel = Mock(), Mock()

    def stream(self, state=None, question="  Hello\n\u025b  ", stop=None):
        return comparison.compare(state or comparison.new_session(), question, MANIFEST, SHA, inference,
            self.store, stop=stop, remote=self.remote, cancel=self.cancel, audition_fn=self.fake)

    async def collect(self, **kwargs):
        return [state async for state in self.stream(**kwargs)]

    async def test_shared_exact_history_and_candidate_canonical_followup(self):
        values = await self.collect()
        state = values[-1]
        pair = state["rounds"][-1]
        self.assertEqual([v for v, m in self.fake.calls], ["base", "r2_meaning"])
        encoded = [json.dumps(m, ensure_ascii=False).encode() for v, m in self.fake.calls]
        self.assertEqual(encoded[0], encoded[1])
        self.assertEqual(pair["shared_messages"], [{"role": "user", "content": "  Hello\n\u025b  "}])
        self.assertEqual(pair["status"], "complete")
        self.assertTrue(pair["verified"])
        self.assertNotIn("winner", pair)
        self.assertEqual(state["shared_messages"][-1]["content"], "r2_meaning answer")
        next_state = (await self.collect(state=state, question="Follow up"))[-1]
        expected = state["shared_messages"] + [{"role": "user", "content": "Follow up"}]
        self.assertEqual(self.fake.calls[-2][1], expected)
        self.assertEqual(self.fake.calls[-1][1], expected)
        self.assertEqual(next_state["rounds"][0], state["rounds"][0])

    async def test_persisted_before_call_through_stream_and_timing_is_observed(self):
        stream = self.stream()
        first = await anext(stream)
        self.assertEqual(comparison.load_session(self.store, first["id"]), first)
        self.assertFalse(self.fake.calls)
        values = [first]
        async for state in stream:
            self.assertEqual(comparison.load_session(self.store, state["id"]), state)
            values.append(state)
        self.assertIsNone(first["rounds"][0]["sides"]["base"]["ttft_seconds"])
        for side in values[-1]["rounds"][0]["sides"].values():
            self.assertGreaterEqual(side["ttft_seconds"], 0)
            self.assertGreaterEqual(side["wall_seconds"], side["ttft_seconds"])
        self.assertEqual(self.store.stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.chat_store.exists())

    async def test_stop_before_call_skips_both(self):
        stop = threading.Event()
        stop.set()
        state = (await self.collect(stop=stop))[-1]
        self.assertEqual(state["rounds"][0]["status"], "cancelled")
        self.assertEqual([s["status"] for s in state["rounds"][0]["sides"].values()], ["skipped", "skipped"])
        self.assertFalse(self.fake.calls)

    async def test_stop_during_base_preserves_partial_and_skips_candidate(self):
        stop = threading.Event()
        async for state in self.stream(stop=stop):
            turn = state["rounds"][0]["sides"]["base"]["turn"]
            if turn and turn["answer"]:
                stop.set()
        pair = state["rounds"][0]
        self.assertEqual(pair["sides"]["base"]["turn"]["answer"], "base ")
        self.assertEqual(pair["sides"]["candidate"]["status"], "skipped")
        self.assertEqual(pair["status"], "cancelled")
        self.assertEqual(self.fake.controls[0], (stop, self.remote, self.cancel))
        self.assertEqual(len(self.fake.calls), 1)

    async def test_stop_between_sides_skips_second_call(self):
        stop = threading.Event()
        async for state in self.stream(stop=stop):
            if state["rounds"][0]["sides"]["base"]["status"] == "complete":
                stop.set()
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(state["rounds"][0]["sides"]["candidate"]["status"], "skipped")

    async def test_partial_failure_preserves_other_side_and_blocks_followup(self):
        for variant in ("base", "r2_meaning"):
            self.fake = FakeAudition({variant: "error"})
            state = (await self.collect())[-1]
            pair = state["rounds"][0]
            self.assertEqual(pair["status"], "error")
            self.assertEqual(len(self.fake.calls), 2)
            self.assertFalse(pair["verified"])
            self.assertEqual(state["shared_messages"], [])
            self.assertIn("simulated partial failure", pair["error"])
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                await self.collect(state=state)
            self.assertEqual(len(self.fake.calls), 2)

    async def test_prompt_hash_mismatch_and_invalid_identity_cannot_verify(self):
        for changes in ({"prompt_ids_sha256": "d" * 64}, {"identity": {}},
                        {"messages_sha256": "d" * 64}, {"decoding": {}}, "missing_final"):
            self.fake = FakeAudition({"r2_meaning": changes})
            state = (await self.collect())[-1]
            self.assertFalse(state["rounds"][0]["verified"])
            self.assertEqual(state["rounds"][0]["status"], "error")

    async def test_verified_length_and_time_stops_allow_followup_without_retry(self):
        for variant in ("base", "r2_meaning"):
            for reason in ("time_limit", "max_new_tokens"):
                self.fake = FakeAudition({variant: {"finish_reason": reason, "complete": False}})
                state = (await self.collect())[-1]
                original = copy.deepcopy(state["rounds"][0])
                self.assertEqual(original["status"], "incomplete")
                self.assertTrue(original["verified"])
                self.assertTrue(comparison.can_continue(original))
                next_state = (await self.collect(state=state, question="A new follow-up"))[-1]
                self.assertEqual(next_state["rounds"][0], original)
                self.assertEqual(len(self.fake.calls), 4)
                expected = original["shared_messages"] + [
                    {"role": "assistant", "content": "r2_meaning answer"},
                    {"role": "user", "content": "A new follow-up"}]
                self.assertEqual(self.fake.calls[-2][1], expected)
                self.assertEqual(self.fake.calls[-1][1], expected)

    async def test_old_saved_incomplete_session_recovers_canonical_context(self):
        self.fake = FakeAudition({"base": {"finish_reason": "max_new_tokens", "complete": False}})
        state = (await self.collect())[-1]
        state["shared_messages"] = []
        comparison._save(self.store, state)
        state = comparison.load_session(self.store, state["id"])
        next_state = (await self.collect(state=state, question="Follow up"))[-1]
        self.assertEqual(len(next_state["rounds"][-1]["shared_messages"]), 3)
        self.assertEqual(next_state["rounds"][0]["status"], "incomplete")

    async def test_disconnect_and_stale_state_never_retry(self):
        stream = self.stream()
        pending = await anext(stream)
        await stream.aclose()
        state = comparison.load_session(self.store, pending["id"])
        self.assertEqual(state["rounds"][0]["status"], "cancelled")
        with self.assertRaises(ValueError):
            await self.collect(state=pending)
        with self.assertRaises(ValueError):
            await self.collect(state=state)
        self.assertFalse(self.fake.calls)

    async def test_reviews_append_and_cannot_cross_sessions(self):
        state = (await self.collect())[-1]
        round_id = state["rounds"][0]["id"]
        for rating in ("unsure", "candidate", "tie", "neither", "base"):
            comparison.save_review(self.store, state["id"], round_id, rating, "Manual judgment")
        with comparison._db(self.store) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM comparison_reviews").fetchone()[0], 5)
        with self.assertRaises(ValueError):
            comparison.save_review(self.store, state["id"], "other-round", "base", "")
        self.assertEqual(comparison.load_session(self.store, state["id"]), state)
        self.assertEqual(comparison.conversations(self.store)[0][1], state["id"])

    async def test_import_saved_pair_preserves_system_history_and_no_fake_timings(self):
        messages = [{"role": "system", "content": "Research"}, {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Earlier answer"}, {"role": "user", "content": "Next"}]
        state = comparison.import_pair(self.store, messages, result(messages, "base"), result(messages, "r2_meaning"),
            MANIFEST, SHA, inference, label="Saved diagnostic", source="local-results.json")
        self.assertEqual(state["origin"], "imported_diagnostic")
        self.assertEqual(state["rounds"][0]["shared_messages"], messages)
        for side in state["rounds"][0]["sides"].values():
            self.assertIsNone(side["ttft_seconds"])
            self.assertIsNone(side["wall_seconds"])
        self.assertFalse(self.fake.calls)
        await self.collect(state=state, question="Continue")
        self.assertEqual(self.fake.calls[0][1][0], messages[0])
        self.assertEqual(self.fake.calls[0][1], self.fake.calls[1][1])

    async def test_invalid_import_does_not_persist(self):
        messages = [{"role": "user", "content": "Hello"}]
        right = {**result(messages, "r2_meaning"), "prompt_ids_sha256": "d" * 64}
        with self.assertRaisesRegex(ValueError, "Native prompt"):
            comparison.import_pair(self.store, messages, result(messages, "base"), right, MANIFEST, SHA, inference)
        self.assertEqual(comparison.conversations(self.store), [])

    async def test_real_audition_with_fake_transport_preserves_canonical_history(self):
        calls = []

        async def remote(messages, variant, expected_sha, request_id):
            calls.append((variant, copy.deepcopy(messages)))
            value = result(messages, variant)
            yield {"type": "start", "protocol": demo.PROTOCOL, "request_id": request_id,
                   "call_id": "fc-test", "identity": value["identity"], "messages_sha256": demo.digest(messages)}
            yield {"type": "delta", "request_id": request_id, "index": 0, "text": value["output"]}
            yield {"type": "final", "request_id": request_id, "result": value}

        seed = [{"role": "system", "content": "Research\n  Keep context. "},
                {"role": "user", "content": "  Twi: \u025b\n"}]
        state = comparison.import_pair(self.store, seed, result(seed, "base"), result(seed, "r2_meaning"),
                                       MANIFEST, SHA, inference)
        for question in (" First\n", "Followup  "):
            expected = state["shared_messages"] + [{"role": "user", "content": question}]
            async for state in comparison.compare(state, question, MANIFEST, SHA, inference,
                    self.store, remote=remote):
                pair = state["rounds"][-1]
                self.assertEqual(pair["shared_messages"], expected)
                for side in pair["sides"].values():
                    if side["turn"]:
                        self.assertEqual(side["turn"]["messages"], expected)
        self.assertEqual(len(calls), 4)
        for left, right in ((calls[0], calls[1]), (calls[2], calls[3])):
            self.assertEqual(json.dumps(left[1], ensure_ascii=False).encode(), json.dumps(right[1], ensure_ascii=False).encode())
        self.assertEqual(calls[2][1][0], seed[0])
        self.assertEqual(calls[2][1][-2], {"role": "assistant", "content": "r2_meaning answer"})
        self.assertTrue(state["rounds"][-1]["verified"])
        self.assertEqual(comparison.load_session(self.store, state["id"]), state)

    async def test_dropped_history_injection_fails_before_remote_submission(self):
        messages = [{"role": "user", "content": "Prior question"}]
        state = comparison.import_pair(self.store, messages, result(messages, "base"), result(messages, "r2_meaning"),
                                       MANIFEST, SHA, inference)

        def broken_audition(state, question, variant, manifest, expected_sha, validate_messages, store, **kwargs):
            return demo.audition(state, question, variant, manifest, expected_sha,
                                 inference.validate_messages, store, **kwargs)

        async for state in comparison.compare(state, "Next", MANIFEST, SHA, inference, self.store,
                remote=self.remote, cancel=self.cancel, audition_fn=broken_audition):
            pass
        self.remote.assert_not_called()
        self.assertEqual(state["rounds"][-1]["status"], "error")
        self.assertFalse(state["rounds"][-1]["verified"])
        self.assertEqual(state["rounds"][-1]["error"], "Side input history differs")

    async def test_real_audition_stop_calls_transport_cancel_and_skips_candidate(self):
        stop, calls, cancelled = threading.Event(), [], []

        async def remote(messages, variant, expected_sha, request_id):
            calls.append(variant)
            yield {"type": "start", "protocol": demo.PROTOCOL, "request_id": request_id,
                   "call_id": "fc-test", "identity": demo.identity(MANIFEST, SHA, variant),
                   "messages_sha256": demo.digest(messages)}
            yield {"type": "delta", "request_id": request_id, "index": 0, "text": "partial"}
            await asyncio.Event().wait()

        async def cancel(call_id):
            cancelled.append(call_id)

        async for state in comparison.compare(comparison.new_session(), "Hello", MANIFEST, SHA, inference,
                self.store, remote=remote, cancel=cancel, stop=stop):
            turn = state["rounds"][0]["sides"]["base"]["turn"]
            if turn and turn["answer"]:
                stop.set()
        self.assertEqual(calls, ["base"])
        self.assertEqual(cancelled, ["fc-test"])
        self.assertEqual(state["rounds"][0]["status"], "cancelled")
        self.assertEqual(state["rounds"][0]["sides"]["base"]["turn"]["cancellation"], "remote cancellation acknowledged")


if __name__ == "__main__":
    unittest.main()
