"""CPU-only tests. Real frozen audition, fake transport; no server or GPU calls."""
import asyncio
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import r2_inference as foundation
import understanding_r4_workbench as ui

SHA = "a" * 64
MANIFEST = {"version": ui.VERSION, "model": foundation.MODEL, "revision": foundation.REVISION,
            "base_sha256": foundation.BASE_SHA, "decoding": foundation.DECODING,
            "private": True, "training": False,
            "actor": {"app_name": "ghana-understanding-r4-demo-private", "class_name": "UnderstandingR4Demo"},
            "variants": {"base": None, **{v: {"directory": "/artifacts/" + v,
                "files": {"adapter_model.safetensors": c * 64}} for v, c in (("r3_response", "b"), ("r4_response", "c"))}}}


class FakeTransport:
    def __init__(self, demo, *, guarded=True, changes=None):
        self.demo, self.guarded, self.changes = demo, guarded, changes or {}
        self.calls, self.cancelled = [], []

    async def cancel(self, call_id):
        self.cancelled.append(call_id)

    async def __call__(self, messages, variant, expected_sha, request_id):
        self.calls.append((variant, copy.deepcopy(messages)))
        change = self.changes.get(variant, {})
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
        self.path = Path(self.tmp.name).resolve() / "r4.sqlite3"
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
        self.assertNotIn("r4_response", self.demo.LABELS)

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
        for rating in ("base", "r3_response", "r4_response", "tie", "neither", "unsure"):
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
            self.transport = FakeTransport(self.demo, changes={"r4_response": mutation})
            state = (await self.collect())[-1]
            self.assertFalse(self.workbench.ready(state), mutation)
            if mutation != {"complete": False}:
                self.assertFalse(state["rounds"][-1]["verified"], mutation)
            count = len(self.transport.calls)
            with self.assertRaisesRegex(ValueError, "Unfinished"):
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
        item["input_messages"]["base"] = copy.deepcopy(item["input_messages"]["r4_response"])
        self.store.save(state)
        before = len(self.transport.calls)
        with self.assertRaisesRegex(ValueError, "not independent"):
            await self.collect(state=state, question="Third")
        self.assertEqual(len(self.transport.calls), before)

    async def test_distinct_histories_permit_distinct_native_prompt_hashes(self):
        state = (await self.collect())[-1]
        self.transport = FakeTransport(self.demo, changes={"r4_response": {"prompt_ids_sha256": "f" * 64}})
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
            self.transport = FakeTransport(self.demo, changes={"r4_response": change})
            state = (await self.collect())[-1]
            original = copy.deepcopy(state["rounds"][0])
            self.assertTrue(self.workbench.ready(state))
            self.assertNotEqual(original["status"], "complete")
            state = (await self.collect(state=state, question="Please continue"))[-1]
            self.assertEqual(state["rounds"][0], original)
            self.assertEqual(len(self.transport.calls), 6)
            self.assertEqual(self.transport.calls[-1][1][-2]["content"], change.get("output", "r4_response answer"))

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
            self.transport = FakeTransport(self.demo, changes={"r4_response": {"complete": False, "finish_reason": reason}})
            state = (await self.collect())[-1]
            self.assertFalse(self.workbench.ready(state))
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                await self.collect(state=state)

    async def test_same_histories_in_two_panels_still_require_matching_prompt_proof(self):
        self.transport = FakeTransport(self.demo, changes={"base": {"output": "shared answer"},
                                                          "r3_response": {"output": "shared answer"}})
        state = (await self.collect())[-1]
        self.transport = FakeTransport(self.demo, changes={"r3_response": {"prompt_ids_sha256": "f" * 64}})
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
        self.assertEqual(state["rounds"][0]["sides"]["r4_response"]["status"], "skipped")
        self.assertEqual(ui.transcript(state, "base")[-1]["content"], "base answer")

    async def test_stop_between_variants_prevents_later_submissions(self):
        stop = threading.Event()
        async for state in self.stream(stop=stop):
            if state["rounds"][-1]["sides"]["r3_response"]["status"] == "complete":
                stop.set()
        self.assertEqual([v for v, _ in self.transport.calls], ["base", "r3_response"])
        self.assertEqual(state["rounds"][-1]["sides"]["r4_response"]["status"], "skipped")

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

    async def test_ui_config_and_callbacks_without_server_or_remote_startup(self):
        remote = Mock(side_effect=AssertionError("No startup inference"))
        app = ui.build_app(self.workbench, remote=remote, cancel=self.transport.cancel)
        config = app.get_config_file()
        self.assertEqual(len([c for c in config["components"] if c["type"] == "chatbot"]), 3)
        self.assertEqual(len([c for c in config["components"] if c["type"] == "browserstate"]), 1)
        checkpoints = [c["props"]["value"] for c in config["components"]
                       if "r4-checkpoint" in c["props"].get("elem_classes", [])]
        self.assertEqual(checkpoints, [
            "AfriqueQwen 9B, adapter off\n\n`" + MANIFEST["base_sha256"][:12] + "`",
            "R3 response stage\n\n`" + "b" * 12 + "`",
            "R4 continuation\n\n`" + "c" * 12 + "`"])
        self.assertFalse(app.is_running)
        restore = next(f.fn for f in app.fns.values() if f.fn.__name__ == "restore")
        first = restore(None)
        second = restore(first[0]["id"])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2:5], ([], [], []))
        proof = ui.evidence(self.workbench, first[0])
        self.assertEqual(proof["identities"]["base"]["base_weights_sha256"], MANIFEST["base_sha256"])
        self.assertEqual(proof["identities"]["r3_response"]["adapter_weights_sha256"], "b" * 64)
        self.assertEqual(proof["identities"]["r4_response"]["adapter_weights_sha256"], "c" * 64)
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
        sources = {"code/r4_inference.py": runtime, "code/r2_inference.py": b'BOUND_SIBLING = True\n'}
        manifest = {**copy.deepcopy(MANIFEST), "files": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}}
        for name, data in sources.items():
            (folder / name).write_bytes(data)
        data = json.dumps(manifest).encode()
        (folder / "manifest.json").write_bytes(data)
        return folder, hashlib.sha256(data).hexdigest(), manifest

    def test_r4_bundle_import_is_explicit_and_validator_bound_without_runtime(self):
        folder, checksum, manifest = self.fixture_bundle()
        before = list(sys.path)
        with patch.dict(sys.modules, {"r4_inference": types.SimpleNamespace(validate_messages=Mock())}):
            loaded, module = ui.bound_bundle(folder, checksum)
        self.assertEqual(loaded, manifest)
        self.assertEqual(module.__file__, str(folder / "code/r4_inference.py"))
        with self.assertRaisesRegex(ValueError, "bound validator"):
            module.validate_messages([])
        self.assertEqual(sys.path, before)
        self.assertFalse((folder / "code/__pycache__").exists())

    def test_manifest_code_sibling_and_path_hash_mismatches_rejected(self):
        folder, checksum, _ = self.fixture_bundle()
        with self.assertRaisesRegex(ValueError, "hash differs"):
            ui.bound_bundle(folder, "f" * 64)
        for name in ("r4_inference.py", "r2_inference.py"):
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
        for mutate in (lambda m: m.update(version="r3-comparison-inference-v1"),
                       lambda m: m["variants"].pop("r4_response"),
                       lambda m: m["actor"].update(class_name="UnderstandingR3Demo")):
            manifest = copy.deepcopy(MANIFEST)
            mutate(manifest)
            with self.assertRaises(ValueError):
                ui.manifest_contract(manifest)
        optional = copy.deepcopy(MANIFEST)
        optional["variants"]["r4_pass1"] = optional["variants"]["r4_response"]
        ui.manifest_contract(optional)

    def test_frozen_helpers_do_not_rebind_global_imports_or_labels(self):
        before = list(sys.path)
        poison = types.SimpleNamespace(VERSION="poison", repeated_passage=Mock(), validate_guard=Mock())
        with patch.dict(sys.modules, {"response_loop_guard": poison}):
            demo = ui.load_helpers()
            self.assertIs(sys.modules["response_loop_guard"], poison)
        self.assertEqual(demo.LOOP_POLICY, "repeat-block-v1")
        self.assertNotIn("r4_response", demo.LABELS)
        self.assertEqual(sys.path, before)
        for name in ui.HELPER_HASHES:
            (self.root / name).write_bytes((ui.DEFAULT_HELPER_CODE / name).read_bytes())
        (self.root / "response_loop_guard.py").write_text("raise AssertionError('changed helper')\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            ui.load_helpers(self.root)

    def test_r3_sqlite_and_different_bundle_rejected_without_modifying_bytes(self):
        prior = self.root / "r3.sqlite3"
        with sqlite3.connect(prior) as db:
            db.execute("CREATE TABLE turns (record TEXT)")
            db.execute("INSERT INTO turns VALUES ('original history')")
        original = prior.read_bytes()
        with self.assertRaisesRegex(ValueError, "separate R4 store"):
            ui.Store(prior, SHA)
        self.assertEqual(prior.read_bytes(), original)
        r4 = self.root / "r4.sqlite3"
        ui.Store(r4, SHA)
        original = r4.read_bytes()
        with self.assertRaisesRegex(ValueError, "another bundle"):
            ui.Store(r4, "f" * 64)
        self.assertEqual(r4.read_bytes(), original)

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


if __name__ == "__main__":
    unittest.main()
