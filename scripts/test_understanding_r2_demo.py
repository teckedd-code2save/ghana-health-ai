"""CPU-only R2 demo tests. No credentials, deployed actor, model, or server needed.

python3 -B -m unittest discover -s scripts -p test_understanding_r2_demo.py -v
"""
import ast
import asyncio
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import r2_inference as inference
import understanding_r2_demo as demo

SHA = "a" * 64
VARIANT = "r2_meaning"
MESSAGES = [{"role": "user", "content": "Test question"}]


def manifest():
    return {"version": "r2-paired-inference-v1", "model": inference.MODEL,
            "revision": inference.REVISION, "base_sha256": inference.BASE_SHA,
            "runtime": inference.RUNTIME, "decoding": inference.DECODING,
            "variants": {v: None if v == "base" else {
                "directory": f"/artifacts/understanding-adaptation-r{1 if v.startswith('r1') else 2}/understanding_20261105T120000Z/responses-adapter",
                "files": {"adapter_config.json": "b" * 64, "adapter_model.safetensors": "c" * 64}}
                for v in inference.VARIANTS}, "files": {}, "private": True, "training": False}


def result(messages=MESSAGES, variant=VARIANT, output="Hello there", **changes):
    value = {"identity": demo.identity(manifest(), SHA, variant), "messages_sha256": demo.digest(messages),
             "prompt_ids_sha256": "d" * 64, "decoding": inference.DECODING,
             "output": output, "output_ids": [8, 9, 10], "seconds": .2,
             "finish_reason": "eos", "complete": True}
    value.update(changes)
    return value


def events(messages, variant, expected_sha, request_id):
    return [{"type": "start", "protocol": demo.PROTOCOL, "request_id": request_id,
             "call_id": "fc-test", "identity": demo.identity(manifest(), expected_sha, variant),
             "messages_sha256": demo.digest(messages)},
            {"type": "delta", "request_id": request_id, "index": 0, "text": "Hello "},
            {"type": "delta", "request_id": request_id, "index": 1, "text": "there"},
            {"type": "final", "request_id": request_id, "result": result(messages, variant)}]


def make_bundle(folder):
    (folder / "code").mkdir()
    (folder / "code/r2_inference.py").write_bytes(Path(inference.__file__).read_bytes())
    value = manifest()
    value["files"] = {"code/r2_inference.py": demo.sha(folder / "code/r2_inference.py")}
    (folder / "manifest.json").write_text(json.dumps(value))
    return demo.sha(folder / "manifest.json")


class BundleTests(unittest.TestCase):
    def test_verified_import_is_bundled_and_does_not_load_heavy_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            expected = make_bundle(folder)
            script = ("import sys; sys.path.insert(0, sys.argv[1]); import understanding_r2_demo as d; "
                      "m,r=d.bound_bundle(sys.argv[2],sys.argv[3]); "
                      "assert r.__file__ == sys.argv[2]+'/code/r2_inference.py'; "
                      "assert sys.path[0] == sys.argv[2]+'/code'; "
                      "assert not ({'modal','torch','transformers','gradio'} & set(sys.modules))")
            run = subprocess.run([sys.executable, "-B", "-c", script, str(demo.ROOT / "scripts"), str(folder), expected],
                                 text=True, capture_output=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_bundle_hash_and_code_tampering_fail_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = make_bundle(folder)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                demo.bound_bundle(folder, "f" * 64)
            (folder / "code/r2_inference.py").write_text("raise AssertionError('executed changed code')")
            with self.assertRaisesRegex(ValueError, "code changed"):
                demo.bound_bundle(folder, expected)

    def test_exact_runtime_identity_for_every_variant(self):
        runtime = inference.Runtime.__new__(inference.Runtime)
        runtime.manifest, runtime.manifest_sha = manifest(), SHA
        for variant in inference.VARIANTS:
            self.assertEqual(demo.identity(manifest(), SHA, variant), runtime.identity(variant))
        self.assertEqual(demo.digest(MESSAGES), inference.digest(MESSAGES))

    def test_verify_cli_never_builds_ui_or_calls_modal(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = make_bundle(folder)
            with patch.object(demo, "build_app", side_effect=AssertionError("UI called")), redirect_stdout(io.StringIO()) as output:
                demo.main(["verify", "--bundle", str(folder), "--expected-sha", expected])
            value = json.loads(output.getvalue())
            self.assertFalse(value["remote_called"])
            self.assertEqual(value["variants"][0]["variant"], "r2_meaning")


class WorkerTests(unittest.TestCase):
    def runtime(self, reply):
        return SimpleNamespace(identity=lambda v: demo.identity(manifest(), SHA, v), reply=reply)

    def stream(self, runtime):
        return demo.stream_reply(lambda: runtime, manifest(), MESSAGES, VARIANT, SHA, "e" * 32, "fc-test")

    def test_first_delta_arrives_before_reply_finishes(self):
        finish = threading.Event()
        finished = threading.Event()

        def reply(messages, variant, on_text, cancelled):
            on_text("Hello ")
            self.assertTrue(finish.wait(2))
            on_text("there")
            finished.set()
            return result()

        stream = self.stream(self.runtime(reply))
        self.assertEqual(next(stream)["type"], "start")
        try:
            self.assertEqual(next(stream)["text"], "Hello ")
            self.assertFalse(finished.is_set())
        finally:
            finish.set()
        rest = list(stream)
        self.assertEqual([e["type"] for e in rest], ["delta", "final"])
        self.assertEqual(rest[-1]["result"]["output"], "Hello there")

    def test_call_identity_arrives_before_loading_runtime(self):
        factory = Mock(side_effect=AssertionError("load should not run"))
        stream = demo.stream_reply(factory, manifest(), MESSAGES, VARIANT, SHA, "e" * 32, "fc-test")
        self.assertEqual(next(stream)["call_id"], "fc-test")
        stream.close()
        factory.assert_not_called()

    def test_disconnect_cooperatively_cancels_and_joins_worker(self):
        finished = threading.Event()

        def reply(messages, variant, on_text, cancelled):
            on_text("Hello ")
            deadline = time.monotonic() + 2
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(cancelled())
            finished.set()
            return result(output="Hello ", finish_reason="cancelled", complete=False)

        stream = self.stream(self.runtime(reply))
        next(stream)
        next(stream)
        stream.close()
        self.assertTrue(finished.is_set())
        self.assertFalse(any(t.name == "r2-demo-generation" for t in threading.enumerate()))

    def test_worker_failure_and_divergent_final_are_not_fabricated(self):
        def fail(messages, variant, on_text, cancelled):
            on_text("Hello ")
            raise RuntimeError("generation failed")
        stream = self.stream(self.runtime(fail))
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            list(stream)

        def mismatch(messages, variant, on_text, cancelled):
            on_text("other text")
            return result()
        with self.assertRaisesRegex(ValueError, "differs from streamed"):
            list(self.stream(self.runtime(mismatch)))

    def test_wrong_runtime_never_generates(self):
        runtime = self.runtime(Mock(side_effect=AssertionError("generated")))
        runtime.identity = lambda v: {**demo.identity(manifest(), SHA, v), "manifest_sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "Runtime identity differs"):
            list(self.stream(runtime))
        runtime.reply.assert_not_called()


class ReplyTests(unittest.TestCase):
    def validate(self, value):
        return demo.validate_reply(value, demo.identity(manifest(), SHA, VARIANT), MESSAGES,
                                   inference.DECODING, "Hello there")

    def test_final_validation_rejects_mismatches_and_malformed_evidence(self):
        self.validate(result())
        changes = [{"identity": demo.identity(manifest(), SHA, "r1_response")},
                   {"messages_sha256": "f" * 64}, {"prompt_ids_sha256": "bad"},
                   {"decoding": {**inference.DECODING, "repetition_penalty": 1.1}},
                   {"output": "different"}, {"output_ids": [True]}, {"output_ids": []},
                   {"output_ids": [1] * 257}, {"seconds": float("nan")}, {"seconds": True},
                   {"finish_reason": "cancelled"}, {"finish_reason": "time_limit"},
                   {"finish_reason": "max_new_tokens", "complete": False}, {"complete": 1},
                   {"diagnostics": {"output_ids": [99]}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(result(**change))


class AuditionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Path(self.temporary.name) / "audition.sqlite3"
        self.calls, self.cancellations = [], []

    async def cancel(self, call_id):
        self.cancellations.append(call_id)

    def remote(self, transform=lambda value: value, hang=False):
        async def call(*args):
            self.calls.append(copy.deepcopy(args))
            for event in transform(events(*args)):
                await asyncio.sleep(0)
                yield event
            if hang:
                await asyncio.Event().wait()
        return call

    async def run_turn(self, state=None, remote=None, **kwargs):
        return [value async for value in demo.audition(state or demo.new_session(), "Test question", VARIANT,
                manifest(), SHA, inference.validate_messages, self.store, remote or self.remote(), self.cancel, **kwargs)]

    async def test_actual_deltas_persist_and_followup_includes_exact_prior_turn(self):
        values = await self.run_turn()
        self.assertEqual([s["turns"][-1]["answer"] for s in values], ["", "Hello ", "Hello there", "Hello there"])
        state = values[-1]
        self.assertEqual(state["turns"][-1]["status"], "complete")
        restored = demo.load_session(self.store, state["id"])
        self.assertEqual(restored, state)
        await self.run_turn(state=restored)
        self.assertEqual(self.calls[-1][0], [*MESSAGES, {"role": "assistant", "content": "Hello there"}, *MESSAGES])
        self.assertEqual(self.cancellations, [])
        self.assertEqual(self.store.stat().st_mode & 0o777, 0o600)

    async def test_missing_final_partial_and_prior_history_are_preserved(self):
        state = (await self.run_turn())[-1]
        values = await self.run_turn(state=state, remote=self.remote(lambda e: e[:-1]))
        failed = values[-1]
        self.assertEqual(failed["turns"][0], state["turns"][0])
        self.assertEqual(failed["turns"][-1]["status"], "error")
        self.assertEqual(failed["turns"][-1]["answer"], "Hello there")
        self.assertEqual(self.cancellations, ["fc-test"])
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            await self.run_turn(state=failed)
        self.assertEqual(len(self.calls), count)

    async def test_length_stopped_reply_can_receive_followup_without_rewriting_it(self):
        def length_stop(items):
            items[-1]["result"].update(complete=False, finish_reason="max_new_tokens", output_ids=[1] * 256)
            return items
        state = (await self.run_turn(remote=self.remote(length_stop)))[-1]
        self.assertEqual(state["turns"][-1]["status"], "incomplete")
        original = copy.deepcopy(state["turns"][0])
        restored = demo.load_session(self.store, state["id"])
        next_state = (await self.run_turn(state=restored))[-1]
        self.assertEqual(next_state["turns"][0], original)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[-1][0], [*MESSAGES, {"role": "assistant", "content": "Hello there"}, *MESSAGES])

    async def test_bad_event_sequences_never_finalize(self):
        def change_start(e):
            e[0]["identity"]["manifest_sha256"] = "f" * 64
            return e
        def change_request(e):
            e[1]["request_id"] = "f" * 32
            return e
        def change_output(e):
            e[-1]["result"]["output"] = "unstreamed"
            return e
        for transform in (lambda e: e[1:], lambda e: [e[0], e[2], e[-1]],
                          lambda e: e + [e[-1]], lambda e: [e[0], e[0]],
                          change_start, change_request, change_output):
            with self.subTest(transform=transform):
                state = (await self.run_turn(remote=self.remote(transform)))[-1]
                self.assertEqual(state["turns"][-1]["status"], "error")

    async def test_stop_after_delta_cancels_remote_and_preserves_partial(self):
        stop = threading.Event()
        values = []
        async for value in demo.audition(demo.new_session(), "Test question", VARIANT, manifest(), SHA,
                inference.validate_messages, self.store, self.remote(lambda e: e[:2], hang=True), self.cancel, stop):
            values.append(value)
            if value["turns"][-1]["answer"]:
                stop.set()
        turn = values[-1]["turns"][-1]
        self.assertEqual((turn["status"], turn["answer"]), ("cancelled", "Hello "))
        self.assertEqual(turn["cancellation"], "remote cancellation acknowledged")
        self.assertEqual(self.cancellations, ["fc-test"])

    async def test_stop_before_submission_and_timeout_without_call_id(self):
        stop = threading.Event()
        stop.set()
        state = (await self.run_turn(stop=stop))[-1]
        self.assertEqual(state["turns"][-1]["status"], "cancelled")
        self.assertFalse(self.calls)
        state = (await self.run_turn(remote=self.remote(lambda e: [], hang=True), timeout=.01))[-1]
        self.assertEqual(state["turns"][-1]["status"], "error")
        self.assertIn("unconfirmed", state["turns"][-1]["cancellation"])

    async def test_cancel_failure_is_visible_and_partial_is_saved(self):
        async def fail(call_id):
            raise RuntimeError("unreachable")
        self.cancel = fail
        state = (await self.run_turn(remote=self.remote(lambda e: e[:-1])))[-1]
        turn = state["turns"][-1]
        self.assertEqual(turn["cancellation"], "unconfirmed: unreachable")
        self.assertEqual(demo.load_session(self.store, state["id"]), state)

    async def test_disconnect_finalizes_pending_row(self):
        stream = demo.audition(demo.new_session(), "Test question", VARIANT, manifest(), SHA,
                              inference.validate_messages, self.store, self.remote(), self.cancel)
        first = await anext(stream)
        await stream.aclose()
        restored = demo.load_session(self.store, first["id"])
        self.assertEqual(restored["turns"][-1]["status"], "cancelled")
        self.assertFalse(self.calls)

    async def test_reviews_are_append_only_and_finalized_turns_immutable(self):
        state = (await self.run_turn())[-1]
        turn = state["turns"][-1]
        for rating in ("Unsure", "Useful"):
            demo.save_review(self.store, state["id"], turn["id"], rating, "A note")
        with demo.connect_store(self.store) as db:
            self.assertEqual(db.execute("SELECT rating FROM reviews ORDER BY id").fetchall(), [("Unsure",), ("Useful",)])
        with self.assertRaises(ValueError):
            demo.save_review(self.store, "other-session", turn["id"], "Useful", "")
        with self.assertRaises(ValueError):
            demo.save_turn(self.store, {**turn, "answer": "edited"})
        self.assertEqual(demo.load_session(self.store, state["id"]), state)
        self.assertIn(state["id"], [key for label, key in demo.conversations(self.store)])

    async def test_changed_bundle_or_variant_blocks_continuation_without_remote_call(self):
        state = (await self.run_turn())[-1]
        state["turns"][0]["identity"]["manifest_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            await self.run_turn(state=state)
        state["variant"] = "base"
        with self.assertRaises(ValueError):
            await self.run_turn(state=state)
        self.assertEqual(len(self.calls), 1)


class PackagingTests(unittest.TestCase):
    def test_meaning_is_diagnostic_default_with_all_comparators_preserved(self):
        self.assertEqual(demo.DEFAULT_VARIANT, "r2_meaning")
        self.assertEqual(demo.new_session()["variant"], "r2_meaning")
        self.assertEqual(demo.LABELS["r2_meaning"], "R2 \u00b7 language + meaning")
        self.assertEqual(demo.LABELS["r2_response"], "R2 \u00b7 final response stage")
        self.assertEqual(set(demo.available_labels(manifest())), {"r2_meaning", "r2_response", "r2_language", "r1_meaning", "r1_response", "base"})
        self.assertNotIn("r3_response", demo.available_labels(manifest()))
        self.assertEqual(demo.new_session("r2_response")["variant"], "r2_response")

    def test_actor_is_private_bounded_readonly_and_bundles_both_importable_modules(self):
        path = demo.ROOT / "modal/research_understanding_r2_demo.py"
        source = path.read_text()
        tree = ast.parse(source)
        actor = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        options = {k.arg: k.value for k in actor.decorator_list[0].keywords}
        for key, value in {"min_containers": 0, "max_containers": 1, "scaledown_window": 120,
                           "timeout": 300, "startup_timeout": 600, "retries": 0,
                           "include_source": False, "gpu": "H100"}.items():
            self.assertEqual(ast.literal_eval(options[key]), value)
        constants = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for destination in ("/demo/scripts/understanding_r2_demo.py", "/demo/modal/research_understanding_r2_demo.py", "/bundle"):
            self.assertIn(destination, constants)
        self.assertIn("/bundle/code:/demo/scripts:/demo/modal", constants)
        for name, version in inference.RUNTIME.items():
            self.assertIn(name + "==" + version, constants)
        self.assertEqual(source.count(".read_only()"), 2)
        self.assertNotIn("fastapi_endpoint", source)
        self.assertNotIn(".remote(", source)
        self.assertNotIn(".spawn(", source)
        self.assertNotIn(".deploy(", source)

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Optional local Gradio dependency not installed")
    def test_ui_construction_has_no_remote_calls_and_no_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            remote = Mock(side_effect=AssertionError("remote startup call"))
            with patch.object(demo, "bound_bundle", return_value=(manifest(), inference)):
                app = demo.build_app(Path(temporary), SHA, Path(temporary) / "reviews.sqlite3", remote=remote)
            remote.assert_not_called()
            self.assertFalse(app.is_running)
            config = app.get_config_file()
            model = next(c for c in config["components"] if c.get("props", {}).get("label") == "Model")
            self.assertEqual(model["props"]["value"], "r2_meaning")
            self.assertEqual({v for label, v in model["props"]["choices"]}, set(demo.available_labels(manifest())))
            self.assertIn(("R2 \u00b7 language + meaning", "r2_meaning"), model["props"]["choices"])
            self.assertIn(("R2 \u00b7 final response stage", "r2_response"), model["props"]["choices"])
            self.assertTrue(all(d["api_visibility"] == "private" for d in config["dependencies"]))

    def test_chat_cli_is_loopback_only_and_does_not_enable_sharing(self):
        app = Mock()
        app.queue.return_value = app
        with tempfile.TemporaryDirectory() as temporary, patch.object(demo, "bound_bundle", return_value=(manifest(), inference)), \
                patch.object(demo, "build_app", return_value=app):
            demo.main(["chat", "--bundle", temporary, "--expected-sha", SHA])
        opts = app.launch.call_args.kwargs
        self.assertEqual((opts["server_name"], opts["server_port"]), ("127.0.0.1", 7866))
        self.assertFalse(opts["share"])
        self.assertFalse(opts["mcp_server"])
        self.assertTrue(opts["strict_cors"])


if __name__ == "__main__":
    unittest.main()
