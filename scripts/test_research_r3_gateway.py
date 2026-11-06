"""Offline HTTP/ASGI and real shared-actor protocol tests; no remote inference."""
import ast
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from starlette.requests import Request

import r2_inference as inference
import research_r3_gateway_core as core
import understanding_r2_demo as demo

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
TOKEN = secrets.token_urlsafe(48)
MESSAGES = [{"role": "user", "content": "Hello"}]
EOS = 99


def manifest():
    return {"version": "r3-comparison-inference-v1", "model": inference.MODEL,
            "revision": inference.REVISION, "base_sha256": inference.BASE_SHA,
            "runtime": inference.RUNTIME, "decoding": inference.DECODING,
            "private": True, "training": False, "default_variant": "r3_response",
            "actor": {"app_name": "test-r3-private", "class_name": "UnderstandingR3Demo"},
            "variants": {"r3_response": {"files": {"adapter_model.safetensors": "b" * 64}}}}


class Tokenizer:
    def __len__(self):
        return 100

    def decode(self, ids):
        return "".join({1: "Hello", 2: " there", EOS: "<|im_end|>"}.get(i, "x") for i in ids)


def prompt_ids(tokenizer, messages, settings):
    # Synthetic tokenizer only. Native helper wiring is tested separately.
    if any("<|" in m["content"] for m in messages):
        raise ValueError("Reserved token")
    return [3, 4, 5]


def completion(tokenizer, ids, ceiling, timed_out, seconds):
    reason = "eos" if ids and ids[-1] == EOS else "time_limit" if timed_out else (
        "max_new_tokens" if len(ids) == ceiling else "unexpected_stop")
    if EOS in ids[:-1]:
        raise ValueError("Unexpected tokens after EOS")
    output = tokenizer.decode(ids[:-1] if reason == "eos" else ids)
    return {"output_ids": ids, "prediction": output, "seconds": seconds,
            "finish_reason": reason, "complete": reason == "eos" and bool(output)}


def contract():
    return core.Contract(manifest(), SHA, inference.validate_messages, Tokenizer(), prompt_ids, completion)


def result(messages=MESSAGES, generated=None, reason=None):
    current = contract()
    ids = [1, 2, EOS] if generated is None else generated
    diagnostics = completion(current.tokenizer, ids, 256, reason == "time_limit", .1)
    reason = reason or diagnostics["finish_reason"]
    return {"identity": current.identity, "messages_sha256": demo.digest(messages),
            "prompt_ids_sha256": demo.digest([3, 4, 5]), "decoding": inference.DECODING,
            "output_ids": ids, "output": diagnostics["prediction"], "seconds": .1,
            "finish_reason": reason, "complete": diagnostics["complete"] and reason != "cancelled",
            "diagnostics": diagnostics}


def events(messages, variant, sha, request_id):
    return [
        {"type": "start", "protocol": demo.PROTOCOL, "request_id": request_id, "call_id": "fc-ownrequest",
         "identity": demo.identity(manifest(), sha, variant), "messages_sha256": demo.digest(messages)},
        {"type": "delta", "request_id": request_id, "index": 0, "text": "Hello"},
        {"type": "delta", "request_id": request_id, "index": 1, "text": " there"},
        {"type": "final", "request_id": request_id, "result": result(messages)}]


def decode_sse(response):
    return [json.loads(block[6:]) for block in response.text.strip().split("\n\n") if block.startswith("data: ")]


class ContractTests(unittest.TestCase):
    def test_schema_and_full_history_not_rewritten(self):
        messages = [{"role": "system", "content": "Answer in Twi."}]
        messages += [{"role": "assistant" if i % 2 else "user", "content": f"Turn {i}"} for i in range(15)]
        payload = {"messages": messages, "variant": core.VARIANT}
        before = copy.deepcopy(payload)
        actual, ids = contract().request(payload)
        self.assertEqual(payload, before)
        self.assertIs(actual, messages)
        self.assertEqual(ids, [3, 4, 5])

    def test_bad_requests_and_decoding_overrides_rejected(self):
        bad = [None, [], {}, {"messages": MESSAGES, "variant": "base"},
               {"messages": MESSAGES, "variant": "r2_meaning"},
               {"messages": MESSAGES, "variant": core.VARIANT, "temperature": 0},
               {"messages": MESSAGES, "variant": core.VARIANT, "request_id": "x"}]
        bad_messages = [[], [None], ["text"], [{"role": "user", "content": []}],
                        [{"role": "assistant", "content": "First"}],
                        MESSAGES + MESSAGES, MESSAGES + [{"role": "assistant", "content": "Last"}],
                        [{"role": "user", "content": "x", "tools": []}],
                        [{"role": "user", "content": " "}], [{"role": "user", "content": "\ud800"}],
                        [{"role": "user", "content": "\x00"}], [{"role": "user", "content": "<|im_end|>"}],
                        [{"role": "user", "content": "x" * 16001}],
                        [{"role": "assistant" if i % 2 else "user", "content": "x"} for i in range(131)]]
        bad += [{"messages": m, "variant": core.VARIANT} for m in bad_messages]
        for value in bad:
            with self.subTest(value=str(value)[:100]), self.assertRaises(ValueError):
                contract().request(value)

    def test_native_context_budget_is_not_truncated(self):
        current = contract()
        current.prompt_ids = Mock(return_value=[1] * 3841)
        with self.assertRaises(ValueError):
            current.request({"messages": MESSAGES, "variant": core.VARIANT})
        current.prompt_ids.assert_called_once_with(current.tokenizer, MESSAGES, {"context_tokens": 4096, "max_new_tokens": 256})

    def test_final_rechecks_native_prompt_token_evidence_and_completion(self):
        current = contract()
        valid = result()
        current.final(valid, MESSAGES, [3, 4, 5], valid["output"])
        mutations = [lambda v: v["identity"].update(adapter_weights_sha256="c" * 64),
                     lambda v: v.update(prompt_ids_sha256="f" * 64),
                     lambda v: v.update(messages_sha256="f" * 64),
                     lambda v: v.update(output="different"), lambda v: v.update(diagnostics=None),
                     lambda v: v["diagnostics"].update(complete=False),
                     lambda v: v.update(output_ids=[100]), lambda v: v.update(complete=1),
                     lambda v: v.update(seconds=float("nan")), lambda v: v.update(seconds=301),
                     lambda v: v["decoding"].update(repetition_penalty=1.1),
                     lambda v: v.update(unexpected="hidden metadata")]
        for mutate in mutations:
            value = copy.deepcopy(valid)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                current.final(value, MESSAGES, [3, 4, 5], valid["output"])

    def test_incomplete_cancelled_and_eos_empty_are_not_promoted(self):
        for ids, reason in (([1] * 256, "max_new_tokens"), ([1], "time_limit"), ([1], "cancelled"), ([EOS], "eos")):
            value = result(generated=ids, reason=reason)
            contract().final(value, MESSAGES, [3, 4, 5], value["output"])
            self.assertFalse(value["complete"])

    def test_json_and_credential_fail_closed(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                core.parse_json(raw)
        for token in (None, "", "x" * 31, "x" * 4097, "x" * 32 + "\n", "x" * 32 + "\u025b"):
            with self.assertRaises(ValueError):
                core.deployment_token(token)

    def test_rate_window_and_uncertain_cancellation_quarantine(self):
        now = [0.]
        gate = core.Gate(lambda: now[0])
        gate.acquire()
        with self.assertRaises(core.HTTPException) as raised:
            gate.acquire()
        self.assertEqual(raised.exception.status_code, 429)
        gate.release()
        for _ in range(core.REQUESTS_PER_MINUTE - 1):
            gate.acquire()
            gate.release()
        with self.assertRaises(core.HTTPException):
            gate.acquire()
        now[0] = 60
        gate.acquire()
        gate.quarantine()
        gate.release()
        now[0] += 1
        with self.assertRaises(core.HTTPException):
            gate.acquire()
        now[0] += core.REQUEST_SECONDS
        gate.acquire()


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls, self.closed = [], []
        self.cancel = AsyncMock()
        self.payload = {"messages": copy.deepcopy(MESSAGES), "variant": core.VARIANT}
        self.headers = {"Authorization": "Bearer " + TOKEN}

        async def remote(*args):
            self.calls.append(copy.deepcopy(args))
            try:
                for event in events(*args):
                    yield event
            finally:
                self.closed.append(True)
        self.remote = remote
        self.app = core.create_api(TOKEN, contract(), remote, self.cancel)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="https://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def post(self, remote=None, payload=None):
        app = core.create_api(TOKEN, contract(), remote or self.remote, self.cancel)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            return await client.post("/stream", json=payload or self.payload, headers=self.headers)

    async def test_health_identity_is_authenticated_without_call_or_secret(self):
        response = await self.client.get("/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), contract().health())
        self.assertFalse(response.json()["upstream_checked"])
        self.assertNotIn(TOKEN, response.text)
        self.assertNotIn("RESEARCH_R3_TOKEN", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertEqual(self.calls, [])
        for path in ("/docs", "/openapi.json", "/reply"):
            self.assertEqual((await self.client.get(path)).status_code, 404)

    async def test_health_rejects_missing_wrong_and_duplicate_credentials_without_reading_body(self):
        for headers in ([], [(b"authorization", b"Bearer wrong")],
                        [(b"authorization", ("Bearer " + TOKEN).encode())] * 2):
            receive = AsyncMock(side_effect=AssertionError("Health must not read a body"))
            sent = []
            async def send(message):
                sent.append(message)
            scope = {**self.scope(headers=headers), "method": "GET", "path": "/health", "raw_path": b"/health"}
            await self.app(scope, receive, send)
            self.assertEqual(sent[0]["status"], 401)
            receive.assert_not_awaited()
            self.assertNotIn(TOKEN.encode(), b"".join(m.get("body", b"") for m in sent))
        self.assertEqual(self.calls, [])
        self.cancel.assert_not_awaited()

    async def test_health_never_acquires_or_updates_rate_gate(self):
        with patch.object(core.Gate, "acquire", side_effect=AssertionError("Health must not acquire rate gate")), \
                patch.object(core.Gate, "release", side_effect=AssertionError("Health must not update rate gate")):
            for _ in range(core.REQUESTS_PER_MINUTE + 1):
                response = await self.client.get("/health", headers=self.headers)
                self.assertEqual(response.status_code, 200)
        self.assertEqual(self.calls, [])
        self.cancel.assert_not_awaited()

    async def test_bad_auth_never_reads_body_or_calls_actor(self):
        for headers in ([], [(b"authorization", b"Bearer wrong")],
                        [(b"authorization", ("Bearer " + TOKEN).encode())] * 2):
            receive = AsyncMock(side_effect=AssertionError("Body must not be read"))
            sent = []
            async def send(message):
                sent.append(message)
            scope = self.scope(headers=headers)
            await self.app(scope, receive, send)
            self.assertEqual(sent[0]["status"], 401)
            receive.assert_not_awaited()
        self.assertEqual(self.calls, [])

    async def test_strict_json_headers_and_body_byte_limits(self):
        cases = [(b"{", {}, 400), (b"\xff", {}, 400), (b'{"x":NaN}', {}, 400),
                 (b'{"messages":[],"variant":"r3_response","variant":"r3_response"}', {}, 400),
                 (b"x", {"content-type": "text/plain"}, 415), (b"x", {"content-encoding": "gzip"}, 415),
                 (b"x", {"content-length": "-1"}, 400), (b"x", {"content-length": "2"}, 400),
                 (b"x" * 65537, {}, 413)]
        for body, headers, expected in cases:
            app = core.create_api(TOKEN, contract(), self.remote, self.cancel)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                response = await client.post("/stream", content=body,
                    headers={**self.headers, "content-type": "application/json", **headers})
            self.assertEqual(response.status_code, expected)
        async def body():
            yield b"x" * 40000
            yield b"x" * 40000
        response = await self.client.post("/stream", content=body(), headers={**self.headers, "content-type": "application/json"})
        self.assertEqual(response.status_code, 413)
        response = await self.client.post("/stream?token=not-allowed", json=self.payload, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_rate_limit_rejects_before_actor_and_no_cors(self):
        for _ in range(core.REQUESTS_PER_MINUTE):
            response = await self.client.post("/stream", json=self.payload, headers=self.headers)
            self.assertEqual(response.status_code, 200)
        response = await self.client.post("/stream", json=self.payload, headers={**self.headers, "Origin": "https://other.test"})
        self.assertEqual(response.status_code, 429)
        self.assertIn("retry-after", response.headers)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertEqual(len(self.calls), core.REQUESTS_PER_MINUTE)

    async def test_sse_preserves_actual_history_identity_and_actor_events(self):
        self.payload["messages"] = [{"role": "system", "content": "Twi please"}] + [
            {"role": "assistant" if i % 2 else "user", "content": f"Turn {i}"} for i in range(15)]
        response = await self.post()
        actual = decode_sse(response)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["cache-control"], "no-store")
        args = self.calls[0]
        self.assertEqual(args[:3], (self.payload["messages"], "r3_response", SHA))
        self.assertRegex(args[3], r"^[a-f0-9]{32}$")
        self.assertEqual(actual, events(*args))
        self.assertEqual(self.closed, [True])
        self.cancel.assert_not_awaited()

    async def test_shared_real_actor_stream_produces_accepted_protocol(self):
        def reply(messages, variant, on_text, cancelled):
            on_text("Hello")
            on_text(" there")
            return result(messages)
        runtime = SimpleNamespace(identity=lambda _: contract().identity, reply=reply)
        async def remote(messages, variant, sha, request_id):
            for event in demo.stream_reply(lambda: runtime, manifest(), messages, variant, sha, request_id, "fc-realshared"):
                yield event
        actual = decode_sse(await self.post(remote))
        self.assertEqual([e["type"] for e in actual], ["start", "delta", "delta", "final"])
        self.assertEqual(actual[0]["call_id"], "fc-realshared")
        self.cancel.assert_not_awaited()

    async def test_mismatched_start_is_never_forwarded_or_used_to_cancel(self):
        for field, value in (("identity", {}), ("messages_sha256", "f" * 64),
                             ("protocol", "r1-live-v1"), ("request_id", "0" * 32), ("call_id", "arbitrary")):
            async def remote(*args):
                start = events(*args)[0]
                start[field] = value
                yield start
            actual = decode_sse(await self.post(remote))
            self.assertEqual([e["type"] for e in actual], ["error"])
        self.cancel.assert_not_awaited()

    async def test_bad_sequence_missing_or_invalid_final_never_succeeds(self):
        def mutation(values, kind):
            if kind == "no_start":
                return values[1:]
            if kind == "no_final":
                return values[:-1]
            if kind == "double_start":
                return values[:1] + values
            if kind == "after_final":
                return values + values[1:2]
            if kind == "bad_index":
                values[1]["index"] = True
            if kind == "empty_delta":
                values[1]["text"] = ""
            if kind == "large_delta":
                values[1]["text"] = "x" * 65537
            if kind == "bad_unicode":
                values[1]["text"] = "\ud800"
            if kind == "wrong_prompt":
                values[-1]["result"]["prompt_ids_sha256"] = "f" * 64
            if kind == "wrong_text":
                values[-1]["result"]["output"] = "Changed"
            return values
        for kind in ("no_start", "no_final", "double_start", "after_final", "bad_index", "empty_delta",
                     "large_delta", "bad_unicode", "wrong_prompt", "wrong_text"):
            async def remote(*args):
                for value in mutation(events(*args), kind):
                    yield value
            actual = decode_sse(await self.post(remote))
            self.assertEqual(actual[-1]["type"], "error", kind)
            self.assertNotIn("final", [e["type"] for e in actual], kind)
        self.assertEqual(self.cancel.await_count, 9)
        self.assertEqual({c.args for c in self.cancel.await_args_list}, {("fc-ownrequest",)})

    async def test_remote_failure_is_sanitized_without_retry_or_fallback(self):
        async def remote(*args):
            yield events(*args)[0]
            raise RuntimeError(TOKEN + " PRIVATE PROMPT")
        response = await self.post(remote)
        self.assertNotIn(TOKEN, response.text)
        self.assertNotIn("PRIVATE PROMPT", response.text)
        self.assertEqual(decode_sse(response)[-1]["error"], "Research model unavailable")
        self.cancel.assert_awaited_once_with("fc-ownrequest")

    async def test_timeout_and_heartbeat_preserve_live_delivery(self):
        async def remote(*args):
            yield events(*args)[0]
            await asyncio.Event().wait()
        with patch.object(core, "REQUEST_SECONDS", .04), patch.object(core, "HEARTBEAT_SECONDS", .005):
            response = await self.post(remote)
        self.assertIn(": keep-alive", response.text)
        self.assertEqual([e["type"] for e in decode_sse(response)], ["start", "error"])
        self.cancel.assert_awaited_once_with("fc-ownrequest")

    async def test_body_timeout_releases_gate_without_remote_submission(self):
        endpoint = next(r.endpoint for r in self.app.routes if r.path == "/stream")
        async def receive():
            await asyncio.Event().wait()
        with patch.object(core, "BODY_SECONDS", .01), self.assertRaises(core.HTTPException) as raised:
            await endpoint(Request(self.scope(), receive))
        self.assertEqual(raised.exception.status_code, 408)
        self.assertEqual(self.calls, [])
        response = await self.client.post("/stream", json=self.payload, headers=self.headers)
        self.assertEqual(decode_sse(response)[-1]["type"], "final")

    async def test_failed_cancellation_quarantines_new_requests(self):
        async def remote(*args):
            yield events(*args)[0]
            raise RuntimeError("upstream failed")
        self.cancel.side_effect = RuntimeError("cancel failed")
        app = core.create_api(TOKEN, contract(), remote, self.cancel)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            first = await client.post("/stream", json=self.payload, headers=self.headers)
            second = await client.post("/stream", json=self.payload, headers=self.headers)
        self.assertEqual(decode_sse(first)[-1]["type"], "error")
        self.assertEqual(second.status_code, 429)
        self.assertGreaterEqual(int(second.headers["retry-after"]), 899)
        self.cancel.assert_awaited_once_with("fc-ownrequest")

    def scope(self, headers=None, spec="2.0"):
        return {"type": "http", "http_version": "1.1", "asgi": {"version": "3.0", "spec_version": spec},
                "method": "POST", "path": "/stream", "raw_path": b"/stream", "query_string": b"",
                "headers": headers if headers is not None else [
                    (b"authorization", ("Bearer " + TOKEN).encode()), (b"content-type", b"application/json")],
                "scheme": "https", "server": ("test", 443), "client": ("test", 1)}

    async def test_asgi_disconnect_and_send_failure_cancel_and_release(self):
        for spec in ("2.0", "2.4"):
            self.cancel.reset_mock()
            sent, disconnected, consumed = [], asyncio.Event(), False
            async def receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": json.dumps(self.payload).encode(), "more_body": False}
                await disconnected.wait()
                return {"type": "http.disconnect"}
            async def send(value):
                sent.append(value)
                if value["type"] == "http.response.body" and b'"type":"start"' in value["body"]:
                    if spec == "2.4":
                        raise OSError("client gone")
                    disconnected.set()
                    await asyncio.sleep(.01)
            try:
                await self.app(self.scope(spec=spec), receive, send)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.assertEqual(type(exc).__name__, "ClientDisconnect")
            self.cancel.assert_awaited_once_with("fc-ownrequest")
            response = await self.client.post("/stream", json=self.payload, headers=self.headers)
            self.assertEqual(decode_sse(response)[-1]["type"], "final")

    async def test_disconnect_before_start_quarantines_without_cancelling_other_calls(self):
        entered = asyncio.Event()
        async def remote(*args):
            entered.set()
            await asyncio.Event().wait()
            yield events(*args)[0]
        app = core.create_api(TOKEN, contract(), remote, self.cancel)
        consumed = False
        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": json.dumps(self.payload).encode(), "more_body": False}
            await entered.wait()
            return {"type": "http.disconnect"}
        async def send(_):
            pass
        await asyncio.wait_for(app(self.scope(), receive, send), 1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            response = await client.post("/stream", json=self.payload, headers=self.headers)
        self.assertEqual(response.status_code, 429)
        self.cancel.assert_not_awaited()

    async def test_first_delta_before_final_and_concurrent_request_rejected(self):
        release = asyncio.Event()
        async def remote(*args):
            for value in events(*args)[:-1]:
                yield value
            await release.wait()
            yield events(*args)[-1]
        app = core.create_api(TOKEN, contract(), remote, self.cancel)
        endpoint = next(r.endpoint for r in app.routes if r.path == "/stream")
        async def receive():
            return {"type": "http.request", "body": json.dumps(self.payload).encode(), "more_body": False}
        response = await endpoint(Request(self.scope(), receive))
        self.assertIn(b'"type":"start"', await anext(response.body_iterator))
        self.assertIn(b'"type":"delta"', await anext(response.body_iterator))
        self.assertFalse(release.is_set())
        with self.assertRaises(core.HTTPException) as raised:
            await endpoint(Request(self.scope(), receive))
        self.assertEqual(raised.exception.status_code, 429)
        await response.body_iterator.aclose()
        response.gate.release()
        self.cancel.assert_awaited_once_with("fc-ownrequest")


class DeploymentTests(unittest.TestCase):
    def test_completed_synthetic_bundle_loads_and_mutated_code_fails_before_import(self):
        import prepare_understanding_r3_inference as builder
        import r3_inference as r3
        from test_prepare_understanding_r3_inference import fixture

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, _, readback, _ = fixture(root)
            for name in core.HELPERS:
                (parent / "code" / (name + ".py")).write_text("# Synthetic helper, not executable native inference.\n")
            value = inference.read(parent / "manifest.json")
            value["files"] = {p.relative_to(parent).as_posix(): demo.sha(p) for p in parent.rglob("*")
                              if p.is_file() and p.name != "manifest.json"}
            inference.save(parent / "manifest.json", value)
            parent_sha = demo.sha(parent / "manifest.json")
            bundle = root / "bundle"
            with patch.object(r3, "R2_BUNDLE_SHA", parent_sha):
                builder.prepare(parent, parent_sha, root / "receipt.json", root / "terminal.json", readback, bundle)
            source = bundle / "code/r3_inference.py"
            source.write_text(source.read_text().replace(r3.R2_BUNDLE_SHA, parent_sha))
            value = inference.read(bundle / "manifest.json")
            value["files"]["code/r3_inference.py"] = demo.sha(source)
            inference.save(bundle / "manifest.json", value)
            checksum = demo.sha(bundle / "manifest.json")
            inference.save(bundle / "SEALED.json", {"manifest_sha256": checksum})
            with patch.object(sys, "path", list(sys.path)):
                loaded, _ = core.load_bundle(bundle, checksum)
                self.assertEqual(loaded["r3_training"]["training_completed"], True)
                (bundle / "code/afrique_response_core.py").write_text("raise AssertionError('must not execute')\n")
                with self.assertRaisesRegex(ValueError, "inference code changed"):
                    core.load_bundle(bundle, checksum)
            self.assertFalse(list(bundle.rglob("__pycache__")))

    def test_bundle_requires_matching_explicit_seal_and_r3_version_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            value = manifest()
            value["files"] = {"code/" + n + ".py": "c" * 64 for n in core.HELPERS}
            (folder / "manifest.json").write_text(json.dumps(value))
            sha = demo.sha(folder / "manifest.json")
            with patch.object(demo, "bound_bundle", return_value=(value, object())) as bound:
                for expected in (None, "", SHA):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        core.load_bundle(folder, expected)
                with self.assertRaises(FileNotFoundError):
                    core.load_bundle(folder, sha)
                (folder / "SEALED.json").write_text(json.dumps({"manifest_sha256": sha}))
                core.load_bundle(folder, sha)
                bound.assert_called_once_with(folder.resolve(), sha)
                value["version"] = "r2-paired-inference-v1"
                (folder / "manifest.json").write_text(json.dumps(value))
                sha = demo.sha(folder / "manifest.json")
                (folder / "SEALED.json").write_text(json.dumps({"manifest_sha256": sha}))
                with self.assertRaisesRegex(ValueError, "Only completed R3"):
                    core.load_bundle(folder, sha)
                self.assertEqual(bound.call_count, 1)

    def test_native_helpers_must_resolve_from_verified_bundle(self):
        fake = {name: SimpleNamespace(__file__="/bundle/code/" + name + ".py") for name in core.HELPERS}
        native = fake["afrique_instruction_baseline_core"]
        native.prompt_ids, native.completion = prompt_ids, completion
        fake["check_afrique_training_contract"].load_tokenizer = Mock(return_value=Tokenizer())
        with patch.object(core, "load_bundle", return_value=(manifest(), inference)), \
                patch.object(core.importlib.metadata, "version", return_value=inference.RUNTIME["transformers"]), \
                patch.object(core.importlib, "import_module", side_effect=fake.__getitem__):
            actual = core.load_contract("/bundle", SHA)
            self.assertIs(actual.prompt_ids, prompt_ids)
            fake[core.HELPERS[0]].__file__ = "/foreign/helper.py"
            with self.assertRaisesRegex(ValueError, "Foreign"):
                core.load_contract("/bundle", SHA)

    def test_tokenizer_version_mismatch_stops_before_loading_helpers(self):
        with patch.object(core, "load_bundle", return_value=(manifest(), inference)), \
                patch.object(core.importlib.metadata, "version", return_value="wrong-version"), \
                patch.object(core.importlib, "import_module") as imported:
            with self.assertRaisesRegex(ValueError, "tokenizer runtime differs"):
                core.load_contract("/bundle", SHA)
            imported.assert_not_called()

    def test_deployment_cpu_bounds_named_secret_and_actual_actor_routing(self):
        path = ROOT / "modal/research_r3_gateway.py"
        source = path.read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gateway")
        options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
        for key, value in {"cpu": 1, "memory": 2048, "timeout": 960, "min_containers": 0,
                           "max_containers": 1, "include_source": False}.items():
            self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertNotIn("gpu", options)
        self.assertNotIn("retries", options)
        self.assertEqual(fn.decorator_list[1].keywords[0].arg, "max_inputs")
        self.assertIn("from_name", ast.unparse(options["secrets"]))
        self.assertNotIn("from_dict", source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "env":
                self.assertNotIn("RESEARCH_R3_TOKEN", ast.unparse(node.args[0]))
        with patch.dict(os.environ, {"GHA_R3_DEMO_BUNDLE": str(ROOT), "GHA_R3_DEMO_SHA": SHA}), \
                patch.object(core, "load_bundle", return_value=(manifest(), inference)):
            spec = importlib.util.spec_from_file_location("test_r3_gateway_deployment", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            self.addCleanup(sys.modules.pop, spec.name, None)
            spec.loader.exec_module(module)
        actor = SimpleNamespace(stream=SimpleNamespace(remote_gen=SimpleNamespace(aio=Mock(return_value="iterator"))))
        with patch.object(module.modal.Cls, "from_name", return_value=lambda: actor) as target:
            self.assertEqual(module.remote(MESSAGES, core.VARIANT, SHA, "1" * 32), "iterator")
            target.assert_called_once_with("test-r3-private", "UnderstandingR3Demo")
            actor.stream.remote_gen.aio.assert_called_once_with(MESSAGES, core.VARIANT, SHA, "1" * 32)
        call = SimpleNamespace(cancel=SimpleNamespace(aio=AsyncMock()))
        with patch.object(module.modal.FunctionCall, "from_id", return_value=call) as lookup:
            asyncio.run(module.cancel("fc-ownrequest"))
        lookup.assert_called_once_with("fc-ownrequest")
        call.cancel.aio.assert_awaited_once_with(terminate_containers=False)


if __name__ == "__main__":
    unittest.main()
