"""CPU-only R3 HTTP boundary. No model construction, fallback, or persistence.

The bearer credential belongs on a trusted server, never in browser code. SSE
objects retain the actor's schema; transport EOF without final is not success.
The in-memory limiter is per container, not a durable account/spending quota.
"""
import asyncio
from collections import deque
import copy
import hmac
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
import unicodedata
import uuid

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import understanding_r2_demo as demo

VERSION = "research-r3-gateway-v1"
VARIANT = "r3_response"
MAX_BODY = 65536
MAX_MESSAGES = 129
MAX_MESSAGE_CHARS = 16000
CONTEXT_TOKENS = 4096
REQUEST_SECONDS = 900
BODY_SECONDS = 15
CLEANUP_SECONDS = 10
HEARTBEAT_SECONDS = 15
REQUESTS_PER_MINUTE = 6
HELPERS = ("afrique_response_core", "check_afrique_training_contract", "afrique_instruction_baseline_core")
RESULT_KEYS = {"identity", "messages_sha256", "prompt_ids_sha256", "decoding", "output_ids",
               "output", "seconds", "diagnostics", "finish_reason", "complete"}
require, digest = demo.require, demo.digest


def parse_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def deployment_token(value):
    require(isinstance(value, str) and 32 <= len(value) <= 4096
            and all(33 <= ord(c) <= 126 for c in value), "Invalid gateway credential")
    return value.encode("ascii")


def load_bundle(folder, expected_sha):
    folder = Path(folder).resolve()
    require(demo.is_sha(expected_sha), "Explicit R3 manifest SHA required")
    require(not (folder / "manifest.json").is_symlink() and not (folder / "SEALED.json").is_symlink(),
            "Linked bundle commitment")
    require(demo.sha(folder / "manifest.json") == expected_sha
            and parse_json((folder / "SEALED.json").read_bytes()) == {"manifest_sha256": expected_sha},
            "Sealed R3 bundle required")
    raw = parse_json((folder / "manifest.json").read_bytes())
    require(raw.get("version") == "r3-comparison-inference-v1", "Only completed R3 bundles are supported")
    require(all("code/" + name + ".py" in raw["files"] for name in HELPERS), "Missing native validation helpers")
    return demo.bound_bundle(folder, expected_sha)


def load_contract(folder, expected_sha):
    manifest, inference = load_bundle(folder, expected_sha)
    require(importlib.metadata.version("transformers") == manifest["runtime"]["transformers"],
            "Native tokenizer runtime differs")
    code = Path(folder).resolve() / "code"
    modules = {}
    for name in HELPERS:
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == code / (name + ".py"), "Foreign validation helper loaded")
        modules[name] = module
    native = modules["afrique_instruction_baseline_core"]
    tokenizer = modules["check_afrique_training_contract"].load_tokenizer(Path(folder) / "tokenizer")
    return Contract(manifest, expected_sha, inference.validate_messages, tokenizer, native.prompt_ids, native.completion)


def cpu_preflight(folder, expected_sha):
    """Native tokenizer/validation probes only; never instantiate an inference Runtime."""
    started = time.monotonic()
    contract = load_contract(folder, expected_sha)
    cases = {
        "english": [{"role": "user", "content": "hello"}],
        "twi": [{"role": "user", "content": "Mepa wo ky\u025bw, boa me."}],
        "unicode": [{"role": "user", "content": "Meda wo ase. \U0001f642 e\u0301\n\tFollow up."}],
        "system": [{"role": "system", "content": "Answer in Twi."}, {"role": "user", "content": "Hello."}],
        "history15": [{"role": "assistant" if i % 2 else "user", "content": f"Turn {i}: hello."} for i in range(15)],
        "history129": [{"role": "assistant" if i % 2 else "user", "content": "x"} for i in range(129)],
    }
    accepted = []
    for name, messages in cases.items():
        before = digest(messages)
        actual, ids = contract.request({"messages": messages, "variant": VARIANT})
        native = contract.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                       enable_thinking=False, add_generation_prompt=True)
        require(actual == messages and digest(messages) == before and ids == native, "Native history/prefix differs")
        accepted.append({"case": name, "messages": len(messages), "messages_sha256": before,
                         "prompt_tokens": len(ids), "prompt_ids_sha256": digest(ids)})
    require(accepted[0]["messages_sha256"] == "d98167dd28f22e330824942ba4d4ce217c2411a0d1141d60b40fe4cb8dc0d232",
            "Cross-language JSON digest differs")

    def repeated(n):
        return [{"role": "user", "content": ("x " * n).rstrip()}]

    low, high = 1, CONTEXT_TOKENS + 1
    while high - low > 1:
        n = (low + high) // 2
        ids = contract.tokenizer.apply_chat_template(repeated(n), tokenize=True, return_dict=False,
                                                     enable_thinking=False, add_generation_prompt=True)
        if len(ids) + contract.manifest["decoding"]["max_new_tokens"] <= CONTEXT_TOKENS:
            low = n
        else:
            high = n
    _, boundary = contract.request({"messages": repeated(low), "variant": VARIANT})
    rejected = []
    negative = {"native_context_overflow": repeated(high), "invalid_unicode": [{"role": "user", "content": "bad\ufffd"}],
                "unpaired_surrogate": [{"role": "user", "content": "bad\ud800"}],
                "invalid_roles": [{"role": "assistant", "content": "Not a user prompt"}]}
    tokens = parse_json((Path(folder) / "tokenizer/tokenizer.json").read_bytes())["added_tokens"]
    for token in tokens:
        if token["special"]:
            negative["reserved_" + str(token["id"])] = [{"role": "user", "content": "Test " + token["content"]}]
    for name, messages in negative.items():
        try:
            contract.request({"messages": messages, "variant": VARIANT})
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError("Invalid source accepted: " + name)
    require(boundary and len(boundary) + 256 <= CONTEXT_TOKENS, "Invalid native context boundary")

    messages, ids = contract.request({"messages": cases["twi"], "variant": VARIANT})
    generated = contract.tokenizer.encode("Meda wo ase.", add_special_tokens=False) + [contract.tokenizer.eos_token_id]
    diagnostics = contract.completion(contract.tokenizer, generated, 256, False, 0.0)
    value = {"identity": contract.identity, "messages_sha256": digest(messages), "prompt_ids_sha256": digest(ids),
             "decoding": contract.manifest["decoding"], "output_ids": generated, "output": diagnostics["prediction"],
             "seconds": 0.0, "diagnostics": diagnostics, "finish_reason": diagnostics["finish_reason"],
             "complete": diagnostics["complete"]}
    contract.final(value, messages, ids, value["output"])
    require(value["complete"] and value["output"] == "Meda wo ase.", "Synthetic native completion differs")
    return {"version": "r3-gateway-cpu-preflight-v1", "state": "passed", "identity": contract.identity,
            "actor": contract.actor, "bundle_artifacts": len(contract.manifest["files"]),
            "runtime": {name: importlib.metadata.version(name) for name in ("transformers", "fastapi", "starlette", "anyio", "jinja2")},
            "vocabulary": len(contract.tokenizer), "eos_token_id": contract.tokenizer.eos_token_id,
            "accepted": accepted, "rejected": rejected,
            "context_boundary": {"prompt_tokens": len(boundary), "reserved_output_tokens": 256,
                                 "prompt_ids_sha256": digest(boundary), "next_larger_prompt_rejected": True},
            "synthetic_completion": value, "seconds": time.monotonic() - started,
            "runtime_constructed": False, "gpu_called": False, "actor_called": False,
            "claims": "CPU bundle/native tokenizer contract only; no generated answer or GPU readiness proof"}


class Contract:
    def __init__(self, manifest, expected_sha, validate_messages, tokenizer, prompt_ids, completion):
        require(manifest.get("version") == "r3-comparison-inference-v1" and demo.is_sha(expected_sha),
                "R3 identity required")
        require(manifest.get("private") is True and manifest.get("training") is False
                and manifest.get("default_variant") == VARIANT, "Private R3 inference required")
        self.manifest, self.sha = copy.deepcopy(manifest), expected_sha
        self.actor = demo.actor_target(manifest)
        self.identity = demo.identity(manifest, expected_sha, VARIANT)
        require(demo.is_sha(self.identity["adapter_weights_sha256"]), "Missing R3 adapter identity")
        self.validate_messages, self.tokenizer = validate_messages, tokenizer
        self.prompt_ids, self.completion = prompt_ids, completion

    def request(self, value):
        require(isinstance(value, dict) and set(value) == {"messages", "variant"}
                and value["variant"] == VARIANT, "Only r3_response requests are supported")
        messages = value["messages"]
        require(isinstance(messages, list) and 1 <= len(messages) <= MAX_MESSAGES, "Invalid conversation")
        for message in messages:
            require(isinstance(message, dict) and set(message) == {"role", "content"}, "Invalid message")
            text = message["content"]
            require(isinstance(text, str) and 0 < len(text) <= MAX_MESSAGE_CHARS, "Invalid message length")
            require(not any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
                    (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text), "Invalid Unicode")
        self.validate_messages(messages)
        ids = self.prompt_ids(self.tokenizer, messages, {
            "context_tokens": CONTEXT_TOKENS, "max_new_tokens": self.manifest["decoding"]["max_new_tokens"]})
        require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < len(self.tokenizer) for i in ids)
                and len(ids) + self.manifest["decoding"]["max_new_tokens"] <= CONTEXT_TOKENS,
                "Native context limit exceeded")
        return messages, ids

    def final(self, value, messages, ids, text):
        require(isinstance(value, dict) and set(value) == RESULT_KEYS, "Invalid final schema")
        demo.validate_reply(value, self.identity, messages, self.manifest["decoding"], text)
        require(value["prompt_ids_sha256"] == digest(ids), "Native prompt differs")
        require(all(i < len(self.tokenizer) for i in value["output_ids"]), "Unknown generated token")
        require(value["seconds"] <= 300, "Actor execution time exceeded")
        diagnostics = value["diagnostics"]
        require(isinstance(diagnostics, dict), "Missing completion evidence")
        actual = self.completion(self.tokenizer, value["output_ids"], self.manifest["decoding"]["max_new_tokens"],
                                 diagnostics.get("finish_reason") == "time_limit", value["seconds"])
        require(actual == diagnostics, "Native completion evidence differs")

    def health(self):
        return {"service": VERSION, "configured": True, "upstream_checked": False,
                "protocol": demo.PROTOCOL, "actor": self.actor, "identity": self.identity,
                "variants": [VARIANT], "decoding": self.manifest["decoding"],
                "limits": {"body_bytes": MAX_BODY, "messages": MAX_MESSAGES,
                           "message_chars": MAX_MESSAGE_CHARS, "context_tokens": CONTEXT_TOKENS,
                           "request_seconds": REQUEST_SECONDS, "active_streams": 1,
                           "requests_per_minute_per_container": REQUESTS_PER_MINUTE}}


class Gate:
    """Single event-loop admission, acquired before reading an authenticated body."""
    def __init__(self, clock=time.monotonic):
        self.clock, self.recent = clock, deque()
        self.active, self.blocked_until = False, 0.0

    def acquire(self):
        now = self.clock()
        while self.recent and now - self.recent[0] >= 60:
            self.recent.popleft()
        delay = self.blocked_until - now
        if self.active or delay > 0 or len(self.recent) >= REQUESTS_PER_MINUTE:
            if len(self.recent) >= REQUESTS_PER_MINUTE:
                delay = max(delay, 60 - (now - self.recent[0]))
            raise HTTPException(429, "Research gateway busy", headers={"Retry-After": str(max(1, math.ceil(delay)))})
        self.recent.append(now)
        self.active = True

    def quarantine(self):
        # An unknown/uncancelled remote submission must not invite immediate retries.
        self.blocked_until = max(self.blocked_until, self.clock() + REQUEST_SECONDS)

    def release(self):
        self.active = False


def sse(event):
    encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    require(len(encoded) <= 1024 * 1024, "Oversized stream event")
    return b"data: " + encoded + b"\n\n"


async def verified_events(contract, messages, ids, remote, cancel, gate):
    request_id = uuid.uuid4().hex
    source, pending, call_id, final = None, None, None, None
    submitted, finished, text, count = False, False, "", 0
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            source = aiter(remote(messages, VARIANT, contract.sha, request_id))
            submitted = True
            while True:
                pending = asyncio.create_task(anext(source))
                while not (await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS))[0]:
                    yield b": keep-alive\n\n"
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    require(final is not None, "Stream ended without final")
                    finished = True
                    yield sse(final)
                    break
                require(final is None and isinstance(event, dict) and event.get("request_id") == request_id,
                        "Stream request/order differs")
                kind = event.get("type")
                if kind == "start":
                    require(call_id is None and set(event) == {"type", "protocol", "request_id", "call_id",
                            "identity", "messages_sha256"} and event["protocol"] == demo.PROTOCOL
                            and event["identity"] == contract.identity and event["messages_sha256"] == digest(messages),
                            "Stream identity differs")
                    require(isinstance(event["call_id"], str) and re.fullmatch(r"fc-[A-Za-z0-9]+", event["call_id"]),
                            "Invalid call identity")
                    call_id = event["call_id"]
                    yield sse(event)
                elif kind == "delta":
                    require(call_id is not None and set(event) == {"type", "request_id", "index", "text"}
                            and type(event["index"]) is int and event["index"] == count and count < 1024,
                            "Invalid delta sequence")
                    chunk = event["text"]
                    require(isinstance(chunk, str) and chunk and len(text) + len(chunk) <= demo.MAX_OUTPUT_CHARS,
                            "Invalid delta text")
                    encoded = sse(event)
                    text += chunk
                    count += 1
                    yield encoded
                elif kind == "final":
                    require(call_id is not None and set(event) == {"type", "request_id", "result"}, "Invalid final event")
                    contract.final(event["result"], messages, ids, text)
                    final = event
                else:
                    raise ValueError("Unknown stream event")
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception:
        yield sse({"type": "error", "request_id": request_id, "error": "Research model unavailable"})
    finally:
        # Shield cleanup from Starlette's disconnect cancellation scope, but bound it.
        with anyio.CancelScope(shield=True):
            if submitted and not finished:
                acknowledged = False
                if call_id is not None:
                    try:
                        await asyncio.wait_for(cancel(call_id), CLEANUP_SECONDS)
                        acknowledged = True
                    except (Exception, asyncio.CancelledError):
                        pass
                if not acknowledged:
                    gate.quarantine()
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                try:
                    await asyncio.wait_for(pending, CLEANUP_SECONDS)
                except (Exception, asyncio.CancelledError):
                    pass
            if source is not None and hasattr(source, "aclose"):
                try:
                    await asyncio.wait_for(source.aclose(), CLEANUP_SECONDS)
                except (Exception, asyncio.CancelledError):
                    pass


class ClosingStream(StreamingResponse):
    def __init__(self, events, gate):
        super().__init__(events, media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"})
        self.gate = gate

    async def __call__(self, scope, receive, send):
        try:
            # Also bound a stalled downstream send, not just upstream generation.
            async with asyncio.timeout(REQUEST_SECONDS + 3 * CLEANUP_SECONDS):
                await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    self.gate.release()


def create_api(token, contract, remote, cancel):
    token_bytes = deployment_token(token)
    gate = Gate()
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, debug=False)

    def authenticate(request):
        auth = request.headers.getlist("authorization")
        raw = auth[0].encode("latin-1") if len(auth) == 1 else b""
        scheme, _, supplied = raw.partition(b" ")
        if not hmac.compare_digest(supplied, token_bytes) or scheme.lower() != b"bearer":
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @api.get("/health")
    async def health(request: Request):
        authenticate(request)
        return JSONResponse(contract.health(), headers={"Cache-Control": "no-store"})

    @api.post("/stream")
    async def stream(request: Request):
        authenticate(request)
        gate.acquire()
        try:
            require(not request.url.query, "Query parameters are not supported")
            types = request.headers.getlist("content-type")
            encodings = request.headers.getlist("content-encoding")
            if (len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json"
                    or len(encodings) > 1 or (encodings and encodings[0].lower() != "identity")):
                raise HTTPException(415, "Uncompressed JSON required")
            lengths = request.headers.getlist("content-length")
            require(len(lengths) <= 1 and (not lengths or (len(lengths[0]) <= 20
                    and lengths[0].isascii() and lengths[0].isdecimal())), "Invalid content length")
            if lengths and int(lengths[0]) > MAX_BODY:
                raise HTTPException(413, "Request too large")
            body = bytearray()
            async with asyncio.timeout(BODY_SECONDS):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BODY:
                        raise HTTPException(413, "Request too large")
                    body.extend(chunk)
            require(not lengths or len(body) == int(lengths[0]), "Content length differs")
            messages, ids = contract.request(parse_json(body.decode("utf-8")))
            return ClosingStream(verified_events(contract, messages, ids, remote, cancel, gate), gate)
        except BaseException as exc:
            gate.release()
            if isinstance(exc, TimeoutError):
                raise HTTPException(408, "Request timed out") from None
            if isinstance(exc, (ValueError, KeyError, TypeError, RecursionError)):
                raise HTTPException(400, "Invalid request or context too long") from None
            raise

    return api
