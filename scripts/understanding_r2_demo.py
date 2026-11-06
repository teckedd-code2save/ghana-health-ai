#!/usr/bin/env python3
"""Local-only R2 audition of an already deployed private Modal class.

GHA_R2_DEMO_BUNDLE and GHA_R2_DEMO_SHA are required (or CLI equivalents).
Commands: verify (local files only), chat (127.0.0.1:7866). Neither deploys,
trains, downloads a model, nor makes a remote call at startup. GPU inference
belongs exclusively to the bundled r2_inference.Runtime in the private actor.
"""
import argparse
import asyncio
from contextlib import closing, contextmanager
import copy
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import sqlite3
import sys
import threading
import time
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ghana-understanding-r2-demo-private"
CLASS_NAME = "UnderstandingR2Demo"
PROTOCOL = "r2-demo-stream-v1"
STORE = ROOT / "tmp/understanding-r2-demo/audition.sqlite3"
DEFAULT_VARIANT = "r2_meaning"
LABELS = {"r2_meaning": "R2 \u00b7 language + meaning", "r2_response": "R2 \u00b7 final response stage",
          "r2_language": "R2 / language", "r1_meaning": "R1 / meaning",
          "r1_response": "R1 / response", "base": "Base / unadapted",
          "r3_response": "R3 / final response stage"}
REQUEST_TIMEOUT_SECONDS = 900  # Includes cold start; actor execution is bounded separately.
MAX_OUTPUT_CHARS = 65536
FINISHED = {"complete", "incomplete", "cancelled", "error"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def bound_bundle(folder, expected_sha):
    """Check bundled code bytes before importing; never construct Runtime locally."""
    require(is_sha(expected_sha), "Expected bundle SHA-256 is required")
    folder = Path(folder).resolve()
    require(sha(folder / "manifest.json") == expected_sha, "Bundle identity changed")
    manifest = json.loads((folder / "manifest.json").read_text())
    r3 = manifest.get("version") == "r3-comparison-inference-v1"
    runtime_file = "code/r3_inference.py" if r3 else "code/r2_inference.py"
    source = folder / runtime_file
    require(source.resolve().is_relative_to(folder) and not source.is_symlink(), "Unsafe runtime path")
    require(sha(source) == manifest["files"][runtime_file], "Bundled inference code changed")
    if r3:
        # R3 imports its sibling runtime; validate every code file before executing it.
        require("code/r2_inference.py" in manifest["files"], "Missing bundled sibling runtime")
        for name, checksum in manifest["files"].items():
            if name.startswith("code/"):
                path = folder / name
                require(path.resolve().is_relative_to(folder) and not path.is_symlink()
                        and sha(path) == checksum, "Bundled inference code changed")
    # Helpers must resolve from this bundle, never from the frozen training tree.
    code = str(folder / "code")
    if code in sys.path:
        sys.path.remove(code)
    sys.path.insert(0, code)
    spec = importlib.util.spec_from_file_location("_r2_demo_inference_" + expected_sha, source)
    module = importlib.util.module_from_spec(spec)
    # Verified bundle code must not create/replace caches inside frozen evidence.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    manifest = module.load_bundle(folder, expected_sha)
    return manifest, module


def available_labels(manifest):
    return {key: label for key, label in LABELS.items() if key in manifest["variants"]}


def default_variant(manifest):
    return "r3_response" if manifest.get("version") == "r3-comparison-inference-v1" else DEFAULT_VARIANT


def actor_target(manifest, app_name=None, class_name=None):
    target = manifest.get("actor", {"app_name": APP_NAME, "class_name": CLASS_NAME})
    target = {"app_name": app_name or target["app_name"], "class_name": class_name or target["class_name"]}
    require(re.fullmatch(r"[a-z][a-z0-9-]{1,62}", target["app_name"])
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,80}", target["class_name"]), "Invalid private actor target")
    if manifest.get("version") == "r3-comparison-inference-v1":
        require(target["app_name"] != APP_NAME and target["class_name"] == "UnderstandingR3Demo",
                "R3 must use its separate actor")
    return target


def routed_events(manifest, app_name=None, class_name=None):
    target = actor_target(manifest, app_name, class_name)

    async def remote(messages, variant, expected_sha, request_id):
        import modal
        actor = modal.Cls.from_name(target["app_name"], target["class_name"])()
        async for event in actor.stream.remote_gen.aio(messages, variant, expected_sha, request_id):
            yield event
    return remote


def bundle_store(manifest, expected_sha, store=None):
    if manifest.get("version") != "r3-comparison-inference-v1":
        return Path(store) if store is not None else STORE
    path = Path(store) if store is not None else ROOT / "tmp/understanding-r3-demo" / expected_sha / "audition.sqlite3"
    # Do not initialize, migrate or write an R2 store accidentally passed to R3.
    for candidate, table in ((path, "turns"), (path.with_name(path.stem + ".comparisons.sqlite3"), "comparison_sessions")):
        if not candidate.exists():
            continue
        require(not candidate.is_symlink(), "Use a separate R3 store")
        with closing(sqlite3.connect(candidate.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                for (record,) in db.execute("SELECT record FROM " + table):
                    value = json.loads(record)
                    identity_sha = value.get("identity", {}).get("manifest_sha256", value.get("manifest_sha256"))
                    require(identity_sha == expected_sha, "Store belongs to another bundle; choose a separate R3 store")
    return path


def identity(manifest, expected_sha, variant):
    require(variant in manifest["variants"], "Unknown variant")
    adapter = manifest["variants"][variant]
    return {"manifest_sha256": expected_sha, "variant": variant, "model": manifest["model"],
            "revision": manifest["revision"], "base_weights_sha256": manifest["base_sha256"],
            "adapter_weights_sha256": None if adapter is None else adapter["files"]["adapter_model.safetensors"]}


def validate_reply(value, expected_identity, messages, decoding, streamed):
    require(isinstance(value, dict) and value.get("identity") == expected_identity, "Remote identity differs")
    require(value.get("messages_sha256") == digest(messages), "Remote history differs")
    require(is_sha(value.get("prompt_ids_sha256")), "Missing native prompt fingerprint")
    require(value.get("decoding") == decoding, "Remote decoding differs")
    output, ids = value.get("output"), value.get("output_ids")
    require(isinstance(output, str) and len(output) <= MAX_OUTPUT_CHARS and output == streamed,
            "Final output differs from streamed text")
    require(isinstance(ids, list) and 0 < len(ids) <= decoding["max_new_tokens"]
            and all(type(i) is int and i >= 0 for i in ids), "Invalid generated token evidence")
    seconds = value.get("seconds")
    require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0, "Invalid timing")
    reason, complete = value.get("finish_reason"), value.get("complete")
    require(reason in {"eos", "max_new_tokens", "time_limit", "unexpected_stop", "cancelled"}
            and type(complete) is bool, "Invalid completion status")
    require(not complete or (reason == "eos" and bool(output.strip())), "Incomplete output marked complete")
    require(reason != "max_new_tokens" or len(ids) == decoding["max_new_tokens"], "Invalid token-limit stop")
    diagnostics = value.get("diagnostics")
    if diagnostics is not None:
        require(isinstance(diagnostics, dict) and diagnostics.get("output_ids") == ids
                and diagnostics.get("prediction") == output and diagnostics.get("seconds") == seconds
                and (reason == "cancelled" or diagnostics.get("finish_reason") == reason)
                and complete == (diagnostics.get("complete") is True and reason != "cancelled"),
                "Completion diagnostics differ")
    return value


def stream_reply(runtime_factory, manifest, messages, variant, expected_sha, request_id, call_id):
    """Callback deltas cross a bounded queue, not a post-hoc split of the reply.

    The actor owns the runtime lock until this generator joins its worker. Modal's
    hard execution timeout remains the bound for an unresponsive CUDA operation.
    """
    require(is_sha(expected_sha), "Invalid request SHA")
    expected = identity(manifest, expected_sha, variant)
    require(isinstance(call_id, str) and call_id.startswith("fc-"), "Missing Modal call identity")
    require(isinstance(request_id, str) and re.fullmatch(r"[a-f0-9]{32}", request_id), "Invalid request ID")
    yield {"type": "start", "protocol": PROTOCOL, "request_id": request_id, "call_id": call_id,
           "identity": expected, "messages_sha256": digest(messages)}
    queue, cancelled, done = Queue(maxsize=32), threading.Event(), threading.Event()
    outcome = {}

    def emit(text):
        require(isinstance(text, str), "Runtime callback must contain text deltas")
        if not text:
            return
        while not cancelled.is_set():
            try:
                queue.put(text, timeout=.1)
                return
            except Full:
                pass

    def generate():
        try:
            runtime = runtime_factory()
            require(runtime.identity(variant) == expected, "Runtime identity differs")
            if cancelled.is_set():
                raise UserCancelled("Stopped during model loading")
            outcome["result"] = runtime.reply(messages, variant, on_text=emit, cancelled=cancelled.is_set)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=generate, name="r2-demo-generation", daemon=False)
    worker.start()
    chunks, length = [], 0
    try:
        while not done.is_set() or not queue.empty():
            try:
                chunk = queue.get(timeout=.1)
            except Empty:
                continue
            length += len(chunk)
            require(length <= MAX_OUTPUT_CHARS, "Stream exceeds output bound")
            chunks.append(chunk)
            yield {"type": "delta", "request_id": request_id, "index": len(chunks) - 1, "text": chunk}
        if "error" in outcome:
            raise outcome["error"]
        result = validate_reply(outcome["result"], expected, messages, manifest["decoding"], "".join(chunks))
        yield {"type": "final", "request_id": request_id, "result": result}
    finally:
        cancelled.set()
        worker.join()


@contextmanager
def connect_store(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    path.chmod(0o600)
    db = sqlite3.connect(path, timeout=10)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, status TEXT NOT NULL, record TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, turn_id TEXT NOT NULL, rating TEXT NOT NULL, correction TEXT NOT NULL, created REAL NOT NULL)")
        db.commit()
        with db:
            yield db
    finally:
        db.close()


def save_turn(path, turn):
    require(turn["status"] in FINISHED | {"pending"}, "Invalid turn status")
    with connect_store(path) as db:
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT record FROM turns WHERE id=?", (turn["id"],)).fetchone()
        if prior:
            old = json.loads(prior[0])
            require(old["status"] == "pending" and all(old[k] == turn[k] for k in
                    ("id", "session_id", "created", "variant", "identity", "question", "messages", "request_id")),
                    "Finalized turns and request ownership are immutable")
        db.execute("INSERT INTO turns VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,record=excluded.record",
                   (turn["id"], turn["session_id"], turn["created"], turn["status"], json.dumps(turn, ensure_ascii=False, allow_nan=False)))


def save_review(path, session_id, turn_id, rating, correction):
    require(rating in {"Useful", "Incorrect", "Unsure"} and isinstance(correction, str)
            and len(correction) <= 8000, "Invalid review")
    with connect_store(path) as db:
        row = db.execute("SELECT session_id,status FROM turns WHERE id=?", (turn_id,)).fetchone()
        require(row and row[0] == session_id and row[1] in {"complete", "incomplete"}, "Select a validated reply in this conversation")
        # Keep earlier assessments as well as revisions; reviews never modify model output.
        db.execute("INSERT INTO reviews(turn_id,rating,correction,created) VALUES (?,?,?,?)",
                   (turn_id, rating, correction, time.time()))


def new_session(variant=DEFAULT_VARIANT):
    require(variant in LABELS, "Unknown demo variant")
    return {"id": uuid.uuid4().hex, "variant": variant, "turns": []}


def load_session(path, session_id):
    with connect_store(path) as db:
        turns = [json.loads(row[0]) for row in db.execute(
            "SELECT record FROM turns WHERE session_id=? ORDER BY created,rowid", (session_id,))]
    require(turns, "Conversation not found")
    return {"id": session_id, "variant": turns[0]["variant"], "turns": turns}


def conversations(path):
    with connect_store(path) as db:
        rows = db.execute("SELECT record FROM turns ORDER BY created DESC,rowid DESC").fetchall()
    found = {}
    for row in rows:
        turn = json.loads(row[0])
        found.setdefault(turn["session_id"], (f"{turn['variant']}: {turn['question'][:60]}", turn["session_id"]))
    return list(found.values())


def history_messages(state, question, variant, expected_identity, validate_messages, decoding):
    require(state["variant"] == variant, "Model changed; start a new conversation")
    messages = []
    for turn in state["turns"]:
        require(turn["status"] in {"complete", "incomplete"} and turn["identity"] == expected_identity,
                "Conversation has an unfinished reply or different bundle; start a new conversation")
        if turn["status"] == "incomplete":
            result = validate_reply(turn.get("remote_result"), expected_identity, turn["messages"], decoding, turn["answer"])
            require(result["finish_reason"] in {"max_new_tokens", "time_limit"} and bool(turn["answer"].strip()),
                    "Conversation has an unfinished reply; start a new conversation")
        messages.extend([{"role": "user", "content": turn["question"]},
                         {"role": "assistant", "content": turn["answer"]}])
    messages.append({"role": "user", "content": question})
    validate_messages(messages)
    return messages


class UserCancelled(Exception):
    pass


async def next_event(iterator, stop, deadline):
    pending = asyncio.create_task(anext(iterator))
    try:
        while True:
            if stop.is_set():
                raise UserCancelled("Stopped by user")
            if time.monotonic() >= deadline:
                raise TimeoutError("Model request exceeded its time limit")
            if pending in (await asyncio.wait({pending}, timeout=.1))[0]:
                return pending.result()
    finally:
        if not pending.done():
            pending.cancel()
        try:
            await pending
        except (Exception, asyncio.CancelledError):
            pass


async def modal_events(messages, variant, expected_sha, request_id):
    import modal
    actor = modal.Cls.from_name(APP_NAME, CLASS_NAME)()
    async for event in actor.stream.remote_gen.aio(messages, variant, expected_sha, request_id):
        yield event


async def cancel_modal(call_id):
    import modal
    await modal.FunctionCall.from_id(call_id).cancel.aio(terminate_containers=True)


async def audition(state, question, variant, manifest, expected_sha, validate_messages, store,
                   remote=modal_events, cancel=cancel_modal, stop=None, timeout=REQUEST_TIMEOUT_SECONDS):
    state = copy.deepcopy(state)
    expected = identity(manifest, expected_sha, variant)
    messages = history_messages(state, question, variant, expected, validate_messages, manifest["decoding"])
    stop = stop if stop is not None else threading.Event()
    turn = {"id": uuid.uuid4().hex, "request_id": uuid.uuid4().hex, "session_id": state["id"],
            "variant": variant, "identity": expected, "question": question, "messages": messages,
            "answer": "", "status": "pending", "finish_reason": "pending", "created": time.time()}
    state["turns"].append(turn)
    save_turn(store, turn)
    iterator, started, final, count = None, False, False, 0
    try:
        yield copy.deepcopy(state)
        if stop.is_set():
            raise UserCancelled("Request stopped before submission")
        iterator = aiter(remote(messages, variant, expected_sha, turn["request_id"]))
        deadline = time.monotonic() + timeout
        while True:
            try:
                event = await next_event(iterator, stop, deadline)
            except StopAsyncIteration:
                require(final, "Remote stream ended without a final event")
                break
            require(isinstance(event, dict) and event.get("request_id") == turn["request_id"], "Stream request differs")
            require(not final, "Unexpected event after final reply")
            kind = event.get("type")
            if kind == "start":
                require(not started and event.get("protocol") == PROTOCOL and event.get("identity") == expected
                        and event.get("messages_sha256") == digest(messages), "Stream identity differs")
                require(isinstance(event.get("call_id"), str) and event["call_id"].startswith("fc-"), "Missing call identity")
                turn["call_id"], started = event["call_id"], True
                save_turn(store, turn)
            elif kind == "delta":
                require(started and event.get("index") == count and type(event.get("index")) is int,
                        "Missing or out-of-order delta")
                chunk = event.get("text")
                require(isinstance(chunk, str) and chunk and len(turn["answer"]) + len(chunk) <= MAX_OUTPUT_CHARS,
                        "Invalid text delta")
                turn["answer"] += chunk
                count += 1
                save_turn(store, turn)
                yield copy.deepcopy(state)
            elif kind == "final":
                require(started, "Final event before stream identity")
                result = validate_reply(event.get("result"), expected, messages, manifest["decoding"], turn["answer"])
                turn["remote_result"] = result
                final = True
            else:
                raise ValueError("Unknown stream event")
        result = turn["remote_result"]
        turn.update(finish_reason=result["finish_reason"], status="cancelled" if result["finish_reason"] == "cancelled"
                    else "complete" if result["complete"] else "incomplete")
    except UserCancelled as exc:
        turn.update(status="cancelled", finish_reason="cancelled", error=str(exc))
    except (asyncio.CancelledError, GeneratorExit):
        turn.update(status="cancelled", finish_reason="cancelled", error="Local connection closed")
        raise
    except Exception as exc:
        turn.update(status="error", finish_reason="error", error_type=type(exc).__name__, error=str(exc)[:2000])
    finally:
        if turn["status"] == "pending":
            turn.update(status="cancelled", finish_reason="cancelled")
        if turn["status"] in {"error", "cancelled"} and iterator is not None:
            turn["cancellation"] = "unconfirmed: no call identity received"
            if turn.get("call_id"):
                try:
                    await asyncio.wait_for(cancel(turn["call_id"]), timeout=10)
                    turn["cancellation"] = "remote cancellation acknowledged"
                except Exception as exc:
                    turn["cancellation"] = "unconfirmed: " + str(exc)[:500]
        if iterator is not None and hasattr(iterator, "aclose"):
            try:
                await asyncio.wait_for(iterator.aclose(), timeout=10)
            except Exception as exc:
                turn["stream_close_error"] = str(exc)[:500]
        save_turn(store, turn)
    yield state


def displayed(state):
    values = []
    for turn in state["turns"]:
        values.append({"role": "user", "content": turn["question"]})
        if turn["answer"]:
            values.append({"role": "assistant", "content":
                           f"**{turn['variant']} / {turn['status']}**\n\n{turn['answer']}"})
    return values


CSS = """
.gradio-container { max-width:1080px !important; margin:auto; padding:20px !important; }
body,.gradio-container { background:#fafbfc !important; color:#24262a !important; }
h1 { font-size:24px !important; font-weight:600 !important; letter-spacing:0 !important; }
#conversation { border:0 !important; background:transparent !important; }
#conversation .message { border-radius:8px !important; box-shadow:none !important; }
#conversation .message p { line-height:1.6; overflow-wrap:anywhere; }
#status,#notice { font-size:13px; overflow-wrap:anywhere; min-height:30px; }
"""


def build_app(bundle, expected_sha, store=None, remote=None, cancel=cancel_modal,
              initialize=True, heading=True, *, app_name=None, class_name=None):
    manifest, module = bound_bundle(bundle, expected_sha)
    store = bundle_store(manifest, expected_sha, store)
    remote = remote if remote is not None else routed_events(manifest, app_name, class_name)
    labels, default = available_labels(manifest), default_variant(manifest)
    import gradio as gr
    active, lock = {}, threading.Lock()

    title = "Twi understanding " + ("R3" if default == "r3_response" else "R2")
    with gr.Blocks(title=title, analytics_enabled=False) as demo:
        state = gr.State(value=None)
        if initialize:
            demo.load(lambda: new_session(default), outputs=state, api_visibility="private")
        if heading:
            gr.Markdown("# " + title)
            gr.Markdown("Messages are processed on private Modal compute. Chat history and reviews are saved on this device. Research, not medical advice.", elem_id="notice")
        with gr.Row():
            history = gr.Dropdown(conversations(store), value=None, label="Saved conversation")
            fresh = gr.Button("New conversation", size="sm")
        with gr.Accordion("Checkpoint", open=False):
            choice = gr.Dropdown([(label, key) for key, label in labels.items()], value=default, label="Model")
        with gr.Accordion("Model identity", open=False):
            details = gr.JSON(identity(manifest, expected_sha, default), show_label=False)
        chat = gr.Chatbot(show_label=False, container=False, layout="bubble", elem_id="conversation",
                          height="clamp(240px, calc(100dvh - 450px), 600px)", buttons=["copy"],
                          feedback_options=[], allow_file_downloads=False, sanitize_html=True)
        status = gr.HTML("Ready", elem_id="status")
        composer = gr.Textbox(label="Message", placeholder="Twi or English", lines=2, max_lines=5, max_length=16000, submit_btn=True)
        stop_button = gr.Button("Stop", variant="stop", size="sm")
        with gr.Accordion("Review a reply", open=False):
            turn_choice = gr.Dropdown([], label="Reply")
            rating = gr.Radio(["Useful", "Incorrect", "Unsure"], value="Unsure", label="Assessment")
            correction = gr.Textbox(label="Correction or note", lines=2, max_length=8000)
            save = gr.Button("Save review", size="sm")
            review_status = gr.Markdown("")

        def view(current, note=None):
            last = current["turns"][-1] if current["turns"] else None
            with lock:
                pending = bool(last and last["status"] == "pending" and current["id"] in active)
            choices = [(f"{i + 1}: {t['variant']} / {t['finish_reason']}", t["id"])
                       for i, t in enumerate(current["turns"]) if t["status"] in {"complete", "incomplete"}]
            if note is None:
                note = (last["status"] + " / " + last["finish_reason"] +
                        (" / " + last["error"] if last.get("error") else "")) if last else "Ready"
            if last and last.get("cancellation", "").startswith("unconfirmed"):
                note += " / Remote cancellation not confirmed"
            return (current, displayed(current), html.escape(note), gr.update(value="", interactive=not pending),
                    gr.update(value=current["variant"], interactive=not pending), gr.update(interactive=not pending),
                    gr.update(choices=conversations(store), value=current["id"] if last else None, interactive=not pending),
                    gr.update(choices=choices, value=choices[-1][1] if choices else None),
                    last["identity"] if last else identity(manifest, expected_sha, current["variant"]), "", "")

        outputs = [state, chat, status, composer, choice, fresh, history, turn_choice, details, review_status, correction]

        async def respond(question, prior, variant):
            prior = prior or new_session(variant)
            signal = threading.Event()
            with lock:
                require(prior["id"] not in active, "Conversation already has an active request")
                active[prior["id"]] = signal
            try:
                async for current in audition(prior, question, variant, manifest, expected_sha,
                        module.validate_messages, store, remote, cancel, signal):
                    yield view(current)
            except Exception as exc:
                yield view(prior, str(exc)[:2000])
            finally:
                with lock:
                    active.pop(prior["id"], None)

        event_options = {"api_visibility": "private", "concurrency_limit": 1, "concurrency_id": "r2-demo"}
        composer.submit(respond, [composer, state, choice], outputs, **event_options)

        def stop(prior):
            with lock:
                signal = active.get((prior or {}).get("id"))
                if signal:
                    signal.set()

        stop_button.click(stop, [state], None, queue=False, api_visibility="private")
        for event in (choice.input, fresh.click):
            event(lambda variant: view(new_session(variant)), [choice], outputs, **event_options)
        history.input(lambda selected: view(load_session(store, selected)) if selected else view(new_session(default)),
                      [history], outputs, **event_options)

        def review(prior, turn_id, value, note):
            try:
                save_review(store, prior["id"], turn_id, value, note)
                return "Review saved locally."
            except ValueError as exc:
                raise gr.Error(str(exc)) from None

        save.click(review, [state, turn_choice, rating, correction], review_status, api_visibility="private")
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "chat"])
    parser.add_argument("--bundle", type=Path, default=os.environ.get("GHA_R2_DEMO_BUNDLE"))
    parser.add_argument("--expected-sha", default=os.environ.get("GHA_R2_DEMO_SHA"))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--app-name")
    parser.add_argument("--class-name")
    parser.add_argument("--port", type=int, default=7866)
    args = parser.parse_args(argv)
    require(args.bundle is not None, "Set GHA_R2_DEMO_BUNDLE or --bundle")
    manifest, _ = bound_bundle(args.bundle, args.expected_sha)
    args.store = bundle_store(manifest, args.expected_sha, args.store)
    target = actor_target(manifest, args.app_name, args.class_name)
    if args.command == "verify":
        print(json.dumps({"private_class": target["app_name"] + "/" + target["class_name"], "remote_called": False,
                          "variants": [identity(manifest, args.expected_sha, v) for v in available_labels(manifest)]}, indent=2))
        return
    require(1024 <= args.port <= 65535, "Choose an unprivileged localhost port")
    import gradio as gr
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    theme = gr.themes.Base(font=["system-ui", "sans-serif"], primary_hue="gray", secondary_hue="gray", neutral_hue="gray")
    theme.set(body_background_fill="#fafbfc", body_background_fill_dark="#fafbfc",
              body_text_color="#24262a", body_text_color_dark="#24262a",
              background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
              background_fill_secondary="#f0f2f4", background_fill_secondary_dark="#f0f2f4")
    build_app(args.bundle, args.expected_sha, args.store, app_name=args.app_name, class_name=args.class_name).queue(max_size=4).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        show_error=False, strict_cors=True, enable_monitoring=False, mcp_server=False,
        ssr_mode=False, footer_links=[], theme=theme, css=CSS,
        blocked_paths=[str(Path(args.bundle).resolve()), str(Path(args.store).resolve().parent)],
        app_kwargs={"middleware": [Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])]})


if __name__ == "__main__":
    main()
