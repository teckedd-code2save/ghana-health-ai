#!/usr/bin/env python3
"""Offline verification or loopback-only base/R3/R4 workbench.

  python3 scripts/understanding_r4_workbench.py verify --bundle DIR --expected-sha SHA
  python3 scripts/understanding_r4_workbench.py chat --bundle DIR --expected-sha SHA

Requires a real, completed R4 bundle. Startup never instantiates Runtime or calls
the GPU. --helper-code defaults to BUNDLE/code and selects the exact frozen R2
demo and loop-guard bytes pinned below; neither globals nor import paths change.
Each model sees the same new question and only its own previous answers.
"""
import argparse
import asyncio
import builtins
from contextlib import closing, contextmanager
import copy
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
import threading
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = "r4-comparison-inference-v1"
STORE_VERSION = "r4-three-way-workbench-v1"
HISTORY_POLICY = "independent_model_histories_v1"
VARIANTS = ("base", "r3_response", "r4_response")
LABELS = {"base": "Base", "r3_response": "R3", "r4_response": "R4"}
DEFAULT_HELPER_CODE = Path("/Users/welcome/.codex/worktrees/r2-train/ghana-health-ai/scripts")
HELPER_HASHES = {
    "understanding_r2_demo.py": "e11649ff095b6315a06b02ac72661e4e4006aefb73c9eceb8e851539ce127781",
    "response_loop_guard.py": "2b0830e33dd3a9bea182d2dbbd4d9d8c091f14eaea4b23e95a6664391ac3b474",
}
TIMING_NOTE = ("First text is the local time to the first nonempty streamed text, not a measured "
               "GPU first-token timestamp. Requests run serially: Base, R3, R4. Loading, "
               "queueing, transport and local saving are included; this is not a fair latency benchmark.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def checked_bytes(root, name, checksum):
    relative = PurePosixPath(name)
    require(is_sha(checksum) and not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe artifact binding")
    path = root / name
    require(path.resolve().is_relative_to(root) and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "Unsafe artifact path")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == checksum, "Artifact hash differs: " + name)
    return data


def execute_module(path, data, imports=None):
    """Private module namespace and import hook, never sys.path/sys.modules edits."""
    module = types.ModuleType("_r4_workbench_" + hashlib.sha256(data).hexdigest())
    module.__file__ = str(path)
    if imports:
        original = builtins.__import__

        def bound_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in imports:
                return imports[name]
            return original(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": bound_import}
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def load_helpers(code=DEFAULT_HELPER_CODE):
    code = Path(code).resolve()
    sources = {name: checked_bytes(code, name, sha) for name, sha in HELPER_HASHES.items()}
    guard = execute_module(code / "response_loop_guard.py", sources["response_loop_guard.py"])
    return execute_module(code / "understanding_r2_demo.py", sources["understanding_r2_demo.py"],
                          {"response_loop_guard": guard})


def manifest_contract(manifest):
    require(manifest.get("version") == VERSION, "An R4 comparison bundle is required")
    variants = manifest.get("variants", {})
    require(set(VARIANTS) <= set(variants) <= set(VARIANTS) | {"r4_pass1"}
            and variants["base"] is None, "Missing or unexpected R4 checkpoint variants")
    require(manifest.get("private") is True and manifest.get("training") is False,
            "A private inference-only bundle is required")
    actor = manifest.get("actor", {})
    require(re.fullmatch(r"[a-z][a-z0-9-]{1,62}", actor.get("app_name", ""))
            and actor["app_name"] not in {"ghana-understanding-r2-demo-private", "ghana-understanding-r3-demo-private"}
            and actor.get("class_name") == "UnderstandingR4Demo", "Use the separate R4 actor")


def bound_bundle(folder, expected_sha):
    require(is_sha(expected_sha), "Expected bundle SHA-256 is required")
    folder = Path(folder).resolve()
    raw = checked_bytes(folder, "manifest.json", expected_sha)
    manifest = read_json(raw)
    manifest_contract(manifest)
    require({"code/r4_inference.py", "code/r2_inference.py"} <= set(manifest["files"]),
            "Missing R4 runtime or sibling runtime")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked bundle artifact")
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            - {"manifest.json", "SEALED.json"} == set(manifest["files"]), "Unbound or missing bundle artifact")
    # Check every artifact before executing the runtime, including its sibling imports.
    sources = {name: checked_bytes(folder, name, sha) for name, sha in manifest["files"].items()}
    source = folder / "code/r4_inference.py"
    module = execute_module(source, sources["code/r4_inference.py"])
    require(callable(getattr(module, "validate_messages", None)), "Missing bundled message validator")
    require(module.load_bundle(folder, expected_sha) == manifest, "Runtime manifest binding differs")
    return manifest, module


def new_session(checksum, *, loop_guard=True):
    require(is_sha(checksum) and type(loop_guard) is bool, "Invalid session policy")
    return {"id": uuid.uuid4().hex, "revision": 0, "manifest_sha256": checksum,
            "history_policy": HISTORY_POLICY, "loop_guard": loop_guard,
            "execution_order": list(VARIANTS), "model_sessions": {v: uuid.uuid4().hex for v in VARIANTS},
            "rounds": []}


def model_session(state, variant):
    """R4 is deliberately not passed to demo.new_session, whose LABELS exclude it."""
    require(variant in VARIANTS, "Unknown comparison variant")
    return {"id": state["model_sessions"][variant], "variant": variant,
            "turns": [copy.deepcopy(r["sides"][variant]["turn"]) for r in state["rounds"]]}


class Store:
    def __init__(self, path, checksum):
        require(is_sha(checksum), "Invalid store bundle identity")
        self.path, self.checksum = Path(path).absolute(), checksum
        require(not any(p.is_symlink() for p in (self.path, *self.path.parents)), "Linked store path")
        self.binding = {"version": STORE_VERSION, "manifest_sha256": checksum,
                        "helper_hashes": HELPER_HASHES}
        if self.path.exists():
            # Read-only preflight before chmod, migrations, WAL or any writes.
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                self._check(db)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
            with self.connection(check=False) as db:
                db.execute("CREATE TABLE r4_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.executemany("INSERT INTO r4_meta VALUES (?,?)", [
                    ("binding", json.dumps(self.binding)), ("browser_secret", secrets.token_hex(32))])
                db.execute("CREATE TABLE r4_sessions (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, updated REAL NOT NULL, record TEXT NOT NULL)")
                db.execute("CREATE TABLE r4_reviews (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, round_id TEXT NOT NULL, rating TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL)")
        with self.connection() as db:
            self.browser_secret = db.execute("SELECT value FROM r4_meta WHERE key='browser_secret'").fetchone()[0]

    def _check(self, db):
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require("r4_meta" in tables, "Use a new separate R4 store; existing history is read-only")
        row = db.execute("SELECT value FROM r4_meta WHERE key='binding'").fetchone()
        require(row and read_json(row[0]) == self.binding, "Store belongs to another bundle or helper version")

    @contextmanager
    def connection(self, *, check=True):
        require(not self.path.is_symlink(), "Linked store path")
        db = sqlite3.connect(self.path, timeout=10)
        try:
            if check:
                self._check(db)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def save(self, state):
        require(state["manifest_sha256"] == self.checksum, "Session bundle differs")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision,record FROM r4_sessions WHERE id=?", (state["id"],)).fetchone()
            require((row[0] if row else 0) == state["revision"], "Stale session; reload without retrying")
            if row:
                old = read_json(row[1])
                require(all(old[k] == state[k] for k in ("manifest_sha256", "history_policy", "loop_guard",
                                                        "execution_order", "model_sessions")),
                        "Policy changed; start a new session")
            revision = state["revision"] + 1
            record = json.dumps({**state, "revision": revision}, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT INTO r4_sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,updated=excluded.updated,record=excluded.record",
                       (state["id"], revision, time.time(), record))
        state["revision"] = revision

    def load(self, session_id):
        with self.connection() as db:
            row = db.execute("SELECT record FROM r4_sessions WHERE id=?", (session_id,)).fetchone()
        require(row is not None, "Saved conversation not found")
        return read_json(row[0])

    def conversations(self):
        with self.connection() as db:
            states = [read_json(r[0]) for r in db.execute("SELECT record FROM r4_sessions ORDER BY updated DESC,rowid DESC")]
        return [(('Guarded' if s['loop_guard'] else 'Raw') + ': ' +
                 (s['rounds'][0]['question'][:60] if s['rounds'] else 'New chat'), s['id']) for s in states]

    def review(self, session_id, round_id, rating, note):
        require(rating in {*VARIANTS, "tie", "neither", "unsure"}
                and isinstance(note, str) and len(note) <= 8000, "Invalid assessment")
        state = self.load(session_id)
        require(any(r["id"] == round_id and r["status"] != "pending" for r in state["rounds"]),
                "Select a finished round in this conversation")
        with self.connection() as db:
            db.execute("INSERT INTO r4_reviews(session_id,round_id,rating,note,created) VALUES (?,?,?,?,?)",
                       (session_id, round_id, rating, note, time.time()))

    def reviews(self, session_id):
        with self.connection() as db:
            return [dict(zip(("round_id", "rating", "note", "created"), r)) for r in db.execute(
                "SELECT round_id,rating,note,created FROM r4_reviews WHERE session_id=? ORDER BY id", (session_id,))]


class Workbench:
    def __init__(self, manifest, module, checksum, store, demo):
        manifest_contract(manifest)
        require(callable(module.validate_messages), "Missing bound message validator")
        require(store.checksum == checksum, "Store binding differs")
        self.manifest, self.module, self.checksum = manifest, module, checksum
        self.store, self.demo = store, demo

    def fresh(self, *, loop_guard=True):
        state = new_session(self.checksum, loop_guard=loop_guard)
        self.store.save(state)
        return state

    def restore(self, session_id):
        return self.store.load(session_id) if session_id else self.fresh()

    def accept(self, side, turn, messages, guarded):
        expected = self.demo.identity(self.manifest, self.checksum, side["variant"])
        require(turn["messages"] == messages and turn["question"] == messages[-1]["content"],
                "Side input history differs")
        require(turn["identity"] == expected and turn["variant"] == side["variant"], "Side checkpoint differs")
        require(turn.get("loop_guard", False) is guarded, "Side repetition policy differs")
        side.update(turn=copy.deepcopy(turn), status=turn["status"], error=turn.get("error"), verified=False)
        if turn["status"] in {"complete", "incomplete", "paused", "cancelled"} and turn.get("remote_result"):
            result = self.demo.validate_reply(turn["remote_result"], expected, messages,
                self.manifest["decoding"], turn["answer"], guarded)
            status = ("paused" if result.get("loop_guard", {}).get("trigger") else "cancelled"
                      if result["finish_reason"] == "cancelled" else "complete" if result["complete"] else "incomplete")
            require(turn["status"] == status and turn["finish_reason"] == result["finish_reason"],
                    "Side completion evidence differs")
            side["verified"] = True
        elif turn["status"] in {"complete", "incomplete", "paused"}:
            raise ValueError("Missing final remote evidence")

    @staticmethod
    def continue_round(item):
        if not item.get("verified") or item["status"] not in {"complete", "incomplete", "paused"}:
            return False
        for side in item["sides"].values():
            if not side.get("verified") or side["status"] not in {"complete", "incomplete", "paused"}:
                return False
            turn = side["turn"]
            result = turn.get("remote_result", {})
            if not turn["answer"].strip():
                return False
            if side["status"] == "paused":
                if not item["loop_guard"] or not result.get("loop_guard", {}).get("trigger"):
                    return False
            elif side["status"] == "incomplete" and result.get("finish_reason") not in {"max_new_tokens", "time_limit"}:
                return False
            elif side["status"] == "complete" and (result.get("finish_reason") != "eos" or result.get("complete") is not True):
                return False
        return True

    @staticmethod
    def matched_prompts(item):
        sides = list(item["sides"].values())
        for i, left in enumerate(sides):
            for right in sides[i + 1:]:
                if left["turn"]["messages"] == right["turn"]["messages"]:
                    require(left["turn"]["remote_result"]["prompt_ids_sha256"] ==
                            right["turn"]["remote_result"]["prompt_ids_sha256"],
                            "Native prompt hashes differ for identical inputs")

    def histories(self, state):
        require(state["manifest_sha256"] == self.checksum and state["history_policy"] == HISTORY_POLICY
                and state["execution_order"] == list(VARIANTS) and type(state["loop_guard"]) is bool,
                "Conversation policy or bundle differs")
        messages = {v: [] for v in VARIANTS}
        for item in state["rounds"]:
            require(self.continue_round(item), "Unfinished or unverified round; start a new chat, no automatic retry")
            require(item["loop_guard"] is state["loop_guard"] and set(item["sides"]) == set(VARIANTS),
                    "Saved repetition policy or checkpoint set differs")
            for v in VARIANTS:
                expected = messages[v] + [{"role": "user", "content": item["question"]}]
                require(item["input_messages"][v] == expected
                        and item["input_messages_sha256"][v] == digest(expected), "Saved model history is not independent")
                turn = item["sides"][v]["turn"]
                require(turn["session_id"] == state["model_sessions"][v], "Model session ownership differs")
                self.accept(copy.deepcopy(item["sides"][v]), turn, expected, state["loop_guard"])
                self.module.validate_messages(expected)
                messages[v] = expected + [{"role": "assistant", "content": turn["answer"]}]
            self.matched_prompts(item)
        return messages

    def ready(self, state):
        try:
            self.histories(state)
            return True
        except (ValueError, KeyError, TypeError):
            return False

    async def compare(self, state, question, *, stop=None, remote=None, cancel=None, audition_fn=None):
        state = copy.deepcopy(state)
        require(self.store.load(state["id"]) == state, "Stale or modified session; reload without retrying")
        require(isinstance(question, str) and bool(question.strip()) and len(question) <= 16000,
                "Enter a nonempty message of at most 16000 characters")
        inputs = {v: history + [{"role": "user", "content": question}] for v, history in self.histories(state).items()}
        for messages in inputs.values():
            self.module.validate_messages(messages)
        sessions = {v: model_session(state, v) for v in VARIANTS}
        guarded = state["loop_guard"]
        remote = remote if remote is not None else self.demo.routed_events(self.manifest, loop_guard=guarded)
        cancel = cancel if cancel is not None else self.demo.cancel_modal
        audition_fn = audition_fn or self.demo.audition
        stop = stop if stop is not None else threading.Event()
        item = {"id": uuid.uuid4().hex, "question": question, "created": time.time(), "status": "pending",
                "verified": False, "error": None, "loop_guard": guarded, "input_messages": copy.deepcopy(inputs),
                "input_messages_sha256": {v: digest(m) for v, m in inputs.items()},
                "sides": {v: {"variant": v, "status": "queued", "verified": False, "turn": None,
                              "ttft_seconds": None, "wall_seconds": None, "error": None} for v in VARIANTS}}
        state["rounds"].append(item)
        self.store.save(state)
        try:
            yield copy.deepcopy(state)
            for v, side in item["sides"].items():
                if stop.is_set():
                    side.update(status="skipped", error="Stopped before submission")
                    self.store.save(state)
                    yield copy.deepcopy(state)
                    continue
                started, iterator = time.monotonic(), None
                try:
                    iterator = audition_fn(sessions[v], question, v, self.manifest, self.checksum,
                        self.module.validate_messages, self.store.path, remote=remote, cancel=cancel,
                        stop=stop, loop_guard=guarded)
                    async for current in iterator:
                        turn = current["turns"][-1]
                        elapsed = time.monotonic() - started
                        if side["ttft_seconds"] is None and turn["status"] == "pending" and turn["answer"]:
                            side["ttft_seconds"] = elapsed
                        side["wall_seconds"] = elapsed
                        self.accept(side, turn, inputs[v], guarded)
                        self.store.save(state)
                        yield copy.deepcopy(state)
                    require(side["status"] not in {"queued", "pending"}, "Stream ended without a final turn")
                except (asyncio.CancelledError, GeneratorExit):
                    stop.set()
                    side.update(status="cancelled", verified=False, error="Local connection closed")
                    raise
                except Exception as exc:
                    side.update(status="error", verified=False, error=str(exc)[:2000])
                finally:
                    if iterator is not None:
                        await iterator.aclose()
                    side["wall_seconds"] = time.monotonic() - started
                if side["status"] == "cancelled":
                    stop.set()
                self.store.save(state)
                yield copy.deepcopy(state)
            statuses = {s["status"] for s in item["sides"].values()}
            item["verified"] = all(s["verified"] for s in item["sides"].values())
            item["status"] = ("error" if "error" in statuses else "cancelled" if statuses & {"cancelled", "skipped"}
                              else "paused" if "paused" in statuses else "complete" if statuses == {"complete"} else "incomplete")
            item["error"] = next((s["error"] for s in item["sides"].values() if s["error"]), None)
            if item["verified"]:
                try:
                    self.matched_prompts(item)
                except ValueError as exc:
                    item.update(status="error", verified=False, error=str(exc))
            self.store.save(state)
            yield copy.deepcopy(state)
        finally:
            if item["status"] == "pending":
                stop.set()
                for side in item["sides"].values():
                    if side["status"] in {"queued", "pending"}:
                        side.update(status="skipped" if side["status"] == "queued" else "cancelled", verified=False)
                item.update(status="cancelled", verified=False, error="Local connection closed; no automatic retry")
                self.store.save(state)


def transcript(state, variant):
    rows = []
    for item in state["rounds"]:
        rows.append({"role": "user", "content": item["question"]})
        turn = item["sides"][variant].get("turn")
        if turn and turn["answer"]:
            rows.append({"role": "assistant", "content": turn["answer"]})
    return rows


def side_status(side):
    if not side:
        return "Ready"
    turn = side.get("turn") or {}
    result = turn.get("remote_result") or {}
    status = "Paused by repetition guard" if side["status"] == "paused" else side["status"].capitalize()
    if side["status"] in {"complete", "incomplete"} and not turn.get("answer", "").strip():
        status = "Empty reply"
    parts = [status, "Evidence verified" if side["verified"] else "Evidence not verified"]
    if side.get("ttft_seconds") is not None:
        parts.append(f"First text: {side['ttft_seconds']:.2f}s")
    if result:
        parts.append("Stop: " + result["finish_reason"])
    if side.get("error"):
        parts.append(side["error"])
    if turn.get("cancellation", "").startswith("unconfirmed"):
        parts.append("Remote cancellation not confirmed")
    return "<br>".join(html.escape(p) for p in parts)


def evidence(workbench, state):
    item = state["rounds"][-1] if state["rounds"] else None
    return {"manifest_sha256": workbench.checksum, "history_policy": HISTORY_POLICY,
            "repetition_policy": workbench.demo.LOOP_POLICY if state["loop_guard"] else "raw / off",
            "execution_order": list(VARIANTS), "timing_scope": TIMING_NOTE,
            "verified": bool(item and item["verified"]), "completion_status": item["status"] if item else "empty",
            "input_messages_by_model": item["input_messages"] if item else {},
            "input_messages_sha256": item["input_messages_sha256"] if item else {},
            "identities": {v: workbench.demo.identity(workbench.manifest, workbench.checksum, v) for v in VARIANTS},
            "checkpoint_directories": {v: workbench.manifest["variants"][v]["directory"] for v in VARIANTS if v != "base"},
            "decoding": workbench.manifest["decoding"], "actor": workbench.manifest["actor"],
            "helpers": HELPER_HASHES,
            "results": {v: {"status": s["status"], "verified": s["verified"],
                "observed_first_text_seconds": s["ttft_seconds"], "local_wall_seconds": s["wall_seconds"],
                "request_id": (s.get("turn") or {}).get("request_id"),
                "call_id": (s.get("turn") or {}).get("call_id"),
                "remote_result": (s.get("turn") or {}).get("remote_result")}
                for v, s in item["sides"].items()} if item else {},
            "assessment": "Identity and input checks do not establish meaning, native-language quality or clinical safety."}


CSS = """
.gradio-container { max-width:1560px !important; margin:auto; padding:20px !important;
  background:#fafbfc !important; color:#24262a !important; }
body { background:#fafbfc !important; }
h1 { font-size:24px !important; letter-spacing:0 !important; }
h3 { font-size:18px !important; letter-spacing:0 !important; }
#r4-panes { gap:20px !important; }
.r4-panel { min-width:0; border:0 !important; box-shadow:none !important; }
.r4-chat { border:0 !important; background:transparent !important; }
.r4-chat .message { border-radius:6px !important; box-shadow:none !important; }
.r4-chat p,.r4-status { overflow-wrap:anywhere; line-height:1.6; }
.r4-status { min-height:86px; font-size:12px; }
.r4-checkpoint code { font-size:11px; white-space:normal; overflow-wrap:anywhere; }
.r4-checkpoint { min-height:64px; }
#r4-status { min-height:28px; font-size:13px; overflow-wrap:anywhere; }
#r4-panes > div:nth-child(1) { border-top:2px solid #66756c; }
#r4-panes > div:nth-child(2) { border-top:2px solid #347a88; }
#r4-panes > div:nth-child(3) { border-top:2px solid #a9556c; }
@media(max-width:800px) { .gradio-container { padding:12px !important; } #r4-panes { gap:12px !important; } }
"""


def build_app(workbench, *, remote=None, cancel=None):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr
    active, lock = {}, threading.Lock()
    with gr.Blocks(title="Twi R4 | Base, R3, R4", analytics_enabled=False) as app:
        state = gr.State(None)
        remembered = gr.BrowserState(None, storage_key="r4-workbench-" + digest(str(workbench.store.path)),
                                     secret=workbench.store.browser_secret)
        gr.Markdown("# Twi R4")
        gr.Markdown("Private model processing. Conversations and assessments stay on this device. Research, not medical advice.")
        with gr.Row():
            saved = gr.Dropdown(workbench.store.conversations(), label="Saved conversation", scale=3)
            mode = gr.Radio([("Guarded", "guarded"), ("Raw", "raw")], value="guarded", label="Response mode", scale=1)
            fresh = gr.Button("New chat", size="sm", scale=0)
        chats, statuses = [], []
        with gr.Row(elem_id="r4-panes", equal_height=True):
            for variant in VARIANTS:
                with gr.Column(min_width=280, elem_classes="r4-panel"):
                    gr.Markdown("### " + LABELS[variant])
                    identity = workbench.demo.identity(workbench.manifest, workbench.checksum, variant)
                    checkpoint = identity["adapter_weights_sha256"] or identity["base_weights_sha256"]
                    description = {"base": "AfriqueQwen 9B, adapter off", "r3_response": "R3 response stage",
                                   "r4_response": "R4 continuation"}[variant]
                    gr.Markdown(f"{description}\n\n`{checkpoint[:12]}`", elem_classes="r4-checkpoint")
                    chats.append(gr.Chatbot(show_label=False, container=False, layout="bubble", height=390,
                        elem_classes="r4-chat", buttons=["copy"], feedback_options=[], sanitize_html=True,
                        allow_file_downloads=False, allow_tags=False))
                    statuses.append(gr.HTML("Ready", elem_classes="r4-status"))
        status = gr.HTML("Ready", elem_id="r4-status")
        composer = gr.Textbox(label="Question", placeholder="Twi or English", lines=2, max_lines=5,
                              max_length=16000, submit_btn=True)
        stop_button = gr.Button("Stop", size="sm", variant="stop")
        with gr.Accordion("Your assessment", open=False):
            selected_round = gr.Dropdown([], label="Round")
            rating = gr.Radio([(LABELS[v], v) for v in VARIANTS] + [("Equal", "tie"), ("Neither", "neither"), ("Unsure", "unsure")],
                              value="unsure", label="Most useful")
            note = gr.Textbox(label="Correction or note", lines=2, max_length=8000)
            save = gr.Button("Save assessment", size="sm")
            review_status = gr.Markdown("")
            reviews = gr.JSON([], label="Saved assessments", open=False)
        with gr.Accordion("Completion and actual input proof", open=False):
            gr.Markdown(TIMING_NOTE)
            proof = gr.JSON({}, label="Checkpoints, exact histories and stream evidence", open=False)

        outputs = [state, remembered, *chats, *statuses, status, composer, mode, fresh, saved,
                   selected_round, proof, review_status, note, rating, reviews]

        def view(current, message=None):
            item = current["rounds"][-1] if current["rounds"] else None
            with lock:
                pending = current["id"] in active
            ready = workbench.ready(current)
            if message is None:
                message = ("Ready" if not item else item["status"].capitalize() +
                           (" | Evidence verified" if item["verified"] else " | Evidence not verified"))
                if item and not pending and not ready:
                    message += " | Start a new chat"
                if item and item.get("error"):
                    message += " | " + item["error"]
            choices = [(f"{i + 1}: {r['question'][:65]}", r["id"]) for i, r in enumerate(current["rounds"])
                       if r["status"] != "pending"]
            return (current, current["id"], *[transcript(current, v) for v in VARIANTS],
                *[side_status(item["sides"][v] if item else None) for v in VARIANTS], html.escape(message),
                gr.update(value="", interactive=ready and not pending),
                gr.update(value="guarded" if current["loop_guard"] else "raw", interactive=not pending),
                gr.update(interactive=not pending),
                gr.update(choices=workbench.store.conversations(), value=current["id"], interactive=not pending),
                gr.update(choices=choices, value=choices[-1][1] if choices else None),
                evidence(workbench, current), "", gr.update(value="", interactive=not pending),
                gr.update(value="unsure", interactive=not pending), workbench.store.reviews(current["id"]))

        async def respond(question, prior, selected_mode):
            current = prior or workbench.fresh(loop_guard=selected_mode == "guarded")
            require(current["loop_guard"] is (selected_mode == "guarded"), "Mode changed; start a new chat")
            signal, error = threading.Event(), None
            with lock:
                require(current["id"] not in active, "Conversation already active")
                active[current["id"]] = signal
            iterator = workbench.compare(current, question, stop=signal, remote=remote, cancel=cancel)
            try:
                async for current in iterator:
                    yield view(current)
            except Exception as exc:
                error = str(exc)[:2000]
            finally:
                try:
                    await iterator.aclose()
                finally:
                    with lock:
                        active.pop(current["id"], None)
            yield view(workbench.store.load(current["id"]), error)

        options = {"api_visibility": "private", "concurrency_limit": 1, "concurrency_id": "r4-workbench"}
        composer.submit(respond, [composer, state, mode], outputs, **options)

        def stop(current):
            with lock:
                signal = active.get((current or {}).get("id"))
                if signal:
                    signal.set()

        stop_button.click(stop, [state], None, queue=False, api_visibility="private")
        for event in (mode.input, fresh.click):
            event(lambda selected: view(workbench.fresh(loop_guard=selected == "guarded")), [mode], outputs, **options)
        saved.input(lambda key: view(workbench.restore(key)), [saved], outputs, **options)

        def restore(key):
            try:
                return view(workbench.restore(key))
            except ValueError as exc:
                return view(workbench.fresh(), str(exc))

        app.load(restore, [remembered], outputs, **options)

        def review(current, round_id, value, comment):
            try:
                workbench.store.review(current["id"], round_id, value, comment)
                return "Assessment saved locally.", workbench.store.reviews(current["id"])
            except (ValueError, TypeError) as exc:
                raise gr.Error(str(exc)) from None

        save.click(review, [state, selected_round, rating, note], [review_status, reviews], api_visibility="private")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "chat"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--helper-code", type=Path, default=os.environ.get("GHA_R4_HELPER_CODE"),
                        help="Frozen helper directory; default: BUNDLE/code")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--port", type=int, default=7868)
    args = parser.parse_args(argv)
    manifest, module = bound_bundle(args.bundle, args.expected_sha)
    args.helper_code = args.helper_code or args.bundle / "code"
    demo = load_helpers(args.helper_code)
    path = args.store or ROOT / "tmp/understanding-r4-workbench" / args.expected_sha / "workbench.sqlite3"
    require(not path.resolve().is_relative_to(args.bundle.resolve())
            and not path.resolve().is_relative_to(args.helper_code.resolve()), "Keep the store outside frozen evidence")
    if args.command == "verify":
        print(json.dumps({"remote_called": False, "server_started": False, "manifest_sha256": args.expected_sha,
                          "actor": manifest["actor"], "variants": [demo.identity(manifest, args.expected_sha, v) for v in VARIANTS],
                          "helper_hashes": HELPER_HASHES, "store": str(path)}, indent=2))
        return
    require(1024 <= args.port <= 65535, "Choose an unprivileged localhost port")
    workbench = Workbench(manifest, module, args.expected_sha, Store(path, args.expected_sha), demo)
    import gradio as gr
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    theme = gr.themes.Base(font=["system-ui", "sans-serif"], primary_hue="gray", secondary_hue="gray", neutral_hue="gray")
    theme.set(body_background_fill="#fafbfc", body_background_fill_dark="#fafbfc",
              body_text_color="#24262a", body_text_color_dark="#24262a",
              background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
              background_fill_secondary="#f0f2f4", background_fill_secondary_dark="#f0f2f4")
    build_app(workbench).queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=False, show_error=False, strict_cors=True, enable_monitoring=False,
        mcp_server=False, ssr_mode=False, footer_links=[], theme=theme, css=CSS,
        blocked_paths=[str(args.bundle.resolve()), str(path.resolve().parent), str(args.helper_code.resolve())],
        app_kwargs={"middleware": [Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])]})


if __name__ == "__main__":
    main()
