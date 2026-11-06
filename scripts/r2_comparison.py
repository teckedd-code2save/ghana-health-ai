"""Private, sequential base/candidate comparisons. No remote calls on import.

UI CONTRACT:
  comparison_store(audition_store) -> Path
  new_session(candidate='r2_meaning', label=None) -> state
  load_session(store, session_id) -> state
  conversations(store) -> [(label, session_id)]
  save_review(store, session_id, round_id, rating, note) -> None
  async compare(state, question, manifest, expected_sha, module, store=STORE,
                *, stop=None, remote=demo.modal_events, cancel=demo.cancel_modal,
                audition_fn=demo.audition) -> async iterator of state snapshots
  import_pair(store, messages, base_result, candidate_result, manifest,
              expected_sha, module, *, candidate='r2_meaning', label=None,
              source=None) -> validated, saved state (no remote call)

State: {id, candidate, label, manifest_sha256, revision, origin,
        history_policy: 'candidate_history_canonical',
        execution_order: ['base', 'candidate'], shared_messages: [], rounds: []}
Round: {id, created, question, shared_messages, messages_sha256,
        status, verified, error, sides: {'base': Side, 'candidate': Side}}
Side: {variant, status, turn, verified, ttft_seconds, wall_seconds, error}
turn is the existing audition turn (answer, identity, messages, remote_result).
Side status: queued/pending/complete/incomplete/error/cancelled/skipped.
Round status: pending/complete/incomplete/error/cancelled.
verified means identities/messages/decoding/native-prompt hashes verified,
NEVER a semantic winner. Accepted shared history uses candidate answers only.
Verified length/time-limited replies can receive follow-ups without being marked
complete or retried. Cancellation, missing evidence and transport errors cannot.
Imported diagnostics have origin='imported_diagnostic' and null timings.
Review ratings: base/candidate/tie/neither/unsure; append-only.
"""
import asyncio
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import threading
import time
import uuid
import understanding_r2_demo as demo


def comparison_store(audition_store):
    path = Path(audition_store)
    return path.with_name(path.stem + ".comparisons.sqlite3")


STORE = comparison_store(demo.STORE)


@contextmanager
def _db(store):
    path = Path(store)
    demo.require(not path.is_symlink() and path.resolve() != demo.STORE.resolve(), "Use a separate comparison store")
    with demo.connect_store(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS comparison_sessions (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, updated REAL NOT NULL, record TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS comparison_reviews (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, round_id TEXT NOT NULL, rating TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL)")
        yield db


def new_session(candidate="r2_meaning", label=None):
    demo.require(candidate in demo.LABELS and candidate != "base", "Choose a non-base candidate")
    demo.require(label is None or isinstance(label, str), "Invalid label")
    return {"id": uuid.uuid4().hex, "candidate": candidate, "label": label, "manifest_sha256": None,
            "revision": 0, "origin": "live", "history_policy": "candidate_history_canonical",
            "execution_order": ["base", "candidate"], "shared_messages": [], "rounds": []}


def _save(store, state):
    with _db(store) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT revision FROM comparison_sessions WHERE id=?", (state["id"],)).fetchone()
        demo.require((row[0] if row else 0) == state["revision"], "Stale comparison; reload, never retry pending calls")
        revision = state["revision"] + 1
        record = json.dumps({**state, "revision": revision}, ensure_ascii=False, allow_nan=False)
        db.execute("INSERT INTO comparison_sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,updated=excluded.updated,record=excluded.record",
                   (state["id"], revision, time.time(), record))
    state["revision"] = revision


def load_session(store, session_id):
    with _db(store) as db:
        row = db.execute("SELECT record FROM comparison_sessions WHERE id=?", (session_id,)).fetchone()
    demo.require(row is not None, "Comparison not found")
    return json.loads(row[0])


def conversations(store):
    with _db(store) as db:
        states = [json.loads(row[0]) for row in db.execute("SELECT record FROM comparison_sessions ORDER BY updated DESC,rowid DESC")]
    return [(s["label"] or f"{s['candidate']}: {s['rounds'][0]['question'][:60]}", s["id"]) for s in states]


def save_review(store, session_id, round_id, rating, note):
    demo.require(rating in {"base", "candidate", "tie", "neither", "unsure"}
                 and isinstance(note, str) and len(note) <= 8000, "Invalid comparison review")
    state = load_session(store, session_id)
    rounds = [r for r in state["rounds"] if r["id"] == round_id]
    demo.require(rounds and rounds[0]["status"] != "pending", "Select a finished round in this session")
    with _db(store) as db:
        db.execute("INSERT INTO comparison_reviews(session_id,round_id,rating,note,created) VALUES (?,?,?,?,?)",
                   (session_id, round_id, rating, note, time.time()))


def _round(state, messages):
    return {"id": uuid.uuid4().hex, "created": time.time(), "question": messages[-1]["content"],
            "shared_messages": copy.deepcopy(messages), "messages_sha256": demo.digest(messages),
            "status": "pending", "verified": False, "error": None,
            "sides": {name: {"variant": variant, "status": "queued", "turn": None, "verified": False,
                              "ttft_seconds": None, "wall_seconds": None, "error": None}
                      for name, variant in (("base", "base"), ("candidate", state["candidate"]))}}


def _accept(side, turn, messages, manifest, expected_sha):
    demo.require(turn["messages"] == messages, "Side input history differs")
    side["turn"] = copy.deepcopy(turn)
    side.update(status=turn["status"], error=turn.get("error"), verified=False)
    if turn["status"] in {"complete", "incomplete", "cancelled"} and turn.get("remote_result"):
        expected = demo.identity(manifest, expected_sha, side["variant"])
        value = demo.validate_reply(turn["remote_result"], expected, messages, manifest["decoding"], turn["answer"])
        status = "cancelled" if value["finish_reason"] == "cancelled" else "complete" if value["complete"] else "incomplete"
        demo.require(turn["identity"] == expected and turn["status"] == status, "Side completion evidence differs")
        side["verified"] = True
    elif turn["status"] in {"complete", "incomplete"}:
        raise ValueError("Missing final remote evidence")


def can_continue(pair):
    if not pair.get("verified") or pair.get("status") not in {"complete", "incomplete"}:
        return False
    for side in pair["sides"].values():
        if not side.get("verified") or side.get("status") not in {"complete", "incomplete"}:
            return False
        result = (side.get("turn") or {}).get("remote_result") or {}
        if result.get("finish_reason") not in {"eos", "max_new_tokens", "time_limit"}:
            return False
    return bool(pair["sides"]["candidate"]["turn"]["answer"].strip())


def canonical_history(state):
    demo.require(all(can_continue(pair) for pair in state["rounds"]),
                 "Unfinished or unverified comparison; start a new conversation, no automatic retry")
    if not state["rounds"]:
        return copy.deepcopy(state["shared_messages"])
    last = state["rounds"][-1]
    # Older saved sessions did not advance context after a verified length stop.
    # Reconstruct it from immutable evidence; never rewrite the stopped result.
    return copy.deepcopy(last["shared_messages"]) + [
        {"role": "assistant", "content": last["sides"]["candidate"]["turn"]["answer"]}]


def _finish(state, pair):
    sides = list(pair["sides"].values())
    if all(s["verified"] for s in sides):
        left, right = (s["turn"]["remote_result"] for s in sides)
        if any(left[k] != right[k] for k in ("messages_sha256", "prompt_ids_sha256", "decoding")):
            pair.update(status="error", error="Native prompt, messages, or decoding differ")
            return
        pair["verified"] = True
    statuses = {s["status"] for s in sides}
    pair["status"] = ("cancelled" if statuses & {"cancelled", "skipped"} else "error" if "error" in statuses
                      else "complete" if statuses == {"complete"} and pair["verified"] else "incomplete")
    pair["error"] = next((s["error"] for s in sides if s["error"]), None)
    if can_continue(pair):
        state["shared_messages"] = copy.deepcopy(pair["shared_messages"]) + [
            {"role": "assistant", "content": pair["sides"]["candidate"]["turn"]["answer"]}]


async def compare(state, question, manifest, expected_sha, module, store=STORE, *, stop=None,
                  remote=demo.modal_events, cancel=demo.cancel_modal, audition_fn=demo.audition):
    """Sequential streams; TTFT is local time to first nonempty pending text update.

    Timings include local persistence, transport and cold start, not isolated GPU
    latency. Neither side is retried. Failed/unfinished rounds block continuation.
    """
    state = copy.deepcopy(state)
    if state["revision"]:
        demo.require(load_session(store, state["id"]) == state, "Stale or modified comparison state")
    demo.require(demo.is_sha(expected_sha) and state["manifest_sha256"] in (None, expected_sha), "Bundle differs")
    state["shared_messages"] = canonical_history(state)
    demo.require(state["candidate"] != "base" and state["candidate"] in manifest["variants"], "Invalid candidate")
    messages = copy.deepcopy(state["shared_messages"]) + [{"role": "user", "content": question}]
    module.validate_messages(messages)
    state["manifest_sha256"] = expected_sha
    pair = _round(state, messages)
    state["rounds"].append(pair)
    stop = stop if stop is not None else threading.Event()
    _save(store, state)
    try:
        yield copy.deepcopy(state)
        for side in pair["sides"].values():
            if stop.is_set():
                side.update(status="skipped", error="Stopped before submission")
                _save(store, state)
                yield copy.deepcopy(state)
                continue
            started = time.monotonic()
            side["status"] = "pending"
            iterator = None
            try:
                # Install canonical context before audition hashes/persists/calls it.
                # This also preserves a system message without fabricating base turns.
                # _accept checks the initial pending turn before remote submission;
                # an audition change that drops this mutation fails closed.
                def shared_history(target):
                    target[:] = copy.deepcopy(messages)
                    module.validate_messages(target)

                iterator = audition_fn(demo.new_session(side["variant"]), question, side["variant"],
                    manifest, expected_sha, shared_history, store, remote=remote, cancel=cancel, stop=stop)
                async for current in iterator:
                    turn = current["turns"][-1]
                    elapsed = time.monotonic() - started
                    if side["ttft_seconds"] is None and turn["status"] == "pending" and turn["answer"]:
                        side["ttft_seconds"] = elapsed
                    side["wall_seconds"] = elapsed
                    _accept(side, turn, messages, manifest, expected_sha)
                    _save(store, state)
                    yield copy.deepcopy(state)
                demo.require(side["status"] not in {"queued", "pending"}, "Audition ended without a final turn")
            except (asyncio.CancelledError, GeneratorExit):
                stop.set()
                side.update(status="cancelled", verified=False, error="Local comparison closed")
                raise
            except Exception as exc:
                side.update(status="error", verified=False, error=str(exc)[:2000])
                _save(store, state)
                yield copy.deepcopy(state)
            finally:
                if iterator is not None:
                    await iterator.aclose()
            if side["status"] == "cancelled":
                stop.set()
        _finish(state, pair)
        _save(store, state)
        yield copy.deepcopy(state)
    finally:
        if pair["status"] == "pending":
            stop.set()
            for side in pair["sides"].values():
                if side["status"] in {"queued", "pending"}:
                    side.update(status="skipped" if side["status"] == "queued" else "cancelled", verified=False)
            pair.update(status="cancelled", verified=False, error="Local comparison closed")
            _save(store, state)


def import_pair(store, messages, base_result, candidate_result, manifest, expected_sha, module, *,
                candidate="r2_meaning", label=None, source=None):
    """Import one existing diagnostic pair; validate evidence, never invent timings.

    Each call creates an independent session, allowing unrelated diagnostic
    histories to remain separate. Successful imports can subsequently continue.
    """
    demo.require(demo.is_sha(expected_sha), "Expected bundle SHA required")
    messages = copy.deepcopy(messages)
    module.validate_messages(messages)
    state = new_session(candidate, label)
    state.update(manifest_sha256=expected_sha, origin="imported_diagnostic", source=source,
                 shared_messages=copy.deepcopy(messages[:-1]))
    pair = _round(state, messages)
    for name, result in (("base", base_result), ("candidate", candidate_result)):
        side = pair["sides"][name]
        status = "cancelled" if result["finish_reason"] == "cancelled" else "complete" if result["complete"] else "incomplete"
        turn = {"id": uuid.uuid4().hex, "session_id": state["id"], "variant": side["variant"],
                "identity": result["identity"], "messages": messages, "question": messages[-1]["content"],
                "answer": result["output"], "status": status, "finish_reason": result["finish_reason"],
                "remote_result": result, "origin": "imported_diagnostic"}
        _accept(side, turn, messages, manifest, expected_sha)
    _finish(state, pair)
    demo.require(pair["verified"], pair["error"] or "Unverified diagnostic pair")
    state["rounds"].append(pair)
    _save(store, state)
    return copy.deepcopy(state)
