#!/usr/bin/env python3
"""Freeze development comparisons and actual multi-turn rollout specifications."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import prepare_understanding_r3 as pack
from r2_inference import DECODING, digest, read, require, save, sha
from understanding_r2_demo import validate_reply

R2_EVALUATION_SHA = "b8426be6dc938f3a81f97e0b27103d6a8ae2729dbe25241f05baa618b68c4d10"


def prepare(r2_evaluation, rollout_file, store, sessions, out, chat_store=None, chat_turns=()):
    require(not out.exists(), "Use a fresh evaluation directory")
    require(sha(r2_evaluation / "manifest.json") == R2_EVALUATION_SHA, "Frozen R2 evaluation changed")
    old = read(r2_evaluation / "manifest.json")
    for name, checksum in old["artifacts"].items():
        require(sha(r2_evaluation / name) == checksum, "Changed R2 evaluation input")
    cases = read(r2_evaluation / "cases.json")
    imported, proofs = [], []
    with sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True) as db:
        for session in sessions:
            row = db.execute("SELECT record FROM comparison_sessions WHERE id=?", (session,)).fetchone()
            require(row is not None, "Missing requested feedback session")
            state = json.loads(row[0])
            for index, pair in enumerate(state["rounds"]):
                if pair["status"] not in {"complete", "incomplete"} or not pair["verified"]:
                    continue
                identity = "feedback-" + session[:8] + "-" + str(index)
                messages = pair["shared_messages"]
                require(digest(messages) == pair["messages_sha256"], "Saved feedback history changed")
                imported.append({"id": identity, "messages": messages, "split": "development",
                    "cohort": "owner_observed_fixed_history", "training_eligible": False,
                    "human_review": {"status": "reported_failure_or_observation_not_formal_score"}})
                proofs.append({"case_id": identity, "round_id": pair["id"], "session_id": session,
                    "messages_sha256": pair["messages_sha256"], "pair_sha256": digest(pair),
                    "sides": pair["sides"]})
    require(not chat_turns or chat_store is not None, "Chat store required for requested feedback turns")
    if chat_store is not None:
        with sqlite3.connect(f"file:{chat_store.resolve()}?mode=ro", uri=True) as db:
            for turn_id in chat_turns:
                row = db.execute("SELECT record FROM turns WHERE id=?", (turn_id,)).fetchone()
                require(row is not None, "Missing requested feedback turn")
                turn = json.loads(row[0])
                require(turn["status"] in {"complete", "incomplete"}, "Unfinished/unverified feedback turn")
                validate_reply(turn["remote_result"], turn["identity"], turn["messages"], DECODING, turn["answer"])
                identity = "feedback-chat-" + turn_id
                imported.append({"id": identity, "messages": turn["messages"], "split": "development",
                    "cohort": "owner_observed_fixed_history", "training_eligible": False,
                    "human_review": {"status": "reported_failure_or_observation_not_formal_score"}})
                proofs.append({"case_id": identity, "turn_id": turn_id, "session_id": turn["session_id"],
                    "messages_sha256": digest(turn["messages"]), "turn_sha256": digest(turn),
                    "remote_result": turn["remote_result"], "observed_status": turn["status"]})
    cases += imported
    rollouts = read(rollout_file)
    require(rollouts and len({r["id"] for r in rollouts}) == len(rollouts), "Invalid rollout inventory")
    for item in rollouts:
        require(2 <= len(item["user_turns"]) <= 4 and item["language"] in {"tw", "en", "mixed"}, "Invalid rollout")
        require(item["rubric"] and all(isinstance(t, str) and t.strip() for t in item["user_turns"]), "Empty rubric/turn")
    out.mkdir(parents=True)
    save(out / "cases.json", cases)
    save(out / "rollouts.json", rollouts)
    save(out / "observed-feedback-proofs.json", proofs)
    # Preserve the original protected language-loss inputs verbatim.
    for name in old["artifacts"]:
        if "language" in name:
            destination = out / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r2_evaluation / name, destination)
    held = []
    for case in cases:
        held.extend({"id": case["id"] + ":" + str(i), "text": text} for i, text in enumerate(pack.fields(case)))
    for rollout in rollouts:
        held.extend({"id": rollout["id"] + ":" + str(i), "text": text} for i, text in enumerate(rollout["user_turns"]))
    pack.data.write_rows(out / "held-content.jsonl", held)
    manifest = {"version": "r3-conversation-evaluation-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "kind": "development_not_blind_final", "training_eligible": False,
        "r2_evaluation_sha256": R2_EVALUATION_SHA, "rollout_source_sha256": sha(rollout_file),
        "builder_sha256": sha(Path(__file__)), "counts": {"fixed_prompts": len(cases),
            "owner_observed_prompts": len(imported), "rollout_scenarios": len(rollouts),
            "rollout_user_turns": sum(len(r["user_turns"]) for r in rollouts)},
        "variants": ["base", "r2_meaning", "r3_response"], "decoding": DECODING,
        "protocol": {"fixed_history": "Identical messages and native prompt tokens for all checkpoints",
            "rollouts": "Identical scripted user turns; each checkpoint retains its own generated assistant history",
            "prompts": "No added system message, fallback, repetition guard, translation layer or rewritten reply",
            "selection": "Keep every frozen prompt and report failures; no post-result replacement",
            "metrics": ["existing strict JSON tasks", "held-out Twi NLL", "within-reply repeated ngrams",
                        "consecutive repeated replies", "requested language", "meaning", "useful next step", "unsupported claims"],
            "review": "Meaning, usefulness and naturalness require explicit per-case assessment; not inferred from EOS or perplexity"},
        "limits": ["Development cases are researcher-authored or previously observed, not a blind final test",
            "Scripted follow-ups are not adaptive human conversations", "No native-human or clinical certification",
            "Greedy short-context comparison does not establish population reliability"],
        "artifacts": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}}
    save(out / "manifest.json", manifest)
    save(out / "FROZEN.json", {"manifest_sha256": sha(out / "manifest.json")})
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r2-evaluation", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--session", action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chat-store", type=Path)
    p.add_argument("--chat-turn", action="append", default=[])
    a = p.parse_args()
    print(json.dumps(prepare(a.r2_evaluation, a.rollouts, a.store, a.session, a.out, a.chat_store, a.chat_turn)["counts"], indent=2))
