#!/usr/bin/env python3
"""Preserve all lab sessions and reviews privately without copying browser secrets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def snapshot(store, out):
    store, out = Path(store).resolve(strict=True), Path(out)
    if out.exists():
        raise ValueError("Use a fresh snapshot directory; original evidence is immutable")
    with sqlite3.connect(store.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        sessions = [json.loads(r[0]) for r in db.execute("SELECT record FROM r4_sessions ORDER BY id")]
        db.row_factory = sqlite3.Row
        reviews = [dict(r) for r in db.execute("SELECT * FROM r4_reviews ORDER BY id")]
    replies = []
    for session in sessions:
        for index, item in enumerate(session["rounds"]):
            for variant, side in item["sides"].items():
                turn = side.get("turn") or {}
                output = turn.get("answer", "")
                replies.append({"id": session["id"] + ":" + item["id"] + ":" + variant,
                    "session_id": session["id"], "round_id": item["id"], "round_index": index,
                    "variant": variant, "policy": "guarded" if session["loop_guard"] else "raw",
                    "question": item["question"], "messages": turn.get("messages"), "output": output,
                    "status": side["status"], "evidence_verified": side["verified"],
                    "error": side.get("error"), "record_sha256": digest(side),
                    "invalid_unicode": "\ufffd" in output or any(0xD800 <= ord(c) <= 0xDFFF for c in output),
                    "training_eligible": False, "split": "observed_development",
                    "semantic_review": "pending", "human_reviewed": False})
    summary = {"sessions": len(sessions), "nonempty_sessions": sum(bool(s["rounds"]) for s in sessions),
        "rounds": sum(len(s["rounds"]) for s in sessions), "reply_slots": len(replies),
        "saved_assessments": len(reviews), "states": dict(Counter(r["status"] for r in replies)),
        "invalid_unicode_outputs": [r["id"] for r in replies if r["invalid_unicode"]],
        "private": True, "browser_credentials_included": False,
        "note": "Mechanical inventory only. Missing native review is not replaced by an automatic pass."}
    out.mkdir(parents=True, mode=0o700)
    for name, value in (("sessions.json", sessions), ("reviews.json", reviews),
                        ("replies.json", replies), ("summary.json", summary)):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        (out / name).chmod(0o600)
    manifest = {"version": "private-lab-snapshot-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_store": str(store), "records_sha256": digest(sessions),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
        "training_eligible": False, "private": True}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out / "manifest.json").chmod(0o600)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.store, args.out), indent=2))
