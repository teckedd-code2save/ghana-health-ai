#!/usr/bin/env python3
"""Read-only export of selected R6 failures, excluded from R7 training."""
import argparse
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import understanding_r6_workbench as workbench

SESSIONS = ("18d13da7638848258cd345c3b5f163d2", "d9bd93ed80fe4fb0a50f3dbbaa418aab")


def archive(bundle, database, out):
    checksum = database.parent.name
    manifest, module = workbench.bound_bundle(bundle.resolve(), checksum)
    helper = workbench.load_helpers(bundle.resolve() / "code")
    audit = workbench.Workbench(manifest, module, checksum, SimpleNamespace(checksum=checksum), helper)
    records = []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for identity in SESSIONS:
            row = db.execute("SELECT record FROM r6_sessions WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("Missing selected evidence: " + identity)
            state = json.loads(row[0])
            audit.audit_context(state)
            records.append(state)
    out.mkdir(parents=True, exist_ok=False)
    payload = {"version": "r7-observed-regressions-v1", "bundle_sha256": checksum,
               "evidence": records, "evidence_sha256": workbench.digest(records),
               "training_eligible": False, "fresh_test": False,
               "evidence_kind": "user_observed_development_regressions",
               "context_audit": "passed_independent_history_and_remote_request_hashes",
               "clinical_or_native_certification": False}
    (out / "evidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    held = []
    for state in records:
        for item in state["rounds"]:
            for variant, side in item["sides"].items():
                turn = side.get("turn")
                if not turn:
                    continue
                held.append({"id": "r7-observed:" + item["id"] + ":" + variant,
                             "group_id": "r7-observed:" + state["id"],
                             "source_id": "r7-observed:" + state["id"],
                             "text": "\n".join(m["content"] for m in turn["messages"])
                                     + "\n" + turn["answer"],
                             "training_eligible": False})
    (out / "held-content.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in held))
    return {"sessions": len(records), "turns": sum(len(s["rounds"]) for s in records),
            "held_units": len(held), "evidence_sha256": payload["evidence_sha256"],
            "out": str(out), "training_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.bundle, args.database, args.out), indent=2))
