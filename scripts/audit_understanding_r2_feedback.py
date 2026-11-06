#!/usr/bin/env python3
"""Private, source-bound live-feedback audit and bounded checkpoint comparison.

Never modifies training/evaluation packs or the user's conversation database.
Observed replies are reused only after exact prompt/protocol/identity checks.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from r2_inference import digest, read, require, save, sha
from understanding_r2_demo import (
    audition, bound_bundle, identity, new_session, validate_reply,
)

VARIANTS = ("base", "r2_language", "r2_meaning")


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def cross_turn_flags(messages, answer):
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    key = normalized(answer)
    matches = [i for i, text in enumerate(previous) if key and normalized(text) == key]
    consecutive = 1 if key else 0
    for text in reversed(previous):
        if not key or normalized(text) != key:
            break
        consecutive += 1
    return {"matches_previous_assistant_indices": matches,
            "same_as_previous_assistant": bool(previous and key and normalized(previous[-1]) == key),
            "consecutive_identical_assistant_replies": consecutive,
            "review_required": bool(matches),
            "semantic_or_safety_score": None}


def observed_turn(store, turn_id):
    require(re.fullmatch(r"[a-f0-9]{32}", turn_id), "Explicit turn ID required")
    with sqlite3.connect(Path(store).resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT record FROM turns WHERE id=?", (turn_id,)).fetchone()
    require(row, "Requested feedback turn not found")
    turn = json.loads(row[0])
    require(turn["id"] == turn_id and turn["status"] == "complete", "Incomplete observed reply")
    return turn


def verify_turn(turn, manifest, checksum):
    expected = identity(manifest, checksum, turn["variant"])
    require(turn["identity"] == expected, "Feedback checkpoint changed")
    require(turn["messages"][-1] == {"role": "user", "content": turn["question"]}, "Feedback prompt differs")
    validate_reply(turn["remote_result"], expected, turn["messages"], manifest["decoding"], turn["answer"])


def prepare(bundle, checksum, store, turn_ids, out):
    require(not out.exists(), "Use a fresh feedback directory")
    require(1 <= len(turn_ids) <= 12 and len(set(turn_ids)) == len(turn_ids), "Provide 1-12 distinct turn IDs")
    manifest, runtime = bound_bundle(bundle, checksum)
    turns = [observed_turn(store, value) for value in turn_ids]
    for turn in turns:
        verify_turn(turn, manifest, checksum)
        runtime.validate_messages(turn["messages"])
    cases = [{"id": "observed-" + t["id"], "messages": t["messages"],
              "origin": "user_reported_live_feedback", "observed_variant": t["variant"],
              "observed": t, "history_type": "fixed_observed_history_not_independent_rollout"} for t in turns]
    cases.append({"id": "isolated-identity-en", "messages": [{"role": "user", "content": "Who are you?"}],
                  "origin": "project_authored_diagnostic", "history_type": "single_turn"})
    slots = len(cases) * len(VARIANTS)
    new_calls = slots - sum(c.get("observed_variant") in VARIANTS for c in cases)
    require(new_calls <= 36, "Diagnostic request limit exceeded")
    out.mkdir(parents=True, mode=0o700)
    save(out / "cases.json", cases)
    plan = {"schema": "r2-live-feedback-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": checksum, "cases_sha256": sha(out / "cases.json"),
            "variants": list(VARIANTS), "slots": slots, "new_request_limit": new_calls,
            "private": True, "raw_feedback_local_only": True, "frozen_benchmark_changed": False,
            "human_review": "pending", "new_training": False,
            "limits": "Fixed-history diagnostic; no semantic, clinical or population score. No automatic retries."}
    save(out / "plan.json", plan)
    save(out / "observed-audit.json", [
        {"turn_id": t["id"], "history_messages": len(t["messages"]),
         "prompt_and_identity_verified": True, "cross_turn": cross_turn_flags(t["messages"], t["answer"]),
         "within_reply": t["remote_result"].get("diagnostics", {}).get("repetition")}
        for t in turns])
    return plan


def fixed_history_state(messages, variant, expected):
    state = new_session(variant)
    require(messages and len(messages) % 2 == 1, "Alternating non-system history required")
    for i in range(0, len(messages) - 1, 2):
        require(messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant", "Invalid fixed history")
        state["turns"].append({"question": messages[i]["content"], "answer": messages[i+1]["content"],
                              "identity": expected, "status": "complete", "origin": "fixed_observed_history"})
    require(messages[-1]["role"] == "user", "Last message must be user")
    return state


async def run(bundle, checksum, out):
    manifest, runtime = bound_bundle(bundle, checksum)
    plan = read(out / "plan.json")
    require(plan["bundle_sha256"] == checksum and sha(out / "cases.json") == plan["cases_sha256"], "Feedback plan changed")
    cases = read(out / "cases.json")
    require(plan["variants"] == list(VARIANTS) and plan["slots"] == len(cases) * len(VARIANTS), "Plan shape changed")
    require(plan["new_request_limit"] <= 36, "Too many requests")
    for variant in VARIANTS:
        for case in cases:
            path = out / "results" / (variant + "--" + case["id"] + ".json")
            if path.exists():
                saved = read(path)
                require(saved["case_sha256"] == digest(case) and saved["variant"] == variant, "Saved result differs")
                require(saved["state"] == "complete", "Unresolved request: inspect it; do not automatically resubmit")
                require(saved["turn"]["messages"] == case["messages"], "Saved history differs")
                verify_turn(saved["turn"], manifest, checksum)
                continue
            if case.get("observed_variant") == variant:
                turn, origin = case["observed"], "exact_saved_live_reply"
            else:
                save(path, {"state": "submitting", "case_sha256": digest(case), "variant": variant})
                state = fixed_history_state(case["messages"], variant, identity(manifest, checksum, variant))
                async for state in audition(state, case["messages"][-1]["content"], variant, manifest,
                        checksum, runtime.validate_messages, out / "probe.sqlite3"):
                    turn = state["turns"][-1]
                    save(path, {"state": "running", "case_sha256": digest(case), "variant": variant, "turn": turn})
                origin = "new_paired_reply"
            require(turn["messages"] == case["messages"], "Compared history differs")
            verify_turn(turn, manifest, checksum)
            record = {"state": "complete", "case_sha256": digest(case), "variant": variant,
                      "origin": origin, "turn": turn, "cross_turn": cross_turn_flags(turn["messages"], turn["answer"])}
            save(path, record)
            print(json.dumps({"variant": variant, "case": case["id"], "origin": origin, "state": "complete"}), flush=True)
    files = sorted((out / "results").glob("*.json"))
    require(len(files) == plan["slots"], "Missing reply slots")
    records = [read(p) for p in files]
    summary = {"state": "complete", "slots": len(files), "origins": dict(Counter(r["origin"] for r in records)),
               "cross_turn_flags": {v: sum(r["cross_turn"]["same_as_previous_assistant"] for r in records if r["variant"] == v) for v in VARIANTS},
               "semantic_review": "pending", "clinical_review": "pending",
               "files": {p.relative_to(out).as_posix(): sha(p) for p in files}}
    save(out / "summary.json", summary)
    return summary


def render_report(out):
    plan, summary = read(out / "plan.json"), read(out / "summary.json")
    # The narrative below describes this exact diagnostic, not arbitrary future cases.
    require(plan["cases_sha256"] == "6f70045fe94e60d5604d3e2592033c2a0bf45b0981e805e0443d2360f757d0b1",
            "This report narrative is bound to the 6 November feedback cases")
    require(sha(out / "cases.json") == plan["cases_sha256"], "Cases changed")
    require(summary["state"] == "complete" and summary["slots"] == plan["slots"], "Comparison incomplete")
    cases = read(out / "cases.json")
    esc = lambda value: html.escape(str(value))
    sections = []
    for case in cases:
        replies = []
        for variant in VARIANTS:
            name = "results/" + variant + "--" + case["id"] + ".json"
            require(sha(out / name) == summary["files"][name], "Reply checksum differs")
            record = read(out / name)
            require(record["case_sha256"] == digest(case) and record["turn"]["messages"] == case["messages"], "Reply prompt differs")
            replies.append(f'<section><h3>{esc(variant)}</h3><p class="meta">{esc(record["origin"])}; '
                f'{esc(record["turn"]["finish_reason"])}; same as preceding assistant: '
                f'{esc(record["cross_turn"]["same_as_previous_assistant"])}</p>'
                f'<pre>{esc(record["turn"]["answer"])}</pre></section>')
        history = ''.join(f'<p><b>{esc(m["role"])}</b>: {esc(m["content"])}</p>' for m in case["messages"])
        sections.append(f'<article><h2>{esc(case["messages"][-1]["content"])}</h2>'
                        f'<details><summary>Exact input history ({len(case["messages"])} messages)</summary>{history}</details>'
                        f'<div class="replies">{"".join(replies)}</div></article>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R2: what did the adapter change?</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#fafbfc;color:#23262d;font:16px/1.6 system-ui;letter-spacing:0}}
main{{max-width:1240px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}h2{{font-size:20px}}h3{{font-size:16px}}
article{{border-top:1px solid #ccd4df;margin-top:28px;padding-top:12px}}.replies{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}}
.replies section{{min-width:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui}}.meta{{font-size:13px;color:#596473}}
details{{padding:8px 0}}summary{{cursor:pointer}}code{{overflow-wrap:anywhere}}a{{color:#245fb4}}
@media(max-width:850px){{.replies{{grid-template-columns:1fr}}main{{padding:20px 16px}}}}
</style><main><p class="meta">Ghana Health AI | 6 November 2026 | Private live-feedback diagnostic</p>
<h1>What did our adapter change?</h1>
<p>I compared the unadapted foundation, the language-stage adapter and the language-plus-meaning adapter using exactly the same input histories and decoding. All three load the same pinned base weights. There is no external answering model.</p>
<p><strong>15 reply slots: 11 new generations and 4 verified saved replies.</strong> Reusing the user's actual replies avoids regenerating a more favourable answer. These are five development cases, not a representative accuracy estimate.</p>
<p>The third-person headache question is echoed by the base and language-only checkpoint; meaning adaptation changes it into a longer reply. The name question goes the other direction: language-only responds directly, while meaning adaptation gives a generic offer. All three repeat the same sympathy sentence when given the stalled follow-up history.</p>
<p>The isolated identity question elicits an OpenAI claim even from the unadapted base. That establishes this behaviour exists before our adapter; it does not identify which original training document caused it. Checkpoint metadata, not a model's self-description, identifies what is running.</p>
<p><strong>What this proves:</strong> adapters measurably change these outputs, with both gains and regressions. It does not establish clinical quality or general reasoning. The longer response remains repetitive and non-specific. Native and clinical judgments remain separate.</p>
<p>The full observed histories below are held constant across checkpoints. This is a fixed-history comparison, not three independent conversations. Existing assistant mistakes in those histories can affect every checkpoint.</p>
<p>New cross-turn diagnostics catch three consecutive identical sympathy replies. The older within-reply repeated-4-gram check reported zero for that short sentence. A repeat flag requires review; absence of a flag is not correctness.</p>
<p><a href="http://127.0.0.1:7866/?__theme=light">Open the local model test</a> | <a href="summary.json">Result checksums</a></p>
{''.join(sections)}<footer><p class="meta">Bundle <code>{esc(plan['bundle_sha256'])}</code>. Raw feedback and this report stay local. Frozen benchmark and training inputs remain unchanged. Protect these cases from future training; use independent repair examples.</p></footer></main></html>'''
    target = out / "comparison.html"
    require(not target.exists(), "Report already exists; preserve previous evidence")
    target.write_text(page)
    save(out / "report-checksum.json", {"comparison.html": sha(target), "cases.json": sha(out / "cases.json"),
                                        "summary.json": sha(out / "summary.json")})
    return {"report": str(target), "reply_files_verified": len(summary["files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--turn", action="append", default=[])
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.store, "Read-only conversation store required")
        result = prepare(args.bundle, args.expected_sha, args.store, args.turn, args.out)
    elif args.action == "run":
        result = asyncio.run(run(args.bundle, args.expected_sha, args.out))
    else:
        bound_bundle(args.bundle, args.expected_sha)
        result = render_report(args.out)
    print(json.dumps(result, indent=2))
