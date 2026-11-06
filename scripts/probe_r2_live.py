#!/usr/bin/env python3
"""Small synthetic live transport check; separate from the frozen evaluation."""
import argparse
import asyncio
from pathlib import Path
import time
from understanding_r2_demo import audition, bound_bundle, modal_events, new_session
from r2_inference import save


async def probe(bundle, checksum, out):
    if out.exists():
        raise ValueError("Use a fresh probe directory")
    out.mkdir(parents=True)
    manifest, runtime = bound_bundle(bundle, checksum)
    state = new_session("r2_meaning")
    results = []
    for question in ("wo ho te s3n", "Please reply in English. What did I just ask?"):
        began, first, deltas = time.monotonic(), None, 0
        async def measured(*args):
            nonlocal first, deltas
            async for event in modal_events(*args):
                if event.get("type") == "delta":
                    first = first if first is not None else time.monotonic() - began
                    deltas += 1
                yield event
        async for current in audition(state, question, "r2_meaning", manifest, checksum,
                                      runtime.validate_messages, out / "probe.sqlite3", remote=measured):
            state = current
        turn = state["turns"][-1]
        result = {"question": question, "status": turn["status"], "answer": turn["answer"],
                  "first_text_seconds": first, "delta_events": deltas, "elapsed_seconds": time.monotonic() - began,
                  "turn": turn, "source": "synthetic_transport_probe", "native_review": "pending"}
        results.append(result)
        save(out / "results.json", results)
        print({k: result[k] for k in ("status", "first_text_seconds", "delta_events", "elapsed_seconds")}, flush=True)
        if turn["status"] != "complete":
            raise RuntimeError("Live probe incomplete; inspect saved evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(probe(args.bundle, args.expected_sha, args.out))
