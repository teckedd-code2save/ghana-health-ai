#!/usr/bin/env python3
"""Import verified local diagnostic pairs into the local comparison UI, once."""
import argparse
from pathlib import Path

import r2_comparison as comparison
from r2_inference import read, require, sha
from understanding_r2_demo import bound_bundle


def ingest(folder, store, bundle, checksum):
    manifest, module = bound_bundle(bundle, checksum)
    plan, summary = read(folder / "plan.json"), read(folder / "summary.json")
    require(plan["bundle_sha256"] == checksum and plan["cases_sha256"] == sha(folder / "cases.json"), "Diagnostic identity differs")
    require(summary["state"] == "complete", "Diagnostic incomplete")
    seen = {comparison.load_session(store, key).get("source") for _, key in comparison.conversations(store)}
    count = 0
    # The third-person case is imported last so it opens first for this feedback pass.
    cases = sorted(read(folder / "cases.json"), key=lambda c: c["id"].endswith("dd247e3f81cd4752a0d471cffaa74efc"))
    for case in cases:
        source = "diagnostic:" + plan["cases_sha256"] + ":" + case["id"]
        if source in seen:
            continue
        results = []
        for variant in ("base", "r2_meaning"):
            path = f"results/{variant}--{case['id']}.json"
            require(sha(folder / path) == summary["files"][path], "Diagnostic reply changed")
            record = read(folder / path)
            require(record["turn"]["messages"] == case["messages"], "Diagnostic histories differ")
            results.append(record["turn"]["remote_result"])
        comparison.import_pair(store, case["messages"], *results, manifest, checksum, module,
                               label=case["messages"][-1]["content"][:80], source=source)
        count += 1
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--expected-sha", required=True)
    a = p.parse_args()
    print({"imported": ingest(a.folder, a.store, a.bundle, a.expected_sha), "remote_calls": 0})
