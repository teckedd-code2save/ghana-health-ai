# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.4.3", "pyarrow"]
# ///
"""Immutable public-source snapshots and durable bounded Modal run receipts."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal/train"))
from dialogue_translation_core import METHODS, digest, requests, structural_flags, validate_report

OUT = ROOT / "tmp/dialogue-translation-calibration/v1"
APP = "ghana-dialogue-translation-calibration"


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    from build_language_adaptation import OverlapIndex, protected_texts, read_jsonl
    parent = ROOT / "tmp/general-language-corpus/adaptation-v1"
    selection_path = ROOT / "data/response-adaptation/dialogue-translation-calibration.v1.json"
    selection = load(selection_path)
    manifest = load(parent / "manifest.json")
    for split in ("train", "validation"):
        artifact = next(r for r in manifest["artifacts"] if r["file"] == split + ".jsonl")
        if artifact["sha256"] != sha(parent / artifact["file"]):
            raise ValueError("Parent corpus changed")
    train, validation = [read_jsonl(parent / (s + ".jsonl")) for s in ("train", "validation")]
    blocked = OverlapIndex(protected_texts() + [m["content"] for r in validation for m in r["messages"]])
    reserved_groups = {r["group"] for r in validation}
    by_id = {r["id"]: r for r in train}
    rows = [by_id[r["id"]] for r in selection["cases"]]
    for row in rows:
        if (row["source"] != selection["source"] or row["revision"] != selection["revision"]
                or row["split"] != "train" or row["group"] in reserved_groups
                or any(blocked.contains(m["content"]) for m in row["messages"])):
            raise ValueError("Wrong source or held-out overlap: " + row["id"])
    demo_path = ROOT / "tmp/native-question-translation-v1/demonstrations.json"
    demo_originals = load(demo_path)
    demo_manifest = load(demo_path.with_name("manifest.json"))
    if digest(demo_originals) != demo_manifest["snapshots"][demo_path.name]:
        raise ValueError("Demonstrations changed")
    demos = [{"en": r["messages"][0]["content"].split("\n\n", 1)[1],
              "tw": r["messages"][-1]["content"]} for r in demo_originals if r["language"] == "tw"]
    inputs = [{k: r[k] for k in ("id", "messages")} for r in rows]
    for variant in METHODS:
        requests(inputs, variant, demos)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / "sources.json", rows)
    save(OUT / "inputs.json", inputs)
    save(OUT / "demonstrations.json", demos)
    save(OUT / "selection.json", selection)
    save(OUT / "manifest.json", {
        "purpose": selection["purpose"], "conversations": len(rows), "source_turns": sum(len(r["messages"]) for r in rows),
        "snapshots": {name: digest(load(OUT / name)) for name in ("sources.json", "inputs.json", "demonstrations.json", "selection.json")},
        "parents": {str(p.relative_to(ROOT)): sha(p) for p in (parent / "manifest.json", selection_path, demo_path)},
        "methods": METHODS, "training_eligible": False, "human_verified": False,
        "limit_per_model_seconds": 1200, "max_output_tokens_per_message": 768,
        "limitations": ["Hand-selected development calibration, not representative corpus accuracy.",
            "Translations do not repair source quality; originals and caveats remain separate.",
            "Only Gemma isolated/context is a within-model comparison. Afrique changes model and prompt together.",
            "Surface flags are review aids, not semantic or native-language certification."]})
    print(json.dumps(load(OUT / "manifest.json"), indent=2))


def snapshot(name):
    value = load(OUT / name)
    if digest(value) != load(OUT / "manifest.json")["snapshots"][name]:
        raise ValueError("Snapshot changed: " + name)
    return value


def result_stem(variant, attempt):
    return variant + ("" if attempt == 1 else ".attempt-" + str(attempt))


def completed(variant, attempt=1):
    stem = result_stem(variant, attempt)
    receipt = load(OUT / (stem + ".receipt.json"))
    result = load(OUT / (stem + ".results.json"))
    if receipt["state"] != "completed" or receipt["result_sha256"] != digest(result):
        raise ValueError("Incomplete or changed results")
    validate_report(result, receipt, snapshot("inputs.json"), snapshot("demonstrations.json"))
    return result


def export(gemma_attempt=1):
    sources = snapshot("sources.json")
    reports = {variant: completed(variant, gemma_attempt if variant == "gemma" else 1) for variant in METHODS}
    review = []
    counts = Counter()
    caveats = {r["id"]: r for r in snapshot("selection.json")["cases"]}
    for source in sources:
        alternatives = []
        for variant, report in reports.items():
            for method in METHODS[variant]:
                output = sorted((r for r in report["results"] if r["source_id"] == source["id"] and r["method"] == method), key=lambda r: r["turn"])
                turns = []
                for original, translated in zip(source["messages"], output, strict=True):
                    flags = structural_flags(original["content"], translated["translation"], translated["finish_reason"])
                    counts[variant + ":" + method + ":turns"] += 1
                    counts[variant + ":" + method + ":flagged_turns"] += bool(flags)
                    turns.append({"role": original["role"], "original_en": original["content"], **translated, "flags": flags})
                alternatives.append({"model": report["model"], "revision": report["revision"], "variant": variant,
                    "method": method, "run_id": report["run_id"], "turns": turns, "semantic_review": None})
        review.append({"source": source, "calibration": caveats[source["id"]], "alternatives": alternatives,
                       "training_eligible": False, "human_verified": False})
    save(OUT / "review.json", review)
    save(OUT / "summary.json", {"conversations": len(review), "counts": dict(counts),
        "review_sha256": digest(review), "training_eligible": 0,
        "semantic_quality_claim": None, "results": {v: digest(r) for v, r in reports.items()}})
    print(json.dumps(load(OUT / "summary.json"), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "status", "export"))
    parser.add_argument("--variant", choices=tuple(METHODS), default="afrique")
    parser.add_argument("--wait-seconds", type=int, default=5)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gemma-attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 1200:
        parser.error("Invalid wait")
    if args.action == "prepare":
        prepare()
        return
    if args.action == "export":
        export(args.gemma_attempt)
        return
    import modal
    inputs, demos = snapshot("inputs.json"), snapshot("demonstrations.json")
    stem = result_stem(args.variant, args.attempt)
    path = OUT / (stem + ".receipt.json")
    if args.action == "submit":
        if args.attempt > 1 and load(OUT / (result_stem(args.variant, args.attempt - 1) + ".receipt.json"))["state"] != "failed":
            raise ValueError("Retry requires a confirmed failed prior attempt")
        receipt = {"state": "submitting", "variant": args.variant,
            "attempt": args.attempt,
            "run_id": "dialogue_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "inputs_sha256": digest(inputs), "demonstrations_sha256": digest(demos),
            "request_count": len(requests(inputs, args.variant, demos)), "timeout_seconds": 1200}
        save(path, receipt)
        call = modal.Function.from_name(APP, "translate_" + args.variant).spawn(receipt["run_id"], inputs, demos)
        receipt.update(state="submitted", call_id=call.object_id)
    else:
        receipt = load(path)
        if receipt["state"] == "completed":
            completed(args.variant, args.attempt)
            print(json.dumps(receipt))
            return
        if not receipt.get("call_id"):
            raise ValueError("Uncertain submission; inspect Modal before any retry")
        try:
            report = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=args.wait_seconds)
        except TimeoutError:
            print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__, error=str(exc))
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        validate_report(report, receipt, inputs, demos)
        save(OUT / (stem + ".results.json"), report)
        receipt.update(state="completed", result_sha256=digest(report))
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
