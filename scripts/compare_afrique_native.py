# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.4.3", "sacrebleu==2.5.1"]
# ///
"""Preserve source references locally and compare suitable tasks for base models."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_native_understanding_mix import read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/afrique-native-comparison"
APP = "ghana-afrique-native-comparison"
MODEL_REVISIONS = {
    "base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
    "afrique": "4358dcbc062421751174279da1efe9f88f85e1d5",
}
# Inspected before inference; unchanged source pairs, not human-verified gold.
DEMO_SOURCE_IDS = ("69432", "68787", "64065")
DEMONSTRATIONS = """Choose the correct answer for each question. Options can change between questions.

Premise: The shop is closed today.
Hypothesis: The shop is not open today.
A. entailment
B. neutral
C. contradiction
Answer: A

Premise: The shop is closed today.
Hypothesis: The owner is ill.
A. contradiction
B. neutral
C. entailment
Answer: B

Premise: The shop is closed today.
Hypothesis: The shop is open today.
A. neutral
B. entailment
C. contradiction
Answer: C

"""


def translation_text(row):
    first, answer = row["messages"]
    prefix, text = first["content"].split("\n\n", 1)
    if not prefix.startswith("Translate from ") or not text or "\n" in text or "\n" in answer["content"]:
        raise ValueError("Expected an unchanged single-line source pair")
    source, target = ("English", "Twi") if row["language"] == "tw" else ("Twi", "English")
    return f"{source}: {text}\n{target}:", answer["content"]


def prepare_cases():
    nli = json.loads((ROOT / "tmp/semantic-response-v4/afrixnli/cases.json").read_text())
    inputs = [{"id": r["id"], "task": "nli", "prompt": DEMONSTRATIONS + r["messages"][0]["content"] + "\nAnswer: "} for r in nli]
    folder = ROOT / "tmp/general-language-corpus/adaptation-v1"
    source = {split: [r for r in read(folder, split) if r["task"] == "translation" and
                      r["source"] == "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT"] for split in ("train", "validation")}
    if {r["group"] for r in source["train"]} & {r["group"] for r in source["validation"]}:
        raise ValueError("Source leakage")
    ordered = lambda rows: sorted(rows, key=lambda r: hashlib.sha256(r["source_record_hash"].encode()).hexdigest())
    # Groups are connected source components, not individual parallel records.
    buckets = {}
    for row in ordered(source["validation"]):
        if row["language"] == "en":
            buckets.setdefault(row["group"], []).append(row["source_record_hash"])
    hashes = []
    while len(hashes) < 32 and any(buckets.values()):
        for group in sorted(buckets):
            if buckets[group] and len(hashes) < 32:
                hashes.append(buckets[group].pop(0))
    selected = [r for r in source["validation"] if r["source_record_hash"] in hashes]
    if len(nli) != 900 or len(selected) != 64:
        raise ValueError("Unexpected source count")
    demo_ids = []
    for language in ("tw", "en"):
        demos = [r for source_id in DEMO_SOURCE_IDS for r in source["train"]
                 if r["source_record_id"] == source_id and r["language"] == language]
        if len(demos) != 3 or len({r["group"] for r in demos}) != 3:
            raise ValueError("Missing or duplicated source demonstrations")
        demo_ids.extend(r["id"] for r in demos)
        prefix = "\n\n".join(f"{q} {a}" for q, a in map(translation_text, demos)) + "\n\n"
        for row in selected:
            if row["language"] == language:
                question, _ = translation_text(row)
                inputs.append({"id": row["id"], "task": "translation", "prompt": prefix + question})
    references = {"nli": nli, "translation": selected, "demonstration_source_ids": demo_ids,
                  "demonstrations_human_verified": False,
                  "translation_source_groups": len({r["group"] for r in selected})}
    return inputs, references


def score(references, results):
    nli = {r["id"]: r for r in references["nli"]}
    translation = {r["id"]: r for r in references["translation"]}
    groups = {}
    for variant in ("base", "afrique"):
        rows = results[variant]["results"]
        if len(rows) != len(nli)+len(translation) or {r["id"] for r in rows} != nli.keys() | translation.keys():
            raise ValueError("Incomplete or duplicated source results")
        by_id = {r["id"]: r for r in rows}
        for language in ("tw", "en"):
            cases = [r for r in nli.values() if r["language"] == language]
            correct = sum(by_id[r["id"]]["prediction"].strip() == r["expected"] for r in cases)
            distribution = Counter(by_id[r["id"]]["prediction"].strip() for r in cases)
            groups[f"{variant}:nli:{language}"] = {"rows": len(cases), "correct": correct,
                "accuracy": correct/len(cases), "predicted_label_counts": dict(distribution)}
            pairs = [r for r in translation.values() if r["language"] == language]
            from sacrebleu.metrics import CHRF
            metric = CHRF(word_order=2)
            value = metric.corpus_score([by_id[r["id"]]["prediction"].strip() for r in pairs],
                [[r["messages"][-1]["content"] for r in pairs]])
            groups[f"{variant}:translation:{language}"] = {"rows": len(pairs), "chrf_plus_plus": value.score,
                "metric_signature": str(metric.get_signature()),
                "length_limit_ids": [r["id"] for r in pairs if by_id[r["id"]]["finish_reason"] == "length"]}
    return {"scores": groups, "not_chat_accuracy": True, "trained_by_project": False,
        "limitations": ["Restricted one-token classification differs from earlier unconstrained instruction-model evaluations.",
            "Three English NLI demonstrations are project-authored; translation demonstrations are unchanged training-source pairs.",
            "No held-out reference answers enter prompts. Official AfriXNLI test remains untouched.",
            "Publisher pretraining overlap is unknown. chrF++ is reference similarity, not verified semantic quality.",
            "This evaluates base-model language tasks, not direct replies, tools, or clinical safety."]}


def verified_snapshot(receipt, name):
    raw = (OUT / (name + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt[name + "_sha256"]:
        raise ValueError("Saved evaluation snapshot changed: " + name)
    return json.loads(raw)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "prepare-status", "submit", "status"])
    parser.add_argument("--wait-seconds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Use a fresh directory for an explicit retry; never erase old receipts")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if not 1 <= args.wait_seconds <= 1800:
        parser.error("Invalid wait")
    import modal
    OUT.mkdir(parents=True, exist_ok=True)
    prep = args.action.startswith("prepare")
    path = OUT / ("preparation.receipt.json" if prep else "run.receipt.json")
    if args.action in ("prepare", "submit"):
        if path.exists():
            raise ValueError("Already submitted; inspect saved receipt")
        receipt = {"state": "submitting", "app": APP, "calls": {}}
        if not prep:
            ready = json.loads((OUT / "preparation.results.json").read_text())
            if {k: v["revision"] for k, v in ready.items()} != MODEL_REVISIONS:
                raise ValueError("Preparation does not match pinned models")
            inputs, refs = prepare_cases()
            receipt["run_id"] = "afrique9_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            for name, value in (("inputs", inputs), ("references", refs)):
                raw = json.dumps(value, ensure_ascii=False, indent=2)
                (OUT / (name + ".json")).write_text(raw)
                receipt[name + "_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            receipt.update(rows=len(inputs), training=False, gpu="A100-40GB", seconds_per_variant=1200)
        with path.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        for variant in (["preparation"] if prep else ["base", "afrique"]):
            fn = modal.Function.from_name(APP, "prepare" if prep else "evaluate")
            call = fn.spawn() if prep else fn.spawn(receipt["run_id"], variant, inputs)
            receipt["calls"][variant] = call.object_id
            path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["state"] = "submitted"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    else:
        receipt = json.loads(path.read_text())
        expected = {"preparation"} if prep else {"base", "afrique"}
        if set(receipt["calls"]) != expected:
            raise ValueError("Uncertain submission; inspect Modal before any retry")
        results = {}
        for key, call_id in receipt["calls"].items():
            result_path = OUT / ("preparation.results.json" if prep else key + ".results.json")
            if result_path.exists():
                results[key] = json.loads(result_path.read_text())
                continue
            try:
                results[key] = modal.FunctionCall.from_id(call_id).get(timeout=args.wait_seconds)
            except TimeoutError:
                print(json.dumps({"state": "pending", "variant": key, "call_id": call_id}))
                return
            except Exception as exc:
                receipt.update(state="failed", failed_variant=key, error_type=type(exc).__name__)
                path.write_text(json.dumps(receipt, indent=2) + "\n")
                raise
            result_path.write_text(json.dumps(results[key], ensure_ascii=False, indent=2) + "\n")
        if not prep:
            inputs = verified_snapshot(receipt, "inputs")
            expected_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if any(r["inputs_sha256"] != expected_hash or r["revision"] != MODEL_REVISIONS[k]
                   or r["run_id"] != receipt["run_id"] for k, r in results.items()):
                raise ValueError("Remote result does not match submitted evaluation")
            report = score(verified_snapshot(receipt, "references"), results)
            (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        receipt["state"] = "completed"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"state": "completed", "receipt": str(path)}))


if __name__ == "__main__":
    main()
