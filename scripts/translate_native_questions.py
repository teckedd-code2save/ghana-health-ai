# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.4.3"]
# ///
"""Translate review questions; retain original answers and separate model judgments."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import build_native_answers as native
from compare_afrique_native import MODEL_REVISIONS, translation_text
from build_native_understanding_mix import read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/native-question-translation-v1"
APP = "ghana-afrique-native-comparison"
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs"
# Unchanged, agent-inspected training pairs outside the candidate source groups.
DEMO_SOURCE_IDS = ("66150", "71191", "67230")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load(name):
    return json.loads((OUT / name).read_text())


def select_questions(rows, blocked):
    selected = []
    for row in rows:
        source = row["source"]
        if source["source_record_hash"] in blocked or source["split"] != "train":
            continue
        if row["answer_tw"] != source["messages"][-1]["content"] or hashlib.sha256(row["answer_tw"].encode()).hexdigest() != row["answer_tw_sha256"]:
            raise ValueError("Native source answer changed")
        for index, candidate in enumerate(row["candidates"]):
            if candidate.get("source_review_candidate"):
                selected.append({"id": f"{row['id']}:q{index}", "source_id": row["id"],
                                 "candidate_index": index, "question_en": candidate["question_en"]})
    if not selected or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Missing or duplicate question candidates")
    return selected


def prepare():
    rows = native.load("ground-review.v2.json")
    native.validate_review_snapshot(rows, native.fixed_sources(), native.load("ground.inputs.json"))
    flags = [json.loads(line) for line in native.ROOT.joinpath("data/response-adaptation/source-review-findings.jsonl").read_text().splitlines()]
    selected = select_questions(rows, {r["source_record_hash"] for r in flags})
    folder = ROOT / "tmp/general-language-corpus/adaptation-v1"
    train, validation = read(folder, "train"), read(folder, "validation")
    demos = [r for key in DEMO_SOURCE_IDS for r in train if r["source"] == "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT"
             and r["source_record_id"] == key and r["task"] == "translation"]
    if len(demos) != 6 or any(r["source_record_hash"] in {f["source_record_hash"] for f in flags} for r in demos):
        raise ValueError("Missing or flagged source demonstrations")
    selected_ids = {r["source_id"] for r in selected}
    selected_sources = [r for r in rows if r["id"] in selected_ids]
    groups = {r["source"]["group"] for r in selected_sources}
    if groups & ({r["group"] for r in validation} | {r["group"] for r in demos}):
        raise ValueError("Held-out or demonstration source-group overlap")
    save("sources.json", selected_sources)
    save("questions.json", selected)
    save("demonstrations.json", demos)
    save("manifest.json", {"source_review_sha256": digest(rows), "flags_sha256": digest(flags),
        "snapshots": {"sources.json": digest(selected_sources), "questions.json": digest(selected), "demonstrations.json": digest(demos)},
        "model": MODEL, "revision": MODEL_REVISIONS["afrique"], "sources": len(selected_sources), "questions": len(selected),
        "demonstration_source_ids": DEMO_SOURCE_IDS, "demonstrations_human_verified": False,
        "generated_answers": 0, "training": False, "human_reviewed": False,
        "limitations": ["Generated questions and existing translations are not human-verified.",
            "Back-translation by the same model is a diagnostic, not independent verification.",
            "Do not turn dated or unidentified source events into standalone assistant claims."]})
    print(json.dumps(load("manifest.json"), indent=2))


def snapshot(name):
    value = load(name)
    if digest(value) != load("manifest.json")["snapshots"][name]:
        raise ValueError("Source snapshot changed: " + name)
    return value


def completed(stage):
    receipt, report, inputs = load(stage + ".receipt.json"), load(stage + ".results.json"), load(stage + ".inputs.json")
    if receipt["state"] != "completed" or receipt["results_sha256"] != digest(report):
        raise ValueError("Incomplete or changed saved translation results")
    validate_report(report, receipt, inputs)
    return {r["id"]: r for r in report["results"]}


def validate_report(report, receipt, inputs):
    expected = {r["id"] for r in inputs}
    ids = [r["id"] for r in report["results"]]
    if report["run_id"] != receipt["run_id"] or report["revision"] != MODEL_REVISIONS["afrique"] or report["model"] != MODEL:
        raise ValueError("Wrong or incomplete translation run")
    if report["inputs_sha256"] != digest(inputs) or receipt["inputs_sha256"] != digest(inputs):
        raise ValueError("Translation snapshot changed")
    if len(inputs) != len(expected) or len(ids) != len(expected) or set(ids) != expected or any(r["task"] != "translation" for r in report["results"]):
        raise ValueError("Missing or duplicate translations")


def translation_inputs(stage):
    language = "tw" if stage == "forward" else "en"
    demos = [r for r in snapshot("demonstrations.json") if r["language"] == language]
    prefix = "\n\n".join(f"{q} {a}" for q, a in map(translation_text, demos)) + "\n\n"
    forward = completed("forward") if stage == "back" else {}
    rows = []
    for question in snapshot("questions.json"):
        text = question["question_en"] if stage == "forward" else forward[question["id"]]["prediction"].strip()
        if not text or "\n" in text:
            raise ValueError("Empty or multiline question; inspect without silently repairing")
        source, target = ("English", "Twi") if stage == "forward" else ("Twi", "English")
        rows.append({"id": question["id"], "task": "translation", "prompt": prefix + f"{source}: {text}\n{target}:"})
    if stage == "back":
        for row in snapshot("sources.json"):
            rows.append({"id": row["id"] + ":answer", "task": "translation",
                         "prompt": prefix + f"Twi: {row['answer_tw']}\nEnglish:"})
    return rows


def export():
    sources, questions = snapshot("sources.json"), snapshot("questions.json")
    forward, back = completed("forward"), completed("back")
    findings = question_findings(set(forward))
    by_source = {}
    for question in questions:
        key = question["id"]
        by_source.setdefault(question["source_id"], []).append({**question,
            "question_tw": forward[key]["prediction"].strip(), "back_translation_en": back[key]["prediction"].strip(),
            "forward_finish_reason": forward[key]["finish_reason"], "back_finish_reason": back[key]["finish_reason"],
            "agent_translation_finding": findings.get(key),
            "translation_verified": False})
    rows = [{**r, "translated_question_candidates": by_source[r["id"]],
             "answer_back_translation_en": back[r["id"] + ":answer"]["prediction"].strip(),
             "answer_translation_finish_reason": back[r["id"] + ":answer"]["finish_reason"],
             "status": "native_question_translation_review", "standalone_conversation_eligible": False,
             "train_eligible": False, "human_reviewed": False,
             "context_policy": "Retain source statement as quoted context unless a reviewer establishes that standalone use is justified."}
            for r in sources]
    save("review.json", rows)
    save("summary.json", {"sources": len(rows), "question_candidates": len(questions), "source_answer_translation_checks": len(rows),
        "question_candidates_flagged_by_agent": len(findings),
        "agent_proposed_corrections_for_review": sum(bool(r.get("proposed_question_tw")) for r in findings.values()),
        "length_limit_forward": [k for k, r in forward.items() if r["finish_reason"] == "length"],
        "length_limit_back": [k for k, r in back.items() if r["finish_reason"] == "length"],
        "changed_original_answers": 0, "generated_answers": 0, "human_reviewed": 0, "train_eligible": 0,
        "review_sha256": digest(rows), "manifest": load("manifest.json")})
    print(json.dumps(load("summary.json"), indent=2))


def question_findings(expected_ids):
    path = ROOT / "data/response-adaptation/native-question-review-findings.json"
    reviewed = json.loads(path.read_text())
    if reviewed["forward_result_logical_sha256"] != load("forward.receipt.json")["results_sha256"]:
        raise ValueError("Question findings refer to different translations")
    findings = {r["id"]: r for r in reviewed["findings"]}
    if len(findings) != len(reviewed["findings"]) or not findings.keys() <= expected_ids:
        raise ValueError("Unknown or duplicate reviewed question")
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "status", "export"))
    parser.add_argument("--stage", choices=("forward", "back"), default="forward")
    parser.add_argument("--wait-seconds", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 1200:
        parser.error("Invalid wait")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare()
    elif args.action == "export":
        export()
    else:
        import modal
        name = args.stage + ".receipt.json"
        if args.action == "submit":
            inputs = translation_inputs(args.stage)
            run_id = "afrique9_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            receipt = {"state": "submitting", "run_id": run_id, "inputs_sha256": digest(inputs),
                       "rows": len(inputs), "stage": args.stage, "gpu": "A100-40GB", "timeout_seconds": 1200}
            save(name, receipt)
            save(args.stage + ".inputs.json", inputs)
            call = modal.Function.from_name(APP, "evaluate").spawn(run_id, "afrique", inputs)
            receipt.update(state="submitted", call_id=call.object_id)
        else:
            receipt = load(name)
            if receipt["state"] == "completed":
                completed(args.stage)
                print(json.dumps(receipt))
                return
            if not receipt.get("call_id"):
                raise ValueError("Uncertain submission; inspect before any retry")
            try:
                result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=args.wait_seconds)
            except TimeoutError:
                print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
                return
            except Exception as exc:
                receipt.update(state="failed", error_type=type(exc).__name__)
                (OUT / name).write_text(json.dumps(receipt, indent=2) + "\n")
                raise
            validate_report(result, receipt, load(args.stage + ".inputs.json"))
            save(args.stage + ".results.json", result)
            receipt.update(state="completed", results_sha256=digest(result), elapsed_seconds=result["elapsed_seconds"])
        (OUT / name).write_text(json.dumps(receipt, indent=2) + "\n")
        if args.action == "status":
            completed(args.stage)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
