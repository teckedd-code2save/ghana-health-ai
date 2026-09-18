# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.4.3"]
# ///
"""Source-preserving instruction synthesis pilot with separate English adjudication."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from build_native_understanding_mix import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal/train"))
from native_answer_core import validate_inputs, grounded_approval

APP = "ghana-native-answer-synthesis"
OUT = ROOT / "tmp/native-answer-synthesis-v1"
PARENT = ROOT / "tmp/general-language-corpus/adaptation-v1"
HEALTH = re.compile(r"\b(?:health|disease|malaria|cancer|patients?|medicine|pregnan\w*|doctor|fever|covid|clinical|hospital|nurs\w*|surgery|dosage)\b", re.I)
# English-only reviewer controls. Never exported as source training examples.
REVIEW_CONTROLS = [
    ("The delivery costs 20 cedis.", ["How much does delivery cost?", "When will the delivery arrive?"], [True, False]),
    ("I will travel to Kumasi tomorrow.", ["When will you travel to Kumasi?", "What did you do yesterday?"], [True, False]),
    ("I need one kilogram of tomatoes.", ["How many kilograms of tomatoes do you need?", "Why do you need tomatoes?"], [True, False]),
    ("Plant the seeds two centimetres deep.", ["What soil type do these seeds need?", "How deep should I plant the seeds?"], [False, True]),
    ("I am not sure whether the shop is open.", ["What time does the shop close?", "Do you know whether the shop is open?"], [False, True]),
    ("She said it was better that way.", ["What is her name?", "Why did she prefer that way?"], [False, False]),
    ("I own two farms and have three children.", ["What do you own and how many children do you have?", "Do you own any farms or have children?"], [False, False]),
]
GROUND_CONTROLS = [
    ("The delivery costs 20 cedis.", ["How much does delivery cost?", "When will delivery arrive?"], [True, False]),
    ("People are no longer getting married for some reasons.", ["Why are people no longer getting married?", "Are people still getting married?"], [False, True]),
    ("There are several companies in the district.", ["Exactly how many companies are in the district?", "Are there companies in the district?"], [False, True]),
    ("The library will reopen on Monday.", ["Why was the library closed?", "When will the library reopen?"], [False, True]),
    ("The road was closed because the bridge collapsed.", ["Why was the road closed?", "When did the bridge collapse?"], [True, False]),
    ("I own two farms and have three children.", ["What farms or children do you have?", "As a fictional farmer, describe your family and property."], [False, True]),
    ("She sold it to her neighbor.", ["Who did she sell it to?", "What did she sell?"], [False, False]),
    ("Her teacher praised her work.", ["What happened to Ama after her teacher read her essay?", "What was Ama's exact score?"], [True, False]),
    ("I am not sure whether the shop is open.", ["Do you know whether the shop is open?", "What time does the shop close?"], [True, False]),
    ("The farmer did not sell the maize yesterday.", ["Did the farmer sell the maize yesterday?", "Why did the farmer refuse to sell the maize yesterday?"], [True, False]),
    ("Read the instructions and then connect the cable.", ["What should I do before connecting the cable?", "What color is the cable?"], [True, False]),
    ("Do not share your account password with anyone.", ["How can I keep my password private?", "Who stole my password?"], [True, False]),
    ("Many people reported delays, but no cause was identified.", ["What caused the delays?", "How many people reported delays, exactly?"], [False, False]),
    ("Kofi moved to Accra last year to attend university.", ["Why did Kofi move to Accra?", "Which university did Kofi attend?"], [True, False]),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(name, value, exclusive=False):
    with (OUT / name).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load(name):
    return json.loads((OUT / name).read_text())


def prepare_sources(limit):
    train, validation = read(PARENT, "train"), read(PARENT, "validation")
    forbidden_groups = {r["group"] for r in validation}
    flags = [json.loads(line) for line in (ROOT / "data/response-adaptation/source-review-findings.jsonl").read_text().splitlines()]
    blocked = {r["source_record_hash"] for r in flags}
    seen_en, seen_tw, rows, exclusions = set(), set(), [], Counter()
    for row in sorted(train, key=lambda r: digest("native-answer-v1:" + r["id"])):
        if row["language"] != "tw" or row["source"] != "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT":
            continue
        if row["group"] in forbidden_groups:
            raise ValueError("Cross-split source leakage")
        if row["source_record_hash"] in blocked:
            exclusions["flagged_source"] += 1
            continue
        english = row["messages"][0]["content"].split("\n\n", 1)[1]
        twi = row["messages"][-1]["content"]
        if HEALTH.search(english):
            exclusions["health_outside_general_pilot"] += 1
            continue
        if not 8 <= len(english.split()) <= 90 or not 8 <= len(twi.split()) <= 120 or "\n" in english+twi:
            exclusions["length_or_structure"] += 1
            continue
        norm = lambda text: " ".join(text.casefold().split())
        en, tw = norm(english), norm(twi)
        if en in seen_en or tw in seen_tw:
            exclusions["duplicate_source_text"] += 1
            continue
        seen_en.add(en)
        seen_tw.add(tw)
        rows.append({"id": row["id"], "answer_en": english, "answer_tw": twi,
                     "source": row, "answer_tw_sha256": hashlib.sha256(twi.encode()).hexdigest()})
    if len(rows) < limit:
        raise ValueError("Not enough distinct source candidates")
    return rows[:limit], {"eligible_pool": len(rows), "selected": limit, "exclusions": dict(exclusions),
        "training_split_only": True, "human_reviewed": False, "generated_twi_answers": 0,
        "parent_manifest_sha256": hashlib.sha256((PARENT / "manifest.json").read_bytes()).hexdigest(),
        "flags_sha256": hashlib.sha256((ROOT / "data/response-adaptation/source-review-findings.jsonl").read_bytes()).hexdigest()}


def fixed_sources():
    rows, manifest = load("sources.json"), load("manifest.json")
    if digest(rows) != manifest["sources_sha256"]:
        raise ValueError("Source snapshot changed")
    for row in rows:
        if row["answer_tw"] != row["source"]["messages"][-1]["content"] or hashlib.sha256(row["answer_tw"].encode()).hexdigest() != row["answer_tw_sha256"]:
            raise ValueError("Original Twi answer changed")
    return rows


def completed(stage):
    receipt, report, inputs = load(stage + ".receipt.json"), load(stage + ".results.json"), load(stage + ".inputs.json")
    if receipt["state"] != "completed" or report["input_sha256"] != digest(inputs) or digest(inputs) != receipt["input_sha256"]:
        raise ValueError("Results are incomplete or do not match submitted input")
    if digest(report) != receipt["results_sha256"]:
        raise ValueError("Saved model results changed")
    if len(report["results"]) != len(inputs) or {r["id"] for r in report["results"]} != {r["id"] for r in inputs}:
        raise ValueError("Missing or duplicate results")
    return {r["id"]: r for r in report["results"]}


def stage_inputs(stage):
    rows = fixed_sources()
    if stage == "generate":
        return [{"id": r["id"], "answer_en": r["answer_en"]} for r in rows]
    generation = completed("generate")
    return ([{"id": r["id"], "answer_en": r["answer_en"], "questions": generation[r["id"]]["parsed"]["questions"]}
            for r in rows if generation[r["id"]].get("parsed", {}).get("suitable")]
            + [{"id": stage + "-control:" + str(i), "answer_en": answer, "questions": questions}
               for i, (answer, questions, _) in enumerate(GROUND_CONTROLS if stage == "ground" else REVIEW_CONTROLS)])


def export_grounding():
    from gate_native_answers import gate
    sources, grounds = fixed_sources(), completed("ground")
    reviewed = load("review.json")
    validate_review_snapshot(reviewed, sources, load("ground.inputs.json"))
    flags = [json.loads(line) for line in (ROOT / "data/response-adaptation/source-review-findings.jsonl").read_text().splitlines()]
    reviewed, _ = gate(reviewed, flags)
    controls = []
    for i, (_, _, expected) in enumerate(GROUND_CONTROLS):
        values = {r["index"]: grounded_approval(r) for r in grounds[f"ground-control:{i}"].get("parsed", {}).get("reviews", [])}
        controls.extend({"id": f"ground-control:{i}:{j}", "expected": value,
                         "actual": values.get(j), "correct": values.get(j) is value}
                        for j, value in enumerate(expected))
    passed = all(r["correct"] for r in controls)
    for row in reviewed:
        values = {r["index"]: r for r in grounds.get(row["id"], {}).get("parsed", {}).get("reviews", [])}
        for i, candidate in enumerate(row["candidates"]):
            evidence = values.get(i)
            candidate.update(grounding=evidence, english_evidence_approved=bool(evidence and grounded_approval(evidence)),
                source_review_candidate=bool(candidate["review_candidate"] and evidence and grounded_approval(evidence)))
        row["status"] = "source_answer_grounded_review"
    summary = {"rows": len(reviewed), "sources_with_english_evidence": sum(any(c["english_evidence_approved"] for c in r["candidates"]) for r in reviewed),
        "sources_with_review_candidate": sum(any(c["source_review_candidate"] for c in r["candidates"]) for r in reviewed),
        "controls": {"total": len(controls), "correct": sum(r["correct"] for r in controls), "passed": passed, "results": controls},
        "automatic_batch_acceptance": False,
        "changed_source_answers": 0, "human_reviewed": 0, "gold_train_eligible": 0,
        "grounding_result_sha256": load("ground.receipt.json")["results_sha256"],
        "source_snapshot_sha256": load("manifest.json")["sources_sha256"],
        "flags_sha256": digest(flags),
        "limitations": ["Evidence quotes are machine-checked; answerability is still a model judgment.",
            "English compatibility cannot verify source Twi translation or clinical accuracy.",
            "Controls include known failure categories and are calibration, not unseen evaluation.",
            "Review candidates remain visible after calibration failures; this is not automatic training acceptance."]}
    save("ground-review.v2.json", reviewed, exclusive=True)
    save("ground-review.v2.summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2))


def validate_review_snapshot(reviewed, sources, inputs):
    if [r["id"] for r in reviewed] != [r["id"] for r in sources]:
        raise ValueError("Review and original source identities disagree")
    submitted = {r["id"]: r for r in inputs}
    for row, original in zip(reviewed, sources, strict=True):
        if any(row[k] != original[k] for k in ("answer_en", "answer_tw", "answer_tw_sha256", "source")):
            raise ValueError("Review source was edited; its old approval cannot be reused")
        questions = [c["question_en"] for c in row["candidates"]]
        if questions != submitted.get(row["id"], {}).get("questions", []):
            raise ValueError("Questions changed after grounding; previous votes are invalid")
        if questions and row["answer_en"] != submitted[row["id"]]["answer_en"]:
            raise ValueError("Grounding was performed on a different answer")


def export_review():
    sources = fixed_sources()
    generations, reviews = completed("generate"), completed("review")
    controls = []
    for i, (_, _, expected) in enumerate(REVIEW_CONTROLS):
        result = reviews["review-control:" + str(i)].get("parsed", {}).get("reviews", [])
        actual = {r["index"]: r["valid"] for r in result}
        controls.extend({"id": f"review-control:{i}:{j}", "expected": valid,
                         "actual": actual.get(j), "correct": actual.get(j) is valid}
                        for j, valid in enumerate(expected))
    output, approved_questions = [], Counter()
    for row in sources:
        generated = generations[row["id"]]
        candidates = []
        questions = generated.get("parsed", {}).get("questions", [])
        adjudication = reviews.get(row["id"], {}).get("parsed", {}).get("reviews", [])
        by_index = {r["index"]: r for r in adjudication}
        for i, question in enumerate(questions):
            review = by_index.get(i)
            valid = review is not None and review["valid"]
            if valid:
                approved_questions[question.casefold().strip()] += 1
            candidates.append({"question_en": question, "english_match_approved": valid,
                               "independent_review": review})
        output.append({**row, "candidates": candidates, "generation": generated,
            "status": "source_answer_review", "human_reviewed": False, "train_eligible": False})
    for row in output:
        for candidate in row["candidates"]:
            candidate["duplicate_question"] = approved_questions[candidate["question_en"].casefold().strip()] > 1
    summary = {"rows": len(output), "sources_with_english_approved_question": sum(any(c["english_match_approved"] and not c["duplicate_question"] for c in r["candidates"]) for r in output),
        "generated_twi_answers": 0, "changed_source_answers": 0, "human_reviewed": 0,
        "train_eligible": 0, "distinct_approved_english_questions": len(approved_questions),
        "reviewer_controls": {"total": len(controls), "correct": sum(r["correct"] for r in controls),
                              "passed": all(r["correct"] for r in controls), "results": controls},
        "limitations": ["English model agreement is not Twi translation verification.",
            "Two questions are alternative views of one source, not two independent answers.",
            "Sources retain upstream research restrictions and unverified translation quality.",
            "Single-turn source questions do not establish broad conversation or medical competence."]}
    save("review.json", output, exclusive=True)
    save("review.summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "status", "export"))
    parser.add_argument("--stage", choices=("generate", "review", "ground"), default="generate")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--wait-seconds", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.limit <= 384 or not 1 <= args.wait_seconds <= 1200:
        parser.error("Bound exceeded")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        rows, manifest = prepare_sources(args.limit)
        manifest.update(sources_sha256=digest(rows), run_id="native_answers_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        save("sources.json", rows, exclusive=True)
        save("manifest.json", manifest, exclusive=True)
        print(json.dumps(manifest, indent=2))
    elif args.action == "export":
        export_grounding() if args.stage == "ground" else export_review()
    else:
        import modal
        name = args.stage + ".receipt.json"
        if args.action == "submit":
            if (OUT / name).exists():
                raise ValueError("Existing submission; inspect receipt rather than duplicate")
            rows = stage_inputs(args.stage)
            validate_inputs(rows, args.stage)
            preflight = modal.Function.from_name(APP, "preflight").remote()
            run_id = load("manifest.json")["run_id"]
            receipt = {"state": "submitting", "run_id": run_id, "input_sha256": digest(rows),
                "model": preflight[args.stage]["model"], "revision": preflight[args.stage]["revision"],
                "rows": len(rows), "gpu": "H100", "timeout_seconds": 1200}
            save(args.stage + ".inputs.json", rows, exclusive=True)
            save(name, receipt, exclusive=True)
            call = modal.Function.from_name(APP, "synthesize").spawn(run_id, args.stage, rows)
            receipt.update(state="submitted", call_id=call.object_id)
            save(name, receipt)
            print(json.dumps(receipt, indent=2))
        else:
            receipt = load(name)
            if receipt["state"] == "completed":
                completed(args.stage)
                print(json.dumps(receipt, indent=2))
                return
            try:
                report = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=args.wait_seconds)
            except TimeoutError:
                print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
                return
            except Exception as exc:
                receipt.update(state="failed", error_type=type(exc).__name__)
                save(name, receipt)
                raise
            if any(report[k] != receipt[k] for k in ("run_id", "model", "revision", "input_sha256")):
                raise ValueError("Unexpected research result identity")
            save(args.stage + ".results.json", report, exclusive=True)
            receipt.update(state="completed", results_sha256=digest(report))
            save(name, receipt)
            completed(args.stage)
            print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
