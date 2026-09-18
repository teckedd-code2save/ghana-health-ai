"""Source-preserving, bounded V6 instruction mix; no private chat ingestion."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_native_understanding_mix import read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/balanced-afrique-v6/corpus"
EXPERIMENT = "balanced_afrique_response_v6"
PARENT = ROOT / "tmp/semantic-response-v4/corpus"
NATIVE = ROOT / "tmp/native-intent-v3"
PARALLEL = ROOT / "tmp/general-language-corpus/adaptation-v1"
REVIEW = ROOT / "tmp/native-question-translation-v1/review.json"
SELECTION = ROOT / "data/response-adaptation/source-context-selection.v6.json"
FLAGS = ROOT / "data/response-adaptation/source-review-findings.jsonl"
KEYS = ("id", "group", "source_record_hash")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked(rows, salt="v6"):
    return sorted(rows, key=lambda r: hashlib.sha256((salt + r["id"]).encode()).hexdigest())


def keys(rows):
    return {key: {r[key] for r in rows} for key in KEYS}


def overlaps(row, reserved):
    return any(row[k] in reserved[k] for k in KEYS)


def intent_rows(split):
    return [{**r, "group": "injongo:" + r["source_record_hash"]} for r in read(NATIVE, split)]


def context_rows(review, selection):
    questions = {q["id"]: (r, q) for r in review for q in r["translated_question_candidates"]}
    result, sources = [], set()
    for key in selection["selected_questions"]:
        row, question = questions[key]
        source = row["source"]
        if (question["agent_translation_finding"] or source["split"] != "train"
                or source["source_record_hash"] in sources
                or hashlib.sha256(row["answer_tw"].encode()).hexdigest() != row["answer_tw_sha256"]
                or row["answer_tw"] != source["messages"][-1]["content"]):
            raise ValueError("Changed, duplicate or flagged contextual source")
        sources.add(source["source_record_hash"])
        # Context prevents a news-source sentence from becoming an invented current fact.
        messages = [
            {"role": "system", "content": "Answer using only the supplied source statement. Reply in Twi unless asked for English. Do not claim the statement is current or describes you."},
            {"role": "user", "content": "Source statement:\n" + row["answer_en"] + "\n\nQuestion:\n" + question["question_tw"]},
            {"role": "assistant", "content": row["answer_tw"]},
        ]
        common = {**source, "source_id": source["source_record_id"], "license": "Noncommercial attributed research; original Ghana NLP terms retained.",
            "attribution": "Ghana NLP Community", "human_validated": False, "production_eligible": False,
            "evidence": {"question_id": key, "reviewer": selection["reviewer"], "selection_sha256": sha(SELECTION),
                "source_answer_unchanged": True, "quality_tier": "agent_inspected_research_silver",
                "original_answer_tw_sha256": row["answer_tw_sha256"], "original_question_en": question["question_en"]}}
        result.append({**common, "id": key + ":context", "task": "native_source_context", "messages": messages})
        result.append({**common, "id": key + ":followup", "task": "source_language_followup", "language": "en",
            "messages": [*messages, {"role": "user", "content": "Now give that answer in English."},
                         {"role": "assistant", "content": row["answer_en"]}]})
    return result


def validate(splits, reserved):
    for split, rows in splits.items():
        if len(rows) != len({r["id"] for r in rows}):
            raise ValueError("Duplicate IDs in " + split)
        if any(r["messages"][-1]["role"] != "assistant" for r in rows):
            raise ValueError("Missing response target")
    if any(overlaps(row, reserved) for row in splits["train"]):
        raise ValueError("Reserved source leakage")
    if any(overlaps(row, keys(splits["validation"])) for row in splits["train"]):
        raise ValueError("Validation source leakage")


def build():
    selection = json.loads(SELECTION.read_text())
    if sha(REVIEW) != selection["review_sha256"]:
        raise ValueError("Selection refers to another source review")
    if json.loads((NATIVE / "manifest.json").read_text())["prompt_version"] != "schema-v2":
        raise ValueError("Incomplete native intent task contract")
    flagged = {r["source_record_hash"] for r in map(json.loads, FLAGS.read_text().splitlines())}
    blocked_ids = set(selection["additional_source_exclusions"])
    reserved = keys([*read(PARENT, "validation"), *intent_rows("validation"), *intent_rows("test"), *read(PARALLEL, "validation")])
    exclusions = []
    train = []
    parent = read(PARENT, "train")
    limits = {"health_response": 180, "tool_call": 600}
    for task in sorted({r["task"] for r in parent}):
        candidates = []
        for row in ranked([r for r in parent if r["task"] == task]):
            if task in ("auxiliary_translation", "general_source_response"):
                exclusions.append({"id": row["id"], "reason": "Translation rebuilt from source pairs; rejected response family remains excluded"})
            elif row["source_record_hash"] in flagged or overlaps(row, reserved):
                exclusions.append({"id": row["id"], "reason": "Flagged or reserved source"})
            else:
                candidates.append(row)
        maximum = limits.get(task, len(candidates))
        train.extend(candidates[:maximum])
        exclusions.extend({"id": r["id"], "reason": "Bounded task weighting"} for r in candidates[maximum:])
    native = intent_rows("train")
    # Keep every intent represented without letting JSON annotation replace conversation.
    for language, maximum in (("tw", 40), ("en", 20)):
        for intent in sorted({r["source_evidence"]["intent"] for r in native}):
            pool = ranked([r for r in native if r["language"] == language and r["source_evidence"]["intent"] == intent])
            train.extend(pool[:maximum])
            exclusions.extend({"id": r["id"], "reason": "Bounded annotation weighting"} for r in pool[maximum:])
    contextual = context_rows(json.loads(REVIEW.read_text()), selection)
    train.extend(contextual)
    context_hashes = {r["source_record_hash"] for r in contextual}
    translations = [r for r in read(PARALLEL, "train") if r["task"] == "translation"
                    and r["source"] == "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT"
                    and r["source_record_hash"] not in flagged | context_hashes
                    and r["source_record_id"] not in blocked_ids and not overlaps(r, reserved)]
    pairs = {}
    for row in translations:
        pairs.setdefault(row["source_record_hash"], {})[row["language"]] = row
    chosen = ranked([v["tw"] for v in pairs.values() if set(v) == {"tw", "en"}], "v6-translation")[:1200]
    if len(chosen) != 1200:
        raise ValueError("Insufficient source-disjoint bilingual pairs")
    for chosen_row in chosen:
        for row in pairs[chosen_row["source_record_hash"]].values():
            train.append({**row, "task": "auxiliary_translation", "human_validated": False, "production_eligible": False})
    validation = [r for r in read(PARENT, "validation") if r["source_record_hash"] not in flagged]
    validation.extend(intent_rows("validation"))
    # Source-group-disjoint original conversation replay adds an English retention check.
    validation.extend(r for r in read(PARALLEL, "validation") if r["task"] == "general_conversation_replay")
    splits = {"train": ranked(train, "v6-final"), "validation": validation}
    validate(splits, reserved)
    return splits, exclusions


def main():
    splits, exclusions = build()
    OUT.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for split, rows in splits.items():
        path = OUT / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        artifacts.append({"file": path.name, "rows": len(rows), "sha256": sha(path)})
    manifest = {"experiment": EXPERIMENT, "base_model": "McGill-NLP/AfriqueQwen3.5-9B-50Langs",
        "base_revision": "4358dcbc062421751174279da1efe9f88f85e1d5",
        "tokenizer_model": "Qwen/Qwen3.5-4B", "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer_policy": "Existing pinned Qwen chat template; assert full vocabulary equality with the 9B foundation before training.",
        "ready_for_production": False, "human_validated": False, "artifacts": artifacts,
        "counts": dict(Counter(f"{s}:{r['task']}:{r['language']}" for s, rows in splits.items() for r in rows)),
        "unique_train_sources": len({r["source_record_hash"] for r in splits["train"]}),
        "unique_train_groups": len({r["group"] for r in splits["train"]}),
        "multiturn_train_views": sum(sum(m["role"] == "user" for m in r["messages"]) > 1 for r in splits["train"]),
        "parents": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in
            (PARENT / "manifest.json", NATIVE / "manifest.json", PARALLEL / "manifest.json", REVIEW, SELECTION, FLAGS)],
        "limitations": ["Private research candidate, not a clinically verified or commercially cleared assistant.",
            "Original source answers are unchanged; known source flags and rejected multi-hop response family excluded.",
            "Only 17 selected native contextual QA sources; their English follow-ups are task views, not new conversations.",
            "Broad free-form Twi multi-turn supervision remains scarce; intent JSON and translation are auxiliary tasks.",
            "Health weighting reduced; no claim that model size or lower loss repairs semantic/clinical failures.",
            "Retained original no-tool contexts; tool-call targets do not establish executed-tool success.",
            "Original source licenses and attribution remain, including noncommercial and share-alike terms.",
            "Compared with V3 and V5, foundation and mixture change together; not a single-variable ablation."]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "exclusions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in exclusions))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
