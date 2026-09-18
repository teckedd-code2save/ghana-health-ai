"""Build an auditable direct-response mix; original source rows stay unchanged."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import jsonschema
import pyarrow.parquet as pq

from prepare_response_sources import FOLDER, ROOT, SOURCES

SYSTEM = "You are a helpful assistant. Answer the user's question directly in their language, unless they request another language. Ask for missing information instead of inventing it. Use available tools when needed; do not claim an action succeeded without its result."
READ_ONLY = re.compile(r"^(?:get|search|find|list|fetch|retrieve|calculate|check|lookup|convert|count|describe|query)_")
SPECIAL = re.compile(r"<\||<turn\|>|<channel\|>|<tool_call\|>|<tool_response\|>")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalized(text):
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def complete(text):
    return bool(re.search(r'[.!?][\s"\u2019\u201d)]*$', text))


def issues(question, answer):
    reasons = []
    if not 8 <= len(question) <= 600 or not 15 <= len(answer) <= 1800:
        reasons.append("length_outside_experiment_bounds")
    if not complete(answer):
        reasons.append("ending_requires_review")
    if SPECIAL.search(question + answer) or "\ufffd" in question + answer:
        reasons.append("source_control_or_encoding")
    words = normalized(answer).split()
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    if grams and 1 - len(set(grams)) / len(grams) > 0.15:
        reasons.append("repetitive_answer")
    if normalized(question) == normalized(answer):
        reasons.append("copied_question")
    if re.search(r"https?://|www\.|\b(?:911|999)\b", answer):
        reasons.append("unverified_contact_or_link")
    return reasons


def parameter_type(name):
    primitives = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
                  "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean"}
    if name in primitives:
        return {"type": primitives[name]}
    array = re.fullmatch(r"(?:List|list)\[(\w+)\]", name)
    if array:
        return {"type": "array", "items": parameter_type(array[1])}
    raise ValueError("Unsupported parameter type")


def tool_example(row):
    raw_tools, answers = json.loads(row["tools"]), json.loads(row["answers"])
    if not isinstance(raw_tools, list) or not isinstance(answers, list) or not 1 <= len(answers) <= 3:
        raise ValueError("Not a bounded call example")
    tools = []
    for raw in raw_tools:
        if not READ_ONLY.match(raw["name"]):
            raise ValueError("Non-read-only tool is outside this experiment")
        properties, required = {}, []
        for name, parameter in raw.get("parameters", {}).items():
            schema = parameter_type(parameter["type"])
            schema["description"] = parameter.get("description", "")
            properties[name] = schema
            if "default" not in parameter:
                required.append(name)
        parameters = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        jsonschema.Draft202012Validator.check_schema(parameters)
        tools.append({"type": "function", "function": {"name": raw["name"],
                      "description": raw.get("description", ""), "parameters": parameters}})
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    if len(definitions) != len(tools):
        raise ValueError("Duplicate function declaration")
    calls = []
    for index, answer in enumerate(answers):
        definition = definitions[answer["name"]]
        jsonschema.validate(answer["arguments"], definition["parameters"])
        calls.append({"id": f"call_{index}", "type": "function",
                      "function": {"name": answer["name"], "arguments": answer["arguments"]}})
    if SPECIAL.search(json.dumps([row["query"], tools, calls], ensure_ascii=False)):
        raise ValueError("Source control tokens")
    return tools, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["query"]},
                   {"role": "assistant", "content": "", "tool_calls": calls}]


def source_frame(name, filename):
    source = SOURCES[name]
    metadata = json.loads((FOLDER / "sources.json").read_text())["sources"][name]
    artifact = next(item for item in metadata["files"] if item["file"] == filename)
    path = FOLDER / "sources" / name / filename
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != artifact["sha256"]:
            raise ValueError("Downloaded source changed")
    return pq.read_table(path).to_pylist(), source


def main():
    selected, excluded, seen, seen_questions = [], [], set(), set()
    counts = Counter()
    exclusions_path = ROOT / "data/response-adaptation/source-exclusions.v2.json"
    exclusions = json.loads(exclusions_path.read_text())
    flagged = {row["source_id"]: row["reason"] for row in exclusions["rows"]}

    def add(source, source_id, messages, task, language, *, group=None, split=None, evidence=None, tools=None):
        if source_id in flagged:
            excluded.append({"source_id": source_id, "reason": "agent_spot_check", "detail": flagged[source_id]})
            return
        if task in ("health_response", "general_source_response"):
            question_key = language + ":" + normalized(messages[-2]["content"])
            if question_key in seen_questions:
                excluded.append({"source_id": source_id, "reason": "duplicate_normalized_question"})
                return
            seen_questions.add(question_key)
        identity = digest([source["repo"], source["revision"], source_id])
        content = digest(messages)
        if content in seen:
            excluded.append({"source_id": source_id, "reason": "duplicate_messages"})
            return
        seen.add(content)
        group = group or identity
        partition = split or ("validation" if int(digest(group)[:8], 16) % 20 == 0 else "train")
        record = {"id": identity + ":" + language, "group": group, "source_record_hash": identity,
                  "source": source["repo"], "revision": source["revision"], "source_id": source_id,
                  "license": source["license"], "attribution": source["attribution"],
                  "language": language, "task": task, "split": partition, "messages": messages,
                  "human_validated": False, "production_eligible": False}
        if evidence:
            record["evidence"] = evidence
        if tools:
            record["tools"] = tools
        selected.append(record)
        counts[partition + ":" + task + ":" + language] += 1

    # A bounded, source-faithful health subset. Do not repair source answers by
    # silently inventing endings or copy the old 95-source pilot's annotations.
    health_path = ROOT / "data/medical-response-corpus/afrihealth-ghana-response-train.v1.jsonl"
    health = [json.loads(line) for line in health_path.read_text().splitlines() if line.strip()]
    locked_pilot = ROOT / "tmp/medical-response-pilot/v1/holdout.jsonl"
    locked_ids = {json.loads(line)["provenance"]["source_record_id"] for line in locked_pilot.read_text().splitlines() if line.strip()}
    random.Random(42).shuffle(health)
    health_used = 0
    for row in health:
        if row["language"] != "tw":
            continue
        if row["source_record_id"] in locked_ids:
            excluded.append({"source_id": row["source_record_id"], "reason": "locked_pilot_holdout"})
            continue
        reasons = issues(row["question"], row["answer"])
        if row["quality_annotation"]["status"] != "pass":
            reasons.append("prior_source_audit_flag")
        if reasons:
            excluded.append({"source_id": row["source_record_id"], "reasons": reasons})
            continue
        if health_used >= 1000:
            continue
        source = {"repo": row["source_dataset"], "revision": row["source_revision"],
                  "license": row["license"], "attribution": row["attribution"]}
        add(source, row["source_record_id"], [{"role": "system", "content": SYSTEM}, *row["messages"]],
            "health_response", "tw", evidence={"original_record_hash": row["record_source_hash"],
            "source_claimed_quality": row["source_claimed_answer_quality"], "source_answer_unchanged": True})
        health_used += 1

    # Each parallel row joins all assistant turns. Only aligned first paragraphs
    # are eligible; keep the full source pointer and English counterpart for audit.
    contextual = re.compile(r"\b(?:according|this|these|those|above|article|study|researchers|findings|paper)\b", re.I)
    medical = re.compile(r"\b(?:health|patient|disease|treatment|drug|malaria|hiv|pregnan|cancer|medicine|surgery|symptom)", re.I)
    chat_pool = []
    for filename in SOURCES["ghana_chat"]["files"]:
        rows, source = source_frame("ghana_chat", filename)
        for row in rows:
            if row["source_type"] != "research":
                continue
            english = row["text"].split("\n\n")
            twi = row["text_ak"].split("\n\n")
            q, answer = row["generated_question_ak"].strip(), twi[0].strip()
            eq, ea = row["generated_question"].strip(), english[0].strip()
            reasons = issues(q, answer)
            if len(english) != len(twi) or not complete(ea):
                reasons.append("unaligned_or_incomplete_paragraphs")
            if contextual.search(eq + " " + ea) or medical.search(eq + " " + ea):
                reasons.append("context_dependent_or_medical_translation")
            if not re.search("[ɛɔƐƆ]", q + answer) or not 0.55 <= len(answer) / max(1, len(ea)) <= 2.5:
                reasons.append("translation_requires_review")
            if reasons:
                excluded.append({"source_id": row["source"], "reasons": reasons})
                continue
            chat_pool.append((row, q, answer, eq, ea))
    random.Random(43).shuffle(chat_pool)
    for row, q, answer, eq, ea in chat_pool[:1800]:
        group = "ghana_chat:" + row["source"]
        add(SOURCES["ghana_chat"], row["source"], [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": q}, {"role": "assistant", "content": answer}],
            "general_source_response", "tw", group=group,
            evidence={"english_question": eq, "english_answer": ea, "selection": "First aligned complete paragraph, not all joined answers",
                      "source_quality": "Uncalibrated machine translation; not gold"})

    everyday_test, _ = source_frame("everyday", SOURCES["everyday"]["files"][1])
    reserved_topics = {row["full_topic"] for row in everyday_test}
    for filename in SOURCES["everyday"]["files"]:
        rows, source = source_frame("everyday", filename)
        for row in rows:
            messages = row["messages"]
            if not messages or len(messages) < 6 or any(m["role"] not in ("user", "assistant") for m in messages):
                continue
            group = "everyday:" + digest(row["full_topic"])
            # Supervise one substantive later turn, preserving its earlier context.
            target = messages[-1]
            if target["role"] != "assistant" or issues(messages[-2]["content"], target["content"]):
                continue
            add(source, digest(messages), [{"role": "system", "content": SYSTEM}, *messages],
                "general_conversation", "en", group=group,
                split="validation" if row["full_topic"] in reserved_topics or int(digest(group)[:8], 16) % 20 == 0 else "train")

    rows, source = source_frame("apigen", SOURCES["apigen"]["files"][0])
    random.Random(44).shuffle(rows)
    tool_count = 0
    for row in rows:
        if tool_count >= 1400:
            break
        try:
            tools, messages = tool_example(row)
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError, jsonschema.SchemaError):
            continue
        names = sorted(call["function"]["name"] for call in messages[-1]["tool_calls"])
        partitions = {"validation" if int(digest(name)[:8], 16) % 20 == 0 else "train" for name in names}
        if len(partitions) != 1:
            continue
        # Called function names stay disjoint, including multi-call examples.
        group = "apigen:" + digest(names)
        add(source, str(row["id"]), messages, "tool_call", "en", group=group, split=partitions.pop(), tools=tools,
            evidence={"origin": row["origin"], "schema_validation": "passed", "execution_claim": "No execution; call-supervision only"})
        tool_count += 1

    # Translation is explicitly auxiliary, limited to 400 existing training pairs.
    prior = ROOT / "tmp/general-language-corpus/adaptation-v1"
    prior_manifest = json.loads((prior / "manifest.json").read_text())
    prior_sources = {r["repo"]: r for r in prior_manifest["sources"]}
    prior_train = [json.loads(line) for line in (prior / "train.jsonl").read_text().splitlines()]
    for row in prior_train:
        if row["task"] == "general_conversation_replay":
            source = {"repo": row["source"], "revision": row["revision"], "license": "apache-2.0", "attribution": "OpenAssistant contributors"}
            add(source, row["source_record_hash"], row["messages"], "general_conversation_replay", "en", group=row["group"], split="train")
    parallel = [r for r in prior_train if r["task"] == "translation" and r["language"] == "tw"]
    random.Random(45).shuffle(parallel)
    for row in parallel[:400]:
        source = {"repo": row["source"], "revision": row["revision"], "license": prior_sources[row["source"]]["terms"], "attribution": "Ghana NLP Community"}
        add(source, row["source_record_hash"], row["messages"], "auxiliary_translation", "tw", group=row["group"], split="train")

    for row in selected:
        if SPECIAL.search(json.dumps([row["messages"], row.get("tools")], ensure_ascii=False)):
            raise ValueError("Source injected a native model control token")
    output = FOLDER / "corpus"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split in ("train", "validation"):
        subset = [row for row in selected if row["split"] == split]
        random.Random(46).shuffle(subset)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset).encode()
        (output / (split + ".jsonl")).write_bytes(payload)
        artifacts.append({"file": split + ".jsonl", "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(subset)})
    overlap = {row["group"] for row in selected if row["split"] == "train"} & {row["group"] for row in selected if row["split"] == "validation"}
    if overlap:
        raise ValueError("Source-group split leakage")
    manifest = {"experiment": "balanced_direct_response_v2", "base_model": "google/gemma-4-31B-it",
                "base_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475", "ready_for_production": False,
                "human_validated": False, "counts": dict(counts), "unique_sources": len({r["source_record_hash"] for r in selected}),
                "artifacts": artifacts, "source_manifest_sha256": hashlib.sha256((FOLDER / "sources.json").read_bytes()).hexdigest(),
                "spot_check_exclusions_sha256": hashlib.sha256(exclusions_path.read_bytes()).hexdigest(),
                "limitations": ["Research mixture, not a verified gold corpus.", "General Twi answers are source-derived machine translations and require calibration.",
                "AfriHealth consensus is a publisher claim; project clinical review is incomplete.", "Tool targets teach selection and arguments, not executed tool trajectories.",
                "Noncommercial source terms apply. No public or production promotion.", "Original sources and transcripts unchanged; supervised first-paragraph selections are explicit."]}
    (output / "excluded.jsonl").write_text("".join(json.dumps(row) + "\n" for row in excluded))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
