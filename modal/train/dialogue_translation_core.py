"""Pure contracts for source-preserving dialogue translation calibration."""
import hashlib
import json
import re
from collections import Counter

MODELS = {
    "afrique": ("McGill-NLP/AfriqueQwen3.5-9B-50Langs", "4358dcbc062421751174279da1efe9f88f85e1d5"),
    "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}
METHODS = {"afrique": ("isolated",), "gemma": ("isolated", "context")}
SYSTEM = (
    "Translate the specified English message faithfully into natural Twi (Akan). "
    "You are translating a recorded conversation, not participating in it. "
    "Do not answer questions, repair factual errors, add advice, or continue the dialogue. "
    "Preserve negation, uncertainty, quantities, names, quoted words, code and speaker perspective. "
    "When a task is about English spelling or wording, keep those English terms quoted. "
    "Output only the translation of the target message, preserving its paragraphs."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_sources(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError("Expected 1-32 bounded public source conversations")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source ID")
    for row in rows:
        if set(row) != {"id", "messages"} or not isinstance(row["id"], str):
            raise ValueError("Only public model-visible source inputs")
        messages = row["messages"]
        if len(messages) not in (4, 6):
            raise ValueError("Expected complete multi-turn conversations")
        for i, message in enumerate(messages):
            if (set(message) != {"role", "content"}
                    or message["role"] != ("user" if i % 2 == 0 else "assistant")
                    or not isinstance(message["content"], str)
                    or not message["content"].strip() or len(message["content"]) > 2200):
                raise ValueError("Invalid source message; never silently truncate")


def requests(rows, variant, demonstrations):
    validate_sources(rows)
    if variant not in MODELS:
        raise ValueError("Unknown model")
    if len(demonstrations) != 3 or any(set(d) != {"en", "tw"} or not all(d.values()) for d in demonstrations):
        raise ValueError("Expected three source translation demonstrations")
    output = []
    prefix = "\n\n".join("English: " + d["en"] + "\nTwi: " + d["tw"] for d in demonstrations)
    for method in METHODS[variant]:
        for row in rows:
            for index, message in enumerate(row["messages"]):
                item = {"id": row["id"] + ":" + str(index) + ":" + method,
                        "source_id": row["id"], "turn": index, "method": method}
                if variant == "afrique":
                    item["prompt"] = prefix + "\n\nEnglish: " + message["content"] + "\nTwi:"
                else:
                    payload = {"target_message": message}
                    if method == "context":
                        payload.update(conversation=row["messages"], target_index=index)
                    item["messages"] = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                output.append(item)
    return output


def structural_flags(source, translation, finish_reason):
    flags = []
    if not translation.strip():
        flags.append("empty")
    if finish_reason != "stop":
        flags.append("incomplete_generation")
    number = r"\d+(?:[.,]\d+)*"
    if Counter(re.findall(number, source)) != Counter(re.findall(number, translation)):
        flags.append("numeric_surface_mismatch_review")
    words = translation.casefold().split()
    if any(count >= 3 for count in Counter(tuple(words[i:i+8]) for i in range(len(words)-7)).values()):
        flags.append("repetition")
    if source.strip().casefold() == translation.strip().casefold():
        flags.append("unchanged_english_review")
    if any(marker in translation for marker in ("<|", "<turn|", "[end_of_turn]")):
        flags.append("control_token_leak")
    return flags


def validate_report(report, receipt, rows, demos):
    variant = receipt["variant"]
    if (report["run_id"] != receipt["run_id"] or report["variant"] != variant
            or (report["model"], report["revision"]) != MODELS[variant]
            or report["inputs_sha256"] != digest(rows)
            or report["demonstrations_sha256"] != digest(demos)
            or report["inputs_sha256"] != receipt["inputs_sha256"]
            or report["demonstrations_sha256"] != receipt["demonstrations_sha256"]):
        raise ValueError("Run identity or source inputs changed")
    expected = requests(rows, variant, demos)
    if report["requests_sha256"] != digest(expected):
        raise ValueError("Prompt contract changed")
    indexed = {r["id"]: r for r in expected}
    actual = [r["id"] for r in report["results"]]
    if len(actual) != len(set(actual)) or set(actual) != set(indexed):
        raise ValueError("Incomplete or duplicate outputs")
    for result in report["results"]:
        if any(result[k] != indexed[result["id"]][k] for k in ("source_id", "turn", "method")):
            raise ValueError("Result turn association changed")


def restore_progress(folder, expected):
    """A preempted invocation may re-enter; reuse only matching durable results."""
    for filename in ("results.json", "partial.json", "loading.json"):
        path = folder / filename
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        for key in ("run_id", "variant", "model", "revision", "inputs_sha256", "demonstrations_sha256",
                    "requests_sha256", "runtime", "decoding", "thinking"):
            if saved[key] != expected[key]:
                raise ValueError("Cannot resume changed experiment: " + key)
        if any(digest(saved[key]) != saved[hash_key] for key, hash_key in (
                ("inputs", "inputs_sha256"), ("demonstrations", "demonstrations_sha256"),
                ("requests", "requests_sha256"))):
            raise ValueError("Saved checkpoint payload changed")
        if "started_at_unix" not in saved:
            raise ValueError("Legacy non-resumable run; retain evidence and use an explicit new attempt")
        indexed = {r["id"]: r for r in expected["requests"]}
        ids = [r["id"] for r in saved["results"]]
        if len(ids) != len(set(ids)) or not set(ids) <= set(indexed):
            raise ValueError("Invalid partial output IDs")
        for result in saved["results"]:
            if any(result[k] != indexed[result["id"]][k] for k in ("source_id", "turn", "method")):
                raise ValueError("Invalid partial turn association")
        if filename == "results.json" and set(ids) != set(indexed):
            raise ValueError("Incomplete file marked complete")
        saved["resume_count"] = saved.get("resume_count", 0) + 1
        return saved
    if any(p.name not in ("loading.json.tmp", "partial.json.tmp", "results.json.tmp") for p in folder.iterdir()):
        raise ValueError("Existing artifacts without a recoverable checkpoint")
    return expected
