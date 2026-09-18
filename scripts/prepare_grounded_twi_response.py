"""Pinned human-translated Twi QA, with source spans and split exclusions retained."""
import hashlib
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from prepare_tool_context_repair import ROOT, load_module

REVISION = "5c29933753692c28773c34ac1f68482007f1d2bb"
BASE_URL = f"https://raw.githubusercontent.com/masakhane-io/afriqa/{REVISION}/data"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def valid_span(row):
    answer = row.get("answer_pivot")
    return (isinstance(row.get("context"), str) and isinstance(answer, dict)
            and isinstance(answer.get("text"), list) and isinstance(answer.get("answer_start"), list)
            and len(answer["text"]) == len(answer["answer_start"]) > 0
            and all(isinstance(text, str) and text and type(start) is int and start >= 0 and row["context"][start:start + len(text)] == text
                    for start, text in zip(answer["answer_start"], answer["text"], strict=True)))


def main():
    destination = ROOT / "tmp/semantic-response-v4/corpus"
    raw_folder = ROOT / "tmp/semantic-response-v4/sources"
    if (destination / "manifest.json").exists():
        raise ValueError("Completed corpus already exists; do not replace it")
    destination.mkdir(parents=True, exist_ok=True)
    raw_folder.mkdir(parents=True, exist_ok=True)
    data, inputs = {}, []
    for split in ("train", "dev", "test"):
        data[split] = {}
        for kind, prefix in (("queries", "queries"), ("gold_passages", "gold_span_passages")):
            name = f"{prefix}.afriqa.twi.en.{split}.json"
            url = f"{BASE_URL}/{kind}/twi/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            path = raw_folder / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError("Pinned download differs from saved source")
            if not path.exists():
                path.write_bytes(content)
            rows = [json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
            if len({str(row["id"]) for row in rows}) != len(rows):
                raise ValueError("Duplicate source IDs")
            data[split][kind] = rows
            inputs.append({"url": url, "rows": len(rows), "sha256": hashlib.sha256(content).hexdigest()})
    parent = ROOT / "tmp/afrique-tool-context-v4/corpus"
    core = load_module("response_core", ROOT / "modal/train/response_adaptation_core.py")
    manifest, splits = core.load_inputs(parent, "afrique_tool_context_repair_v4")
    reserved_titles = {normalized(row["title"]) for split in ("dev", "test") for row in data[split]["gold_passages"] if isinstance(row.get("title"), str)}
    reserved_questions = {normalized(row["question"]) for split in ("dev", "test") for row in data[split]["queries"]}
    test_titles = {normalized(row["title"]) for row in data["test"]["gold_passages"] if isinstance(row.get("title"), str)}
    excluded, added, seen = [], Counter(), set()
    for split in ("train", "dev"):
        queries = {str(row["id"]): row for row in data[split]["queries"]}
        for row in data[split]["gold_passages"]:
            source_id = f"{split}:{row['id']}"
            query = queries[str(row["id"])]
            reason = None
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("title", "context", "question_lang", "question_translated", "answer_lang")):
                reason = "missing_grounded_source_field"
            elif query["translation_type"] != "human_translation" or query["question"] != row["question_lang"] or query["translated_question"] != row["question_translated"]:
                reason = "source_alignment"
            elif not valid_span(row):
                reason = "pivot_span_mismatch"
            elif not row["answer_lang"].strip() or not row["question_lang"].strip() or not row["context"].strip():
                reason = "empty_field"
            elif (split == "train" and (normalized(row["title"]) in reserved_titles or normalized(row["question_lang"]) in reserved_questions)) or (split == "dev" and normalized(row["title"]) in test_titles):
                reason = "reserved_source_overlap"
            elif normalized(row["question_lang"]) in seen:
                reason = "duplicate_question"
            if reason:
                excluded.append({"source_id": source_id, "reason": reason})
                continue
            seen.add(normalized(row["question_lang"]))
            partition = "train" if split == "train" else "validation"
            identity = hashlib.sha256((REVISION + ":" + source_id).encode()).hexdigest()
            splits[partition].append({"id": identity + ":tw", "source_record_hash": identity,
                "group": "afriqa:" + hashlib.sha256(normalized(row["title"]).encode()).hexdigest(),
                "source": "masakhane/afriqa", "revision": REVISION, "source_id": source_id,
                "license": "cc-by-sa-4.0", "attribution": "AfriQA authors and Masakhane contributors; original Wikipedia passage contributors",
                "language": "tw", "task": "grounded_question_answer", "split": partition, "human_validated": False,
                "production_eligible": False, "evidence": {"upstream_translation_type": "human_translation", "raw_record": row,
                    "question_english": query["translated_question"], "upstream_answer_english": query["translated_answer"],
                    "source_context_not_current_world_truth": True},
                "messages": [{"role": "system", "content": "Answer the question using only the supplied source passage. Give a brief answer in Twi. Do not add facts or claim the passage is current."},
                    {"role": "user", "content": "Source passage:\n" + row["context"] + "\n\nQuestion:\n" + row["question_lang"]},
                    {"role": "assistant", "content": row["answer_lang"]}]})
            added[partition] += 1
    artifacts, counts = [], Counter()
    for split, rows in splits.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        filename = split + ".jsonl"
        (destination / filename).write_bytes(content)
        artifacts.append({"file": filename, "rows": len(rows), "sha256": hashlib.sha256(content).hexdigest()})
        counts.update(split + ":" + row["task"] + ":" + row["language"] for row in rows)
    manifest = {**manifest, "experiment": "grounded_response_v4", "base_model": "google/gemma-4-31B-it",
        "base_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475", "artifacts": artifacts, "counts": dict(counts),
        "parent_manifest_sha256": hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest(),
        "unique_sources": len({row["source_record_hash"] for rows in splits.values() for row in rows}),
        "source_imports": inputs, "new_source_rows": sum(added.values()), "grounded_source_counts": dict(added),
        "limitations": [
            "Research adaptation of the strongest measured instruction foundation; no production acceptance.",
            "General-source multi-hop Twi translations remain excluded. Source questions and answers are not rewritten.",
            "AfriQA adds source-grounded short answers, not native general conversation or patient advice.",
            "Upstream human_translation labels and exact English answer spans do not certify every native answer.",
            "AfriQA official test is not trained or used for selection; reserved article titles and questions are excluded from training.",
            "Existing health source remains narrow; broad Twi conversational and clinical supervision remains insufficient.",
            "1,000 no-tool training contexts preserve original answers; tool execution itself is not a training target.",
            "Original source restrictions and Wikipedia attribution remain; no redistribution clearance asserted.",
            "Multiple variables change relative to V3; this is not a single-variable causal claim.",
        ]}
    manifest.pop("tokenizer_model", None)
    manifest.pop("tokenizer_revision", None)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "source-exclusions.json").write_text(json.dumps(excluded, indent=2) + "\n")
    print(json.dumps({"added": added, "excluded": Counter(row["reason"] for row in excluded), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
