"""Training contracts shared by CPU preflight and the bounded GPU experiment."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def load_inputs(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["experiment"] != "general_language_adaptation_v1" or manifest["ready_for_production"] is not False:
        raise ValueError("Expected the explicitly experimental language-adaptation corpus")
    outputs = {}
    for split in ("train", "validation"):
        artifact = next(a for a in manifest["artifacts"] if a["file"] == split + ".jsonl")
        data = (folder / artifact["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("Corpus hash mismatch")
        outputs[split] = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(outputs[split]) != artifact["rows"]:
            raise ValueError("Corpus count mismatch")
    for field in ("id", "group", "source_record_hash"):
        if {r[field] for r in outputs["train"]} & {r[field] for r in outputs["validation"]}:
            raise ValueError("Evaluation leakage: " + field)
    if not all(any(r["language"] == lang for r in outputs["train"]) for lang in ("en", "tw")):
        raise ValueError("Both languages must participate in training")
    return manifest, outputs


def encode(tokenizer, row, max_length=896):
    if row["messages"][-1]["role"] != "assistant":
        raise ValueError("Missing assistant target")
    prefix = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
    ending = tokenizer.encode("<turn|>", add_special_tokens=False)
    if len(ending) != 1 or ending[0] not in tokenizer.all_special_ids:
        raise ValueError("Gemma end-of-turn token not recognized")
    answer = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
    if not answer or any(i in tokenizer.all_special_ids for i in answer):
        raise ValueError("Empty target or source control-token injection")
    target = answer + ending
    if len(prefix) + len(target) > max_length:
        return None
    return {"input_ids": prefix + target, "attention_mask": [1] * (len(prefix) + len(target)),
            "labels": [-100] * len(prefix) + target}


def tokenize_inputs(tokenizer, outputs, max_length=896):
    encoded, accepted, rejected, counts, tokens = {}, {}, [], Counter(), Counter()
    for split, rows in outputs.items():
        encoded[split], accepted[split] = [], []
        for row in rows:
            value = encode(tokenizer, row, max_length)
            if value is None:
                rejected.append({"id": row["id"], "reason": "over_token_limit_no_truncation"})
                continue
            encoded[split].append(value)
            accepted[split].append(row)
            key = split + ":" + row["task"] + ":" + row["language"]
            counts[key] += 1
            tokens[key] += sum(x != -100 for x in value["labels"])
    report = {"examples": dict(counts), "supervised_tokens": dict(tokens), "rejected": rejected,
              "max_length": max_length, "truncated_targets": 0,
              "train_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in accepted["train"]]).encode()).hexdigest()}
    return encoded, accepted, report


def translation_diagnostics(validation, offset=32, count=32):
    """Separate new generation checks from the 32 loss-monitoring source pairs."""
    pairs = {}
    for row in validation:
        if row["task"] != "translation":
            continue
        source = row["source_record_hash"]
        by_language = pairs.setdefault(source, {})
        if row["language"] in by_language:
            raise ValueError("Duplicate translation direction for one source")
        by_language[row["language"]] = row
    eligible = [key for key in sorted(pairs) if set(pairs[key]) == {"en", "tw"}]
    selected = eligible[offset:offset + count]
    if len(selected) != count:
        raise ValueError("Not enough unused paired validation sources")
    return [{"id": row["id"], "source_record_hash": key, "source": row["source"],
             "source_revision": row["revision"], "language": language,
             "category": "heldout_translation", "messages": row["messages"][:-1],
             "reference": row["messages"][-1]["content"]}
            for key in selected for language in ("en", "tw") for row in [pairs[key][language]]]
