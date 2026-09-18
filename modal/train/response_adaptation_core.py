"""Direct-response and native-tool SFT boundaries for the pinned Gemma template."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def load_inputs(folder: Path, experiment="balanced_direct_response_v2"):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["experiment"] != experiment or manifest["ready_for_production"] is not False:
        raise ValueError("Expected the unpromoted response experiment")
    splits = {}
    for split in ("train", "validation"):
        data = (folder / (split + ".jsonl")).read_bytes()
        artifact = next(row for row in manifest["artifacts"] if row["file"] == split + ".jsonl")
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("Input hash mismatch")
        splits[split] = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(splits[split]) != artifact["rows"]:
            raise ValueError("Input count mismatch")
    for key in ("id", "group", "source_record_hash"):
        if {r[key] for r in splits["train"]} & {r[key] for r in splits["validation"]}:
            raise ValueError("Source leakage: " + key)
    return manifest, splits


def encode(tokenizer, row, max_length=1024):
    messages, tools = row["messages"], row.get("tools")
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Missing final assistant target")
    # Only the template may introduce control tokens, never source prose/schema.
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    special = set(tokenizer.all_special_ids)
    for text in strings([messages, tools]):
        if special.intersection(tokenizer.encode(text, add_special_tokens=False)):
            raise ValueError("Source control-token injection")
    options = dict(tools=tools, tokenize=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    # Gemma's generation prefix emits an empty thought channel; a training
    # assistant message does not. Keep the inference prefix while taking the
    # entire native completion (including tool handoff) from the official template.
    bare = prefix.removesuffix("<|channel>thought\n<channel|>")
    if not rendered.startswith(bare):
        raise ValueError("Native template changed its assistant boundary")
    completion = rendered[len(bare):].rstrip("\n")
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target = tokenizer.encode(completion, add_special_tokens=False)
    endings = {tokenizer.convert_tokens_to_ids("<turn|>"), tokenizer.convert_tokens_to_ids("<|tool_response>")}
    if not target or target[-1] not in endings:
        raise ValueError("Missing native turn ending or tool handoff")
    if len(prompt_ids) + len(target) > max_length:
        return None
    return {"input_ids": prompt_ids + target, "attention_mask": [1] * (len(prompt_ids) + len(target)),
            "labels": [-100] * len(prompt_ids) + target}


def tokenize_inputs(tokenizer, splits, max_length=1024):
    encoded, accepted, rejected, counts, supervised = {}, {}, [], Counter(), Counter()
    for split, rows in splits.items():
        encoded[split], accepted[split] = [], []
        for row in rows:
            value = encode(tokenizer, row, max_length)
            if value is None:
                rejected.append({"id": row["id"], "task": row["task"], "reason": "overlength_no_truncation"})
                continue
            encoded[split].append(value)
            accepted[split].append(row)
            key = split + ":" + row["task"] + ":" + row["language"]
            counts[key] += 1
            supervised[key] += sum(token != -100 for token in value["labels"])
    report = {"examples": dict(counts), "supervised_tokens": dict(supervised), "max_length": max_length,
              "truncated_targets": 0, "rejected": rejected,
              "train_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in accepted["train"]]).encode()).hexdigest()}
    return encoded, accepted, report
