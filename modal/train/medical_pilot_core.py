"""CPU-testable contracts for the isolated annotated-response pilot."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class PilotCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(row["input_ids"]) for row in features)
        return {
            key: torch.tensor([
                row[key] + [padding] * (width - len(row[key])) for row in features
            ], dtype=torch.long)
            for key, padding in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))
        }


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = row["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("Training row must end with an assistant target")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    # This base inserts an empty think block only for inference prompts. Train
    # against that exact prefix instead of its different completed-chat template.
    answer = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if not answer or tokenizer.eos_token_id is None:
        raise ValueError("Assistant text and EOS are required")
    full = prompt + answer + [tokenizer.eos_token_id]
    if len(full) > max_length:
        raise ValueError(f"Untruncated target exceeds {max_length}: {row['id']} ({len(full)})")
    if len(full) <= len(prompt):
        raise ValueError("Empty assistant target")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prompt) + full[len(prompt):]}


def parsed_interpretation(text: str) -> dict[str, Any] | None:
    try:
        row = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(row, dict):
        return None
    fields = {"normalized_text", "natural_english", "intent", "entities", "reply", "reply_language"}
    if set(row) != fields or row["reply_language"] != "tw":
        return None
    if any(not isinstance(row[key], str) or not row[key].strip() for key in fields - {"entities"}):
        return None
    intents = {"health_information_request", "health_symptom_report", "health_prevention_question", "health_testing_question", "health_treatment_question", "health_medication_question", "health_service_question", "health_personal_safety_question", "health_emergency_report", "health_followup", "unclear_or_out_of_scope"}
    categories = {"symptoms", "conditions", "medicines", "body_parts", "durations", "quantities", "persons", "locations", "other"}
    entities = row["entities"]
    if row["intent"] not in intents or not isinstance(entities, dict) or set(entities) != categories:
        return None
    if any(not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values) for values in entities.values()):
        return None
    return row


def check_pilot_files(manifest: dict[str, Any], train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> None:
    if manifest.get("experiment") != "afrihealth_teacher_v3_pilot_v1" or manifest.get("ready_for_production") is not False:
        raise ValueError("Not an explicitly uncalibrated pilot")
    if len(train) != manifest["training_examples"] or len(holdout) != manifest["holdout_examples"]:
        raise ValueError("Manifest row count mismatch")
    if not train or not holdout:
        raise ValueError("Training and holdout are both required")
    for key in ("row_id", "group_id", "source_record_hash"):
        if {r["provenance"][key] for r in train} & {r["provenance"][key] for r in holdout}:
            raise ValueError(f"Training/holdout leakage: {key}")
    for expected, rows in (("pilot_train", train), ("pilot_holdout", holdout)):
        if any(r["split"] != expected or r.get("experimental_only") is not True or r.get("eligible_for_research_training") is not False for r in rows):
            raise ValueError("Mixed pilot and accepted-corpus rows")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate examples")
        for row in rows:
            if row["task"] == "interpret_and_reply" and parsed_interpretation(row["messages"][-1]["content"]) is None:
                raise ValueError(f"Invalid interpretation target: {row['id']}")
