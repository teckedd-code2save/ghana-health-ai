"""Strict per-stage export contracts; semantic/source checks run separately."""
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field

Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Text = Annotated[str, Field(min_length=1)]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: Text


class SourceView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    source_id: Text
    source_hash: Hash
    source_file: Text
    source_file_sha256: Hash
    source_row_index: Annotated[int, Field(ge=0)]
    source_revision: Text
    group_id: Hash
    split: Literal["train", "validation"]
    language: Literal["tw", "en"]
    record_type: Literal["sentence", "document", "utterance", "dictionary_entry", "intent_example", "conversation", "grounded_qa"]
    origin: Text
    topic: Text
    license: Any
    human_reviewed: Literal[False]
    production_eligible: Literal[False]
    screening: Literal["automated_source_preserved"]
    semantic_certification: Literal[False]
    id: Text
    tokens: Annotated[int, Field(ge=0)]


class LanguageView(SourceView):
    text: Text


class AlignmentView(SourceView):
    twi: Text
    english: Text
    reference_preserved: Literal[True]


class UnderstandingView(SourceView):
    text: Text
    understanding: dict[str, Any]
    messages: list[Message]
    meaning_english: str | None
    missing_fields: list[str]


class ConversationView(SourceView):
    messages: list[Message]


class GroundedQAView(ConversationView):
    context: Text
    reference_answer: Text
    upstream_human_translation: Literal[True]


class ToolView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: Hash
    group_id: Hash
    split: Literal["train", "validation"]
    language: Literal["en"]
    scenario: Literal["search_and_compare", "clarify_location", "correct_quantity"]
    origin: Literal["synthetic_executable_fixture"]
    simulated: Literal[True]
    real_orders: Literal[False]
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    catalogue_sha256: Hash
    human_reviewed: Literal[False]
    twi_training_eligible: Literal[False]
    screening: Literal["tool_execution_verified"]


SCHEMAS = {"language": LanguageView, "alignment": AlignmentView,
           "understanding": UnderstandingView, "conversation": ConversationView,
           "grounded_qa": GroundedQAView, "tools": ToolView}


def validate(view, row):
    if view.startswith("language/"): kind = "language"
    else:
        kind = {"meaning/alignment": "alignment", "meaning/structured": "understanding",
                "conversation/english_retention": "conversation", "conversation/grounded_twi_qa": "grounded_qa",
                "domain-actions/tools-english-simulated": "tools"}[view]
    SCHEMAS[kind].model_validate(row)
