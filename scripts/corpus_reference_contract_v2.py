"""Successor contract: reported goals are not requests directed at an assistant.

Not qualified for generation or training. Keep v1 receipts and scoring unchanged.
"""
from corpus_reference_core import Analysis, Evidence, validate


class ReferenceAnalysisV2(Analysis):
    expressed_goal: Evidence | None


INSTRUCTIONS = """Additional field expressed_goal: {label,quote} or null. Extract an explicitly stated desired action, even when the sentence is a statement rather than a request. The label describes that action and the quote includes its actor and goal. Do not infer an actor, invent a goal from a factual event, or convert quoted speech into a request to the assistant. A reported person's goal may be extracted, but conversational_intent remains null for narrative or reported speech. Separate grammatical/discourse form, stated goal, and direct conversational request. For example, a statement of a travel wish is narrative with a travel goal; a report that someone already travelled does not state a goal."""


def validate_v2(reference, raw):
    value = ReferenceAnalysisV2.model_validate(raw).model_dump()
    goal = value.pop("expressed_goal")
    value, flags = validate(reference, value)
    if goal and (not goal["label"].strip() or not goal["quote"].strip() or goal["quote"] not in reference):
        raise ValueError("Goal must have an exact, nonempty source span")
    return {**value, "expressed_goal": goal}, flags
