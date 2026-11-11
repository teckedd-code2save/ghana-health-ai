"""Explainable R5 candidate screens. An empty finding list is not semantic approval."""
import re
import unicodedata

from response_loop_guard import repeated_passage

IDENTITY = re.compile(r"\b(?:I (?:am|was)|I'm|we are)\b.{0,100}\b(?:developed|created|trained|built) by (?:OpenAI|Google|Anthropic)\b", re.I | re.S)
CURRENT_REQUEST = re.compile(r"\b(?:today|now|current(?:ly)?|this year(?:['’]?s)?|next year|latest|upcoming|weather|until christmas)\b", re.I)
CURRENT_ASSERTION = re.compile(r"\b(?:today(?:['’]s date)? is|current (?:president|leader|goals)|this year(?:['’]?s)?|to be launched|planned missions)\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
SYMPATHY_ONLY = re.compile(r"^(?:I'm sorry to hear that|I am sorry to hear that|Ɛyɛ me yaw sɛ wote nka saa)[.! ]*$", re.I)


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def inspect_messages(messages):
    findings = []
    if not isinstance(messages, list) or not messages:
        return [{"code": "missing_history", "severity": "invalid"}]
    start = int(messages[0].get("role") == "system")
    if (len(messages) - start) % 2 or len(messages) - start < 2:
        findings.append({"code": "incomplete_history", "severity": "invalid"})
    for i, message in enumerate(messages):
        role = "system" if start and i == 0 else "user" if (i - start) % 2 == 0 else "assistant"
        text = message.get("content")
        if message.get("role") != role or not isinstance(text, str) or not text.strip():
            findings.append({"code": "invalid_role_or_text", "turn": i, "severity": "invalid"})
            continue
        if any(c == "\ufffd" or unicodedata.category(c) in {"Cs", "Co"} or
               (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text):
            findings.append({"code": "invalid_unicode", "turn": i, "severity": "invalid"})
        if role != "assistant":
            continue
        request = messages[i - 1].get("content", "")
        context = "\n".join(m.get("content", "") for m in messages[:i] if m.get("role") != "assistant")
        if IDENTITY.search(text):
            findings.append({"code": "inherited_provider_identity", "turn": i, "severity": "review"})
        if YEAR.search(text) and (CURRENT_REQUEST.search(request) or CURRENT_ASSERTION.search(text)) and not YEAR.search(context):
            findings.append({"code": "unanchored_current_information", "turn": i, "severity": "review"})
        loop = repeated_passage(text)
        if loop:
            findings.append({"code": "repeated_passage", "turn": i, "severity": "review", "trigger": loop})
        if len(normalized(text).split()) >= 5 and normalized(text) == normalized(request):
            findings.append({"code": "request_echo", "turn": i, "severity": "review"})
        if SYMPATHY_ONLY.fullmatch(text):
            findings.append({"code": "sympathy_without_followup", "turn": i, "severity": "review"})
    return findings


def validate_decision(record, decision, digest):
    """Require evidence-bound explicit review, never infer approval from the screens."""
    if decision.get("record_sha256") != digest(record):
        raise ValueError("Reviewed source record changed")
    if decision.get("reviewer_type") != "agent" or decision.get("human_reviewed") is not False:
        raise ValueError("Agent decisions cannot claim human review")
    if not decision.get("reviewer") or not decision.get("reason"):
        raise ValueError("Missing decision accountability")
    if decision.get("decision") not in {"retain", "repair", "exclude"}:
        raise ValueError("Unknown admission decision")
    if decision["decision"] == "exclude":
        return None
    messages = decision.get("messages") if decision["decision"] == "repair" else record["messages"]
    findings = inspect_messages(messages)
    if any(f["severity"] == "invalid" for f in findings):
        raise ValueError("Invalid training history")
    accounted = {(r["code"], r["turn"]) for r in decision.get("screen_resolutions", []) if r.get("reason")}
    if any((f["code"], f["turn"]) not in accounted for f in findings):
        raise ValueError("Unresolved screening findings")
    if decision["decision"] == "repair":
        turns = decision.get("per_turn_review", [])
        if ([t.get("index") for t in turns] != list(range(len(messages)))
                or not all(t.get("note") and t.get("back_translation") and t.get("findings") == [] for t in turns)):
            raise ValueError("Repair needs source-bound per-turn meaning checks")
    return messages
