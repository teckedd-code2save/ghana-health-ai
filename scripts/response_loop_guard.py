"""Conservative repeated-passage stop, separate from model decoding and scoring."""
import re
import unicodedata

VERSION = "repeat-block-v1"
WORDS = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)


def repeated_passage(text):
    normalized = unicodedata.normalize("NFC", text).lower()
    words = [m.group() for m in WORDS.finditer(normalized)]
    for end in range(32, len(words) + 1):
        for width in range(6, min(80, end // 2) + 1):
            copies = 3 if width < 24 else 2
            if end < width * copies:
                continue
            block = words[end-width:end]
            if all(words[end-width*(i+1):end-width*i] == block for i in range(1, copies)):
                return {"block_words": width, "repetitions": copies, "end_word": end}
    return None


def validate_guard(value, output, reason, complete, enabled):
    if not enabled:
        if value is not None:
            raise ValueError("Unexpected repetition policy")
        return
    if not isinstance(value, dict) or set(value) != {"version", "trigger"} or value["version"] != VERSION:
        raise ValueError("Missing or unknown repetition policy")
    trigger = value["trigger"]
    if trigger is not None:
        if trigger != repeated_passage(output) or reason != "cancelled" or complete:
            raise ValueError("Invalid repetition-stop evidence")
