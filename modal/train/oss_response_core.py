"""Native Harmony regions using the Transformers response parser; no raw fallback."""
TEMPLATE = {"defaults": {"role": "assistant"}, "start_anchor": "<|start|>assistant", "fields": {
    "thinking": {"open": "<|channel|>analysis<|message|>", "close": "<|end|>",
                 "repeats": True, "join": "\n", "content": "text"},
    "content": {"open": "<|channel|>final<|message|>", "close": ["<|return|>", "<|end|>"], "content": "text"},
    "tool_calls": {
        "open_pattern": r"(?:<\|channel\|>commentary\s+)?to=functions\.(?P<name>\w+)(?:<\|channel\|>commentary)?(?:\s+(?:<\|constrain\|>)?json)?\s*<\|message\|>",
        "close": "<|call|>", "repeats": True, "content": "json",
        "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
    },
}}


def parse(tokenizer, prompt, token_ids, tools=None):
    parser = tokenizer.get_response_parser(response_template=TEMPLATE, prefix=prompt, tools=tools)
    parser.feed(tokenizer.decode(token_ids, skip_special_tokens=False))
    value, _ = parser.finalize()
    return value
