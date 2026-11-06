"""Native Qwen response targets for instruction-tuning the Twi CPT foundation."""
from collections import Counter
import re

CONTROL = re.compile(r"<\||</?(?:think|tool_call|tool_response)>")


def encode(tokenizer, row, max_length=1536):
    messages, tools = row["messages"], row.get("tools")
    if messages[-1]["role"] != "assistant":
        raise ValueError("Missing assistant target")
    special = set(tokenizer.all_special_ids)

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

    if any(CONTROL.search(text) or special.intersection(tokenizer.encode(text, add_special_tokens=False))
           for text in strings([messages, tools])):
        raise ValueError("Source contains model control tokens")
    options = dict(tools=tools, tokenize=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    if not rendered.startswith(prefix):
        raise ValueError("Qwen assistant template boundary changed")
    completion = rendered[len(prefix):].rstrip("\n")
    prompt = tokenizer.encode(prefix, add_special_tokens=False)
    target = tokenizer.encode(completion, add_special_tokens=False)
    if not target or target[-1] != tokenizer.eos_token_id:
        raise ValueError("Target must end at native end-of-turn")
    if len(prompt) + len(target) > max_length:
        return None
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target}


def tokenize(tokenizer, rows):
    encoded, accepted, rejected, counts, tokens = {}, {}, [], Counter(), Counter()
    for split, values in rows.items():
        encoded[split], accepted[split] = [], []
        for row in values:
            result = encode(tokenizer, row)
            if result is None:
                rejected.append({"id": row["id"], "reason": "overlength_no_truncation"})
                continue
            encoded[split].append(result)
            accepted[split].append(row)
            key = split + ":" + row["task"] + ":" + row["language"]
            counts[key] += 1
            tokens[key] += sum(t != -100 for t in result["labels"])
    return encoded, accepted, {"counts": dict(counts), "supervised_tokens": dict(tokens), "rejected": rejected, "truncated_targets": 0}
