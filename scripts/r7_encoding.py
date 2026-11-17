"""R7 native text/tool targets; no synthetic control tokens or truncated history."""
import math
import re
import unicodedata

from check_afrique_training_contract import EOS, encode_complete as encode_text, require

WITNESS_VERSION = "r7-source-witness-v1"
CONTROL = re.compile(r"<\||</?(?:think|tool_call|tool_response|tools|function|parameter)(?=[\s=>/])")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def source_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), "Non-string source key")
            yield key
            yield from source_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_strings(child)
    else:
        require(value is None or type(value) in (bool, int, float), "Non-JSON source value")
        require(type(value) is not float or math.isfinite(value), "Nonfinite source value")


def validate_history(row, tokenizer=None):
    """Validate the ID-bearing source before the native template omits IDs."""
    messages, tools = row.get("messages"), row.get("tools", [])
    require(isinstance(messages, list) and messages and all(isinstance(m, dict) for m in messages),
            "Missing/invalid messages")
    require(isinstance(tools, list), "Tools must be a schema list")
    for text in source_strings([messages, tools]):
        require(not CONTROL.search(text), "Source contains model control tokens")
        require(not any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
                        (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text),
                "Invalid Unicode in source")
        if tokenizer is not None:
            special = set(tokenizer.all_special_ids) | set(tokenizer.contract_special_ids)
            require(not special.intersection(tokenizer.encode(text, add_special_tokens=False)),
                    "Reserved source token")
    schemas = {}
    for tool in tools:
        require(isinstance(tool, dict) and set(tool) == {"type", "function"}
                and tool["type"] == "function", "Invalid tool schema")
        function = tool["function"]
        require(isinstance(function, dict) and {"name", "parameters"} <= set(function)
                and set(function) <= {"name", "description", "parameters", "strict"}, "Invalid tool function")
        name = function["name"]
        require(isinstance(name, str) and NAME.fullmatch(name) and name not in schemas,
                "Duplicate/unsafe tool name")
        parameters = function["parameters"]
        require(isinstance(parameters, dict) and parameters.get("type") == "object"
                and isinstance(parameters.get("properties", {}), dict), "Tool parameters must be an object schema")
        required = parameters.get("required", [])
        require(isinstance(required, list) and all(isinstance(k, str) for k in required)
                and len(set(required)) == len(required) and set(required) <= set(parameters.get("properties", {})),
                "Invalid required tool parameters")
        require(all(NAME.fullmatch(k) for k in parameters.get("properties", {})), "Unsafe parameter name")
        require("description" not in function or isinstance(function["description"], str), "Invalid tool description")
        require("strict" not in function or type(function["strict"]) is bool, "Invalid tool strict flag")
        schemas[name] = parameters
    expected, pending, used = "user", [], set()
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role == "system":
            require(index == 0 and set(message) == {"role", "content"}
                    and isinstance(content, str) and content.strip(), "Invalid system message")
            continue
        require(role == expected, "Invalid history roles or unresolved tool calls")
        if role == "user":
            require(set(message) == {"role", "content"} and isinstance(content, str) and content.strip(),
                    "Invalid user message")
            expected = "assistant"
        elif role == "assistant":
            require({"role", "content"} <= set(message) <= {"role", "content", "tool_calls"}
                    and isinstance(content, str), "Invalid assistant message")
            calls = message.get("tool_calls", [])
            require(isinstance(calls, list) and (bool(calls) or content.strip()), "Empty assistant target")
            require("tool_calls" not in message or calls, "Empty tool_calls field")
            for call in calls:
                require(isinstance(call, dict) and set(call) == {"id", "type", "function"}
                        and call["type"] == "function", "Invalid tool call")
                identity, function = call["id"], call["function"]
                require(isinstance(identity, str) and CALL_ID.fullmatch(identity) and identity not in used,
                        "Duplicate/invalid tool call ID")
                require(isinstance(function, dict) and set(function) == {"name", "arguments"}
                        and isinstance(function["name"], str) and function["name"] in schemas,
                        "Unknown/invalid called function")
                arguments, schema = function["arguments"], schemas[function["name"]]
                require(isinstance(arguments, dict) and all(NAME.fullmatch(k) for k in arguments),
                        "Tool arguments must be a structured object with safe names")
                require(set(schema.get("required", [])) <= set(arguments), "Missing required tool arguments")
                require(schema.get("additionalProperties") is not False or
                        set(arguments) <= set(schema.get("properties", {})), "Unknown tool argument")
                pending.append((identity, function["name"]))
                used.add(identity)
            expected = "tool" if calls else "user"
        elif role == "tool":
            require({"role", "content", "tool_call_id"} <= set(message)
                    <= {"role", "content", "tool_call_id", "name"}
                    and isinstance(content, str) and content.strip(), "Invalid tool response")
            identity, name = pending.pop(0)
            require(message["tool_call_id"] == identity and message.get("name", name) == name,
                    "Mismatched/out-of-order tool response ID or name")
            expected = "tool" if pending else "assistant"
    require(messages[-1]["role"] == "assistant", "Missing final assistant target")
    return bool(tools)


def encode_complete(tokenizer, row, max_length):
    """Preserve native schemas/history and supervise only the final assistant turn."""
    require(type(max_length) is int and 0 < max_length <= tokenizer.model_max_length, "Invalid context length")
    has_tools = validate_history(row, tokenizer)
    if not has_tools:
        return encode_text(tokenizer, row, max_length)
    from afrique_response_core import encode
    encoded = encode(tokenizer, row, max_length=max_length)
    require(encoded is not None, "overlength_no_truncation: " + row.get("id", "probe"))
    options = dict(tools=row["tools"], tokenize=False, enable_thinking=False)
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options).rstrip("\n")
    require(rendered.startswith(prefix), "Native assistant boundary changed")
    ids, labels = encoded["input_ids"], encoded["labels"]
    target = [t for t in labels if t != -100]
    require(tokenizer.encode(rendered, add_special_tokens=False) == ids, "Token boundary mismatch")
    require(len(target) > 1 and target[-1] == EOS and target.count(EOS) == 1, "Invalid native target EOS")
    prompt_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    require(prompt_length > 0 and labels == [-100] * prompt_length + ids[prompt_length:]
            and encoded["attention_mask"] == [1] * len(ids), "Invalid final-assistant mask")
    require(tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            == rendered[len(prefix):], "Native tool target changed")
    return encoded
