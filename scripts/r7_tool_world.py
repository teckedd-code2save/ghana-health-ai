"""Deterministic, CPU-only R7 tool world. No network, files, accounts or payments.

tools("qwen") returns native function schemas; execute(name, args, context=None)
accepts a dict or JSON object string and returns an envelope with result,
result_sha256 and messages containing the actual call and serialized result.
Calls without recordable object arguments have messages=[] rather than invented
arguments; the envelope still retains the actual error and any bounded raw call.

Trusted context (never model arguments):
  evidence_records: up to 32 records with id, source_type="web_snapshot", url,
    title, captured_at (UTC ISO timestamp), text, sha256. The pin is sha256_json
    of the entire record EXCEPT sha256. Supply real, externally captured snapshots
    only. There are no built-in web results. Pins establish byte integrity, not
    source authenticity; the caller must authenticate snapshots before admission.
  confirmation: {"confirmed": true, "draft_sha256": <returned draft hash>}.
    Supply only after explicit user confirmation of that exact draft. Even a
    confirmed draft cannot create a cart, place an order or transfer money.

All results are deterministic for identical inputs. Prices and inventory are
fictional simulation values. No context or catalogue state is ever mutated.
"""

from __future__ import annotations

import ast
import copy
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata
from urllib.parse import urlsplit


CONTRACT_VERSION = "r7-tool-world-v1"
MAX_ARGS_BYTES = 8192
MAX_CONTEXT_BYTES = 1_048_576
MAX_RESULT_BYTES = 131_072


class ToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def canonical_json(value):
    """Canonical JSON used for every pin, result hash and tool response."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def _string(max_length, **extra):
    return {"type": "string", "minLength": 1, "maxLength": max_length, **extra}


_ID = _string(64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HASH = _string(64, pattern=r"^[a-f0-9]{64}$")
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 5}
_SCHEMAS = {
    "calculator": (
        "Exact bounded arithmetic: decimal literals, parentheses, + - * / // % **. "
        "Integer exponents must be between -12 and 12. No code execution.",
        _object({"expression": _string(256)}, ["expression"])),
    "catalogue_search": (
        "Search the fictional simulated catalogue. Prices and stock are not real offers.",
        _object({"query": _string(200), "limit": _LIMIT}, ["query"])),
    "catalogue_compare": (
        "Compare 2-5 distinct IDs from the fictional simulated catalogue, in input order.",
        _object({"product_ids": {"type": "array", "items": _ID,
                                 "minItems": 2, "maxItems": 5, "uniqueItems": True}},
                ["product_ids"])),
    "order_draft": (
        "Draft only, using simulated products. Explicit user confirmation is required "
        "through trusted context. Never creates a cart, purchase or payment.",
        _object({"items": {"type": "array", "minItems": 1, "maxItems": 10,
                           "items": _object({"product_id": _ID,
                                             "quantity": {"type": "integer", "minimum": 1,
                                                          "maximum": 100}},
                                            ["product_id", "quantity"])},
                 "fulfillment": _string(8, enum=["pickup", "delivery"]),
                 "location": _string(200)}, ["items", "fulfillment"])),
    "web_search": (
        "Search only caller-supplied pinned web snapshots, with exact text excerpts. "
        "No live browsing or generated results; unavailable evidence returns an empty list.",
        _object({"query": _string(200), "limit": _LIMIT}, ["query"])),
}

_SNAPSHOT = _object({
    "id": _ID, "source_type": _string(12, enum=["web_snapshot"]),
    "url": _string(2048), "title": _string(200), "captured_at": _string(20),
    "text": _string(16_384), "sha256": _HASH,
}, ["id", "source_type", "url", "title", "captured_at", "text", "sha256"])
_CONTEXT = _object({
    "evidence_records": {"type": "array", "maxItems": 32, "items": _SNAPSHOT},
    "confirmation": _object({"confirmed": {"type": "boolean", "const": True},
                             "draft_sha256": _HASH}, ["confirmed", "draft_sha256"]),
}, [])

# Deliberately fictional stock and prices, unrelated to any production catalogue.
_CATALOGUE = {
    "catalogue_id": "r7-simulated-catalogue-v1", "simulated": True,
    "currency": "GHS", "price_scale": 100,
    "products": [
        {"id": "SIM-RICE-1", "name": "Simulated rice, 1 kg", "unit": "1 kg bag",
         "price_minor": 2500, "stock": 20, "tags": ["rice", "food"]},
        {"id": "SIM-RICE-2", "name": "Simulated rice, 2 kg", "unit": "2 kg bag",
         "price_minor": 4600, "stock": 12, "tags": ["rice", "food"]},
        {"id": "SIM-SOAP-1", "name": "Simulated soap, 100 g", "unit": "100 g bar",
         "price_minor": 700, "stock": 30, "tags": ["soap", "household"]},
        {"id": "SIM-WATER-1", "name": "Simulated water, 1 L", "unit": "1 L bottle",
         "price_minor": 500, "stock": 0, "tags": ["water", "drink"]},
    ],
}


def tools(schema="qwen"):
    """Return fresh Qwen/OpenAI-style function definitions, without shared state."""
    if schema not in ("qwen", "openai"):
        raise ValueError("schema must be 'qwen' or 'openai'")
    return [{"type": "function", "function": {
        "name": name, "description": description, "parameters": copy.deepcopy(parameters),
    }} for name, (description, parameters) in _SCHEMAS.items()]


def _bounded_json(value, max_bytes):
    nodes = 0
    size = 0

    def charge(amount):
        nonlocal size
        size += amount
        if size > max_bytes:
            raise ToolError("size_limit", "JSON exceeds the serialized byte budget.")

    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        if depth > 16 or nodes > 10_000:
            raise ToolError("size_limit", "JSON nesting or item count exceeds the limit.")
        if type(item) is dict:
            if len(item) > 128:
                raise ToolError("size_limit", "JSON object exceeds 128 fields.")
            charge(2 + len(item) + max(0, len(item) - 1))
            for key, child in item.items():
                if type(key) is not str:
                    raise ToolError("invalid_json", "JSON object keys must be strings.")
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            if len(item) > 256:
                raise ToolError("size_limit", "JSON array exceeds 256 items.")
            charge(2 + max(0, len(item) - 1))
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            if len(item) > max_bytes:
                raise ToolError("size_limit", "JSON string exceeds the byte budget.")
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise ToolError("invalid_json", "Strings must be valid Unicode.") from None
            charge(len(canonical_json(item)))
        elif type(item) is int:
            if item.bit_length() > 4096:
                raise ToolError("size_limit", "JSON integer exceeds the limit.")
            charge(len(str(item)))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ToolError("invalid_json", "Non-finite JSON numbers are not accepted.")
            charge(len(canonical_json(item)))
        elif item is not None and type(item) is not bool:
            raise ToolError("invalid_json", "Only plain JSON values are accepted.")
        else:
            charge(len(canonical_json(item)))

    visit(value, 0)


def _validate(value, schema, path="args"):
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str,
                "integer": int, "boolean": bool}[kind]
    if type(value) is not expected:
        raise ToolError("invalid_arguments", f"{path} must be {kind}.")
    if kind == "object":
        if set(value) - set(schema["properties"]):
            raise ToolError("invalid_arguments", f"{path} has unknown fields.")
        if set(schema["required"]) - set(value):
            raise ToolError("invalid_arguments", f"{path} is missing required fields.")
        for key, child in value.items():
            _validate(child, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema["maxItems"]:
            raise ToolError("invalid_arguments", f"{path} has an invalid item count.")
        if schema.get("uniqueItems") and len({canonical_json(v) for v in value}) != len(value):
            raise ToolError("invalid_arguments", f"{path} must contain distinct items.")
        for child in value:
            _validate(child, schema["items"], f"{path}[]")
    elif kind == "string":
        if not value.strip() or not schema["minLength"] <= len(value) <= schema["maxLength"]:
            raise ToolError("invalid_arguments", f"{path} has an invalid string length.")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ToolError("invalid_arguments", f"{path} has an invalid format.")
    elif kind == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
        raise ToolError("invalid_arguments", f"{path} is outside its allowed range.")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError("invalid_arguments", f"{path} is not an allowed option.")
    if "const" in schema and value != schema["const"]:
        raise ToolError("invalid_arguments", f"{path} must be explicitly true.")


def _parse_args(args):
    if type(args) is str:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ToolError("invalid_json", "Duplicate JSON keys are not accepted.")
                result[key] = value
            return result

        def constant(_):
            raise ToolError("invalid_json", "Non-finite JSON numbers are not accepted.")

        try:
            args = json.loads(args, object_pairs_hook=pairs, parse_constant=constant)
        except (ValueError, RecursionError) as exc:
            if isinstance(exc, ToolError):
                raise
            raise ToolError("invalid_json", "Arguments must be a valid JSON object.") from None
    _bounded_json(args, MAX_ARGS_BYTES)
    if type(args) is not dict:
        raise ToolError("invalid_arguments", "Arguments must be a JSON object.")
    return copy.deepcopy(args)


def _calculator(args):
    expression = args["expression"].strip()
    if re.fullmatch(r"[0-9\s.+*/%()-]+", expression) is None:
        raise ToolError("invalid_expression", "Only decimal arithmetic syntax is accepted.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ToolError("invalid_expression", "Invalid arithmetic expression.") from None
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ToolError("size_limit", "Expression exceeds 128 syntax nodes.")

    def evaluate(node, depth=0):
        if depth > 24:
            raise ToolError("size_limit", "Expression nesting exceeds 24 levels.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if len(literal) > 32 or not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", literal):
                raise ToolError("invalid_expression", "Decimal literals must be at most 32 characters.")
            value = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.FloorDiv):
                value = Fraction(left // right)
            elif isinstance(node.op, ast.Mod):
                value = left % right
            elif isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right) > 12:
                    raise ToolError("invalid_expression", "Exponent must be an integer from -12 to 12.")
                value = left ** int(right)
            else:
                raise ToolError("invalid_expression", "Unsupported arithmetic operator.")
        else:
            raise ToolError("invalid_expression", "Unsupported arithmetic syntax.")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 2048:
            raise ToolError("size_limit", "Arithmetic result exceeds 2048 bits.")
        return value

    try:
        result = evaluate(tree.body)
    except ZeroDivisionError:
        raise ToolError("division_by_zero", "Division by zero is undefined.") from None
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        decimal = Decimal(result.numerator) / Decimal(result.denominator)
    return {"expression": expression, "numerator": str(result.numerator),
            "denominator": str(result.denominator), "decimal": str(decimal),
            "decimal_is_exact": Fraction(decimal) == result}


def _tokens(text):
    return set(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def _catalogue_info():
    return {key: _CATALOGUE[key] for key in ("catalogue_id", "simulated", "currency", "price_scale")}


def _product(product_id):
    for product in _CATALOGUE["products"]:
        if product["id"] == product_id:
            return copy.deepcopy(product)
    raise ToolError("product_not_found", "A requested ID is not in the simulated catalogue.")


def _catalogue_search(args):
    query = _tokens(args["query"])
    ranked = []
    for product in _CATALOGUE["products"]:
        score = len(query & _tokens(" ".join([product["id"], product["name"], *product["tags"]])))
        if score:
            ranked.append((-score, -product["stock"], product["id"], product))
    products = [copy.deepcopy(row[-1]) for row in sorted(ranked)[:args.get("limit", 5)]]
    return {**_catalogue_info(), "products": products,
            "empty_reason": None if products else "no_match"}


def _catalogue_compare(args):
    products = [_product(product_id) for product_id in args["product_ids"]]
    prices = [product["price_minor"] for product in products]
    return {**_catalogue_info(), "products": products,
            "price_spread_minor": max(prices) - min(prices),
            "comparison_basis": "listed package prices; package sizes may differ"}


def _order_draft(args, context):
    if args["fulfillment"] == "delivery" and "location" not in args:
        raise ToolError("invalid_arguments", "Delivery drafts require an explicit location.")
    if len({item["product_id"] for item in args["items"]}) != len(args["items"]):
        raise ToolError("invalid_arguments", "Draft product IDs must be distinct.")
    lines = []
    for item in args["items"]:
        product = _product(item["product_id"])
        if item["quantity"] > product["stock"]:
            raise ToolError("insufficient_simulated_stock", "Quantity exceeds simulated stock.")
        lines.append({"product_id": product["id"], "name": product["name"],
                      "unit": product["unit"], "quantity": item["quantity"],
                      "unit_price_minor": product["price_minor"],
                      "subtotal_minor": product["price_minor"] * item["quantity"]})
    draft = {**_catalogue_info(), "catalogue_sha256": sha256_json(_CATALOGUE),
             "items": lines, "fulfillment": args["fulfillment"],
             "location": args.get("location"),
             "total_minor": sum(line["subtotal_minor"] for line in lines),
             "total_basis": "simulated items only; excludes delivery and taxes"}
    draft_sha256 = sha256_json(draft)
    confirmation = context.get("confirmation")
    if confirmation and confirmation["draft_sha256"] != draft_sha256:
        raise ToolError("confirmation_mismatch", "Confirmation does not match this exact draft.")
    return {"draft": draft, "draft_sha256": draft_sha256,
            "requires_confirmation": True, "confirmed": bool(confirmation),
            "status": "confirmed_draft" if confirmation else "awaiting_confirmation",
            "purchases_enabled": False, "side_effects": []}


def _validate_snapshots(records):
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ToolError("invalid_evidence", "Snapshot IDs must be distinct.")
        ids.add(record["id"])
        try:
            url = urlsplit(record["url"])
            valid_url = (url.scheme in ("http", "https") and url.hostname
                         and not url.username and not url.password and not url.fragment)
            _ = url.port
            if not valid_url or re.search(r"\s|[\x00-\x1f\x7f]", record["url"]):
                raise ValueError()
            captured = datetime.strptime(record["captured_at"], "%Y-%m-%dT%H:%M:%SZ")
            if captured.strftime("%Y-%m-%dT%H:%M:%SZ") != record["captured_at"]:
                raise ValueError()
        except ValueError:
            raise ToolError("invalid_evidence", "Snapshot URL or UTC capture timestamp is invalid.") from None
        unsigned = {key: value for key, value in record.items() if key != "sha256"}
        if record["sha256"] != sha256_json(unsigned):
            raise ToolError("evidence_hash_mismatch", "Snapshot does not match its supplied SHA-256 pin.")


def _web_search(args, context):
    records = context.get("evidence_records", [])
    query = _tokens(args["query"])
    ranked = []
    for record in records:
        score = len(query & _tokens(record["title"] + " " + record["text"]))
        if score:
            ranked.append((-score, record["id"], record))
    results = []
    for _, _, record in sorted(ranked)[:args.get("limit", 5)]:
        # Return a contiguous original-text slice, never generated paraphrases.
        match = next((m for m in re.finditer(r"[^\W_]+", record["text"])
                      if _tokens(m.group()) & query), None)
        start = max(0, match.start() - 120) if match else 0
        end = min(len(record["text"]), start + 1200)
        results.append({"id": record["id"], "url": record["url"], "title": record["title"],
                        "captured_at": record["captured_at"], "snapshot_sha256": record["sha256"],
                        "excerpt": record["text"][start:end], "excerpt_start": start,
                        "excerpt_end": end, "truncated": start > 0 or end < len(record["text"])})
    return {"source": "caller_supplied_pinned_snapshots", "live_search": False,
            "results": results, "empty_reason": None if results else (
                "no_match" if records else "no_evidence")}


def execute(name, args, context=None):
    """Execute one bounded call; validation/runtime failures become hashed results.

    The tool response content is canonical_json({result, result_sha256}). Successful
    call arguments are dictionaries, as expected by the repo's Qwen chat template.
    Malformed JSON strings are retained verbatim in call, but messages=[] because
    Qwen requires object arguments. Inputs that cannot safely be recorded also get
    call=None. Both cases retain the actual error result and its SHA-256.
    """
    call = None
    provenance = {"contract_version": CONTRACT_VERSION, "executor": "local_cpu",
                  "network_used": False, "side_effects": [],
                  "schemas_sha256": sha256_json(tools())}
    try:
        _validate(name, _ID, "name")
        _bounded_json(args, MAX_ARGS_BYTES)
        call = {"name": name, "arguments": copy.deepcopy(args)}
        parsed = _parse_args(args)
        call["arguments"] = json.loads(canonical_json(parsed))
        if name not in _SCHEMAS:
            raise ToolError("unknown_tool", "No such tool in this sandbox.")
        _validate(parsed, _SCHEMAS[name][1])
        context = {} if context is None else context
        _bounded_json(context, MAX_CONTEXT_BYTES)
        _validate(context, _CONTEXT, "context")
        _validate_snapshots(context.get("evidence_records", []))
        provenance["context_sha256"] = sha256_json(context)
        if name.startswith("catalogue_") or name == "order_draft":
            provenance.update({"simulated": True, "catalogue_sha256": sha256_json(_CATALOGUE)})
        if name == "web_search":
            provenance["evidence"] = [{"id": row["id"], "sha256": row["sha256"]}
                                      for row in context.get("evidence_records", [])]
        if name == "calculator":
            data = _calculator(parsed)
        elif name == "catalogue_search":
            data = _catalogue_search(parsed)
        elif name == "catalogue_compare":
            data = _catalogue_compare(parsed)
        elif name == "order_draft":
            data = _order_draft(parsed, context)
        else:
            data = _web_search(parsed, context)
        _bounded_json(data, MAX_RESULT_BYTES)
        result = {"ok": True, "data": data, "error": None}
    except ToolError as exc:
        result = {"ok": False, "data": None, "error": {"code": exc.code, "message": str(exc)}}
    provenance["call_sha256"] = sha256_json(call) if call is not None else None
    result["provenance"] = provenance
    digest = sha256_json(result)
    messages = []
    if call is not None and type(call["arguments"]) is dict:
        call_id = "call_" + provenance["call_sha256"][:24]
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": copy.deepcopy(call)}]},
            {"role": "tool", "name": name, "tool_call_id": call_id,
             "content": canonical_json({"result": result, "result_sha256": digest})},
        ]
    return {"call": call, "result": result, "result_sha256": digest, "messages": messages}
