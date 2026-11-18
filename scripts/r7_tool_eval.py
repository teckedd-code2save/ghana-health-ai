"""Bounded native tool evaluation, with CPU replay separate from model inference.

Fixed protected histories test the next call and its result-grounded final answer,
not autonomous end-to-end rollouts. One generated call and one final generation
per case; no repair, retry, search access, or inferred user confirmation. Importing
this module does not load a model or contact Modal.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import r7_tool_world as world

VERSION = "r7-tool-eval-v1"
SOURCE_SHA = "afc368b7de0ce6d1a9550459b0146691dda0c9e9b3011159669f05b34a1cd9c7"
EOS, PAD = 248046, 248044
DECODING = {"do_sample": False, "enable_thinking": False, "max_new_tokens": 256,
            "num_beams": 1, "repetition_penalty": 1.0, "seed": 17, "truncate": False}
PROTOCOL = {"version": VERSION, "mode": "fixed_history_next_tool_then_final",
            "decoding": DECODING, "context_tokens": 4096, "max_calls": 1,
            "max_generations": 2, "generation_seconds": 60, "max_cases": 128,
            "parser": "qwen_native_function_parameters_v1", "guard": "raw",
            "web_evidence": "none", "confirmation": "source_explicit_exact_draft_only",
            "grounding": "result_renderer_exact_else_bound_independent_review"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def read(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [parse_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Linked output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.is_symlink(), "Linked temporary output")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_file(root, name):
    root, name = Path(root), Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe artifact path")
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "Missing/linked artifact: " + str(name))
    return path


def verify_seal(folder, expected_sha, seal="SEALED.json", inventory="files"):
    folder = Path(folder)
    require(re.fullmatch("[a-f0-9]{64}", expected_sha or "")
            and sha(safe_file(folder, "manifest.json")) == expected_sha, "Manifest pin differs")
    require(read(safe_file(folder, seal)) == {"manifest_sha256": expected_sha}, "Seal differs")
    manifest = read(folder / "manifest.json")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked package artifact")
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            == set(manifest[inventory]) | {"manifest.json", seal}, "Unbound/missing package artifact")
    for name, checksum in manifest[inventory].items():
        require(sha(safe_file(folder, name)) == checksum, "Artifact changed: " + name)
    return manifest


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def parse_call(text):
    """Parse the pinned native grammar, not OpenAI JSON or repaired XML.

    Native strings are raw text, arrays/objects/numbers are JSON. This is a
    delimiter grammar, not XML: '<function=name>' is not well-formed XML.
    Natural language before the call is retained; any suffix/multiple call fails.
    """
    require(isinstance(text, str) and len(text.encode()) <= 32768, "Oversized generation")
    require(not re.search(r"<\||</?(?:think|tool_response|tools)(?=[\s>/])", text), "Control markup in output")
    marker = text.find("<tool_call>")
    if marker < 0:
        require(not re.search(r"</?(?:tool_call|function|parameter)(?=[\s=>/])", text), "Malformed native call")
        return None
    prefix, body = text[:marker].strip(), text[marker:].strip()
    require(not re.search(r"</?(?:tool_call|function|parameter)(?=[\s=>/])", prefix), "Malformed prefix")
    match = re.fullmatch(r"<tool_call>\s*<function=([A-Za-z0-9][A-Za-z0-9_.-]{0,63})>\s*(.*?)\s*</function>\s*</tool_call>",
                         body, re.S)
    require(match is not None, "Expected exactly one native function call without suffix")
    name, parameters = match.groups()
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in world.tools()}
    properties = schemas.get(name, {}).get("properties", {})
    args, position = {}, 0
    while position < len(parameters):
        if not parameters[position:].strip():
            break
        item = re.match(r"\s*<parameter=([A-Za-z0-9_]{1,64})>\s*\n?(.*?)\s*</parameter>", parameters[position:], re.S)
        require(item is not None, "Malformed native parameter")
        key, value = item.groups()
        require(key not in args and not re.search(r"</?(?:tool_call|function|parameter)(?=[\s=>/])", value),
                "Duplicate/nested native parameter")
        kind = properties.get(key, {}).get("type", "string")
        args[key] = value.strip() if kind == "string" else parse_json(value.strip())
        position += item.end()
    world._bounded_json(args, world.MAX_ARGS_BYTES)
    return {"name": name, "arguments": args, "content": prefix}


def validity(call):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in world.tools()}
    if call["name"] not in schemas:
        return {"tool_name_valid": False, "arguments_valid": False}
    try:
        world._validate(call["arguments"], schemas[call["name"]])
        valid = True
    except world.ToolError:
        valid = False
    return {"tool_name_valid": True, "arguments_valid": valid}


def validate_plan(plan):
    require(set(plan) == {"version", "tools", "cases"} and plan["version"] == VERSION, "Wrong plan")
    require(plan["tools"] == world.tools(), "Native tools changed")
    require(isinstance(plan["cases"], list) and 1 <= len(plan["cases"]) <= PROTOCOL["max_cases"], "Unbounded plan")
    require(len({r["id"] for r in plan["cases"]}) == len(plan["cases"]), "Duplicate case")
    for case in plan["cases"]:
        require(set(case) == {"id", "messages", "context"}, "Reference answer leaked into model plan")
        require(isinstance(case["id"], str) and 0 < len(case["id"]) <= 300, "Invalid ID")
        messages = case["messages"]
        require(isinstance(messages, list) and 2 <= len(messages) <= 32 and messages[-1]["role"] == "user",
                "Expected bounded user-ending source prefix")
        require(len(json.dumps(messages).encode()) <= 131072, "Oversized source history")
        context = case["context"]
        world._validate(context, world._CONTEXT, "context")
        require(not context.get("evidence_records"), "No web snapshots admitted to this evaluation")
        if context.get("confirmation"):
            previous = [m for m in messages if m["role"] == "tool"]
            require(previous, "Confirmation without preceding draft")
            result = parse_json(previous[-1]["content"])["result"]
            require(result["ok"] and result["data"].get("draft_sha256") == context["confirmation"]["draft_sha256"],
                    "Confirmation not bound to source draft")
    return digest(plan)


def native_prompt(tokenizer, messages, tools):
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    require(ids == tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False,
                                                 add_generation_prompt=True, enable_thinking=False),
            "Native prompt encoding differs")
    require(len(ids) + DECODING["max_new_tokens"] <= PROTOCOL["context_tokens"], "Full history exceeds budget; no truncation")
    return ids


def identity(manifest, bundle_sha, variant):
    require(variant in ("r6_response", "r7_response") and variant in manifest["variants"], "Wrong model variant")
    return {"manifest_sha256": bundle_sha, "variant": variant, "model": manifest["model"],
            "revision": manifest["revision"], "base_weights_sha256": manifest["base_sha256"],
            "adapter_weights_sha256": manifest["variants"][variant]["files"]["adapter_model.safetensors"]}


def verify_generation(result, tokenizer, messages, tools, expected_identity):
    ids = native_prompt(tokenizer, messages, tools)
    require(result["identity"] == expected_identity and result["decoding"] == DECODING, "Generation identity/decoding differs")
    require(result["messages_sha256"] == digest(messages) and result["tools_sha256"] == digest(tools)
            and result["prompt_ids_sha256"] == digest(ids), "Generation prompt differs")
    output = result["output_ids"]
    vocabulary_size = len(tokenizer)
    require(type(output) is list and 0 < len(output) <= DECODING["max_new_tokens"]
            and all(type(i) is int and 0 <= i < vocabulary_size for i in output), "Invalid native output IDs")
    eos = output[-1] == EOS
    require(EOS not in output[:-1], "Early EOS followed by extra tokens")
    require(type(result["timed_out"]) is bool and type(result["complete"]) is bool, "Invalid completion flags")
    require(result["complete"] == (eos and not result["timed_out"]), "EOS/completion mismatch")
    require(result["output"] == tokenizer.decode(output[:-1] if eos else output, skip_special_tokens=False),
            "Output text differs from actual native IDs")
    return result


def transition(case, generations):
    """Replay actual executions; return next history or a terminal case record."""
    record = {"id": case["id"], "input_sha256": digest(case), "generations": deepcopy(generations),
              "execution": None, "call": None, "final_answer": None, "state": "pending"}
    if not generations:
        return deepcopy(case["messages"]), record
    first = generations[0]
    if not first["complete"]:
        record["state"] = "incomplete_call"
        return None, record
    try:
        call = parse_call(first["output"])
    except (ValueError, RecursionError) as error:
        record.update(state="invalid_call", parse_error=str(error))
        return None, record
    if call is None:
        record.update(state="missing_call", final_answer=first["output"])
        return None, record
    record["call"] = call
    execution = world.execute(call["name"], call["arguments"], case["context"])
    record["execution"] = execution
    require(execution["messages"], "Parsed call cannot be recorded")
    actual = deepcopy(execution["messages"])
    actual[0]["content"] = call["content"]
    # IDs are bookkeeping aliases only; the template itself omits them.
    alias = "eval_" + digest([case["id"], execution["call"]])[:24]
    actual[0]["tool_calls"][0]["id"] = alias
    actual[1]["tool_call_id"] = alias
    history = deepcopy(case["messages"]) + actual
    if len(generations) == 1:
        record["state"] = "awaiting_final"
        return history, record
    require(len(generations) == 2, "Extra generation beyond fixed bound")
    final = generations[1]
    record["final_answer"] = final["output"]
    try:
        extra = parse_call(final["output"])
        valid = extra is None and bool(final["output"].strip())
    except (ValueError, RecursionError):
        valid = False
    record["state"] = ("incomplete_final" if not final["complete"] else
                       "answered" if valid else "invalid_final")
    return None, record


def run_case(case, runtime, tools, tokenizer, expected_identity):
    generations = []
    rejected = None
    try:
        for _ in range(PROTOCOL["max_generations"]):
            history, record = transition(case, generations)
            if history is None:
                return record
            result = runtime.generate(history, tools)
            rejected = result
            verify_generation(result, tokenizer, history, tools, expected_identity)
            generations.append(result)
            rejected = None
        return transition(case, generations)[1]
    except Exception as error:
        record = transition(case, generations)[1]
        return {**record, "state": "error", "error": type(error).__name__ + ": " + str(error),
                "rejected_generation": rejected}


def verify_record(record, case, tokenizer, tools, expected_identity):
    require(record["id"] == case["id"] and record["input_sha256"] == digest(case), "Saved case changed")
    generations = record["generations"]
    require(type(generations) is list and len(generations) <= 2, "Generation bound exceeded")
    for i, result in enumerate(generations):
        history, _ = transition(case, generations[:i])
        require(history is not None, "Generation after terminal call")
        verify_generation(result, tokenizer, history, tools, expected_identity)
    if record["state"] == "error":
        require(isinstance(record.get("error"), str) and record["error"], "Missing recorded error")
        replay = transition(case, generations)[1]
        require({k: v for k, v in record.items() if k not in ("state", "error", "rejected_generation")}
                == {k: v for k, v in replay.items() if k != "state"}, "Error lost or changed executed evidence")
    else:
        history, replay = transition(case, generations)
        require(history is None and replay == record, "Actual execution/result replay differs")
    return record


def outcome(execution):
    result = execution["result"]
    if not result["ok"]:
        return {"error": result["error"]["code"]}
    name, data = execution["call"]["name"], deepcopy(result["data"])
    if name == "calculator":
        return {"numerator": data["numerator"], "denominator": data["denominator"]}
    # Request spelling/provenance is assessed separately from executed outcomes.
    for key in ("query", "expression"):
        data.pop(key, None)
    return data


def normalize_answer(text):
    return " ".join(text.split())


def score_case(record, reference, renderer, review=None):
    execution, call = record.get("execution"), record.get("call")
    result = {"id": record["id"], "source_group_id": reference["source_group_id"],
              "scenario_id": reference["scenario_id"], "family": reference["family"],
              "prompt_form_family": reference["prompt_form_family"], "state": record["state"],
              "native_call_valid": call is not None, "tool_name_valid": False, "arguments_valid": False,
              "expected_call_exact": False, "execution_ok": False, "expected_outcome": False,
              "grounded_final": "not_evaluable", "grounding_basis": "no_executed_final"}
    if execution:
        result.update(validity(call))
        expected = reference["execution"]
        result["expected_call_exact"] = execution["call"] == expected["call"]
        result["execution_ok"] = execution["result"]["ok"]
        result["actual_result_sha256"] = execution["result_sha256"]
        same = call["name"] == expected["call"]["name"] and outcome(execution) == outcome(expected)
        # Empty evidence/error returns alone cannot certify the requested action.
        if call["name"] == "web_search" or not expected["result"]["ok"]:
            same = same and result["expected_call_exact"]
        result["expected_outcome"] = same
        if record["state"] == "answered":
            try:
                exact = normalize_answer(record["final_answer"]) == normalize_answer(renderer(execution))
            except ValueError:
                exact = False
            result.update(grounded_final="pass" if exact else "needs_review",
                          grounding_basis="exact_actual_result_renderer" if exact else "unreviewed_paraphrase_or_claim")
    if review is not None:
        require(result["grounded_final"] == "needs_review", "Only unresolved grounding can receive review")
        require(review["id"] == record["id"] and review["record_sha256"] == digest(record)
                and review["result_sha256"] == execution["result_sha256"], "Review not bound to actual output/result")
        require(review["verdict"] in ("pass", "fail", "needs_review") and review.get("independent") is True
                and isinstance(review.get("reviewer"), str) and review["reviewer"].strip()
                and isinstance(review.get("reason"), str) and review["reason"].strip(), "Missing independent review evidence")
        require(review.get("answer_quotes") and all(isinstance(q, str) and q.strip()
                and q in record["final_answer"] for q in review["answer_quotes"]), "Review quotes not from actual answer")
        result.update(grounded_final=review["verdict"], grounding_basis="independent_hash_bound_review")
    result["task_success"] = result["expected_outcome"] and result["grounded_final"] == "pass"
    return result


METRICS = ("native_call_valid", "tool_name_valid", "arguments_valid", "expected_call_exact",
           "execution_ok", "expected_outcome", "task_success")


def summarize(scores, expected):
    def counts(rows):
        return {"cases": len(rows), **{key: sum(bool(r[key]) for r in rows) for key in METRICS},
                "grounded_final": dict(Counter(r["grounded_final"] for r in rows))}
    groups, families = defaultdict(list), defaultdict(list)
    for row in scores:
        groups[row["source_group_id"]].append(row)
        families[row["family"]].append(row)
    return {"expected_cases": expected, "scored_cases": len(scores), "pending_cases": expected - len(scores),
            "totals": counts(scores), "by_source_group": {g: counts(r) for g, r in sorted(groups.items())},
            "by_family": {g: counts(r) for g, r in sorted(families.items())},
            "views_are_independent": False, "grounding_unreviewed_is_not_failure_or_pass": True,
            "limitations": ["English templated synthetic dev; shared prompt forms, not wording-held-out.",
                "Fixed source histories, not autonomous multistep rollout success.",
                "Four simulated products; no real commerce, web search or factual web accuracy.",
                "Exact renderer is a sufficient grounding check, not an exhaustive semantic grader.",
                "No foundation pretraining contamination certification."]}


def compare_reports(baseline, candidate):
    require(baseline["binding"]["identity"]["variant"] == "r6_response"
            and candidate["binding"]["identity"]["variant"] == "r7_response", "Expected R6 then final R7")
    for key in ("evaluation_sha256", "protocol_sha256", "plan_sha256"):
        require(baseline["binding"][key] == candidate["binding"][key], "Unmatched " + key)
    for report in (baseline, candidate):
        require(report["summary"]["pending_cases"] == 0 and report["verified"] is True, "Comparison needs complete replayed evidence")
    left, right = ({r["id"]: r for r in report["scores"]} for report in (baseline, candidate))
    require(left.keys() == right.keys(), "Comparison case inventory differs")
    result = {}
    for key in METRICS:
        gains = sum(not left[i][key] and right[i][key] for i in left)
        losses = sum(left[i][key] and not right[i][key] for i in left)
        result[key] = {"r6": sum(left[i][key] for i in left), "r7": sum(right[i][key] for i in left),
                       "total": len(left), "gains": gains, "losses": losses,
                       "delta_points": 100 * (gains - losses) / len(left)}
    unresolved = sum(left[i]["grounded_final"] == "needs_review" or right[i]["grounded_final"] == "needs_review" for i in left)
    result["task_success"]["provisional_lower_bound"] = unresolved > 0
    return {"metrics": result, "paired_grounding_pending": unresolved,
            "source_groups": len({r["source_group_id"] for r in left.values()}),
            "independent_samples_are_source_groups_not_prompt_views": True}


def evaluate(plan, binding, runtime_factory, progress_path, commit=lambda: None):
    """Checkpoint whole cases; never silently rerun an uncertain interrupted call."""
    validate_plan(plan)
    progress_path = Path(progress_path)
    require(not progress_path.exists(), "Progress exists; do not duplicate an uncertain GPU run")
    report = {"version": VERSION, "binding": binding, "results": [], "state": "loading", "inflight": None}

    def checkpoint():
        report["snapshot_sha256"] = digest({k: v for k, v in report.items() if k != "snapshot_sha256"})
        save(progress_path, report)
        commit()

    checkpoint()
    try:
        runtime = runtime_factory()
        require(runtime.identity == binding["identity"], "Loaded model identity differs")
        started = time.monotonic()
        for case in plan["cases"]:
            if time.monotonic() - started > 6900:
                report["state"] = "paused"
                checkpoint()
                return report
            report.update(state="running", inflight=case["id"])
            checkpoint()
            record = run_case(case, runtime, plan["tools"], runtime.tokenizer, binding["identity"])
            report["results"].append(record)
            report["inflight"] = None
            checkpoint()
        report["state"] = "complete"
    except Exception as error:
        report.update(state="error", error=type(error).__name__ + ": " + str(error))
    checkpoint()
    return report


class NativeRuntime:
    """Frozen inference loader with a native tools-aware generation boundary."""
    def __init__(self, bundle, bundle_sha, variant):
        bundle = Path(bundle)
        manifest = verify_seal(bundle, bundle_sha)
        require(manifest["private"] and not manifest["training"] and manifest["decoding"] == DECODING,
                "Wrong private inference bundle")
        module_name = "r7_inference" if manifest["version"] == "r7-comparison-inference-v1" else "r6_inference"
        code = bundle / "code"
        sys.dont_write_bytecode = True
        for path in code.glob("*.py"):
            existing = sys.modules.get(path.stem)
            if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
                del sys.modules[path.stem]
        sys.path.insert(0, str(code))
        module = import_file("_tool_eval_native_" + bundle_sha, code / (module_name + ".py"))
        self.runtime = module.Runtime(bundle, bundle_sha)
        self.variant, self.tokenizer = variant, self.runtime.tokenizer
        self.identity = identity(manifest, bundle_sha, variant)
        require(self.runtime.identity(variant) == self.identity, "Loaded identity differs")

    def generate(self, messages, tools):
        import random
        import numpy as np
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        ids = native_prompt(self.tokenizer, messages, tools)
        with self.runtime.select(self.variant):
            tick = time.monotonic()

            class Stop(StoppingCriteria):
                timed_out = False

                def __call__(self, input_ids, scores, **kwargs):
                    self.timed_out = time.monotonic() - tick >= PROTOCOL["generation_seconds"]
                    return torch.full((input_ids.shape[0],), self.timed_out, dtype=torch.bool, device=input_ids.device)

            timer = Stop()
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
            config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                max_new_tokens=256, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=None,
                use_cache=True, return_dict_in_generate=True)
            with torch.inference_mode():
                output = self.runtime.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    generation_config=config, stopping_criteria=StoppingCriteriaList([timer]))
            torch.cuda.synchronize()
            generated = output.sequences[0, len(ids):].tolist()
        eos = bool(generated) and generated[-1] == EOS
        return {"identity": self.identity, "messages_sha256": digest(messages), "tools_sha256": digest(tools),
                "prompt_ids_sha256": digest(ids), "decoding": DECODING, "output_ids": generated,
                "output": self.tokenizer.decode(generated[:-1] if eos else generated, skip_special_tokens=False),
                "complete": eos and not timer.timed_out, "timed_out": timer.timed_out,
                "seconds": time.monotonic() - tick}
