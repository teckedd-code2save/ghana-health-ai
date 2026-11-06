"""Token-weighted microbatches and objective-specific native adaptation reports."""
from collections import Counter
import math
import inspect


def require_kernel_binding(wrapped, implementation):
    """Verify the installed 5.17 fallback wrapper selected the pinned FLA function."""
    values = inspect.getclosurevars(wrapped).nonlocals
    if values.get("implementation") is not implementation or values.get("is_new_implementation") is not True:
        raise ValueError("Native wrapper did not select the required optimized kernel")
    return implementation.__module__ + "." + implementation.__name__


def target_count(row):
    return sum(t != -100 for t in row["labels"][1:])


def backward(model, rows, collator, device, microbatch):
    import torch
    total = sum(target_count(r) for r in rows)
    if total <= 0 or microbatch < 1:
        raise ValueError("Empty gradient window or invalid microbatch")
    ordered = sorted(rows, key=lambda r: len(r["input_ids"]))
    loss = 0.0
    for start in range(0, len(ordered), microbatch):
        batch = ordered[start:start + microbatch]
        tensors = {k: v.to(device) for k, v in collator(batch).items()}
        value = model(**tensors, use_cache=False).loss
        if not torch.isfinite(value):
            raise ValueError("Nonfinite training loss")
        weighted = value * (sum(target_count(r) for r in batch) / total)
        weighted.backward()
        loss += float(weighted.detach())
    return loss, total


def outcome(before, after, base_generations, adapted_generations, completed, expected_examples):
    from stage1_run_core import flags
    if set(before) != {"twi_text", "english_retention"} or set(after) != set(before):
        raise ValueError("Incomplete objective evaluation")
    for bucket in before:
        if any(before[bucket][k] != after[bucket][k] for k in ("target_tokens", "examples")):
            raise ValueError("Before/after loss uses different held-out inputs")
        if not all(math.isfinite(r[bucket]["loss"]) and r[bucket]["loss"] > 0 for r in (before, after)):
            raise ValueError("Invalid evaluation loss")
    indexed = [{r["id"]: r for r in rows} for rows in (base_generations, adapted_generations)]
    if not indexed[0] or set(indexed[0]) != set(indexed[1]) or any(len(idx) != len(rows) for idx, rows in zip(indexed, (base_generations, adapted_generations))):
        raise ValueError("Incomplete or duplicate paired generations")
    for key in indexed[0]:
        if indexed[0][key]["messages"] != indexed[1][key]["messages"]:
            raise ValueError("Paired generation prompts differ")
    ratios = {k: after[k]["loss"] / before[k]["loss"] for k in before}
    diagnostics = {}
    for name, values in zip(("base", "adapted"), indexed):
        diagnostics[name] = {flag: [key for key, row in values.items() if flags(row["prediction"])[flag]] for flag in ("empty", "repetition")}
        diagnostics[name]["stops"] = dict(Counter(row["stop_reason"] for row in values.values()))
    advanced = completed == expected_examples and ratios["twi_text"] <= .95 and ratios["english_retention"] <= 1.05
    advanced &= all(not (set(diagnostics["adapted"][k]) - set(diagnostics["base"][k])) for k in ("empty", "repetition"))
    advanced &= not any(row["stop_reason"] == "time_limit" for values in indexed for row in values.values())
    return {"complete_epoch": completed == expected_examples, "objective_loss_ratios": ratios,
            "language_learning_numerical_gate": bool(advanced), "generations": diagnostics,
            "semantic_response_review_required": True, "better_response_model_established": False,
            "production_eligible": False, "note": "Language loss and English retention alone cannot satisfy the response-model goal"}
