"""Prepare registered English-only R7 tool candidates, never training admission.

All prose is explicitly templated synthetic material. Catalogue/order facts come
only from the simulated tool world. No web evidence, legacy Twi targets, external
requests, teacher calls, orders or payments are imported or performed.

Scenario groups and train/dev assignments precede prompt-form augmentation.
Execution receipts are shared by their prompt variants, not counted as new sources.
Native message call IDs are occurrence-bound aliases of executor IDs; actual call
arguments and serialized result content are preserved without modification.

API: scenario_specs(), register_scenarios(specs), build_package(forms=FORMS),
prepare(out), verify_package(out). Run with --out <fresh-directory> or --verify.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from r7_encoding import validate_history
import r7_tool_world as world


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/r7-20261117/tool-candidates-v1"
VERSION = "r7-tool-candidates-v1"
DATASET_ID = "r7-executed-tool-english-candidates"
FORMS = ("direct", "question", "context_first")
RENDERER = "result-bound-english-v1"
SPLIT_POLICY = "sha256-r7-tool-split-v1-mod5-dev0"
SYSTEM = ("This is an English synthetic sandbox dialogue. Use actual tool results. "
          "Catalogue prices and stock are simulated. Do not claim a real order, payment, "
          "live search or unavailable evidence.")
PENDING = {"training_ready": False, "training_eligible": False,
           "admission_status": "candidate_pending_twi_qualification",
           "human_reviewed": False, "private_research_only": True,
           "production_eligible": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _step(tool, args=None, *, mode="initial", purpose="checking a sandbox calculation", **extra):
    return {"tool": tool, "mode": mode, "purpose": purpose,
            **({"args": args} if args is not None else {}), **extra}


def _draft(items, fulfillment="pickup", location=None):
    args = {"items": [{"product_id": product, "quantity": quantity} for product, quantity in items],
            "fulfillment": fulfillment}
    if location is not None:
        args["location"] = location
    return args


def scenario_specs():
    """Finite authored semantic plans, not generated claims or translated targets."""
    specs = []

    def add(family, steps):
        specs.append({"family": family, "steps": steps})

    arithmetic = [
        ("0.1 + 0.2", "adding decimal amounts"),
        ("18 + 27 + 35", "summing three counts"),
        ("100 - 37.5", "working out a remaining balance"),
        ("4 * 12.5", "finding the total for equal-cost items"),
        ("96 / 8", "sharing a total equally"),
        ("1 / 3", "keeping a non-terminating fraction exact"),
        ("-7 + 12", "applying a signed adjustment"),
        ("(18 + 6) / 4", "adding before dividing"),
        ("23 // 5", "counting complete groups"),
        ("23 % 5", "counting a remainder"),
        ("2 ** 5", "checking repeated multiplication"),
        ("250 * 0.08", "finding eight percent of an amount"),
    ]
    for expression, purpose in arithmetic:
        add("calculator", [_step("calculator", {"expression": expression}, purpose=purpose)])

    for before, after, correction in [
        ("3 * 25", "5 * 25", "I meant five items, not three, at 25 each"),
        ("80 - 15", "80 - 12", "The deduction is 12 rather than 15"),
        ("48 / 4", "48 / 6", "There are six people sharing, not four"),
        ("12 + 3 * 4", "(12 + 3) * 4", "Add 12 and 3 before multiplying by 4"),
        ("200 * 0.05", "200 * 0.08", "The percentage is eight, not five"),
        ("2.5 + 1.5", "2.5 - 1.5", "Subtract the second amount instead of adding it"),
    ]:
        add("calculator_correction", [_step("calculator", {"expression": before}),
            _step("calculator", {"expression": after}, mode="correction", correction=correction)])

    for expression, operator, operand, instruction in [
        ("18 + 27 + 35", "/", "4", "Divide that total equally among four people"),
        ("4 * 12.5", "+", "7.5", "Add a separate 7.5 allowance to that total"),
        ("1 / 3", "*", "6", "Multiply that exact fraction by six"),
        ("100 - 37.5", "-", "10", "Subtract a further ten from the remaining amount"),
    ]:
        add("calculator_followup", [_step("calculator", {"expression": expression}),
            _step("calculator", mode="followup", derive={"kind": "previous_calculation",
                  "operator": operator, "operand": operand}, instruction=instruction)])

    for query in ["rice", "soap", "water", "food", "unlisted laptop"]:
        add("catalogue_search", [_step("catalogue_search", {"query": query},
            purpose="checking simulated entries, including availability")])
    for first, second in [("rice", "soap"), ("soap", "water"),
                          ("unlisted laptop", "rice"), ("water", "household")]:
        add("catalogue_search_correction", [_step("catalogue_search", {"query": first}),
            _step("catalogue_search", {"query": second}, mode="correction",
                  correction=f"I meant {second}, not {first}")])

    pairs = [["SIM-RICE-1", "SIM-RICE-2"], ["SIM-RICE-1", "SIM-SOAP-1"],
             ["SIM-SOAP-1", "SIM-WATER-1"]]
    for ids in pairs:
        add("catalogue_compare", [_step("catalogue_compare", {"product_ids": ids})])
    add("search_then_compare", [_step("catalogue_search", {"query": "rice"}),
        _step("catalogue_compare", mode="followup", derive={"kind": "search_ids"},
              instruction="Compare the two rice entries you just found")])
    for ids in pairs[:2]:
        add("compare_then_total", [_step("catalogue_compare", {"product_ids": ids}),
            _step("calculator", mode="followup", derive={"kind": "comparison_total"},
                  instruction="Calculate the combined listed GHS cost of one package of each")])

    rice = _draft([("SIM-RICE-1", 2)])
    basket = _draft([("SIM-RICE-1", 2), ("SIM-SOAP-1", 1)])
    for args in [rice, basket, _draft([("SIM-SOAP-1", 3)], "delivery", "Simulated collection point A")]:
        add("order_draft", [_step("order_draft", args)])
    changes = [
        (rice, {"kind": "quantity", "product_id": "SIM-RICE-1", "quantity": 3},
         "Make that three bags, not two; keep the same product and pickup"),
        (rice, {"kind": "replace", "product_id": "SIM-RICE-1", "replacement": "SIM-RICE-2"},
         "Use the two-kilogram rice bags instead; keep the quantity and pickup"),
        (rice, {"kind": "add", "product_id": "SIM-SOAP-1", "quantity": 2},
         "Add two soap bars to that draft and keep everything else"),
        (basket, {"kind": "remove", "product_id": "SIM-SOAP-1"},
         "Remove the soap, leaving the rice and pickup unchanged"),
        (rice, {"kind": "fulfillment", "fulfillment": "delivery", "location": "Simulated collection point B"},
         "Change only fulfillment to delivery at Simulated collection point B"),
        (_draft([("SIM-SOAP-1", 3)], "delivery", "Simulated collection point A"),
         {"kind": "location", "location": "Simulated collection point C"},
         "I gave the wrong location; use Simulated collection point C instead"),
    ]
    for args, change, instruction in changes:
        add("order_draft_correction", [_step("order_draft", args),
            _step("order_draft", mode="correction", derive={"kind": "draft_change", "change": change},
                  instruction=instruction)])
    add("draft_confirmation", [_step("order_draft", rice),
        _step("order_draft", mode="confirmation", derive={"kind": "confirm_previous_draft"},
              instruction="I explicitly confirm this exact simulated draft only; do not place an order")])

    recoveries = [
        ("catalogue_search", {"query": "rice", "limit": 9}, {"query": "rice", "limit": 2},
         "Use a limit of two instead, keeping the rice query", "invalid_arguments"),
        ("catalogue_compare", {"product_ids": ["SIM-RICE-1", "SIM-RICE-1"]},
         {"product_ids": ["SIM-RICE-1", "SIM-RICE-2"]},
         "Replace the repeated second ID with SIM-RICE-2", "invalid_arguments"),
        ("order_draft", _draft([("SIM-RICE-1", "two")]), _draft([("SIM-RICE-1", 2)]),
         "Use the integer 2 for quantity instead of the word two", "invalid_arguments"),
        ("order_draft", _draft([("SIM-SOAP-1", 1)], "delivery"),
         _draft([("SIM-SOAP-1", 1)], "delivery", "Simulated collection point D"),
         "The missing delivery location is Simulated collection point D", "invalid_arguments"),
        ("order_draft", _draft([("SIM-WATER-1", 1)]), _draft([("SIM-SOAP-1", 1)]),
         "Replace the unavailable water with one soap bar, still for pickup", "insufficient_simulated_stock"),
        ("order_draft", _draft([("SIM-RICE-1", 21)]), _draft([("SIM-RICE-1", 2)]),
         "Reduce the quantity to two, keeping the same rice and pickup", "insufficient_simulated_stock"),
        ("calculator", {"expression": "10 / 0"}, {"expression": "10 / 2"},
         "I mistyped the divisor; use two instead of zero", "division_by_zero"),
        ("calculator", {"expression": "abs(-7)"}, {"expression": "0 - (-7)"},
         "Use the supported arithmetic expression 0 - (-7) instead", "invalid_expression"),
    ]
    for tool, bad, good, instruction, error in recoveries:
        add("invalid_args_recovery", [_step(tool, bad, mode="validation_probe", expect_error=error),
            _step(tool, good, mode="recovery", instruction=instruction)])
    for query in ["community library opening hours", "public bus timetable", "local market schedule"]:
        add("empty_web_search", [_step("web_search", {"query": query})])
    return specs


def _semantic(spec):
    return [{key: value for key, value in step.items()
             if key not in {"purpose", "instruction", "correction"}} for step in spec["steps"]]


def split_group(group_id):
    value = world.sha256_json({"policy": SPLIT_POLICY, "source_group_id": group_id})
    return "dev" if int(value[:16], 16) % 5 == 0 else "train"


def register_scenarios(specs):
    """Group common initial requests before seeing any prompt forms or views."""
    result, seen = [], set()
    for spec in specs:
        first = spec["steps"][0]
        anchor = {"world": world.CONTRACT_VERSION, "tool": first["tool"], "args": first["args"]}
        group = "r7-tool-source-" + world.sha256_json(anchor)
        semantic_hash = world.sha256_json(_semantic(spec))
        identity = "r7-tool-scenario-" + semantic_hash
        require(identity not in seen, "Duplicate underlying scenario")
        seen.add(identity)
        result.append({**copy.deepcopy(spec), "id": identity, "source_id": group,
                       "source_group_id": group, "group_anchor": anchor,
                       "semantic_sha256": semantic_hash, "split": split_group(group),
                       **PENDING})
    return sorted(result, key=lambda row: row["id"])


def _draft_from_result(data):
    draft = data["draft"]
    return _draft([(line["product_id"], line["quantity"]) for line in draft["items"]],
                  draft["fulfillment"], draft["location"])


def resolve_call(step, previous):
    """Resolve follow-ups from actual prior output, never from assumed answers."""
    if "args" in step:
        return copy.deepcopy(step["args"]), {}
    require(previous and previous["result"]["ok"], "Follow-up needs a successful preceding result")
    data, derive = previous["result"]["data"], step["derive"]
    if derive["kind"] == "previous_calculation":
        return {"expression": f"({data['numerator']}/{data['denominator']}) {derive['operator']} {derive['operand']}"}, {}
    if derive["kind"] == "search_ids":
        require(len(data["products"]) == 2, "Expected exactly two actual search results")
        return {"product_ids": [row["id"] for row in data["products"]]}, {}
    if derive["kind"] == "comparison_total":
        return {"expression": " + ".join(f"({row['price_minor']}/{data['price_scale']})"
                                         for row in data["products"])}, {}
    args = _draft_from_result(data)
    if derive["kind"] == "confirm_previous_draft":
        return args, {"confirmation": {"confirmed": True, "draft_sha256": data["draft_sha256"]}}
    require(derive["kind"] == "draft_change", "Unknown derivation")
    change = derive["change"]
    kind = change["kind"]
    if kind in {"quantity", "replace"}:
        line = next(row for row in args["items"] if row["product_id"] == change["product_id"])
        line["quantity" if kind == "quantity" else "product_id"] = change[
            "quantity" if kind == "quantity" else "replacement"]
    elif kind == "add":
        args["items"].append({"product_id": change["product_id"], "quantity": change["quantity"]})
    elif kind == "remove":
        args["items"] = [row for row in args["items"] if row["product_id"] != change["product_id"]]
    elif kind == "fulfillment":
        args["fulfillment"], args["location"] = change["fulfillment"], change["location"]
    elif kind == "location":
        args["location"] = change["location"]
    else:
        raise ValueError("Unknown draft change")
    return args, {}


def user_prompt(step, args, form):
    require(form in FORMS, "Unknown prompt form")
    index = FORMS.index(form)
    if "instruction" in step:
        text = step["instruction"]
        clause = text if text.startswith("I ") else text[0].lower() + text[1:]
        if step["mode"] == "confirmation":
            return (text + ".", "Yes, " + clause + ".",
                    "About the draft you just showed: " + clause + ".")[index]
        return (text + ".", text + ". Can you apply that change?",
                "For the previous result: " + clause + ".")[index]
    tool = step["tool"]
    if tool == "calculator":
        expression = args["expression"]
        forms = [f"Calculate {expression}.", f"What is the value of {expression}?",
                 f"I am {step['purpose']}. Work out {expression}."]
    elif tool in {"catalogue_search", "web_search"}:
        place = "simulated catalogue" if tool == "catalogue_search" else "supplied pinned web evidence"
        query = args["query"]
        limit = f" Use result limit {args['limit']}." if "limit" in args else ""
        forms = [f"Search the {place} for {query}.{limit}",
                 f"What matches {query} in the {place}?{limit}",
                 f"I am looking for {query}. Check only the {place}.{limit}"]
    elif tool == "catalogue_compare":
        ids = " and ".join(args["product_ids"])
        forms = [f"Compare {ids} in the simulated catalogue.",
                 f"How do {ids} differ in listed package price, size and simulated stock?",
                 f"My simulated options are {ids}. Compare the packages, without assuming equal sizes."]
    else:
        items = "; ".join(f"quantity {json.dumps(row['quantity'])} of {row['product_id']}" for row in args["items"])
        fulfillment = args["fulfillment"] + (" at " + args["location"] if "location" in args else "")
        forms = [f"Make a simulated draft only: {items}; {fulfillment}. Do not place an order.",
                 f"Can you draft {items} for {fulfillment}, using only the simulated catalogue? No purchase.",
                 f"This is a draft exercise, not a real order. Use {items} for {fulfillment}."]
    if "correction" in step:
        forms = [step["correction"] + ". " + text for text in forms]
    if step["mode"] == "validation_probe":
        forms = ["Validation exercise; report any tool rejection. " + text for text in forms]
    return forms[index]


def _money(minor, scale=100):
    require(type(minor) is int and scale == 100, "Expected actual integer GHS minor units")
    return f"GHS {minor // scale}.{minor % scale:02d}"


def assistant_response(execution):
    """Truthful English surface derived exclusively from the recorded tool result."""
    result, name = execution["result"], execution["call"]["name"]
    if not result["ok"]:
        error = result["error"]
        return (f"The sandbox rejected this call ({error['code']}): {error['message']}"
                + (" No order or payment was made." if name == "order_draft" else ""))
    data = result["data"]
    if name == "calculator":
        exact = data["numerator"] if data["denominator"] == "1" else data["numerator"] + "/" + data["denominator"]
        text = f"{data['expression']} = {exact} exactly."
        if data["denominator"] != "1":
            text += (f" In decimal form, this is exactly {data['decimal']}." if data["decimal_is_exact"]
                     else f" Its decimal approximation is {data['decimal']}.")
        return text
    if name in {"catalogue_search", "catalogue_compare"}:
        if not data["products"]:
            require(data["empty_reason"] == "no_match", "Unexpected empty catalogue result")
            return "No simulated catalogue entries matched. I cannot supply an unlisted product or invent a price."
        text = "Simulated catalogue only: " + "; ".join(
            f"{row['id']} ({row['name']}), {row['unit']}, {_money(row['price_minor'], data['price_scale'])} "
            f"per listed package, simulated stock {row['stock']}" for row in data["products"]) + "."
        if name == "catalogue_compare":
            text += (f" The listed package-price spread is {_money(data['price_spread_minor'])}. "
                     "These package prices are not a normalized unit-price comparison.")
        return text
    if name == "order_draft":
        draft = data["draft"]
        text = "Simulated draft only: " + "; ".join(
            f"{row['quantity']} x {row['product_id']} ({row['unit']}) at {_money(row['unit_price_minor'])} each, "
            f"subtotal {_money(row['subtotal_minor'])}" for row in draft["items"])
        text += f". Items total {_money(draft['total_minor'])}, excluding delivery and taxes. {draft['fulfillment'].capitalize()}"
        text += (" at " + draft["location"] if draft["location"] else "") + ". "
        text += ("Your explicit confirmation is recorded for this exact simulated draft. " if data["confirmed"]
                 else "This exact draft still requires your explicit confirmation. ")
        return text + "No real order, cart change or payment was made."
    require(name == "web_search" and data["results"] == [] and data["empty_reason"] == "no_evidence",
            "This candidate source must never generate web evidence")
    return "No pinned web evidence was supplied, so the search returned no results. I cannot verify the requested information."


def _registry(scenarios):
    groups = defaultdict(list)
    for scenario in scenarios:
        groups[scenario["source_group_id"]].append(scenario)
    return [{"id": group, "source_id": group, "source_group_id": group,
             "origin": "project_authored_templated_synthetic", "split": rows[0]["split"],
             "group_anchor": rows[0]["group_anchor"],
             "group_policy": "all scenarios sharing the initial structured request stay together",
             "scenario_refs": [{"id": row["id"], "sha256": world.sha256_json(row)} for row in rows],
             "language": "en", "tool_world": world.CONTRACT_VERSION,
             "catalogue_is_simulated": True, "external_sources": [],
             "legacy_targets_imported": False, **PENDING} for group, rows in sorted(groups.items())]


def _execute_scenario(scenario):
    records, previous = [], None
    for index, step in enumerate(scenario["steps"]):
        args, context = resolve_call(step, previous)
        execution = world.execute(step["tool"], args, context)
        require(bool(execution["messages"]), "Candidate call cannot be rendered natively")
        require(execution["result_sha256"] == world.sha256_json(execution["result"]), "Result hash differs")
        expected_error = step.get("expect_error")
        require((not execution["result"]["ok"] and execution["result"]["error"]["code"] == expected_error)
                if expected_error else execution["result"]["ok"], "Unexpected execution result")
        identity = f"{scenario['id']}:execution:{index}"
        records.append({"id": identity, "source_group_id": scenario["source_group_id"],
                        "scenario_id": scenario["id"], "step_index": index,
                        "arguments": args, "context": context, "execution": execution,
                        "derived_from_execution_id": records[-1]["id"] if "derive" in step else None,
                        "explicit_confirmation": step["mode"] == "confirmation"})
        previous = execution
    return records


def _candidate(scenario, receipts, registry_record, form):
    messages = [{"role": "system", "content": SYSTEM}]
    bindings, derivations = [], []
    for index, (step, receipt) in enumerate(zip(scenario["steps"], receipts, strict=True)):
        execution = receipt["execution"]
        messages.append({"role": "user", "content": user_prompt(step, receipt["arguments"], form)})
        call, response = copy.deepcopy(execution["messages"])
        original_id = call["tool_calls"][0]["id"]
        alias = "call_" + scenario["semantic_sha256"][:24] + "_" + str(index)
        call["tool_calls"][0]["id"], response["tool_call_id"] = alias, alias
        bindings.append({"execution_id": receipt["id"], "execution_record_sha256": world.sha256_json(receipt),
                         "executor_call_id": original_id, "native_call_id": alias,
                         "assistant_call_index": len(messages), "tool_response_index": len(messages) + 1,
                         "tool_result_sha256": execution["result_sha256"]})
        messages.extend([call, response])
        answer = assistant_response(execution)
        derivations.append({"message_index": len(messages), "execution_id": receipt["id"],
                            "renderer": RENDERER, "content_sha256": digest_bytes(answer.encode())})
        messages.append({"role": "assistant", "content": answer})
    row = {"id": scenario["id"] + ":" + form, "source_id": scenario["source_group_id"],
           "source_group_id": scenario["source_group_id"], "scenario_id": scenario["id"],
           "split": scenario["split"], "collection": "tools", "language": "en",
           "messages": messages, "tools": world.tools(), **PENDING,
           "metadata": {"origin": "project_authored_templated_synthetic", "synthetic": True,
                        "group_id": scenario["source_group_id"], "family": scenario["family"],
                        "prompt_form_family": form, "prompt_forms_are_independent_sources": False,
                        "english_response_origin": "deterministic_actual_result_renderer",
                        "catalogue_and_orders_simulated": True, "external_evidence_used": False,
                        "translation_status": "pending_twi_translation_and_qualification",
                        "twi_targets_present": False, "meaning_review_status": "pending",
                        "prior_twi_targets_imported": False,
                        "tool_bindings": bindings, "assistant_derivations": derivations,
                        "erroneous_call_targets_admitted": False},
           "original_ref": {"dataset_id": DATASET_ID, "dataset_version": VERSION,
                            "scenario_path": "scenarios.jsonl", "scenario_id": scenario["id"],
                            "scenario_sha256": world.sha256_json(scenario),
                            "registry_path": "source-registry.jsonl",
                            "registry_record_sha256": world.sha256_json(registry_record),
                            "executions_path": "executions.jsonl"}}
    validate_history(row)
    return row


def _coverage(scenarios, registry, receipts, rows, forms):
    def count(subset):
        return {"candidate_views": len(subset), "scenario_count": len({r["scenario_id"] for r in subset}),
                "source_groups": len({r["source_group_id"] for r in subset}),
                "prompt_form_families": sorted({r["metadata"]["prompt_form_family"] for r in subset})}

    overlap = {}
    for role in ("user", "assistant"):
        texts = [{m["content"] for r in rows if r["split"] == split for m in r["messages"]
                  if m["role"] == role and m["content"]} for split in ("train", "dev")]
        overlap[role + "_exact_texts_shared_across_splits"] = len(texts[0] & texts[1])
    split_by_scenario = {s["id"]: s["split"] for s in scenarios}
    requests = [{world.sha256_json({"call": r["execution"]["call"], "context": r["context"]})
                 for r in receipts if split_by_scenario[r["scenario_id"]] == split} for split in ("train", "dev")]
    overlap["exact_tool_requests_shared_across_splits"] = len(requests[0] & requests[1])
    return {"source_groups": len(registry), "scenario_count": len(scenarios), "candidate_views": len(rows),
            "prompt_form_families": list(forms), "prompt_form_family_count": len(forms),
            "distinct_user_prompts": len({m["content"] for r in rows for m in r["messages"] if m["role"] == "user"}),
            "distinct_assistant_responses": len({m["content"] for r in rows for m in r["messages"]
                                                  if m["role"] == "assistant" and m["content"]}),
            "execution_receipts": len(receipts),
            "distinct_structured_tool_requests": len({world.sha256_json({"call": r["execution"]["call"],
                                                                           "context": r["context"]}) for r in receipts}),
            "native_call_occurrences_in_views": sum(len(r["metadata"]["tool_bindings"]) for r in rows),
            "tool_names": dict(sorted(Counter(r["execution"]["call"]["name"] for r in receipts).items())),
            "error_results": dict(sorted(Counter(r["execution"]["result"]["error"]["code"] for r in receipts
                                                   if not r["execution"]["result"]["ok"]).items())),
            "families": {family: count([r for r in rows if r["metadata"]["family"] == family])
                         for family in sorted({s["family"] for s in scenarios})},
            "splits": {split: count([r for r in rows if r["split"] == split]) for split in ("train", "dev")},
            "training_ready_views": 0, "training_eligible_views": 0, "twi_qualified_views": 0,
            "web_snapshot_records": 0, "legacy_targets_imported": 0,
            "exact_overlap_diagnostics": overlap,
            "all_views_templated_synthetic": True, "split_before_prompt_augmentation": True,
            "source_groups_disjoint_across_splits": True,
            "views_are_not_independent_sources": True, "family_source_counts_are_additive": False,
            "limitations": ["English-only candidate sources; Twi translation and qualification are pending.",
                            "Shared wording templates across train/dev; this is not a wording-held-out evaluation.",
                            "Related scenarios share source groups; views and call occurrences are not independent samples.",
                            "Different trajectories can reuse exact answers and tool requests across splits; see overlap diagnostics.",
                            "Only four fictional catalogue products; not broad real-world commerce coverage.",
                            "No factual web evidence, health advice, real orders, teacher outputs or training admission.",
                            "No model-token lengths or external holdout-overlap admission checks have been certified."]}


def _json_bytes(value):
    return (world.canonical_json(value) + "\n").encode("ascii")


def _rows_bytes(rows):
    return b"".join(_json_bytes(row) for row in rows)


def build_package(forms=FORMS):
    require(bool(forms) and len(forms) == len(set(forms)) and set(forms) <= set(FORMS), "Invalid prompt forms")
    scenarios = register_scenarios(scenario_specs())
    registry = _registry(scenarios)
    by_group = {row["id"]: row for row in registry}
    receipts, rows = [], []
    for scenario in scenarios:
        executed = _execute_scenario(scenario)
        receipts.extend(executed)
        rows.extend(_candidate(scenario, executed, by_group[scenario["source_group_id"]], form) for form in forms)
    rows.sort(key=lambda row: row["id"])
    train = {r["source_group_id"] for r in rows if r["split"] == "train"}
    dev = {r["source_group_id"] for r in rows if r["split"] == "dev"}
    require(train and dev and not train & dev, "Missing or overlapping source-group split")
    coverage = _coverage(scenarios, registry, receipts, rows, forms)
    files = {"scenarios.jsonl": _rows_bytes(scenarios), "source-registry.jsonl": _rows_bytes(registry),
             "executions.jsonl": _rows_bytes(receipts), "tools.json": _json_bytes(world.tools()),
             "coverage.json": _json_bytes(coverage)}
    for split in ("train", "dev"):
        files[f"candidates/{split}.jsonl"] = _rows_bytes(r for r in rows if r["split"] == split)
    code = [Path(__file__), Path(world.__file__), ROOT / "scripts/r7_encoding.py"]
    manifest = {"dataset_id": DATASET_ID, "dataset_version": VERSION, **PENDING,
                "source_registry_path": "source-registry.jsonl", "records_paths": ["candidates/train.jsonl", "candidates/dev.jsonl"],
                "record_count": len(rows), "source_group_count": len(registry), "scenario_count": len(scenarios),
                "source_kind": "templated_synthetic_english_executed_tools",
                "language": "en", "translation_status": "pending_twi_translation_and_qualification",
                "split_policy": SPLIT_POLICY, "prompt_forms": list(forms),
                "group_identity": "sha256 of tool-world version and initial structured tool request; independent of wording",
                "tool_world_version": world.CONTRACT_VERSION, "response_renderer": RENDERER,
                "code_sha256": {path.relative_to(ROOT).as_posix(): digest_bytes(path.read_bytes()) for path in code},
                "artifacts": {name: digest_bytes(value) for name, value in sorted(files.items())},
                "teacher_calls": 0, "network_requests": 0, "legacy_sources_used": [],
                "admission_requirements": ["Qualified Twi translation preserving full history and native tool records",
                                           "Independent meaning/correction/follow-up review",
                                           "Held-source/held-content overlap checks and native no-truncation encoding",
                                           "Explicit versioned research admission; this package grants none"]}
    files["manifest.json"] = _json_bytes(manifest)
    files["PREPARED.json"] = _json_bytes({"manifest_sha256": digest_bytes(files["manifest.json"]),
                                           "candidate_only": True, "training_ready": False})
    return files


def prepare(out):
    out = Path(out)
    require(not out.exists() and not out.is_symlink(), "Use a fresh candidate output directory; refusing overwrite")
    files = build_package()
    out.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return verify_package(out)


def verify_package(out):
    """Verify exact bytes by deterministic regeneration and actual CPU tool replay."""
    out = Path(out)
    require(out.is_dir() and not out.is_symlink(), "Expected an ordinary candidate directory")
    require(not any(p.is_symlink() for p in out.rglob("*")), "Linked candidate artifact")
    expected = build_package()
    require({p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == set(expected),
            "Missing or unregistered candidate artifact")
    for name, content in expected.items():
        require((out / name).read_bytes() == content, "Candidate artifact differs from executed source: " + name)
    return {"verified": True, "candidate_only": True, "training_ready": False,
            "manifest_sha256": digest_bytes(expected["manifest.json"]),
            "coverage": json.loads(expected["coverage.json"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(verify_package(args.out) if args.verify else prepare(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
