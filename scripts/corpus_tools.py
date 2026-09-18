"""Research-only catalogue trajectories with real local execution, never orders."""
from __future__ import annotations
import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from corpus_release import digest, save, sha

CATALOGUE = [
    {"sku": "SIM-TOMATO-A", "name": "tomatoes", "unit": "kg", "price_ghs": "18.00", "stock": 20},
    {"sku": "SIM-TOMATO-B", "name": "tomatoes", "unit": "kg", "price_ghs": "21.00", "stock": 12},
    {"sku": "SIM-ONION-A", "name": "onions", "unit": "kg", "price_ghs": "16.00", "stock": 18},
    {"sku": "SIM-RICE-A", "name": "rice", "unit": "5kg bag", "price_ghs": "85.00", "stock": 40},
    {"sku": "SIM-SOAP-A", "name": "laundry soap", "unit": "bar", "price_ghs": "8.00", "stock": 200},
    {"sku": "SIM-OIL-A", "name": "cooking oil", "unit": "1L bottle", "price_ghs": "32.00", "stock": 60},
]
TOOLS = [
    {"type": "function", "function": {"name": "catalogue_search", "description": "Search the simulated research catalogue; never the internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculate_total", "description": "Calculate a simulated item subtotal in GHS, excluding delivery.", "parameters": {"type": "object", "properties": {"sku": {"type": "string"}, "quantity": {"type": "string"}}, "required": ["sku", "quantity"], "additionalProperties": False}}},
]


def execute(name, arguments):
    if name == "catalogue_search":
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str):
            raise ValueError("Invalid search arguments")
        return {"simulated": True, "source": "controlled_catalogue_v1", "items": [r for r in CATALOGUE if arguments["query"].lower() in r["name"]]}
    if name == "calculate_total":
        if set(arguments) != {"sku", "quantity"}:
            raise ValueError("Invalid calculator arguments")
        row = next((r for r in CATALOGUE if r["sku"] == arguments["sku"]), None)
        try:
            quantity = Decimal(arguments["quantity"])
        except (InvalidOperation, TypeError):
            raise ValueError("Invalid quantity") from None
        if not row or not quantity.is_finite() or quantity <= 0 or quantity > row["stock"]:
            raise ValueError("Unavailable item or quantity")
        if row["unit"] != "kg" and quantity != quantity.to_integral_value():
            raise ValueError("Packaged items require whole units")
        return {"simulated": True, "sku": row["sku"], "quantity": str(quantity), "unit": row["unit"],
                "currency": "GHS", "subtotal": str((quantity * Decimal(row["price_ghs"])).quantize(Decimal(".01"))),
                "delivery_included": False, "order_placed": False}
    raise ValueError("Only local catalogue and calculator actions are allowed")


def call(messages, name, arguments):
    identity = "call_" + digest([len(messages), name, arguments])[:12]
    result = execute(name, arguments)
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": identity, "name": name, "content": json.dumps(result)},
    ])
    return result


def verify_trajectory(row):
    pending, seen, totals = {}, set(), []
    executed = 0
    for message in row["messages"]:
        for tool_call in message.get("tool_calls", []):
            if message["role"] != "assistant" or tool_call["id"] in seen:
                raise ValueError("Duplicate call ID")
            seen.add(tool_call["id"])
            pending[tool_call["id"]] = tool_call["function"]
        if message["role"] == "tool":
            function = pending.pop(message["tool_call_id"])
            if json.loads(message["content"]) != execute(function["name"], json.loads(function["arguments"])):
                raise ValueError("Fabricated tool result")
            if function["name"] == "calculate_total": totals.append(json.loads(message["content"]))
            executed += 1
    if pending or not executed or row["messages"][-1]["role"] != "assistant":
        raise ValueError("Incomplete tool trajectory")
    if row.get("simulated") is not True or row.get("real_orders") is not False:
        raise ValueError("Missing simulation boundary")
    if not totals: raise ValueError("Missing executed subtotal")
    total = totals[-1]
    product = next(r for r in CATALOGUE if r["sku"] == total["sku"])
    expected = f"Simulated subtotal: GHS {total['subtotal']} for {total['quantity']} {product['unit']} units of {product['name']}. Delivery is not included. No order has been placed."
    if row["messages"][-1]["content"] != expected:
        raise ValueError("Final answer does not match the executed result")


def build(output):
    if output.exists():
        raise ValueError("Use a new immutable tool artifact directory")
    output.mkdir(parents=True)
    save(output / "catalogue.json", {"version": "controlled_catalogue_v1", "simulated": True, "items": CATALOGUE})
    rows = []
    for index, product in enumerate(CATALOGUE):
        for quantity in ("1", "2", "3"):
            for scenario in ("search_and_compare", "clarify_location", "correct_quantity"):
                messages = [{"role": "system", "content": "Use only the simulated research catalogue. Prices are fixtures, not live offers. No purchases can be made. Ask for missing quantities and locations. Ground totals in executed tools."},
                            {"role": "user", "content": f"Find {product['name']} in the simulated catalogue."}]
                results = call(messages, "catalogue_search", {"query": product["name"]})
                available = sorted(results["items"], key=lambda r: Decimal(r["price_ghs"]))
                chosen = available[0]
                prices = "; ".join(f"{r['sku']}: GHS {r['price_ghs']} per {r['unit']}" for r in available)
                messages.append({"role": "assistant", "content": f"Simulated options: {prices}. How many {chosen['unit']} units do you want?"})
                messages.append({"role": "user", "content": f"{quantity} units."})
                total = call(messages, "calculate_total", {"sku": chosen["sku"], "quantity": quantity})
                if scenario == "correct_quantity":
                    messages.extend([{"role": "assistant", "content": f"The simulated subtotal is GHS {total['subtotal']}, excluding delivery. No order has been placed."},
                                     {"role": "user", "content": "Change that to 4 units instead."}])
                    total = call(messages, "calculate_total", {"sku": chosen["sku"], "quantity": "4"})
                if scenario == "clarify_location":
                    messages.extend([{"role": "assistant", "content": "Which delivery area should I record for this simulation?"},
                                     {"role": "user", "content": "Adenta."}])
                messages.append({"role": "assistant", "content": f"Simulated subtotal: GHS {total['subtotal']} for {total['quantity']} {chosen['unit']} units of {chosen['name']}. Delivery is not included. No order has been placed."})
                identity = digest([product["name"], quantity, scenario])
                row = {"id": identity, "group_id": digest(["controlled_catalogue_v1", product["name"]]),
                       "split": "validation" if product["name"] == "cooking oil" else "train", "language": "en",
                       "scenario": scenario, "origin": "synthetic_executable_fixture", "simulated": True,
                       "real_orders": False, "tools": TOOLS, "messages": messages,
                       "catalogue_sha256": sha(output / "catalogue.json"), "human_reviewed": False,
                       "twi_training_eligible": False, "screening": "tool_execution_verified"}
                verify_trajectory(row)
                if identity not in {r["id"] for r in rows}: rows.append(row)
    for split in ("train", "validation"):
        with (output / f"{split}.jsonl").open("x") as handle:
            for row in rows:
                if row["split"] == split: handle.write(json.dumps(row) + "\n")
    save(output / "report.json", {"rows": len(rows), "unique_product_groups": len({r["group_id"] for r in rows}),
         "simulated": True, "executions_verified": True, "twi_derivatives": 0,
         "limitations": ["Small executable fixtures, not a general commerce corpus.", "English targets retained until a Twi derivative passes qualification.", "No real sellers, live search, payments or purchases."]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
