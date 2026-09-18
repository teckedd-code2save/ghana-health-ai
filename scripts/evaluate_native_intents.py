"""Score real source-held-out intent/entity generation, without a model judge."""
import argparse
import json
from collections import Counter
from pathlib import Path
from build_native_understanding_mix import read
from prepare_native_intents import annotation_instruction

ROOT = Path(__file__).resolve().parents[1]


def evaluation_inputs(cases, train, prompt_version):
    inputs = [{"id": row["id"], "messages": [dict(m) for m in row["messages"][:-1]]} for row in cases]
    if prompt_version == "legacy":
        return inputs
    if prompt_version != "schema-v2":
        raise ValueError("Unknown annotation prompt version")
    # Only the training inventory defines the public task, never dev answers.
    annotations = [decode(row["messages"][-1]["content"]) for row in train]
    intents = sorted({row["intent"] for row in annotations})
    types = sorted({entity["type"] for row in annotations for entity in row["entities"]})
    instruction = annotation_instruction(intents, types)
    for row in inputs:
        row["messages"][0]["content"] = instruction
    return inputs


def decode(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"intent", "entities"} or not isinstance(value["intent"], str) or not isinstance(value["entities"], list):
        raise ValueError("Invalid annotation structure")
    for item in value["entities"]:
        if not isinstance(item, dict) or set(item) != {"type", "text"} or not all(isinstance(v, str) for v in item.values()):
            raise ValueError("Invalid entity")
    return value


def score(cases, results):
    reference = {r["id"]: decode(r["messages"][-1]["content"]) for r in cases}
    report = {}
    for variant in ("base", "adapter"):
        rows = [r for r in results if r["variant"] == variant]
        if len(reference) != len(cases) or len(rows) != len(cases) or {r["id"] for r in rows} != set(reference):
            raise ValueError("Missing or duplicated paired predictions")
        correct = label_correct = invalid = format_invalid = exact = tp = fp = fn = 0
        for row in rows:
            truth = reference[row["id"]]
            try:
                text = row["prediction"].strip()
                lines = text.splitlines()
                fenced = len(lines) >= 3 and lines[0] in ("```json", "```") and lines[-1] == "```"
                if fenced:
                    text = "\n".join(lines[1:-1])
                if row["limit_reached"]:
                    raise ValueError("Incomplete generation")
                value = json.loads(text)
                label_correct += isinstance(value, dict) and value.get("intent") == truth["intent"]
                prediction = decode(text)
                format_invalid += fenced
                correct += prediction["intent"] == truth["intent"]
                exact += prediction == truth
            except (ValueError, TypeError):
                prediction = {"intent": None, "entities": []}
                invalid += 1
                format_invalid += 1
            actual = Counter((r["type"], r["text"]) for r in prediction["entities"])
            expected = Counter((r["type"], r["text"]) for r in truth["entities"])
            tp += sum((actual & expected).values())
            fp += sum((actual - expected).values())
            fn += sum((expected - actual).values())
        report[variant] = {"rows": len(rows), "intent_correct": correct, "intent_accuracy": correct/len(rows),
            "intent_label_correct_independent_of_entity_format": label_correct,
            "intent_label_accuracy_independent_of_entity_format": label_correct/len(rows),
            "complete_structure_correct": exact, "unparseable_or_incomplete": invalid, "strict_format_invalid": format_invalid,
            "entity_tp": tp, "entity_fp": fp, "entity_fn": fn,
            "entity_micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return {"scores": report, "scope": "INJONGO Twi official development; auxiliary intent/entity task, not direct-response quality.",
            "format_policy": "One enclosing JSON markdown fence is allowed for semantic scoring but reported as a strict format failure. No prose extraction or label guessing."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "status"])
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint-100")
    parser.add_argument("--attempt", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--prompt-version", choices=["legacy", "schema-v2"], default="legacy")
    args = parser.parse_args()
    import modal
    folder = ROOT / "tmp/native-understanding-v5"
    cases = read(ROOT / "tmp/native-intent-v2", "validation")
    prompt_tag = "" if args.prompt_version == "legacy" else "-" + args.prompt_version
    receipt_path = folder / (args.checkpoint.replace("/", "-") + f"-intents{prompt_tag}-attempt{args.attempt}.receipt.json")
    if args.action == "submit":
        if receipt_path.exists():
            raise ValueError("Already submitted; inspect saved receipt")
        run = json.loads((folder / "run-receipt.json").read_text())["run_id"]
        candidate = modal.Function.from_name("ghana-response-v2-private", "inspect_checkpoint").remote(run, args.checkpoint)
        inputs = evaluation_inputs(cases, read(ROOT / "tmp/native-intent-v2", "train"), args.prompt_version)
        receipt = {"candidate": candidate, "state": "submitting", "source": "tmp/native-intent-v2/manifest.json",
                   "prompt_version": args.prompt_version}
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        call = modal.Function.from_name("ghana-semantic-pair-evaluation", "evaluate").spawn(
            {k: candidate[k] for k in ("run_id", "checkpoint", "adapter_sha256")},
            inputs, args.attempt, "native-intent" + prompt_tag)
        receipt.update(call_id=call.object_id, state="submitted")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    else:
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("call_id"):
            raise ValueError("Uncertain submission; inspect remote state")
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=5)
        except TimeoutError:
            print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        receipt_path.with_suffix(".results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        report = score(cases, result["results"])
        receipt_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
