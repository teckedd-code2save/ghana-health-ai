"""Small, durable direct-response diagnostic before enabling a private model choice."""
import argparse
import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "tmp/response-adaptation-v2"
APP = "ghana-response-v2-private"


def greeting_cases(fixture):
    cases = [{"id": row["id"], "messages": [
        {"role": message["role"], "content": message["content"]}
        for message in row["messages"]]} for row in fixture["cases"]]
    if len(cases) != 4 or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Expected four distinct greeting cases")
    for row in cases:
        messages = row["messages"]
        if not messages or len(messages) % 2 != 1:
            raise ValueError("Diagnostic must end with a user message")
        for index, message in enumerate(messages):
            if (message["role"] != ("user" if index % 2 == 0 else "assistant")
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError("Invalid diagnostic conversation")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "status"])
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint-100")
    parser.add_argument("--variant", choices=["response_v2", "response_v4", "response_v5", "response_base", "afrique_v3", "afrique_base", "afrique_v6", "afrique9_base"], default="response_v2")
    parser.add_argument("--run-receipt", type=Path)
    parser.add_argument("--profile", choices=["greedy", "publisher", "qwen-presence", "gemma-thinking"], default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--suite", choices=["context", "health", "meaning", "greeting"], default="context")
    args = parser.parse_args()
    folder = ROOT / "tmp/afrique-response-v3" if args.variant.startswith("afrique_") else FOLDER
    if args.variant == "response_v4":
        folder = ROOT / "tmp/semantic-response-v4"
    if args.variant == "response_v5":
        folder = ROOT / "tmp/native-understanding-v5"
    if args.variant in ("afrique_v6", "afrique9_base"):
        folder = ROOT / "tmp/balanced-afrique-v6"
    if args.run_receipt:
        folder = args.run_receipt.parent
    run_receipt = args.run_receipt or folder / "run-receipt.json"
    run = json.loads(run_receipt.read_text())["run_id"]
    tag = args.checkpoint.replace("/", "-") + "-" + args.variant
    if args.profile != "greedy" or args.seed != 42:
        tag += f"-{args.profile}-seed{args.seed}"
    if args.attempt > 1:
        tag += f"-attempt{args.attempt}"
    if args.suite != "context":
        tag += "-" + args.suite
    if args.run_receipt:
        tag = run + "-" + tag
    receipt_path = folder / (tag + ".receipt.json")
    if args.action == "submit":
        if receipt_path.exists():
            raise ValueError("Diagnostic already submitted; inspect its saved receipt")
        candidate = modal.Function.from_name(APP, "inspect_checkpoint").remote(run, args.checkpoint)
        if args.suite == "greeting":
            fixture = json.loads((ROOT / "data/response-adaptation/greeting-regressions.v1.json").read_text())
            cases = greeting_cases(fixture)
        else:
            product = json.loads((ROOT / "tmp/general-foundation-comparison/cases-81a7aad880d9.json").read_text())
            wanted = {"context": ("budget-tw", "no-age-invention", "language-switch"),
                      "health": ("tw-breathing-difficulty", "tw-pregnancy-warning-signs", "tw-confirmed-malaria-newborn", "tw-hospital-choice"),
                      "meaning": ("negation-tw", "grounded-tw", "correction-tw", "search-tw-no-tools")}[args.suite]
            cases = [{"id": row["id"], "messages": row["messages"][1:]} for row in product if row["id"] in wanted]
            if len(cases) != len(wanted):
                raise ValueError("Diagnostic cases are incomplete")
        if args.suite == "context":
            cases.append({"id": "executed-total-tw", "messages": [{"role": "user", "content":
                "Tomato kilogram baako bo yɛ cedi 7.50. Mepɛ kilogram 2. Fa calculate_total bu ne nyinaa bo kyerɛ me wɔ Twi mu."}]})
        engine = modal.Cls.from_name(APP, "Response")(**{k: candidate[k] for k in ("run_id", "checkpoint", "adapter_sha256")})
        receipt = {"candidate": candidate, "variant": args.variant, "profile": args.profile, "seed": args.seed,
                   "inputs": cases, "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                   "suite": args.suite, "state": "submitting"}
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        call = engine.check.spawn(cases, args.variant, args.profile, args.seed)
        receipt.update(call_id=call.object_id, state="submitted")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    else:
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("call_id"):
            raise ValueError("Uncertain submission; inspect remote state before restarting")
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=5)
        except TimeoutError:
            print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        path = folder / (tag + ".results.json")
        path.write_text(json.dumps({"receipt": receipt, "results": result}, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"path": str(path), "results": [{"id": row["id"], "answer": row["answer"],
             "error_type": row["error_type"], "tool_executed": any(e["type"] == "tool_result" for e in row["events"])} for row in result]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
