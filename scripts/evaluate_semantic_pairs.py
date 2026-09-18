"""Paired AfriXNLI development evaluation, kept separate from training."""
import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/semantic-response-v4/afrixnli"
REVISION = "e3ca06b30f3e7af2a86f6c8609ea76fee326bc56"
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def make_cases(twi, english):
    if len(twi) != 450 or len(english) != 450:
        raise ValueError("Unexpected AfriXNLI development split size")
    cases = []
    for index, (tw, en) in enumerate(zip(twi, english, strict=True)):
        if tw["label"] != en["label"] or int(tw["label"]) not in LABELS:
            raise ValueError("Parallel labels disagree")
        order = list(LABELS)
        random.Random(f"afrixnli-options-v1:{index}").shuffle(order)
        options = {letter: LABELS[number] for letter, number in zip("ABC", order, strict=True)}
        expected = "ABC"[order.index(int(tw["label"]))]
        for language, row in (("tw", tw), ("en", en)):
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("premise", "hypothesis")):
                raise ValueError("Missing source text")
            content = ("Determine the relationship of the hypothesis to the premise. "
                       "Entailment means the hypothesis must follow; contradiction means it cannot be true; "
                       "neutral means there is not enough information. Reply with only A, B, or C.\n\n"
                       f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\n\n" +
                       "\n".join(f"{letter}. {label}" for letter, label in options.items()))
            cases.append({"id": f"afrixnli-dev-{index:04d}-{language}", "pair_id": index,
                          "language": language, "source_row": row, "options": options,
                          "expected": expected, "messages": [{"role": "user", "content": content}]})
    return cases


def score(cases, results):
    expected = {row["id"]: row for row in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate source IDs")
    groups, paired = {}, {}
    for variant in ("base", "adapter"):
        rows = [row for row in results if row["variant"] == variant]
        if len(rows) != len(cases) or {row["id"] for row in rows} != set(expected):
            raise ValueError("Incomplete or duplicate paired evaluation")
        for language in ("tw", "en"):
            values = []
            for row in rows:
                case = expected[row["id"]]
                if case["language"] != language:
                    continue
                answer = row["prediction"].strip()
                valid = answer in "ABC" and len(answer) == 1 and not row["limit_reached"]
                values.append((valid and answer == case["expected"], valid))
                paired[(variant, row["id"])] = values[-1][0]
            correct = sum(v[0] for v in values)
            majority = max(Counter(expected[row["id"]]["expected"] for row in rows
                                   if expected[row["id"]]["language"] == language).values())
            groups[f"{variant}:{language}"] = {"rows": len(values), "correct": correct,
                "accuracy": correct / len(values), "invalid": sum(not v[1] for v in values),
                "constant_letter_accuracy": majority / len(values)}
    changes = {language: {"improved": sum(not paired[("base", row["id"])] and paired[("adapter", row["id"])]
                                             for row in cases if row["language"] == language),
                          "regressed": sum(paired[("base", row["id"])] and not paired[("adapter", row["id"])]
                                             for row in cases if row["language"] == language)} for language in ("tw", "en")}
    return {"groups": groups, "paired_changes": changes,
            "meaning": "Strict generated-label accuracy on AfriXNLI development, not conversational or clinical accuracy.",
            "limitations": ["English instructions; Twi versus English premise/hypothesis text.",
                            "Existing foundation pretraining contamination is unknown.",
                            "Development set can inform research selection; official test remains untouched."]}


def prepare():
    from huggingface_hub import hf_hub_download
    if (OUT / "manifest.json").exists():
        raise ValueError("Prepared evaluation already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    sources, datasets = [], []
    for language in ("twi", "eng"):
        filename = f"data/{language}/dev.tsv"
        raw = Path(hf_hub_download("masakhane/afrixnli", filename, repo_type="dataset", revision=REVISION)).read_bytes()
        (OUT / f"{language}-dev.tsv").write_bytes(raw)
        datasets.append(list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t")))
        sources.append({"file": filename, "sha256": hashlib.sha256(raw).hexdigest()})
    cases = make_cases(*datasets)
    raw = json.dumps(cases, ensure_ascii=False, indent=2).encode()
    (OUT / "cases.json").write_bytes(raw)
    inputs = [{"id": row["id"], "messages": row["messages"]} for row in cases]
    (OUT / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    manifest = {"dataset": "masakhane/afrixnli", "revision": REVISION, "license": "apache-2.0",
                "split": "validation", "rows": len(cases), "parallel_pairs": len(cases) // 2,
                "sources": sources, "cases_sha256": hashlib.sha256(raw).hexdigest(),
                "label_mapping": LABELS, "train_eligible": False, "official_test_accessed": False}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "status"])
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint-100")
    parser.add_argument("--attempt", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--run-receipt", type=Path, default=OUT.parent / "run-receipt.json")
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare()
    import modal
    result_folder = args.run_receipt.parent / "afrixnli"
    result_folder.mkdir(parents=True, exist_ok=True)
    receipt_path = result_folder / (args.checkpoint.replace("/", "-") + (f"-attempt{args.attempt}" if args.attempt > 1 else "") + ".receipt.json")
    if args.action == "submit":
        if receipt_path.exists():
            raise ValueError("Already submitted; inspect existing receipt")
        run_id = json.loads(args.run_receipt.read_text())["run_id"]
        candidate = modal.Function.from_name("ghana-response-v2-private", "inspect_checkpoint").remote(run_id, args.checkpoint)
        inputs = json.loads((OUT / "inputs.json").read_text())
        import importlib.util
        spec = importlib.util.spec_from_file_location("response_runtime", ROOT / "modal/response_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        for row in inputs:
            runtime.validate_input(row["messages"], runtime.model_spec(run_id)["variants"][0], candidate["run_id"], candidate["checkpoint"], candidate["adapter_sha256"])
        manifest = json.loads((OUT / "manifest.json").read_text())
        receipt = {"candidate": candidate, "manifest": manifest, "state": "submitting"}
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        call = modal.Function.from_name("ghana-semantic-pair-evaluation", "evaluate").spawn(
            {k: candidate[k] for k in ("run_id", "checkpoint", "adapter_sha256")}, inputs, args.attempt)
        receipt.update(call_id=call.object_id, state="submitted")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    else:
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("call_id"):
            raise ValueError("Uncertain submission; inspect remote state first")
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=5)
        except TimeoutError:
            print(json.dumps({"state": "pending", "call_id": receipt["call_id"]}))
            return
        except Exception as exc:
            receipt.update(state="failed", error_type=type(exc).__name__)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise
        path = receipt_path.with_suffix(".results.json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        report = score(json.loads((OUT / "cases.json").read_text()), result["results"])
        path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
