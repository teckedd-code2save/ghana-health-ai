#!/usr/bin/env python3
"""Admit executed English sandbox trajectories; no Twi or live-action claim."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import prepare_r7_tool_candidates as candidates
import prepare_understanding_r7 as pack
import training_data_inputs as data
from check_afrique_training_contract import load_tokenizer
from r7_encoding import encode_complete
from select_understanding_r5_sources import bound_package, bound_read, recheck

VERSION = "r7-executed-english-tools-admission-v1"


def admitted_record(row, scenarios, source):
    record = deepcopy(row)
    scenario = scenarios[row["scenario_id"]]
    probes = {binding["assistant_call_index"] for binding in row["metadata"]["tool_bindings"]
              if scenario["steps"][int(binding["execution_id"].rsplit(":", 1)[1])]["mode"] == "validation_probe"}
    indices = [i for i, m in enumerate(record["messages"]) if m["role"] == "assistant" and i not in probes]
    data.require(indices and indices[-1] == len(record["messages"]) - 1, "Missing final response supervision")
    record.update(training_eligible=True, training_ready=True, admission_status="admitted",
                  private_research_only=True, production_eligible=False, human_reviewed=False)
    record["metadata"].update(training_assistant_indices=indices, context_only_assistant_indices=sorted(probes),
        meaning_review_status="deterministic_executor_and_result_renderer_verified",
        translation_status="english_only_no_twi_claim", admission_scope="executed_simulated_english_actions")
    record["original_ref"]["candidate_release"] = {"path": str(source.resolve()),
        "manifest_sha256": data.sha(source / "manifest.json"), "record_sha256": pack.run.digest(row)}
    return record


def prepare(source, protection, tokenizer_path, out):
    source, out = Path(source), Path(out)
    data.require(not out.exists(), "Use a new admission folder")
    verified = candidates.verify_package(source)
    pins = {}
    bound_package(source, pins, {"candidates/train.jsonl", "candidates/dev.jsonl", "scenarios.jsonl"}, "PREPARED.json")
    scenarios = {r["id"]: r for r in bound_read(source / "scenarios.jsonl", pins, rows=True)}
    train = bound_read(source / "candidates/train.jsonl", pins, rows=True)
    dev = bound_read(source / "candidates/dev.jsonl", pins, rows=True)
    held = []
    for row in dev:
        for i, text in enumerate(pack.content_views(row)):
            held.append({"id": row["id"] + ":prefix:" + str(i), "source_id": row["source_id"],
                "group_id": row["source_group_id"], "text": text, "training_eligible": False,
                "original_ref": {"candidate_manifest_sha256": verified["manifest_sha256"],
                    "record_sha256": pack.run.digest(row)}})
    for folder in protection:
        folder = Path(folder)
        seals = [name for name in ("SEALED.json", "FROZEN.json", "PREPARED.json") if (folder / name).is_file()]
        data.require(len(seals) == 1, "Ambiguous protection release")
        bound_package(folder, pins, {"held-content.jsonl"}, seals[0])
        held.extend(bound_read(folder / "held-content.jsonl", pins, rows=True))
    held = list({r["id"]: r for r in held}.values())
    blocked = pack.protected_records(train, held)
    tokenizer = load_tokenizer(Path(tokenizer_path))
    admitted, decisions, measurements = [], [], []
    for row in train:
        decision = {"id": row["id"], "source_id": row["source_id"], "status": "excluded"}
        if row["id"] in blocked:
            decisions.append({**decision, "reason": "protected_source_or_history", "evidence": blocked[row["id"]]})
            continue
        record = admitted_record(row, scenarios, source)
        ref = {"dataset_id": VERSION, "dataset_version": VERSION, "record_sha256": pack.run.digest(record)}
        measured = []
        try:
            for view in pack.prefix_views(record, {}, ref):
                encoded = encode_complete(tokenizer, view, 4096)
                measured.append({"id": view["id"], "source_record_id": record["id"],
                    "assistant_index": view["metadata"]["r7_view"]["assistant_message_index"],
                    "sequence_tokens": len(encoded["input_ids"]),
                    "target_tokens": sum(t != -100 for t in encoded["labels"])})
        except ValueError as error:
            decisions.append({**decision, "reason": "native_encoding_rejected_no_truncation", "detail": str(error)})
            continue
        admitted.append(record)
        measurements.extend(measured)
        decisions.append({**decision, "status": "admitted", "reason": "executed_result_bound_english_sandbox"})
    data.require(admitted, "No admitted tool trajectories")
    recheck(pins)
    out.mkdir(parents=True)
    records_path = "datasets/tools/records.jsonl"
    (out / records_path).parent.mkdir(parents=True)
    data.write_rows(out / records_path, admitted)
    data.write_rows(out / "held-content.jsonl", held)
    data.write_rows(out / "admission-ledger.jsonl", decisions)
    data.write_rows(out / "token-measurements.jsonl", measurements)
    data.save(out / "source-verification.json", verified)
    dataset = {"dataset_id": "gha-r7-executed-tools-english", "dataset_version": VERSION,
        "catalog_dataset_id": "tools", "admission_status": "admitted", "private_research_only": True,
        "production_eligible": False, "record_count": len(admitted), "records_path": records_path,
        "records_sha256": data.sha(out / records_path), "language": "en", "synthetic": True,
        "scope": "Executed English sandbox only; no native Twi, external factual or clinical certification"}
    data.save(out / "datasets/tools/manifest.json", dataset)
    data.save(out / "source-manifests.json", [{"dataset_id": dataset["dataset_id"], "dataset_version": VERSION,
        "manifest_path": "datasets/tools/manifest.json", "manifest_sha256": data.sha(out / "datasets/tools/manifest.json")}])
    report = {"candidate_histories": len(train), "admitted_histories": len(admitted),
        "admitted_sources": len({r["source_id"] for r in admitted}), "supervised_prefixes": len(measurements),
        "context_only_probe_calls": sum(len(r["metadata"]["context_only_assistant_indices"]) for r in admitted),
        "families": dict(Counter(r["metadata"]["family"] for r in admitted)),
        "decisions": dict(Counter(r["reason"] for r in decisions)),
        "target_tokens": sum(r["target_tokens"] for r in measurements), "training_executed": False,
        "limitations": ["English only; Twi transfer unmeasured.", "Four simulated products, not broad commercial competence.",
            "No live search evidence or real orders.", "Shared templates; dev is not wording-held-out.",
            "Validation-probe calls are context only, never supervised targets."]}
    data.save(out / "summary.json", report)
    data.save(out / "manifest.json", {"version": VERSION, "admission_status": "admitted", "private_research_only": True,
        "production_eligible": False, "input_files": pins, "artifacts": {
            p.relative_to(out).as_posix(): data.sha(p) for p in out.rglob("*") if p.is_file()}})
    data.save(out / "SEALED.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    pack.source_release(out, {})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protection", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.protection, args.tokenizer, args.out), indent=2))


if __name__ == "__main__":
    main()
