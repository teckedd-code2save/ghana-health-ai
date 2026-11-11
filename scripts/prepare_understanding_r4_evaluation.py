#!/usr/bin/env python3
"""Prepare a private R4 evaluation contract; never import inference or emit prompts."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

import training_data_inputs as data

VERSION = "r4-evaluation-plan-v1"
R3_EVALUATION_SHA = "413c6e3b7e5737ffa3360219202bfb4e8e7a86437eb0d84dc9e2c72385612a06"
R2_PACK_SHA = "1e632f4d7382d6b7db910973a83d2483d4e1f843659335951fe59f81bd974084"
R3_BUNDLE_SHA = "62fc8df677147e175fb4f22c63d8a78281cebef9f1f0fc57991bdbf726091af4"
R3_RESULTS_SHA = "59cdb4b14becb42363a7645cf0915850639bd136d3229cadd409c2c5c7566951"
R3_ADAPTER_SHA = "5cc0537b21674231b640ae9109a08ffa6fab426c80ac1969705dfac9c9e6b874"
HISTORY_POLICY = "independent_model_histories_v1"
GUARD_POLICY = "repeat-block-v1"


def bind(sources, path):
    data.require(path.is_file() and not path.is_symlink(), "Missing or linked input: " + str(path))
    ref = {"path": str(path.resolve()), "sha256": data.sha(path)}
    sources[ref["path"]] = ref["sha256"]
    return ref


def package(sources, root, expected, field="artifacts", name="manifest.json"):
    manifest_path = data.safe_path(root, name)
    data.require(data.sha(manifest_path) == expected, "Pinned manifest changed: " + str(manifest_path))
    bind(sources, manifest_path)
    manifest = data.read(manifest_path)
    data.require(isinstance(manifest.get(field), dict) and manifest[field], "Missing artifact inventory")
    for relative, checksum in manifest[field].items():
        p = data.safe_path(root, relative)
        data.require(data.valid_hash(checksum) and data.sha(p) == checksum, "Changed artifact: " + relative)
        bind(sources, p)
    return manifest


def literal(path, name):
    values = [ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    data.require(len(values) == 1, "Missing or ambiguous policy constant: " + name)
    return values[0]


def rubric():
    return {
        "status": "pending_native_reviewer_calibration", "ratings": [],
        "dimensions": {
            "meaning_preservation": "Preserve supplied facts, entities, negation, time, uncertainty and experiencer; no invented facts or completed actions.",
            "actionable_next_step": "Answer the current request with a feasible grounded next step, or ask a necessary clarification; a short correct answer can fully satisfy a task.",
            "context_correction_topic_switch": "Apply latest corrections, retain still-relevant constraints and follow a requested topic switch without importing irrelevant prior claims.",
            "language_adherence": "Follow the requested language, allowing appropriate names, quotations and requested code-switching; assess dialect and naturalness separately.",
            "uncertainty_grounding": "Express uncertainty proportionately, distinguish known from unknown and avoid unsupported identity, diagnosis or action claims."
        },
        "scale": {"0": "Material failure with cited output evidence", "1": "Partial success with a stated omission or error", "2": "Satisfies the dimension in context", "not_applicable": "Dimension not elicited", "unable_to_judge": "Insufficient language or domain competence; not a pass"},
        "calibration": {"minimum_independent_proficient_reviewers": 2,
            "reviewer_fields": ["reviewer_id", "language_proficiency", "dialect", "domain_scope", "conflicts"],
            "procedure": "Calibrate on separately designated development outputs, not new final-test prompts. Freeze rubric and adjudication rules before candidate scoring. Blind checkpoint labels and randomize paired side order. Preserve original ratings and disagreements.",
            "adjudication": "Resolve material disagreements with a third qualified reviewer; report agreement, unresolved counts and denominators, not an invented reliability score.",
            "thresholds": None, "calibration_cases": []},
        "required_rating_fields": ["case_id", "checkpoint_sha256", "history_policy", "runtime_mode", "messages_sha256", "output_sha256", "dimension", "rating", "evidence_span", "rationale", "reviewer_id"],
        "forbidden_semantic_proxies": ["word_count", "reply_length", "keyword_or_phrase_presence", "EOS", "strict_JSON", "perplexity", "absence_of_repetition"],
        "clinical_approval": False,
        "clinical_policy": "Health-content interpretation requires separately qualified review; this contract does not certify clinical safety."
    }


def prepare(r3_evaluation, r2_pack, r3_bundle, r3_results, comparison_tools, out):
    roots = [Path(p) for p in (r3_evaluation, r2_pack, r3_bundle, r3_results, comparison_tools)]
    data.require(all(p.is_dir() and not p.is_symlink() for p in roots), "Use existing non-linked input directories")
    roots = [p.resolve() for p in roots]
    r3_evaluation, r2_pack, r3_bundle, r3_results, comparison_tools = roots
    out = Path(out).absolute()
    data.require(not out.exists() and not out.is_symlink(), "Use a fresh plan directory")
    data.require(not any(out.resolve().is_relative_to(p) for p in roots), "Output must not be inside an input directory")
    sources = {}
    evaluation = package(sources, r3_evaluation, R3_EVALUATION_SHA)
    protection = package(sources, r2_pack, R2_PACK_SHA)
    bundle = package(sources, r3_bundle, R3_BUNDLE_SHA, "files")
    results = package(sources, r3_results, R3_RESULTS_SHA, "files", "results-manifest.json")
    data.require(evaluation["kind"] == "development_not_blind_final" and evaluation["training_eligible"] is False,
                 "R3 development classification changed")
    data.require(results["binding"]["evaluation_sha256"] == R3_EVALUATION_SHA
                 and results["binding"]["inference_sha256"] == R3_BUNDLE_SHA, "R3 results are not the pinned baseline")
    adapter = bundle["variants"]["r3_response"]
    data.require(adapter["files"]["adapter_model.safetensors"] == R3_ADAPTER_SHA
                 and bundle["private"] is True and bundle["training"] is False, "R3 baseline identity changed")
    data.require(bundle["decoding"] == evaluation["decoding"], "Baseline decoding differs from development protocol")
    required_eval = {"cases.json", "rollouts.json", "held-content.jsonl", "language-heldout.json", "language-loss.json"}
    required_protection = {"protection/held-content.jsonl", "protection/split-ledger.jsonl",
                          "protection/snapshot/manifest.json", "protection/snapshot-witnesses.jsonl"}
    data.require(required_eval <= evaluation["artifacts"].keys(), "Incomplete R3 protected evaluation")
    data.require(required_protection <= protection["artifacts"].keys(), "Incomplete existing protection")
    snapshot_root = r2_pack / "protection/snapshot"
    snapshot = package(sources, snapshot_root, protection["snapshot_manifest_sha256"])
    data.require(snapshot["training_authorized"] is False, "Protection snapshot is not an admission")
    data.require(data.sha(r2_pack / "protection/split-ledger.jsonl") == snapshot["artifacts"]["split-ledger.jsonl"],
                 "Protection split ledger differs")
    cases, rollouts = data.read(r3_evaluation / "cases.json"), data.read(r3_evaluation / "rollouts.json")
    case_ids, rollout_ids = [r["id"] for r in cases], [r["id"] for r in rollouts]
    counts = evaluation["counts"]
    data.require(len(set(case_ids)) == len(cases) == counts["fixed_prompts"] and cases, "Fixed inventory changed")
    data.require(len(set(rollout_ids)) == len(rollouts) == counts["rollout_scenarios"] and rollouts, "Rollout inventory changed")
    data.require(sum(len(r["user_turns"]) for r in rollouts) == counts["rollout_user_turns"], "Rollout turn count differs")
    pointers = [r for _, r in data.rows(data.safe_path(snapshot_root, "records.jsonl"))]
    data.require(all(r["split"] in {"train", "validation", "test"} for r in pointers), "Unknown protected split")
    held_counts = Counter((r["collection"], r["split"]) for r in pointers if r["split"] != "train")
    comparison_path = data.safe_path(comparison_tools, "r2_comparison.py")
    guard_path = data.safe_path(comparison_tools, "response_loop_guard.py")
    data.require(literal(comparison_path, "HISTORY_POLICY") == HISTORY_POLICY, "Use own-history comparison tools, not legacy shared histories")
    data.require(literal(guard_path, "VERSION") == GUARD_POLICY, "Guard policy changed")
    tools = {name: bind(sources, data.safe_path(comparison_tools, name)) for name in
             ("r2_comparison.py", "response_loop_guard.py", "probe_independent_r3_comparison.py")}
    bind(sources, Path(__file__).resolve())
    bind(sources, Path(data.__file__).resolve())
    refs = lambda root, names: {name: bind(sources, data.safe_path(root, name)) for name in sorted(names)}
    contract = {
        "version": VERSION, "private": True, "plan_only": True, "launchable": False,
        "training_eligible": False, "model_calls": 0, "training_executed": False, "results": None,
        "baseline": {"variant": "r3_response", "immutable": True, "adapter_sha256": R3_ADAPTER_SHA,
            "inference_manifest_sha256": R3_BUNDLE_SHA, "historical_results_manifest_sha256": R3_RESULTS_SHA,
            "model": bundle["model"], "revision": bundle["revision"], "base_sha256": bundle["base_sha256"],
            "runtime": bundle["runtime"], "decoding": bundle["decoding"],
            "recheck_actual_weight_bytes_before_execution": True,
            "historical_results_policy": "Never overwrite R3 evidence; historical raw results are not guarded or newly paired candidate results."},
        "candidate": {"variant": "r4_candidate", "status": "unbound", "checkpoint_sha256": None,
            "required_before_execution": ["immutable checkpoint and tokenizer/runtime binding", "completed training receipt", "source-group and exact/normalized overlap audit against all protected inputs"]},
        "development": {"kind": "development_not_blind_final", "cases": refs(r3_evaluation, required_eval),
            "case_ids": case_ids, "rollout_ids": rollout_ids, "counts": counts,
            "mutations_permitted": False, "case_replacement_after_results": False},
        "protection": {"manifest": bind(sources, snapshot_root / "manifest.json"),
            "files": refs(r2_pack, required_protection), "snapshot_inventory": bind(sources, snapshot_root / "records.jsonl"),
            "held_inventory_counts": [{"collection": c, "split": s, "records": n} for (c, s), n in sorted(held_counts.items())],
            "policy": "Preserve every existing validation/test label and all connected source/group versions. All R3 development cases, rollouts, held content and loss inputs remain training-ineligible. Never turn observed failures into training data without separate disjoint source-backed material.",
            "training_overlap_clearance": "not_performed_no_R4_training_inputs_supplied",
            "raw_holdout_content_exported": False},
        "comparisons": {
            "checkpoints": ["r3_response", "r4_candidate"],
            "history_modes": {
                "fixed_context": "Use identical original messages and native prompt tokens across checkpoints and modes; preserve existing system messages. No injected wrappers or rewritten replies.",
                "own_history_rollout": "Use identical scripted user turns; each checkpoint/runtime cell retains only its own exact generated assistant replies. Later prompt hashes may differ. No candidate-history leakage, cross-cell sharing or silent repair."},
            "runtime_modes": {
                "raw": {"guard": None, "fallback": False, "reply_rewrite": False},
                "guarded": {"guard_policy": GUARD_POLICY, "guard_source": tools["response_loop_guard.py"],
                    "same_policy_for_both_checkpoints": True, "fallback": False, "reply_rewrite": False,
                    "trigger_state": "paused_not_complete", "interpretation": "Stopping repetition is a runtime intervention, not a semantic model improvement."}},
            "cells": [{"history_mode": h, "runtime_mode": m} for h in ("fixed_context", "own_history_rollout") for m in ("raw", "guarded")],
            "pooling": "Report each checkpoint/history/runtime cell separately; never pool guarded and raw quality or fixed-context and own-history results.",
            "continuation": "Retain verifiable valid-Unicode partial replies and their finish reason; no retries. Invalid Unicode, missing/failed evidence, error or cancellation blocks later turns in that cell. Record blocked slots; never synthesize continuation.",
            "guarded_continuation": "A verified guard pause may continue with its exact observed prefix; keep the trigger and paused status distinct from EOS.",
            "tools_provenance": tools},
        "measurement": {
            "semantic": rubric(),
            "mechanical": {"strict_json": "Existing literal tasks only; schema/JSON success is not meaning correctness.",
                "within_reply_repetition": "Repeated four-word sequences and repeated passages, with generated-reply denominators; diagnostic only, inspect whether repetition is harmful.",
                "cross_turn_repetition": "Consecutive identical replies and repeated passages across turns; report eligible adjacent-pair denominators and human contextual judgments.",
                "completion": "Count complete, incomplete, paused, cancelled, error and blocked slots separately; completion is not task success."},
            "language_retention": {"twi": "Reuse immutable R3 language loss inputs with identical native tokenization and token-weighted NLL/perplexity.",
                "english": {"source": bind(sources, snapshot_root / "records.jsonl"),
                    "validation_records": held_counts[("english_retention", "validation")],
                    "protected_test_records": held_counts[("english_retention", "test")],
                    "selected_test_records": 0, "status": "validation_protocol_pending_binding",
                    "protocol": "Use only existing protected English validation witnesses for token-weighted NLL/perplexity; bind exact IDs, native token counts and masks before running. Retain existing English development fixed/rollout tasks, with human rubric. Do not relabel validation as final test or consume test records."}},
            "reporting": "Preserve scheduled, attempted, generated, completed and human-rated denominators, unreviewed and N/A counts, per-language/cohort results and all failures. Paired deltas require comparable inputs/policies; a sparse development set is not population reliability.",
            "latency": "Cold/warm and transport-inclusive timings are operational diagnostics; no uncontrolled base/candidate latency benchmark."},
        "release": {"status": "research_only_no_promotion", "thresholds": None,
            "required_before_claims": ["predeclared calibrated native-review rubric and thresholds", "paired results with critical-error and regression review", "English retention review", "separate qualified clinical review for health claims"],
            "automatic_winner": False, "runtime_guard_is_semantic_gate": False},
        "final_test": {"status": "empty_pending_native_review", "cases": [], "prompts_authored": 0,
            "execution_authorized": False, "existing_test_records_remain_protected": True,
            "requirements": "Native reviewers and owner must seal a separate, source-group-disjoint final-test protocol before model access; do not expose prompts to training or mine development failures into final tests."}
    }
    # Recheck after all reads, before any output: concurrent edits cannot create a mixed contract.
    data.require(all(data.sha(Path(p)) == checksum for p, checksum in sources.items()), "Input changed during preparation")
    out.mkdir(parents=True, exist_ok=False)
    data.save(out / "evaluation-contract.json", contract)
    data.save(out / "provenance.json", {"version": VERSION, "prepared_at": datetime.now(timezone.utc).isoformat(),
        "input_files": sources, "input_content_copied": False, "model_loaded": False, "network_calls": 0})
    data.save(out / "manifest.json", {"version": VERSION, "launchable": False,
        "artifacts": {name: data.sha(out / name) for name in ("evaluation-contract.json", "provenance.json")}})
    data.save(out / "PREPARED.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return contract


def verify(out):
    out = Path(out).resolve()
    data.require(data.read(out / "PREPARED.json") == {"manifest_sha256": data.sha(out / "manifest.json")}, "Plan seal changed")
    manifest = data.read(out / "manifest.json")
    data.require(manifest["version"] == VERSION and manifest["launchable"] is False, "Not a plan-only contract")
    for name, checksum in manifest["artifacts"].items():
        data.require(data.sha(data.safe_path(out, name)) == checksum, "Plan artifact changed")
    provenance = data.read(out / "provenance.json")
    data.require(all(data.sha(Path(p)) == checksum for p, checksum in provenance["input_files"].items()), "Bound source changed")
    contract = data.read(out / "evaluation-contract.json")
    data.require(contract["baseline"]["adapter_sha256"] == R3_ADAPTER_SHA and contract["launchable"] is False
                 and contract["results"] is None and contract["final_test"]["cases"] == [], "Plan boundary changed")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("r3-evaluation", "r2-pack", "r3-bundle", "r3-results", "comparison-tools", "out"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    sub.add_parser("verify").add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.out) if args.command == "verify" else prepare(
        args.r3_evaluation, args.r2_pack, args.r3_bundle, args.r3_results, args.comparison_tools, args.out)
    print(json.dumps({"version": VERSION, "launchable": result["launchable"],
        "development_counts": result["development"]["counts"], "final_test_cases": len(result["final_test"]["cases"])}))


if __name__ == "__main__":
    main()
