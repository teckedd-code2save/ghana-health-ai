#!/usr/bin/env python3
"""Freeze source-audited developer probes and score recorded paired replies. CPU only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata

from training_data_inputs import (digest, iter_snapshot_records, parse_json, read, require,
                                 rows, safe_path, save, sha, valid_hash, verify_snapshot)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "understanding-evaluation-v1"
DIMENSIONS = ("meaning", "relevance", "requested_language", "naturalness", "grounding")
LIMITS = [
    "Synthetic developer challenge, not an untouched final test or representative population sample.",
    "No native-human approval is inferred. Case rubrics and Twi naturalness still need review.",
    "Exact and normalized whole-text matching does not establish semantic or near-duplicate independence.",
    "Upstream base-model pretraining exposure is unknown. Historical cases are development-only.",
    "Only selected release files, their embedded text fields, and the supplied run witnesses are scanned; not the entire raw inventory.",
    "Fixed assistant history tests response to supplied context, not end-to-end generated conversation quality.",
    "This is neither a clinical assessment nor a tool-execution benchmark. No tools are connected.",
]


def normalize(text):
    """Retain Twi letters, digits and negation; ignore case, spacing and punctuation."""
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                            else " " for c in value).split())


def fingerprints(text):
    return {"exact": hashlib.sha256(text.encode()).hexdigest(),
            "normalized": hashlib.sha256(normalize(text).encode()).hexdigest()}


def strings(value, path="$", depth=0):
    """Include source_text inside JSON chat envelopes, not just the outer prompt."""
    if isinstance(value, str):
        if value.strip():
            yield path, value
        if depth < 3 and value.lstrip().startswith(("{", "[")):
            try:
                decoded = parse_json(value)
            except (ValueError, RecursionError):
                return
            yield from strings(decoded, path + ".decoded", depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, path + "." + key, depth)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f"{path}[{index}]", depth)


def validate_cases(cases):
    require(isinstance(cases, list) and cases, "Empty cases")
    seen, families = set(), {}
    for case in cases:
        cid = case["id"]
        require(isinstance(cid, str) and cid and cid not in seen, "Repeated/empty case ID")
        seen.add(cid)
        require(case["split"] in {"development", "challenge"}, "Only development and challenge splits")
        require(families.setdefault(case["family"], case["split"]) == case["split"], "Family crosses eval splits")
        messages = case["messages"]
        require(isinstance(messages, list) and messages, "Missing complete history")
        start = int(messages[0].get("role") == "system")
        require(len(messages) > start and (len(messages) - start) % 2 == 1, "History must end in user")
        for i, message in enumerate(messages):
            role = "system" if start and i == 0 else "user" if (i - start) % 2 == 0 else "assistant"
            require(set(message) == {"role", "content"} and message["role"] == role
                    and isinstance(message["content"], str) and message["content"].strip(), "Invalid complete history")
        checks = case["expected_meaning_checks"]
        require(isinstance(checks, list) and checks and all(isinstance(c, str) and c.strip() for c in checks), "Meaning checks required")
        require(case["human_review"]["status"] == "pending", "Do not infer human approval")
        obj = case["objective"]
        require(obj is None or (set(obj) == {"type", "expected"} and obj["type"] == "json_exact"
                               and isinstance(obj["expected"], dict) and obj["expected"]), "Unknown objective rule")


def load_cases(spec, root):
    require(spec["version"] == 1 and spec["human_review"]["status"] == "pending", "Unknown or overclaimed suite")
    require(spec["protocol"]["history_mode"] == "fixed_complete_history"
            and spec["protocol"]["tools"] == [], "Only fixed-history, no-tool evaluation")
    require(spec["protocol"]["decoding"] == {"do_sample": False, "num_beams": 1, "max_new_tokens": 256,
            "repetition_penalty": 1.0, "seed": 17, "enable_thinking": False, "truncate": False}, "Unknown decoding protocol")
    cases = []
    for case in spec["cases"]:
        require(case["split"] == "challenge", "New authoring belongs in challenge")
        cases.append({**case, "human_review": spec["human_review"], "provenance": {
            "origin": "ai_authored_synthetic_developer_probe", "source_id": spec["suite_id"] + ":" + case["id"],
            "source_derived": False, "human_authored": False, "training_eligible": False}})
    history = spec["historical"]
    pack = safe_path(root, history["pack"])
    require(sha(pack / "manifest.json") == history["manifest_sha256"], "Historical manifest changed")
    manifest = read(pack / "manifest.json")
    for filename, field in (("cases.json", "cases_sha256"), ("review.json", "review_sha256")):
        require(sha(pack / filename) == history[field] == manifest["artifacts"][filename], "Historical cases/review changed")
    original = {c["id"]: c for c in read(pack / "cases.json")}
    rubrics = {c["id"]: c for c in read(pack / "review.json")}
    require(len(set(history["selection"])) == len(history["selection"]), "Repeated historical selection")
    for cid in history["selection"]:
        row, rubric = original[cid], rubrics[cid]["rubric"]
        require("messages" in row and "turns" not in row, "Do not invent missing rollout history")
        cases.append({"id": "dev-" + cid, "family": "historical-" + cid, "split": "development",
            "input_language": "historical_see_messages", "response_language": "historical_see_rubric",
            "tags": ["historical_regression"], "messages": row["messages"], "objective": None,
            "expected_meaning_checks": rubric if isinstance(rubric, list) else [rubric],
            "human_review": spec["human_review"], "provenance": {"origin": "historical_project_fixture",
            "historical_id": cid, "pack_manifest_sha256": history["manifest_sha256"],
            "source_id": history["pack"] + ":" + cid, "prior_development_exposure": True,
            "training_eligible": False}})
    validate_cases(cases)
    return cases


def eval_units(cases):
    for case in cases:
        for i, message in enumerate(case["messages"]):
            yield case, f"messages[{i}].content", message["content"]
        if case["objective"]:
            # Full expected objects, not ubiquitous scalar values such as true or a name.
            yield case, "objective.expected", json.dumps(case["objective"]["expected"], ensure_ascii=False, sort_keys=True)


def overlap_audit(cases, source_rows):
    index, matches, cross = defaultdict(list), {}, set()
    for case, field, text in eval_units(cases):
        unit = {"case_id": case["id"], "split": case["split"], "field": field}
        for method, fp in fingerprints(text).items():
            for earlier in index[method, fp]:
                if earlier["split"] != case["split"]:
                    cross.add((earlier["case_id"], earlier["field"], case["id"], field, method))
            index[method, fp].append(unit)
    counts, ids = Counter(), set()
    for ref, row in source_rows:
        counts[ref["scope"]] += 1
        sid = row.get("source_id", row.get("id", ""))
        ids.add(sid)
        for field, text in strings(row):
            for method, fp in fingerprints(text).items():
                for unit in index.get((method, fp), []):
                    key = (unit["case_id"], unit["field"], method)
                    item = matches.setdefault(key, {**unit, "method": method, "fingerprint": fp,
                                                   "occurrences": 0, "witnesses": []})
                    item["occurrences"] += 1
                    if len(item["witnesses"]) < 3:
                        item["witnesses"].append({**ref, "source_id": sid, "field": field})
    identity_hits = [c["id"] for c in cases if c["split"] == "challenge" and c["provenance"]["source_id"] in ids]
    hits = sorted(matches.values(), key=lambda v: (v["case_id"], v["field"], v["method"]))
    return {"source_rows_scanned": dict(counts), "unique_source_ids_scanned": len(ids),
        "matches": hits, "cross_eval_split_matches": [list(v) for v in sorted(cross)],
        "challenge_source_identity_matches": identity_hits,
        "challenge_exact_source_overlap_free": not any(v["split"] == "challenge" for v in hits) and not identity_hits,
        "cross_eval_split_overlap_free": not cross, "near_duplicate_certification": False,
        "prior_base_model_exposure": "unknown", "limits": LIMITS}


def source_rows(snapshot, run_pack):
    # Validate every selected pointer before scanning its hash-bound original release files.
    for _ in iter_snapshot_records(snapshot):
        pass
    for batch in read(snapshot / "batches.json"):
        for name in sorted(batch["files"]):
            for line, row in rows(safe_path(Path(batch["release_path"]), name)):
                yield {"scope": "selected_release_files", "batch_id": batch["batch_id"], "file": name,
                       "line": line, "file_sha256": batch["files"][name]}, row
    for line, row in rows(run_pack / "source-witnesses.jsonl"):
        yield {"scope": "run_witnesses", "file": "source-witnesses.jsonl", "line": line}, row


def freeze(spec_path, snapshot, run_pack, out, root=ROOT):
    spec = read(spec_path)
    verify_snapshot(snapshot)
    train_manifest = read(run_pack / "manifest.json")
    require(train_manifest["snapshot_manifest_sha256"] == sha(snapshot / "manifest.json"), "Run/snapshot mismatch")
    witness_sha = sha(run_pack / "source-witnesses.jsonl")
    require(witness_sha == train_manifest["artifacts"]["source-witnesses.jsonl"], "Run witness changed")
    bindings = {"spec_sha256": sha(spec_path), "snapshot_manifest_sha256": sha(snapshot / "manifest.json"),
                "run_manifest_sha256": sha(run_pack / "manifest.json"),
                "run_source_witnesses_sha256": witness_sha}
    if out.exists():
        manifest, _ = load_pack(out)
        require(manifest["bindings"] == bindings, "Existing pack differs; choose a new output version")
        return manifest
    cases = load_cases(spec, root)
    legacy_pack = safe_path(root, spec["historical"]["pack"])
    legacy = read(legacy_pack / "cases.json")
    # Rollout user turns are leakage witnesses, never invented fixed histories.
    rollout_witnesses = [{"id": "dev-" + c["id"], "split": "development", "objective": None,
        "messages": [{"content": turn} for turn in c["turns"]]} for c in legacy if "turns" in c]
    audit = overlap_audit(cases + rollout_witnesses, source_rows(snapshot, run_pack))
    require(audit["source_rows_scanned"]["run_witnesses"] == train_manifest["selected_rows"], "Run witness count mismatch")
    ready = audit["challenge_exact_source_overlap_free"] and audit["cross_eval_split_overlap_free"]
    out.mkdir(parents=True, exist_ok=False)
    save(out / "spec.json", spec)
    save(out / "cases.json", cases)
    save(out / "prompts.json", [{"id": c["id"], "messages": c["messages"]} for c in cases])
    save(out / "legacy-development.json", {"split": "development", "final_test": False,
        "cases": read(legacy_pack / "cases.json"), "review": read(legacy_pack / "review.json"),
        "manifest_sha256": spec["historical"]["manifest_sha256"],
        "protocol_note": "All 16 original cases preserved. Ten fixed-history cases are reused in prompts.json; six actual-history rollouts retain their original evaluator and are not mixed into fixed-history scores."})
    save(out / "overlap-audit.json", audit)
    manifest = {"version": VERSION, "suite_id": spec["suite_id"], "bindings": bindings,
        "code_sha256": sha(__file__), "snapshot_verifier_sha256": sha(ROOT / "scripts/training_data_inputs.py"),
        "protocol": spec["protocol"], "model": train_manifest["model"], "revision": train_manifest["revision"],
        "counts": dict(Counter(c["split"] for c in cases)),
        "objective_cases": sum(c["objective"] is not None for c in cases),
        "human_review": spec["human_review"], "ready_for_developer_evaluation": ready,
        "final_test": False, "training_eligible": False, "limits": LIMITS,
        "artifacts": {name: sha(out / name) for name in ("spec.json", "cases.json", "prompts.json", "overlap-audit.json", "legacy-development.json")}}
    save(out / "manifest.json", manifest)
    return manifest


def load_pack(pack):
    manifest = read(pack / "manifest.json")
    require(manifest["version"] == VERSION and manifest["code_sha256"] == sha(__file__), "Changed evaluation implementation")
    require(set(manifest["artifacts"]) == {"spec.json", "cases.json", "prompts.json", "overlap-audit.json", "legacy-development.json"}, "Incomplete pack")
    for name, checksum in manifest["artifacts"].items():
        require(sha(pack / name) == checksum, "Changed evaluation artifact: " + name)
    require(manifest["ready_for_developer_evaluation"] is True, "Evaluation blocked by overlap; inspect audit")
    cases = read(pack / "cases.json")
    validate_cases(cases)
    require(read(pack / "prompts.json") == [{"id": c["id"], "messages": c["messages"]} for c in cases], "Prompt/rubric separation failed")
    return manifest, cases


def parse_run(value, manifest, cases, manifest_sha, role):
    require(value["version"] == 1 and value["role"] == role, "Invalid result role/version")
    require(value["suite_manifest_sha256"] == manifest_sha, "Results belong to another evaluation")
    require(isinstance(value["run_id"], str) and value["run_id"], "Run ID required")
    require(value["decoding"] == manifest["protocol"]["decoding"], "Decoding differs from frozen protocol")
    model = value["model"]
    require(model["id"] == manifest["model"] and model["revision"] == manifest["revision"], "Unpaired base checkpoint")
    require(valid_hash(model["weights_sha256"]), "Weight identity required")
    require(model["adapter_sha256"] is None if role == "base" else valid_hash(model["adapter_sha256"]), "Adapter identity missing/unexpected")
    require(set(value["runtime"]) == {"environment_sha256", "tokenizer_sha256", "chat_template_sha256"}
            and all(valid_hash(v) for v in value["runtime"].values()), "Pinned runtime/tokenizer/template required")
    expected = {c["id"]: c for c in cases}
    records = {}
    for row in value["records"]:
        cid = row["case_id"]
        require(cid in expected and cid not in records, "Unknown/duplicate result case")
        require(row["messages_sha256"] == digest(expected[cid]["messages"]), "Changed model-visible messages")
        require(valid_hash(row["prompt_token_ids_sha256"]), "Native prompt token fingerprint required")
        require(isinstance(row["output"], str) and row["finish_reason"] in {"eos", "max_new_tokens", "time_limit", "error"}, "Invalid output/finish reason")
        require(type(row["seconds"]) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] >= 0, "Invalid timing")
        require(row["finish_reason"] != "eos" or bool(row["output"].strip()), "Empty successful reply")
        records[cid] = row
    require(set(records) == set(expected), "Missing results: record errors rather than dropping cases")
    return records


def objective_score(case, row):
    if case["objective"] is None:
        return None
    try:
        actual = parse_json(row["output"])
        matched = digest(actual) == digest(case["objective"]["expected"])
        parse_error = None
    except (ValueError, RecursionError) as error:
        matched, parse_error = False, type(error).__name__
    return {"literal_match": matched, "passed": matched and row["finish_reason"] == "eos",
            "parse_error": parse_error, "human_semantic_approval": False}


def review_template(pack, base, adapter):
    manifest, cases = load_pack(pack)
    case_list = []
    for role, run in (("base", base), ("adapter", adapter)):
        result = parse_run(run, manifest, cases, sha(pack / "manifest.json"), role)
        for case in cases:
            row = result[case["id"]]
            case_list.append({"case_id": case["id"], "run_id": run["run_id"], "output_sha256": digest(row["output"]),
                "status": "pending", "reviewer": "", "reviewer_kind": "human", "twi_proficiency": "",
                "case_rubric_valid": None, "checks": [None] * len(case["expected_meaning_checks"]),
                "ratings": {dimension: None for dimension in DIMENSIONS}, "notes": ""})
    return {"version": 1, "suite_manifest_sha256": sha(pack / "manifest.json"), "reviews": case_list}


def validate_reviews(value, cases, runs, manifest_sha):
    if value is None:
        return {}
    require(value["version"] == 1 and value["suite_manifest_sha256"] == manifest_sha, "Review suite mismatch")
    known, result = {c["id"]: c for c in cases}, {}
    for item in value["reviews"]:
        key = (item["run_id"], item["case_id"])
        require(key not in result and key[0] in runs and key[1] in known, "Duplicate/unknown review")
        require(item["output_sha256"] == digest(runs[key[0]][key[1]]["output"]), "Review belongs to another output")
        require(item["status"] in {"pending", "completed"}, "Invalid review status")
        if item["status"] == "completed":
            require(item["reviewer_kind"] == "human" and isinstance(item["reviewer"], str) and item["reviewer"].strip(), "Human reviewer attribution required")
            require(item["twi_proficiency"] in {"native", "fluent", "not_assessed"}, "Reviewer proficiency required")
            require(type(item["case_rubric_valid"]) is bool, "Rubric validity must be assessed")
            require(len(item["checks"]) == len(known[key[1]]["expected_meaning_checks"])
                    and all(type(v) is bool for v in item["checks"]), "Complete meaning checks required")
            require(set(item["ratings"]) == set(DIMENSIONS) and all(type(v) is int and 0 <= v <= 2
                    for v in item["ratings"].values()), "Ratings must be 0/1/2, not automatic confidence")
        result[key] = item
    return result


def compare(pack, base, adapter, reviews=None):
    manifest, cases = load_pack(pack)
    manifest_sha = sha(pack / "manifest.json")
    br = parse_run(base, manifest, cases, manifest_sha, "base")
    ar = parse_run(adapter, manifest, cases, manifest_sha, "adapter")
    require(base["run_id"] != adapter["run_id"], "Distinct run IDs required")
    require(base["runtime"] == adapter["runtime"] and base["model"]["weights_sha256"] == adapter["model"]["weights_sha256"], "Unpaired runtime or base weights")
    require(all(br[c["id"]]["prompt_token_ids_sha256"] == ar[c["id"]]["prompt_token_ids_sha256"] for c in cases), "Base/adapter native prompts differ")
    rated = validate_reviews(reviews, cases, {base["run_id"]: br, adapter["run_id"]: ar}, manifest_sha)
    details, summary = [], {}
    for split in ("development", "challenge"):
        selected = [c for c in cases if c["split"] == split]
        objective = [c for c in selected if c["objective"]]
        item = {"cases": len(selected), "families": len({c["family"] for c in selected}), "objective_cases": len(objective)}
        for role, run, recorded in (("base", base, br), ("adapter", adapter, ar)):
            passed = sum(objective_score(c, recorded[c["id"]])["passed"] for c in objective)
            completed = [rated[run["run_id"], c["id"]] for c in selected
                         if rated.get((run["run_id"], c["id"]), {}).get("status") == "completed"]
            valid = [r for r in completed if r["case_rubric_valid"]]
            item[role] = {"objective_passed": passed, "objective_rate": passed / len(objective) if objective else None,
                "eos_replies": sum(recorded[c["id"]]["finish_reason"] == "eos" for c in selected),
                "human_completed": len(completed), "human_invalid_rubrics": len(completed) - len(valid),
                "human_valid_ratings": len(valid), "human_rating_means": {d: sum(r["ratings"][d] for r in valid) / len(valid)
                    if valid else None for d in DIMENSIONS}}
        paired = []
        for case in selected:
            b, a = (rated.get((run["run_id"], case["id"])) for run in (base, adapter))
            if all(r and r["status"] == "completed" and r["case_rubric_valid"] for r in (b, a)):
                paired.append((b, a))
        item["paired_human_cases"] = len(paired)
        item["paired_human_mean_delta"] = {d: sum(a["ratings"][d] - b["ratings"][d] for b, a in paired) / len(paired)
                                           if paired else None for d in DIMENSIONS}
        item["objective_delta"] = item["adapter"]["objective_rate"] - item["base"]["objective_rate"] if objective else None
        summary[split] = item
    for case in cases:
        details.append({"case_id": case["id"], "split": case["split"], "family": case["family"],
            "expected_meaning_checks": case["expected_meaning_checks"], "messages": case["messages"],
            **{role: {**result[case["id"]], "objective": objective_score(case, result[case["id"]]),
                      "review": rated.get((run["run_id"], case["id"]))}
               for role, run, result in (("base", base, br), ("adapter", adapter, ar))}})
    return {"version": VERSION, "suite_manifest_sha256": manifest_sha, "base_run_sha256": digest(base),
            "adapter_run_sha256": digest(adapter), "reviews_sha256": digest(reviews) if reviews else None,
            "summary": summary, "cases": details, "limits": LIMITS,
            "claim": "Descriptive developer comparison only; no clinical, population, gold or final-test certification."}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("freeze")
    build.add_argument("--spec", type=Path, default=ROOT / "data/eval/understanding-demo-v1.json")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--run-pack", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--pack", type=Path, required=True)
    for command in ("score", "review-template"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--pack", type=Path, required=True)
        cmd.add_argument("--base", type=Path, required=True)
        cmd.add_argument("--adapter", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        if command == "score":
            cmd.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "freeze":
            value = freeze(args.spec, args.snapshot, args.run_pack, args.out)
            print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"), "counts": value["counts"],
                              "ready_for_developer_evaluation": value["ready_for_developer_evaluation"]}, indent=2))
            return 0 if value["ready_for_developer_evaluation"] else 2
        if args.action == "verify":
            manifest, cases = load_pack(args.pack)
            print(json.dumps({"verified": True, "cases": len(cases), "manifest_sha256": sha(args.pack / "manifest.json")}))
        else:
            base, adapter = read(args.base), read(args.adapter)
            value = review_template(args.pack, base, adapter) if args.action == "review-template" else compare(
                args.pack, base, adapter, read(args.reviews) if args.reviews else None)
            write_new(args.out, value)
            print(json.dumps({"output": str(args.out), "sha256": sha(args.out)}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
