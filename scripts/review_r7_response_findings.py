"""Build and replay an issue-only ledger over the sealed R6 response audit.

Decisions are individually authored after full-context inspection, never inferred
from a regex or a previous automatic classification. No model or network calls.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tmp/r7-20261117/response-target-audit-v1"
OUT = ROOT / "tmp/r7-20261117/response-findings-review-v2"
VERSION = "r7-response-findings-review-v2"
AUDIT_MANIFEST_SHA = "4f1bb7b508f9265fa0c91c4b27619d3e5edd25d499fdf1df1a7651d012f04bbd"
DECISIONS = {"clear_issue", "hold_unresolved_meaning", "hold_unsupported_claim"}
SCOPE = {
    "scope": "individual_repetition_or_claim_only",
    "reviewer": "codex_agent",
    "human_reviewed": False,
    "native_certified": False,
    "whole_translation_gold": False,
    "training_eligible": False,
    "training_ready": False,
    "production_eligible": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl(rows):
    return b"".join(encoded(row) + b"\n" for row in rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sealed_audit():
    require(file_sha(AUDIT / "manifest.json") == AUDIT_MANIFEST_SHA, "audit manifest drift")
    manifest = read_json(AUDIT / "manifest.json")
    require(read_json(AUDIT / "SEALED.json")["manifest_sha256"] == AUDIT_MANIFEST_SHA,
            "audit seal mismatch")
    inputs = {str(AUDIT / name): sha for name, sha in manifest["artifacts"].items()}
    inputs.update(manifest["inputs"]["files"])
    inputs[str(AUDIT / "manifest.json")] = AUDIT_MANIFEST_SHA
    inputs[str(AUDIT / "SEALED.json")] = file_sha(AUDIT / "SEALED.json")
    for path, sha in inputs.items():
        require(file_sha(path) == sha, f"sealed input drift: {path}")
    rows = load_rows(AUDIT / "record-audit.jsonl")
    originals = load_rows(manifest["inputs"]["histories"]["file"])
    require(len(rows) == len(originals) == 997, "whole 997 history accounting changed")
    for number, (row, original) in enumerate(zip(rows, originals), 1):
        require(row["original_line"] == number, "source line mismatch")
        require(row["original_record"] == original, "original history mismatch")
        require(digest(original) == row["original_record_sha256"], "original record hash mismatch")
        require(digest(original["english_messages"]) ==
                row["english_original_ref"]["english_messages_sha256"], "English hash mismatch")
    return rows, load_rows(AUDIT / "agent-reviews.jsonl"), inputs


def validate_notes(rows, notes):
    by_line = {row["original_line"]: row for row in rows}
    expected = {(r["original_line"], i) for r in rows for i, f in enumerate(r["findings"])
                if not f["agent_reviewed"]}
    decisions = {}
    seen_lines = set()
    for note in notes:
        line = note["line"]
        require(type(line) is int and line in by_line and line not in seen_lines,
                "invalid or duplicate note line")
        seen_lines.add(line)
        require(isinstance(note["record_concerns"], list) and
                all(isinstance(s, str) and s for s in note["record_concerns"]), "invalid concerns")
        require(note["reviews"], "empty manual reviews")
        for review in note["reviews"]:
            require(review["decision"] in DECISIONS, "unknown decision")
            require(all(isinstance(review[k], str) and review[k].strip() for k in ("reason", "kind")),
                    "missing manual rationale")
            anchor = review["source_anchor"]
            messages = by_line[line]["original_record"]["english_messages"]
            require(type(anchor["turn"]) is int and 0 <= anchor["turn"] < len(messages),
                    "invalid anchor turn")
            require(isinstance(anchor["quote"], str) and anchor["quote"].strip() and
                    anchor["quote"] in messages[anchor["turn"]]["content"], "source anchor mismatch")
            require(review["indices"], "empty issue selection")
            for index in review["indices"]:
                key = (line, index)
                require(type(index) is int and key in expected and key not in decisions,
                        "unknown, prior, or duplicate finding selection")
                decisions[key] = review
    require(set(decisions) == expected, "remaining findings coverage incomplete")
    return decisions


def occurrences(text, unit):
    require(isinstance(unit, str) and unit.strip(), "empty deletion unit")
    starts, offset = [], 0
    while (start := text.find(unit, offset)) >= 0:
        starts.append(start)
        offset = start + len(unit)
    return starts


def deletion_candidate(row, plan):
    require(plan["kind"] in {"exact_duplicate_list_entry", "exact_duplicate_prose_sentences"},
            "only individually reviewed prose/list deletions allowed")
    turn = plan["turn"]
    original = row["original_record"]
    require(digest(original) == row["original_record_sha256"], "repair original hash mismatch")
    require(type(turn) is int and 0 <= turn < len(original["messages"]), "invalid repair turn")
    require(original["messages"][turn]["role"] == "assistant", "user prompts are protected")
    require(original["english_messages"][turn]["role"] == "assistant", "source role mismatch")
    target = original["messages"][turn]["content"]
    source = original["english_messages"][turn]["content"]
    unit, source_unit = plan["target_unit"], plan["source_unit"]
    # No generic deduplication: the plan names exact, individually read occurrences.
    require(not any(marker in unit or marker in source_unit
                    for marker in ("```", "~~~", "`", '"', "\u201c", "\u201d")),
            "quoted/code ambiguity in deletion unit")
    target_starts, source_starts = occurrences(target, unit), occurrences(source, source_unit)
    keep, remove, source_remove = (plan[k] for k in
                                  ("keep_occurrence", "delete_occurrence", "source_delete_occurrence"))
    require(all(type(n) is int for n in (keep, remove, source_remove)) and
            1 <= keep < remove <= len(target_starts) and
            1 < source_remove <= len(source_starts), "duplicate occurrence proof failed")
    require(plan["reason"].strip(), "missing deletion rationale")
    selected = plan["finding_indices"]
    require(selected and len(set(selected)) == len(selected), "invalid repair findings")
    for index in selected:
        require(type(index) is int and 0 <= index < len(row["findings"]) and
                row["findings"][index]["turn"] == turn, "repair finding mismatch")
    start = target_starts[remove - 1]
    end = start + len(unit)
    require(target[:start].count("```") % 2 == 0 and target[:start].count("~~~") % 2 == 0,
            "deletion inside fenced code")
    changed = target[:start] + target[end:]
    require(changed and changed != target and unit in changed, "retained duplicate missing")
    messages = copy.deepcopy(original["messages"])
    messages[turn]["content"] = changed
    candidate_id = "r7-deletion-candidate:" + digest({"source": row["original_record_sha256"],
                                                    "plan": plan, "messages": messages})
    return {
        **SCOPE, "id": candidate_id, "source_record_id": row["id"],
        "source_id": row["source_id"], "source_group_id": row["source_group_id"],
        "original_line": row["original_line"], "original_record_sha256": row["original_record_sha256"],
        "split": original["split"], "turn": turn, "decision": "approve_issue_only_deletion",
        "method": "direct_paired_context_review_then_exact_deletion_replay",
        "reason": plan["reason"], "record_concerns": plan["record_concerns"],
        "whole_record_admission": "pending_independent_review",
        "finding_ids": [row["findings"][i]["id"] for i in selected],
        "original_target": target, "final_changed_target": changed,
        "original_target_sha256": text_sha(target), "changed_target_sha256": text_sha(changed),
        "messages": messages, "messages_sha256": digest(messages),
        "english_messages": original["english_messages"],
        "english_messages_sha256": digest(original["english_messages"]),
        "transformation": {"operation": "delete_exact_span", "offset_unit": "unicode_code_points",
                           "start": start, "end": end, "deleted_text": unit,
                           "retained_start": target_starts[keep - 1],
                           "source_deleted_equivalent_start": source_starts[source_remove - 1],
                           "source_unit": source_unit, "inserted_characters": 0},
        "manual_plan_sha256": digest(plan),
    }


def compile_artifacts(rows, prior_reviews, notes, plans):
    decisions = validate_notes(rows, notes)
    by_line = {r["original_line"]: r for r in rows}
    prior = {r["review_id"]: r for r in prior_reviews}
    require(len(prior) == len(prior_reviews), "duplicate prior reviews")
    concerns = {n["line"]: n["record_concerns"] for n in notes}
    candidates, repaired_turns = [], set()
    for plan in plans:
        key = (plan["line"], plan["turn"])
        require(key not in repaired_turns, "multiple repair plans for one turn")
        repaired_turns.add(key)
        require(plan["line"] in by_line, "repair source missing")
        candidates.append(deletion_candidate(by_line[plan["line"]], plan))
    registry, ledger = [], []
    for row in rows:
        if not row["findings"]:
            continue
        record = row["original_record"]
        registry.append({**SCOPE, "id": row["id"], "original_line": row["original_line"],
            "source_group_id": row["source_group_id"], "original_record": record,
            "original_record_sha256": row["original_record_sha256"],
            "english_original_ref": row["english_original_ref"],
            "original_record_usage": "evidence_only_not_admission_despite_original_flags",
            "full_english_and_twi_read_in_this_pass": (row["original_line"] in concerns or
                any(c["original_line"] == row["original_line"] for c in candidates)),
            "record_concerns": concerns.get(row["original_line"], []),
            "prior_agent_review_ids": row["agent_review_ids"],
            "whole_record_admission": "pending_independent_review"})
        for index, finding in enumerate(row["findings"]):
            turn = finding["turn"]
            entry = {**SCOPE, "finding_id": finding["id"], "record_id": row["id"],
                "original_line": row["original_line"], "turn": turn,
                "source_group_id": row["source_group_id"], "finding": finding,
                "original_record_sha256": row["original_record_sha256"],
                "target_sha256": text_sha(record["messages"][turn]["content"]),
                "english_target_sha256": text_sha(record["english_messages"][turn]["content"]),
                "context_sha256": digest({k: record[k] for k in ("messages", "english_messages")}),
                "candidate_correction_ids": [c["id"] for c in candidates
                                             if finding["id"] in c["finding_ids"]]}
            if finding["agent_reviewed"]:
                links = finding["linked_agent_review_ids"]
                require(links and all(ref in prior for ref in links), "missing prior agent review")
                require(all(prior[ref]["id"] == row["id"] and prior[ref]["turn"] == turn
                            for ref in links), "prior review binding mismatch")
                entry.update(review_pass="prior_27", decision="preserve_prior_review",
                             prior_reviews=[prior[ref] for ref in links])
            else:
                review = decisions[(row["original_line"], index)]
                entry.update(review_pass="remaining_164", decision=review["decision"],
                    classification=review["kind"], rationale=review["reason"],
                    source_anchor=review["source_anchor"], method="direct_full_paired_context_inspection",
                    automatic_semantic_pass=False, manual_decision_sha256=digest(review),
                    record_concerns=concerns[row["original_line"]])
            entry["review_id"] = "r7-issue-review:" + digest(entry)
            ledger.append(entry)
    fresh = [r for r in ledger if r["review_pass"] == "remaining_164"]
    counts = dict(sorted(Counter(r["decision"] for r in fresh).items()))
    summary = {**SCOPE, "version": VERSION, "state": "complete_issue_review_not_training_admission",
        "all_original_histories_accounted": len(rows), "flagged_histories": len(registry),
        "total_selected_findings": len(ledger), "prior_agent_reviewed_findings": len(ledger) - len(fresh),
        "remaining_findings_directly_reviewed": len(fresh), "remaining_findings_unreviewed": 0,
        "histories_read_for_remaining_findings": len(notes), "decisions": counts,
        "deletion_only_candidates": len(candidates),
        "new_remaining_findings_corrected": sum(not by_line[c["original_line"]]["findings"][i]["agent_reviewed"]
            for c, p in zip(candidates, plans) for i in p["finding_indices"]),
        "prior_findings_with_new_deletion_candidates": sum(by_line[c["original_line"]]["findings"][i]["agent_reviewed"]
            for c, p in zip(candidates, plans) for i in p["finding_indices"]),
        "automatic_semantic_clearances": 0, "automatic_edits": 0, "new_twi_words": 0,
        "user_test_prompts_imported": 0, "new_training_admissions": 0,
        "notes": ["Clearances concern selected issues only; record concerns remain separately visible.",
                  "Previous 27 reviews are imported without claiming a new whole-translation review.",
                  "Two original past-search flags are language views of one turn, not two events.",
                  "Additional unsupported dated internship listings at line355 remain held.",
                  "Deletion-only candidates revisit prior lines132/875, not extra remaining-164 findings.",
                  "Domain/factual uncertainties in record concerns are not established rejections."]}
    return {
        "review-ledger.jsonl": jsonl(ledger), "source-registry.jsonl": jsonl(registry),
        "issue-clearances.jsonl": jsonl([r for r in fresh if r["decision"] == "clear_issue"]),
        "unresolved-findings.jsonl": jsonl([r for r in fresh if r["decision"] != "clear_issue"]),
        "record-concerns.jsonl": jsonl([{**SCOPE, "original_line": n["line"],
            "record_id": by_line[n["line"]]["id"], "record_concerns": n["record_concerns"],
            "disposition": "review_notes_not_automatic_rejection_or_admission"} for n in notes]),
        "candidate-corrections.jsonl": jsonl(candidates),
        "correction-registry.jsonl": jsonl([{**SCOPE, "id": c["id"],
            "candidate_sha256": digest(c), "original_record_sha256": c["original_record_sha256"],
            "source_group_id": c["source_group_id"], "split": c["split"],
            "original_line": c["original_line"], "finding_ids": c["finding_ids"],
            "transformation_sha256": digest(c["transformation"])} for c in candidates]),
        "prior-agent-reviews.jsonl": jsonl(prior_reviews), "summary.json": encoded(summary) + b"\n",
    }


def package(out=OUT, verify=False):
    out = Path(out)
    if not verify:
        require(not (out / "SEALED.json").exists(), "sealed package: choose a new version")
    rows, prior, inputs = sealed_audit()
    notes, plans = load_rows(out / "manual-decisions.jsonl"), load_rows(out / "manual-repairs.jsonl")
    for name in ("manual-decisions.jsonl", "manual-repairs.jsonl"):
        inputs[str(out / name)] = file_sha(out / name)
    artifacts = compile_artifacts(rows, prior, notes, plans)
    summary = json.loads(artifacts["summary.json"])
    require((summary["flagged_histories"], summary["total_selected_findings"],
             summary["prior_agent_reviewed_findings"], summary["remaining_findings_directly_reviewed"],
             summary["histories_read_for_remaining_findings"]) == (122, 191, 27, 164, 109),
            "review coverage drift")
    manifest = {**SCOPE, "version": VERSION, "input_sha256": inputs,
        "code_sha256": {str(Path(__file__).resolve()): file_sha(__file__)},
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
        "source_attribution": "OpenAssistant/oasst1 English source prose, Apache-2.0; exact source refs retained.",
        "hash_contract": "JSON: UTF-8 sorted compact ensure_ascii=False; text/files: raw UTF-8 bytes",
        "protected_prompts": "No observed user test prompt or evaluation file is an input; source user turns unchanged."}
    artifacts["manifest.json"] = encoded(manifest) + b"\n"
    artifacts["SEALED.json"] = encoded({"version": VERSION,
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
        "training_ready": False}) + b"\n"
    if verify:
        expected = set(artifacts) | {"manual-decisions.jsonl", "manual-repairs.jsonl"}
        require({p.name for p in out.iterdir()} == expected, "unexpected/missing package file")
        for name, data in artifacts.items():
            require((out / name).read_bytes() == data, f"replay mismatch: {name}")
    else:
        expected_inputs = {"manual-decisions.jsonl", "manual-repairs.jsonl"}
        require({p.name for p in out.iterdir()} == expected_inputs, "output directory is not fresh")
        for name, data in artifacts.items():
            (out / name).write_bytes(data)
    return summary


def remaining():
    return [r for r in load_rows(AUDIT / "record-audit.jsonl")
            if any(not f["agent_reviewed"] for f in r["findings"])]


def inspect(start, count):
    records = remaining()
    for offset, row in enumerate(records[start:start + count], start):
        print(f"\n=== INDEX {offset} ORIGINAL LINE {row['original_line']} ===")
        for key in ("english_messages", "messages"):
            print("\n" + key.upper())
            for turn, message in enumerate(row["original_record"][key]):
                print(f"[{turn} {message['role']}]\n{message['content']}")
        print("\nREMAINING FINDINGS")
        for index, finding in enumerate(row["findings"]):
            if finding["agent_reviewed"]:
                continue
            print(json.dumps({"index": index, "turn": finding["turn"],
                "code": finding["inventory_flag"]["code"] if finding.get("inventory_flag") else finding["classification"],
                "units": [{"text": g["occurrences"][0]["text"], "count": len(g["occurrences"]),
                           "labels": [o.get("label") for o in g["occurrences"] if o.get("label")]}
                          for g in finding["target_matches"][:3]],
                "total_pattern_units": len(finding["target_matches"])}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", nargs=2, type=int, metavar=("START", "COUNT"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.inspect:
        inspect(*args.inspect)
    else:
        require(args.build != args.verify, "choose exactly one of --build or --verify")
        print(json.dumps(package(args.output, verify=args.verify), ensure_ascii=False, indent=2))
