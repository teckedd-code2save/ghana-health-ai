#!/usr/bin/env python3
"""Re-register inherited research records with scoped, source-bound R7 issue review."""
import argparse
from collections import Counter, defaultdict
import copy
from itertools import chain
import json
from pathlib import Path
import sys

import admit_r7_alignment as shared
import prepare_r7_alignment_candidates as alignment
import prepare_r7_source_inventory as inventory
import check_afrique_training_contract as contract
import r7_encoding
import run_understanding_r7 as runner
import review_r7_response_findings as issue_review
from prepare_understanding_r2 import encode_language
from prepare_understanding_run import HoldoutIndex

ROOT = Path(__file__).resolve().parents[1]
PROJECT = inventory.PROJECT
RESTORED = Path("/Users/welcome/ghana-health-twi-r6")
VERSION = "r7-inherited-research-replay-v2"
FINDINGS_REVIEW = ROOT / "tmp/r7-20261117/response-findings-review-v2"
REVIEW_MANIFEST_SHA = "4683e075f476ca2f162b98c521611e9be1c24247c5ae165e4762a234d5a8a860"
# Individually read record notes, not regex judgments or whole-translation passes.
# These line selections are valid only for the exact pinned review release above.
RECORD_CONCERN_HOLDS = {
    1, 42, 46, 59, 88, 97, 99, 124, 166, 211, 230, 231, 238, 247, 275, 279,
    284, 307, 317, 323, 350, 355, 373, 396, 406, 430, 445, 453, 462, 465,
    473, 484, 485, 499, 503, 532, 541, 549, 561, 576, 577, 588, 594, 613,
    643, 645, 646, 672, 702, 707, 723, 748, 754, 760, 764, 780, 790, 792,
    800, 803, 809, 895, 901, 904, 907, 908, 909, 929, 939,
}
INHERITED_LIMITATION_LINES = {
    20, 22, 47, 54, 56, 77, 87, 95, 100, 104, 105, 173, 219, 263, 276, 297,
    322, 334, 363, 372, 375, 378, 389, 429, 443, 461, 476, 516, 525, 528,
    614, 618, 806, 838, 844, 875, 884, 900, 916, 945,
}
COLLECTIONS = {"responses", "meaning", "language", "english_retention"}
RESOLVED = {"preserve_section_specific_repetition", "confirmed_normalization_collision", "preserve_shared_attribute",
            "preserve_algorithm_structure", "preserve_code_coordinates", "preserve_definition_summary_structure", "preserve_template_slots"}
GENUINE = {"nonadjacent_exact_duplicate_review", "unsupported_search_claim_review", "unsupported_internet_capability_review",
           "nonadjacent_exact_duplicate_prose_review"}
require, sha, digest = inventory.require, inventory.sha, inventory.digest


def content(record):
    keys = ("messages", "tools") if "messages" in record else ("text",)
    return {key: record[key] for key in keys if key in record}


def training_content(entry):
    return content(entry.get("reviewed_correction", entry["record"]))


def checked_deletion(original, candidate):
    require(candidate["decision"] == "approve_issue_only_deletion"
            and candidate["original_record_sha256"] == digest(original)
            and candidate["source_record_id"] == original["id"]
            and candidate["source_id"] == original["source_id"], "Deletion source differs")
    turn, edit = candidate["turn"], candidate["transformation"]
    require(type(turn) is int and 0 <= turn < len(original["messages"])
            and original["messages"][turn]["role"] == "assistant", "Deletion is not an assistant edit")
    target = original["messages"][turn]["content"]
    start, end, kept = edit["start"], edit["end"], edit["retained_start"]
    require(edit["operation"] == "delete_exact_span" and edit["offset_unit"] == "unicode_code_points"
            and edit["inserted_characters"] == 0 and all(type(n) is int for n in (start, end, kept))
            and 0 <= kept < start < end <= len(target)
            and target[start:end] == edit["deleted_text"]
            and target[kept:kept + end - start] == edit["deleted_text"], "Invalid exact deletion")
    expected = copy.deepcopy(original["messages"])
    expected[turn]["content"] = target[:start] + target[end:]
    require(candidate["original_target"] == target and candidate["messages"] == expected
            and candidate["final_changed_target"] == expected[turn]["content"]
            and candidate["english_messages"] == original["english_messages"]
            and candidate["messages_sha256"] == digest(expected)
            and candidate["original_target_sha256"] == issue_review.text_sha(target)
            and candidate["changed_target_sha256"] == issue_review.text_sha(expected[turn]["content"]),
            "Deletion changed unapproved content")
    return {**content(original), "messages": expected}


def integrate_findings_review(entries, folder, inputs):
    folder = Path(folder).resolve(strict=True)
    inputs.bind(folder / "manifest.json", REVIEW_MANIFEST_SHA)
    review_manifest = shared.bind_sealed(folder, "SEALED.json", inputs)
    require(review_manifest["version"] == "r7-response-findings-review-v2"
            and review_manifest["scope"] == "individual_repetition_or_claim_only"
            and review_manifest["whole_translation_gold"] is False, "Wrong issue-review scope")
    for mapping in ("input_sha256", "code_sha256"):
        for path, checksum in review_manifest[mapping].items():
            inputs.bind(path, checksum)
    issue_review.package(folder, verify=True)
    original_audit = {r["id"]: r for r in inputs.rows(issue_review.AUDIT / "record-audit.jsonl")}
    sources = {r["id"]: r for r in inputs.rows(folder / "source-registry.jsonl")}
    ledger = inputs.rows(folder / "review-ledger.jsonl")
    notes = inputs.rows(folder / "record-concerns.jsonl")
    require({r["original_line"] for r in notes} == RECORD_CONCERN_HOLDS | INHERITED_LIMITATION_LINES
            and not RECORD_CONCERN_HOLDS & INHERITED_LIMITATION_LINES, "Record concern disposition coverage differs")
    concerns = {r["record_id"]: r for r in notes}
    findings = defaultdict(list)
    for row in ledger:
        findings[row["record_id"]].append(row)
    candidates = {r["source_record_id"]: r for r in inputs.rows(folder / "candidate-corrections.jsonl")}
    release = {"path": str(folder), "manifest_sha256": REVIEW_MANIFEST_SHA,
               "review_ledger_sha256": review_manifest["artifacts"]["review-ledger.jsonl"],
               "record_concerns_sha256": review_manifest["artifacts"]["record-concerns.jsonl"]}
    dispositions = []
    for entry in entries:
        if entry["version"] != "r6" or entry["collection"] != "responses":
            continue
        require("audit" in entry, "Missing original response audit")
        report, rich = entry["audit"], entry["rich"]
        require(report["original_record"] == rich and report["original_record_sha256"] == digest(rich),
                "Response audit source differs")
        linked = findings.get(rich["id"], [])
        require({r["finding_id"] for r in linked} == {f["id"] for f in report["findings"]}, "Incomplete issue-review coverage")
        require(report == original_audit[rich["id"]], "Response audit differs from bound source")
        if not linked:
            continue
        registry = sources[rich["id"]]
        require(registry["original_record"] == rich and registry["original_record_sha256"] == digest(rich),
                "Issue-review original differs")
        for review in linked:
            require(review["original_record_sha256"] == digest(rich)
                    and review["context_sha256"] == digest({k: rich[k] for k in ("messages", "english_messages")})
                    and review["finding"] in report["findings"]
                    and review["target_sha256"] == issue_review.text_sha(rich["messages"][review["turn"]]["content"])
                    and review["english_target_sha256"] == issue_review.text_sha(rich["english_messages"][review["turn"]]["content"]),
                    "Issue-review history or target differs")
            if review["decision"] == "preserve_prior_review":
                require(all(r in entry["audit_reviews"] for r in review["prior_reviews"]), "Prior review differs")
        note = concerns.get(rich["id"])
        held = note is not None and note["original_line"] in RECORD_CONCERN_HOLDS
        require(not held or note["record_concerns"], "Missing record hold evidence")
        proof = {"release": release, "original_record_sha256": digest(rich),
                 "review_ids": [r["review_id"] for r in linked], "finding_ids": [r["finding_id"] for r in linked],
                 "record_concerns": note["record_concerns"] if note else [],
                 "record_scope_disposition": "hold_unresolved_record_concern" if held else "retain_inherited_tier_not_whole_record_certification",
                 "record_scope_method": "direct_read_of_pinned_record_concern_notes_not_new_whole_translation_review",
                 "whole_translation_gold": False, "new_human_review": False}
        entry["issue_reviews_v2"], entry["record_concern_hold"] = linked, held
        if candidate := candidates.get(rich["id"]):
            entry["reviewed_correction"] = checked_deletion(rich, candidate)
            entry["correction_candidate"] = candidate
            proof["deletion_candidate"] = {"id": candidate["id"], "sha256": digest(candidate),
                "original_line": candidate["original_line"], "record_concerns": candidate["record_concerns"],
                "scope_decision": "exact_duplicate_deletion_only_inherited_research_tier",
                "whole_translation_gold": False}
            entry["extra_evidence"]["exact_deletion_candidate"] = candidate
        entry["extra_evidence"]["issue_review_v2"] = proof
        dispositions.append({"record_id": rich["id"], "original_line": report["original_line"], **proof})
    return {"release": release, "dispositions": dispositions, "review_summary": inputs.read(folder / "summary.json")}


def read_pack(folder, version, inputs):
    folder = Path(folder).resolve(strict=True)
    manifest = shared.bind_sealed(folder, "PREPARED.json", inputs)
    require(manifest["version"] == "understanding-adaptation-" + version
            and manifest["private_research_only"] is True and manifest["production_eligible"] is False, "Wrong prior pack scope")
    witnesses = defaultdict(list)
    witness_sha = sha(folder / "source-witnesses.jsonl")
    for line, witness in enumerate(inputs.rows(folder / "source-witnesses.jsonl"), 1):
        ref = witness["dataset_ref"]
        witnesses[(ref["dataset_id"], ref["dataset_version"], ref["record_id"])].append({"record": witness,
            "ref": {"file": str(folder / "source-witnesses.jsonl"), "line": line,
                    "file_sha256": witness_sha, "record_sha256": digest(witness)}})
    entries = []
    for ref in inputs.read(folder / "source-manifests.json"):
        require(manifest["artifacts"][ref["manifest_path"]] == ref["manifest_sha256"], "Unbound prior dataset")
        source = inputs.read(folder / ref["manifest_path"])
        require(source["admission_status"] == "admitted" and source["private_research_only"] is True
                and source["production_eligible"] is False, "Prior dataset not admitted")
        collection = source.get("catalog_dataset_id", Path(source["records_path"]).parts[-2])
        require(collection in COLLECTIONS and manifest["artifacts"][source["records_path"]] == source["records_sha256"], "Unbound prior records")
        values = inputs.rows(folder / source["records_path"])
        require(len(values) == source["record_count"], "Prior record count differs")
        for line, record in enumerate(values, 1):
            require(record["training_eligible"] is True and record["split"] == "train", "Prior record not admitted")
            require((collection == "language" and "text" in record and "messages" not in record)
                    or (collection != "language" and isinstance(record.get("messages"), list)),
                    "Prior collection/objective mismatch; no invented chat wrapper")
            linked = witnesses[(source["dataset_id"], source["dataset_version"], record["id"])]
            require(linked and all(w["record"]["dataset_ref"]["record_sha256"] == runner.digest(record)
                                  and w["record"]["source_id"] == record["source_id"] for w in linked), "Prior witness differs")
            entries.append({"version": version, "collection": collection, "record": record, "rich": record,
                "prior_witnesses": linked, "prior_dataset": source, "prior_pack": {"path": str(folder), "manifest_sha256": sha(folder / "manifest.json")},
                "record_ref": {"file": str(folder / source["records_path"]), "line": line,
                               "file_sha256": source["records_sha256"], "record_sha256": digest(record)}, "extra_evidence": {}})
    return entries


def restore_r6(entries, audit_folder, restored, project, inputs):
    audit_manifest = shared.bind_sealed(audit_folder, "SEALED.json", inputs)
    require(audit_manifest["version"] == "r7-response-target-audit-v1", "Wrong response audit")
    for path, checksum in audit_manifest["inputs"]["files"].items():
        inputs.bind(path, checksum)
    audited = {row["id"]: row for row in inputs.rows(audit_folder / "record-audit.jsonl")}
    reviews = {row["review_id"]: row for row in inputs.rows(audit_folder / "agent-reviews.jsonl")}
    response_manifest = inputs.read(restored / "work/responses-r6/manifest.json")
    for name, path in {"authored": restored / "work/response-authoring-v1/authored.jsonl",
                       "histories": restored / "twi-r5-resume/conversation-suitability-v1/twi-histories.jsonl",
                       "decisions": restored / "twi-r5-resume/conversation-second-review-v1/decisions.jsonl",
                       "repaired": restored / "work/conversation-repair-v1/repaired.jsonl"}.items():
        inputs.bind(path, response_manifest[name + "_sha256"])
    decisions = {r["tree_id"]: r for r in inputs.rows(restored / "twi-r5-resume/conversation-second-review-v1/decisions.jsonl")}
    repairs = {r["tree_id"]: r for r in inputs.rows(restored / "work/conversation-repair-v1/repaired.jsonl")}
    english_folder = restored / "work/english-retention-growth-v1"
    english_manifest = inputs.read(english_folder / "manifest.json")
    inputs.bind(english_folder / "records.jsonl", english_manifest["records_sha256"])
    meaning_folder = project / "tmp/understanding-scale-20261111/r6-meaning-v1"
    meaning_manifest = inputs.read(meaning_folder / "summary.json")
    inputs.bind(meaning_folder / "admitted.jsonl", meaning_manifest["admitted_sha256"])
    originals = {"english_retention": {r["id"]: r for r in inputs.rows(english_folder / "records.jsonl")},
                 "meaning": {r["id"]: r for r in inputs.rows(meaning_folder / "admitted.jsonl")},
                 "language": {e["record"]["id"]: e["record"] for e in entries if e["version"] == "r5" and e["collection"] == "language"}}
    for entry in entries:
        if entry["version"] != "r6":
            continue
        old, collection = entry["record"], entry["collection"]
        if collection == "responses":
            report = audited[old["id"]]
            rich = report["original_record"]
            require(digest(rich) == report["original_record_sha256"], "Audit original hash differs")
            linked_reviews = [reviews[key] for key in report["agent_review_ids"]]
            for review in linked_reviews:
                require(review["record_sha256"] == digest(rich)
                        and review["target_sha256"] == digest(rich["messages"][review["turn"]]["content"]), "Audit review target differs")
            entry["audit"], entry["audit_reviews"] = report, linked_reviews
            entry["extra_evidence"] = {"r6_registration_manifest": response_manifest,
                "response_audit_manifest_sha256": sha(audit_folder / "manifest.json"),
                "response_audit": report, "saved_agent_reviews": linked_reviews,
                "prior_second_review": decisions.get(rich["tree_id"]), "prior_repair": repairs.get(rich["tree_id"])}
        else:
            rich = originals[collection][old["id"]]
            entry["extra_evidence"] = {"r6_original_registration": english_manifest if collection == "english_retention" else
                                      meaning_manifest if collection == "meaning" else {"mode": "exact_R5_language_inheritance"}}
        require(content(rich) == content(old) and rich["source_id"] == old["source_id"], "Restored original differs from sealed R6 record")
        entry["rich"] = rich


def audit_status(entry):
    if "audit" not in entry:
        return None, []
    if "issue_reviews_v2" in entry:
        pending, genuine = [], []
        corrected = set(entry.get("correction_candidate", {}).get("finding_ids", []))
        for review in entry["issue_reviews_v2"]:
            decision = review["decision"]
            if decision == "clear_issue":
                continue
            if decision == "preserve_prior_review":
                if all(r["decision"] in RESOLVED for r in review["prior_reviews"]):
                    continue
                if review["finding_id"] in corrected and all(r["decision"] in {
                    "nonadjacent_exact_duplicate_review", "nonadjacent_exact_duplicate_prose_review"
                } for r in review["prior_reviews"]):
                    continue
                if any(r["decision"] in GENUINE for r in review["prior_reviews"]):
                    genuine.append(review["review_id"])
                    continue
            if decision in {"hold_unresolved_meaning", "hold_unsupported_claim"}:
                genuine.append(review["review_id"])
            else:
                pending.append(review["review_id"])
        if genuine:
            return "unresolved_genuine_audit_finding", genuine
        if entry["record_concern_hold"]:
            return "unresolved_record_scope_concern", entry["extra_evidence"]["issue_review_v2"]["record_concerns"]
        return ("unresolved_audit_selector_not_proven_error", pending) if pending else (None, [])
    reviews = {r["review_id"]: r for r in entry["audit_reviews"]}
    genuine = [r["review_id"] for r in reviews.values() if r["decision"] in GENUINE]
    if genuine:
        return "unresolved_genuine_audit_finding", genuine
    pending = []
    for finding in entry["audit"]["findings"]:
        linked = [reviews[k] for k in finding["linked_agent_review_ids"]]
        if linked and all(r["decision"] in RESOLVED for r in linked):
            continue
        if finding["classification"] == "protected_code_repetition" and not linked:
            continue
        pending.append(finding["id"])
    return ("unresolved_audit_selector_not_proven_error", pending) if pending else (None, [])


def protection(entries, folder, held, inputs):
    graph, blocked_keys = inventory.Families(), []
    for node in inputs.rows(folder / "oasst-tree-index.jsonl"):
        graph.connect([node["message_id"], node["message_tree_id"]])
        if node["source_split"] != "train":
            blocked_keys.append((node["message_tree_id"], {"kind": "original_oasst_validation_family"}))
    for entry in entries:
        entry["family_keys"] = inventory.identity_keys(entry["rich"]) | inventory.identity_keys(entry["record"])
        graph.connect(entry["family_keys"])
    for row in held:
        keys = inventory.identity_keys(row)
        graph.connect(keys)
        blocked_keys.extend((key, {"kind": "held_identity", "held_id": row["id"]}) for key in keys)
    for row in inputs.rows(folder / "protection-index.jsonl"):
        graph.connect(row["identity_keys"])
        blocked_keys.extend((key, {"kind": "inherited_held_identity", "held_id": row["id"]}) for key in row["identity_keys"])
    for name in ("oasst-decisions", "r5-replay-decisions", "parallel-meaning-decisions"):
        for row in inputs.rows(folder / (name + ".jsonl")):
            keys = inventory.identity_keys(row) | set(row.get("source_family_keys", []))
            graph.connect(keys)
            if row["status"] == "excluded" and (row.get("protection_evidence") or any(
                reason.startswith(("protected_", "original_validation", "inherited_validation", "inherited_test")) for reason in row["reasons"])):
                blocked_keys.extend((key, {"kind": "inherited_inventory_exclusion", "inventory_id": row["id"],
                                           "evidence": row.get("protection_evidence")}) for key in keys)
    index = HoldoutIndex((r["id"], r["text"]) for r in held)
    for entry in entries:
        for text in chain(inventory.text_views(entry["rich"]),
                          inventory.text_views(training_content(entry)) if "reviewed_correction" in entry else ()):
            if hit := index.match(text):
                blocked_keys.extend((key, hit) for key in entry["family_keys"])
                break
    blocked = {graph.root(key): proof for key, proof in blocked_keys}
    for entry in entries:
        entry["family"] = graph.root(entry["record"]["source_id"])
    return blocked


def dedupe_key(entry):
    return entry["collection"], entry["family"], digest(training_content(entry))


def inherited_tier(entry):
    rich = entry["rich"]
    metadata = rich.get("metadata", {})
    return (rich.get("quality_status") or rich.get("review", {}).get("status") or metadata.get("screening")
            or metadata.get("disposition") or entry["prior_dataset"]["dataset_version"])


def measure(tokenizer, record, max_length):
    if "messages" in record:
        return shared.native_measure(tokenizer, record, max_length)
    try:
        encoded = encode_language(tokenizer, record["text"], tokenizer.model_max_length)
    except ValueError as error:
        return None, {"status": "rejected", "reason": "language_encoding_rejected", "detail": str(error)}
    size = len(encoded["input_ids"])
    result = {"sequence_tokens": size, "target_tokens": size, "effective_causal_target_tokens": size - 1,
              "encoded_arrays_sha256": digest(encoded), "exact_original_target": True, "truncated": False}
    if size > max_length:
        return None, {**result, "status": "rejected", "reason": "overlength_no_truncation"}
    return encoded, {**result, "status": "passed"}


def verify(out, native=False):
    out = Path(out).resolve(strict=True)
    inputs = inventory.Inputs()
    manifest = shared.bind_sealed(out, "SEALED.json", inputs)
    require(manifest["version"] == VERSION and manifest["admission_status"] == "admitted" and manifest["training_pack_ready"] is False
            and manifest["parent_boundary_approval"] is False, "Wrong replay scope")
    for path, info in manifest["input_files"].items():
        inputs.bind(path, info["sha256"])
    require(not any(p.is_symlink() for p in out.rglob("*")) and
            {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == set(manifest["artifacts"]) | {"manifest.json", "SEALED.json"},
            "Unbound replay artifact")
    rows, witnesses = inputs.rows(out / "rows.jsonl"), inputs.rows(out / "source-witnesses.jsonl")
    snapshots = runner.admitted_sources(out, rows, witnesses, manifest["artifacts"])
    runner.verify_overlap(out, witnesses, manifest["artifacts"])
    tokenizer = contract.load_tokenizer(Path(manifest["tokenizer_path"])) if native else None
    response_entries = []
    for witness in witnesses:
        if witness["collection"] == "responses" and witness["metadata"]["prior_version"] == "r6":
            extra = witness["original_ref"]["extra_evidence"]
            entry = {"version": "r6", "collection": "responses", "record": witness["original_ref"]["prior_record_snapshot"],
                     "rich": witness["metadata"]["prior_full_record"], "extra_evidence": copy.deepcopy(extra)}
            if "response_audit" in extra:
                entry.update(audit=extra["response_audit"], audit_reviews=extra["saved_agent_reviews"])
            response_entries.append(entry)
    integrate_findings_review(response_entries, Path(manifest["findings_review"]["path"]), inputs)
    response_entries = {e["rich"]["id"]: e for e in response_entries}
    seen = set()
    for row, witness in zip(rows, witnesses, strict=True):
        require(row["id"] not in seen and row["collection"] in COLLECTIONS
                and all(row[k] == witness[k] for k in ("id", "collection", "source_id")), "Replay identity differs")
        seen.add(row["id"])
        original = witness["metadata"]["prior_full_record"]
        if witness["collection"] == "responses" and witness["metadata"]["prior_version"] == "r6":
            entry = response_entries[original["id"]]
            require(audit_status(entry)[0] is None, "Replay admitted an unresolved response")
            require(content(witness) == training_content(entry), "Replay differs from reviewed source or exact deletion")
        else:
            require(content(witness) == content(original), "Replay content differs from prior full record")
        ids, labels = row["input_ids"], row["labels"]
        require(len(ids) == len(labels) == len(row["attention_mask"]) <= manifest["max_length"]
                and row["attention_mask"] == [1] * len(ids) and ids[-1] == contract.EOS, "Invalid replay tokens")
        if row["objective"] == "causal_lm":
            require(row["collection"] == "language" and labels == ids, "Invalid language objective")
        else:
            first = next(i for i, value in enumerate(labels) if value != -100)
            require(first > 0 and labels == [-100] * first + ids[first:] and labels.count(contract.EOS) == 1, "Invalid full-history final target")
        if native:
            encoded, _ = measure(tokenizer, witness, manifest["max_length"])
            require(encoded is not None and all(row[k] == encoded[k] for k in ("input_ids", "labels", "attention_mask")), "Native replay encoding differs")
    return {"verified": True, "native_reencoded": native, "full_records": len(rows), "datasets": len(snapshots)}


def prepare(r5, r6, inventory_folder, audit_folder, tokenizer_path, evaluations, out, restored=RESTORED, project=PROJECT,
            max_length=4096, findings_review=FINDINGS_REVIEW):
    destination = Path(out).absolute()
    out = destination.with_name(destination.name + ".building")
    inventory_folder, audit_folder = Path(inventory_folder).resolve(strict=True), Path(audit_folder).resolve(strict=True)
    require(not destination.exists() and not destination.is_symlink() and not out.exists() and not out.is_symlink(),
            "Use a new replay admission directory; preserve existing attempts")
    require(type(max_length) is int and 1 <= max_length <= 4096, "Invalid replay length ceiling")
    inputs = inventory.Inputs()
    for module in (sys.modules[__name__], shared, inventory, alignment, contract, runner, r7_encoding, issue_review):
        inputs.bind(module.__file__)
    for path in (ROOT / "scripts/prepare_understanding_r2.py", ROOT / "scripts/run_understanding_r2.py",
                 ROOT / "scripts/prepare_understanding_run.py", ROOT / "modal/train/afrique_response_core.py"):
        inputs.bind(path)
    inventory_manifest = inventory.verify(inventory_folder)
    shared.bind_sealed(inventory_folder, "SEALED.json", inputs)
    for folder in (r5, r6):
        path = str(Path(folder).resolve(strict=True) / "manifest.json")
        require(path in inventory_manifest["input_files"], "Pack outside original inventory")
        inputs.bind(path, inventory_manifest["input_files"][path]["sha256"])
    entries = read_pack(r5, "r5", inputs) + read_pack(r6, "r6", inputs)
    restore_r6(entries, audit_folder, Path(restored), Path(project), inputs)
    reviewed = integrate_findings_review(entries, findings_review, inputs)
    references = alignment.References(inventory_manifest, inputs)
    held = shared.held_sources(inventory_folder, inventory_manifest, inputs, references)
    eval_held, eval_refs = shared.evaluation_sources(evaluations, inputs)
    held += eval_held
    blocked = protection(entries, inventory_folder, held, inputs)
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    for name, checksum in contract.ASSET_HASHES.items():
        inputs.bind(tokenizer_path / name, checksum)
    tokenizer = contract.load_tokenizer(tokenizer_path)
    datasets, rows, witnesses, decisions, measures, chosen = defaultdict(list), [], [], [], [], {}
    print("Prior admissions and audit bound; measuring protected full-history replay", file=sys.stderr, flush=True)
    entries.sort(key=lambda e: (e["version"] != "r6", e["collection"], e["record"]["id"]))
    for number, entry in enumerate(entries, 1):
        prior, rich, collection = entry["record"], entry["rich"], entry["collection"]
        identity = "replay:" + entry["version"] + ":" + prior["id"]
        decision = {"id": identity, "prior_version": entry["version"], "prior_record_id": prior["id"], "collection": collection,
                    "source_id": prior["source_id"], "original_ref": entry["record_ref"], "source_family": entry["family"],
                    "training_eligible": False, "new_semantic_review": False, "full_history_only": True}
        if "issue_review_v2" in entry["extra_evidence"]:
            decision["issue_review_v2"] = entry["extra_evidence"]["issue_review_v2"]
        problem, evidence = audit_status(entry)
        if entry["family"] in blocked:
            decision.update(status="excluded", reasons=["protected_source_family_or_content"], protection_evidence=blocked[entry["family"]])
        elif problem:
            decision.update(status="excluded" if problem == "unresolved_genuine_audit_finding" else "pending_audit_review",
                            reasons=[problem], audit_finding_refs=evidence)
        elif dedupe_key(entry) in chosen:
            decision.update(status="duplicate", reasons=["same_collection_family_and_exact_complete_content"], retained_id=chosen[dedupe_key(entry)])
        else:
            original_ref = {"prior_pack": entry["prior_pack"], "prior_dataset": entry["prior_dataset"],
                "prior_record_ref": entry["record_ref"], "prior_record_snapshot": prior,
                "prior_original_ref": rich.get("original_ref", {}), "prior_review": rich.get("review", {}),
                "prior_witnesses": entry["prior_witnesses"], "extra_evidence": entry["extra_evidence"]}
            metadata = {**rich.get("metadata", {}), "prior_full_record": rich, "prior_review": rich.get("review", {}),
                "prior_quality_tier": inherited_tier(entry), "prior_version": entry["version"],
                "group_id": "r7-replay-family:" + digest(entry["family"]), "group_aliases": sorted(entry["family_keys"]),
                "origin": rich.get("origin", rich.get("metadata", {}).get("origin", entry["prior_witnesses"][0]["record"]["metadata"].get("origin"))),
                "human_reviewed": False, "semantic_certification": False, "clinical_certification": False,
                "native_certified": False, "review_policy": "inherited_exact_prior_admission_no_new_semantic_judgment",
                "english_messages": rich.get("english_messages", []), "original_english_messages": rich.get("original_english_messages", [])}
            if "issue_review_v2" in entry["extra_evidence"]:
                metadata["issue_review_v2"] = entry["extra_evidence"]["issue_review_v2"]
            if "correction_candidate" in entry:
                metadata["review_policy"] = "inherited_research_tier_with_individually_reviewed_exact_duplicate_deletion"
                metadata["source_transformation"] = entry["correction_candidate"]["transformation"]
            record = {"id": identity, "source_id": prior["source_id"], "collection": collection, "split": "train",
                **training_content(entry), "metadata": metadata, "original_ref": original_ref,
                "training_eligible": True, "research_admitted": True, "private_research_only": True,
                "production_eligible": False, "parent_boundary_verification_required": True}
            encoded, measured = measure(tokenizer, record, max_length)
            measurement = {"id": identity, "collection": collection, "prior_version": entry["version"], **measured}
            measures.append(measurement)
            if encoded is None:
                decision.update(status="excluded", reasons=[measured["reason"]], encoding=measured)
            else:
                chosen[dedupe_key(entry)] = identity
                decision.update(status="admitted", reasons=["inherited_research_with_reviewed_exact_deletion" if "correction_candidate" in entry
                                                            else "unchanged_prior_research_admission"], training_eligible=True,
                                prior_quality_tier=inherited_tier(entry), encoding=measured)
                datasets[collection].append(record)
                ref = {"dataset_id": "gha-r7-replay-" + collection, "dataset_version": VERSION, "record_id": identity,
                       "source_id": prior["source_id"], "record_sha256": runner.digest(record)}
                common = {"id": identity, "source_id": prior["source_id"], "collection": collection, "dataset_ref": ref}
                rows.append({**common, "split": "train", "objective": "causal_lm" if collection == "language" else "final_assistant",
                             **{k: encoded[k] for k in ("input_ids", "labels", "attention_mask")},
                             "sequence_tokens": measured["sequence_tokens"], "target_tokens": measured["target_tokens"]})
                witnesses.append({**common, "version": r7_encoding.WITNESS_VERSION, **training_content(entry), "metadata": metadata, "original_ref": original_ref})
        decisions.append(decision)
        if number % 2000 == 0:
            print(f"Checked {number}/{len(entries)} prior records; {len(rows)} admitted full records", file=sys.stderr, flush=True)
    require(rows, "No prior replay records admitted")
    admitted_ids = {r["id"] for r in rows}
    counts = {c: {"admitted_records": len(datasets[c]), "admitted_by_prior_version": dict(Counter(
        d["prior_version"] for d in decisions if d["collection"] == c and d["status"] == "admitted")),
        "decisions_by_prior_version": {v: dict(Counter(d["status"] for d in decisions if d["collection"] == c and d["prior_version"] == v)) for v in ("r5", "r6")},
        "tokens": shared.token_summary([m for m in measures if m["collection"] == c and m["id"] in admitted_ids])} for c in sorted(COLLECTIONS)}
    summary = {"version": VERSION, "prior_registered_records": len(entries), "admitted_full_records": len(rows), "counts_by_collection": counts,
        "reasons_by_record": dict(Counter(reason for d in decisions for reason in d["reasons"])),
        "tokens": shared.token_summary([m for m in measures if m["id"] in admitted_ids]), "max_length": max_length,
        "prefix_views_created": 0, "new_targets_created": 0, "new_semantic_reviews": 0, "api_calls": 0, "gpu_calls": 0,
        "inherited_issue_review": reviewed["review_summary"],
        "record_scope_dispositions": dict(Counter(d["record_scope_disposition"] for d in reviewed["dispositions"])),
        "admitted_exact_deletion_histories": sum("inherited_research_with_reviewed_exact_deletion" in d["reasons"] for d in decisions),
        "changed_assistant_messages": sum("inherited_research_with_reviewed_exact_deletion" in d["reasons"] for d in decisions),
        "new_generated_twi_words": 0, "whole_translation_gold": False,
        "training_pack_ready": False, "parent_boundary_approval": False,
        "limits": ["Inherited R6 meaning includes original machine dispositions and AI-proposed corrections, not newly certified references.",
                   "Issue clearances preserve inherited research tiers, not whole-record gold; separate record concerns are held explicitly.",
                   "Two individually reviewed deletion candidates remove only exact repeated spans; no inserted wording or factual certification.",
                   "Token counts measure one full-history final-assistant view per chat record; language uses its inherited causal objective.",
                   "Parent assembles prefixes, final holdout verification and a bounded token/exposure schedule.",
                   "No new failed-calibration English review judgments are imported; no new human, native or clinical certification."]}
    inputs.recheck()
    out.mkdir(parents=True, mode=0o700)
    for name, values in (("admission-ledger", decisions), ("token-measurements", measures), ("rows", rows),
                         ("source-witnesses", witnesses), ("held-content", held), ("record-scope-decisions", reviewed["dispositions"])):
        inventory.write_rows(out / (name + ".jsonl"), values)
    refs = []
    for collection, values in sorted(datasets.items()):
        if not values:
            continue
        target = out / "datasets" / collection
        target.mkdir(parents=True)
        inventory.write_rows(target / "records.jsonl", values)
        source = {"dataset_id": "gha-r7-replay-" + collection, "dataset_version": VERSION, "catalog_dataset_id": collection,
                  "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
                  "record_count": len(values), "records_path": f"datasets/{collection}/records.jsonl", "records_sha256": sha(target / "records.jsonl"),
                  "scope": "inherited prior research tier; scoped issue review and registered exact deletions; current protection and full-history encoding",
                  "new_semantic_review": False, "whole_translation_gold": False}
        inventory.write_json(target / "manifest.json", source)
        refs.append({"dataset_id": source["dataset_id"], "dataset_version": VERSION,
                     "manifest_path": f"datasets/{collection}/manifest.json", "manifest_sha256": sha(target / "manifest.json")})
    inventory.write_json(out / "source-manifests.json", refs)
    inventory.write_json(out / "evaluation-manifest.json", {"r7_evaluations": eval_refs, "inventory_manifest_sha256": sha(inventory_folder / "manifest.json")})
    inventory.write_json(out / "overlap-audit.json", {"protected_units": len(held), "overlap_free": True,
        "held_content_sha256": sha(out / "held-content.jsonl"), "source_witnesses_sha256": sha(out / "source-witnesses.jsonl"),
        "evaluation_manifest_sha256": sha(out / "evaluation-manifest.json"), "parent_boundary_verification_required": True})
    inventory.write_json(out / "summary.json", summary)
    inputs.recheck()
    inventory.write_json(out / "manifest.json", {"version": VERSION, "source_admission_scope": "inherited_R5_R6_research_with_scoped_issue_review",
        "admission_status": "admitted", "admission_scope": "Only source-manifests registered datasets; ledger exclusions are not training data",
        "private_research_only": True, "production_eligible": False, "training_pack_ready": False, "parent_boundary_approval": False,
        "input_files": inputs.files, "source_packs": {"r5": sha(Path(r5) / "manifest.json"), "r6": sha(Path(r6) / "manifest.json")},
        "inventory_manifest_sha256": sha(inventory_folder / "manifest.json"), "audit_manifest_sha256": sha(audit_folder / "manifest.json"),
        "findings_review": reviewed["release"], "whole_translation_gold": False, "new_generated_twi_words": 0,
        "evaluations": eval_refs, "tokenizer_path": str(tokenizer_path), "tokenizer_assets": contract.ASSET_HASHES,
        "model": contract.MODEL, "revision": contract.REVISION, "max_length": max_length, "enable_thinking": False,
        "artifacts": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    inventory.write_json(out / "SEALED.json", {"manifest_sha256": sha(out / "manifest.json")})
    verify(out)
    out.rename(destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--r5", type=Path, default=PROJECT / "tmp/understanding-r5-20261109/pack-v1")
    parser.add_argument("--r6", type=Path, default=PROJECT / "tmp/understanding-r6-20261113/pack-v1")
    parser.add_argument("--inventory", type=Path, default=ROOT / "tmp/r7-20261117/source-inventory/v4")
    parser.add_argument("--audit", type=Path, default=ROOT / "tmp/r7-20261117/response-target-audit-v1")
    parser.add_argument("--findings-review", type=Path, default=FINDINGS_REVIEW)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tmp/r6-demo/bundle-v1/tokenizer")
    parser.add_argument("--evaluation", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/r7-20261117/replay-admission-v2")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    result = (prepare(args.r5, args.r6, args.inventory, args.audit, args.tokenizer,
                      args.evaluation or [ROOT / "tmp/r7-20261117/behavior-eval-v1"], args.out, findings_review=args.findings_review)
              if args.action == "prepare" else verify(args.out, args.native))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
