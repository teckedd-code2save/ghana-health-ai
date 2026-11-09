"""Admit fully reviewed, source-bound R5 translations without modifying sources."""
import argparse
from collections import Counter
from pathlib import Path

import training_data_inputs as data
import prepare_understanding_r3 as previous
from prepare_understanding_run import HoldoutIndex
from screen_understanding_r5_data import inspect_messages
from select_understanding_r5_sources import (
    bound_package, bound_read, check_source, nonblank, protected_text, recheck, unique_rows,
)


def materialize(item, proposal, decision, refs):
    source = item["source_record"]
    check_source(source)
    data.require(type(item.get("index")) is int and item["index"] >= 0
                 and item.get("training_eligible") is False, "Invalid unadmitted source work")
    data.require(item["source_record_sha256"] == data.digest(source), "Source binding differs")
    data.require(proposal["source_id"] == source["id"] and proposal["index"] == item["index"]
                 and proposal["source_record_sha256"] == item["source_record_sha256"], "Proposal/source differs")
    data.require(decision["index"] == item["index"] and decision["proposal_record_sha256"] == data.digest(proposal),
                 "Review/proposal differs")
    data.require(type(proposal.get("index")) is int and type(decision.get("index")) is int,
                 "Invalid review index")
    data.require(decision["decision"] in {"accepted", "repair", "reject"}
                 and nonblank(decision.get("reason")), "Missing decision")
    data.require(proposal.get("decision") in {"translate", "exclude"}, "Unknown proposal decision")
    if proposal["decision"] == "exclude" or decision["decision"] == "reject":
        return None
    data.require(proposal["decision"] == "translate", "Unknown proposal decision")
    data.require(decision.get("source_findings") == [] and decision.get("findings") == [],
                 "Explicit empty source and target findings required")
    source_review = decision.get("source_review")
    data.require(nonblank(source_review) or (isinstance(source_review, dict)
                 and any(nonblank(value) for value in source_review.values())), "Missing source quality review")
    original = source["messages"]
    text = decision["corrected_twi"] if decision["decision"] == "repair" else proposal["twi"]
    data.require(isinstance(original, list) and isinstance(text, list)
                 and len(text) == len(original) and all(nonblank(t) for t in text), "Incomplete translation")
    turns = decision["per_turn"]
    data.require(isinstance(turns, list) and all(isinstance(t, dict) and type(t.get("index")) is int for t in turns)
                 and [t["index"] for t in turns] == list(range(len(original))), "Every source turn needs review")
    data.require(all(t.get("back_translation", "").strip() and t.get("note", "").strip()
                     and t.get("findings") == [] for t in turns), "Unresolved per-turn meaning")
    amendments = proposal.get("english_overrides", {})
    data.require(isinstance(amendments, dict), "Invalid reference amendments")
    data.require(not amendments or (proposal.get("amendment_reason", "").strip()
                 and decision.get("amendment_review", "").strip()), "Unreviewed reference amendment")
    reference = [dict(m) for m in original]
    for key, text_value in amendments.items():
        data.require(str(int(key)) == key and 0 <= int(key) < len(original)
                     and isinstance(text_value, str) and text_value.strip(), "Invalid reference amendment")
        reference[int(key)]["content"] = text_value
    messages = [{"role": m["role"], "content": t} for m, t in zip(original, text, strict=True)]
    findings = inspect_messages(messages) + inspect_messages(reference)
    screen_resolutions = decision.get("screen_resolutions", [])
    data.require(isinstance(screen_resolutions, list) and all(
        isinstance(r, dict) and nonblank(r.get("code")) and type(r.get("turn")) is int
        and 0 <= r["turn"] < len(original) and nonblank(r.get("reason")) for r in screen_resolutions),
        "Invalid decision-specific screen resolution")
    resolutions = {(r["code"], r["turn"]) for r in screen_resolutions}
    data.require(not any(f["severity"] == "invalid" for f in findings), "Invalid history")
    data.require(all((f["code"], f["turn"]) in resolutions for f in findings), "Unresolved screen finding")
    metadata = source["metadata"]
    return {"id": "r5-translation:" + source["id"], "source_id": source["source_id"],
        "source_hash": metadata.get("source_hash", item["source_record_sha256"]),
        "group_id": metadata["group_id"],
        "source_ids": sorted(set(metadata.get("source_ids", [])) | {source["source_id"]}),
        **{key: metadata[key] for key in ("family_id", "tree_id", "original_group_id") if key in metadata},
        "messages": messages, "english_messages": reference, "original_english_messages": original,
        "english_amendments": amendments, "amendment_reason": proposal.get("amendment_reason"),
        "origin": ("source_grounded_agent_translated_and_separately_reviewed" if metadata.get("origin") == "source_preserved"
                   else "project_authored_agent_translated_and_separately_reviewed"),
        "language": "tw", "split": "train", "training_eligible": True,
        "private_research_only": True, "production_eligible": False, "human_reviewed": False,
        "source_metadata": metadata, "source_quality_review": decision.get("source_review"),
        "review": {"status": "agent_checked", "findings": [], "per_turn": turns,
                   "source_review": source_review,
                   "screen_resolutions": [dict(r) for r in screen_resolutions],
                   "decision_sha256": data.digest(decision),
                   "screened_messages_sha256": data.digest(messages),
                   "screened_english_messages_sha256": data.digest(reference),
                   "reviewer": refs["reviewer"], "human_reviewed": False},
        "original_ref": {**source["original_ref"], **refs,
                         "source_record_sha256": item["source_record_sha256"],
                         "proposal_record_sha256": data.digest(proposal), "review_decision_sha256": data.digest(decision)}}


def admit(pairs, r4_pack, fresh, out):
    data.require(not out.exists() and not out.is_symlink(), "Use a new immutable admission directory")
    pins = {}
    bound_package(r4_pack, pins, {"held-content.jsonl"})
    fresh_manifest = bound_package(fresh, pins, {"held-content.jsonl"}, "FROZEN.json")
    index = HoldoutIndex(protected_text(r4_pack, pins, "r4") + protected_text(fresh, pins, "fresh"))
    candidates, excluded, evidence, seen = [], [], [], set()
    protection_views = []
    for proposal_path, review_path in pairs:
        proposals, review = bound_read(proposal_path, pins), bound_read(review_path, pins)
        data.require(proposals["human_reviewed"] is False and proposals["training_eligible"] is False, "Proposal cannot self-admit")
        data.require(nonblank(proposals.get("agent")) and nonblank(review.get("reviewer")),
                     "Missing accountable author or reviewer")
        data.require(review["kind"] == "agent_review" and review["human_reviewed"] is False
                     and review["reviewer"].strip() != proposals["agent"].strip()
                     and review["proposal_sha256"] == pins[str(proposal_path.absolute())], "Independent source-bound review required")
        source_path = Path(proposals["source_file"])
        source_rows = bound_read(source_path, pins, rows=True)
        data.require(pins[str(source_path.absolute())] == proposals["source_file_sha256"], "Source work changed")
        sources = unique_rows(source_rows, "index", integer=True)
        decisions = unique_rows(review["records"], "index", integer=True)
        submitted = unique_rows(proposals["records"], "index", integer=True)
        data.require(len(submitted) == len(proposals["records"]) and len(decisions) == len(review["records"])
                     and set(sources) == set(submitted) == set(decisions), "Every selected source needs one proposal and review")
        for key in sources:
            source = sources[key]
            identity = source["source_record"]["id"]
            data.require(identity not in seen, "Repeated source in admissions")
            seen.add(identity)
            refs = {"source_file": str(source_path.resolve()), "source_file_sha256": pins[str(source_path.absolute())],
                    "proposal_file": str(proposal_path.resolve()), "proposal_sha256": pins[str(proposal_path.absolute())],
                    "review_file": str(review_path.resolve()), "review_sha256": pins[str(review_path.absolute())],
                    "reviewer": review["reviewer"], "translator": proposals["agent"]}
            candidate = materialize(source, submitted[key], decisions[key], refs)
            # Rejected branches still connect their source families to protected text.
            protection_views.append({**source["source_record"], "id": "r5-translation:" + identity})
            evidence.append({"source": source, "proposal": submitted[key], "review": decisions[key], "references": refs})
            if candidate:
                candidates.append(candidate)
            else:
                excluded.append({"id": identity, "reason": "source_or_translation_rejected", "decision": decisions[key]})
    for record in candidates:
        protection_views.extend([record, {**record, "messages": record["english_messages"]}])
    blocked = previous.protected_groups(protection_views, index)
    accepted = []
    for record in candidates:
        if record["id"] in blocked:
            excluded.append({"id": record["id"], "reason": "protected_content", "evidence": blocked[record["id"]]})
        else:
            accepted.append(record)
    data.require(accepted, "No admissible data")
    recheck(pins)
    out.mkdir(parents=True)
    for name, records in (("conversations.jsonl", accepted), ("excluded.jsonl", excluded), ("source-review-evidence.jsonl", evidence)):
        data.write_rows(out / name, records)
    data.save(out / "report.json", {"selected_sources": len(seen), "admitted_histories": len(accepted), "excluded": len(excluded),
        "source_trees": len({r["group_id"] for r in accepted}),
        "multiturn_histories": sum(len(r["messages"]) > 2 for r in accepted),
        "assistant_occurrences": sum(m["role"] == "assistant" for r in accepted for m in r["messages"]),
        "origin_counts": dict(Counter(r["origin"] for r in accepted)), "human_reviewed": 0,
        "clinical_certification": False, "training_executed": False,
        "inherited_training_clearance": "Not checked: requires separately bound R2/R3/R4 actual-training audit",
        "pretraining_independence_claimed": False})
    data.save(out / "manifest.json", {"dataset_id": "r5-source-bound-conversations", "dataset_version": out.name,
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "r4_protection_sha256": pins[str((r4_pack / "manifest.json").absolute())],
        "fresh_suite_sha256": pins[str((fresh / "manifest.json").absolute())],
        "fresh_suite_version": fresh_manifest["version"],
        "input_files": pins,
        "artifacts": {p.name: data.sha(p) for p in out.iterdir() if p.is_file()}, "code_sha256": data.sha(__file__)})
    recheck(pins)
    data.save(out / "READY.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return data.read(out / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True, type=Path)
    for name in ("r4-pack", "fresh", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(admit(args.pair, args.r4_pack, args.fresh, args.out), indent=2))
