"""Admit source-bound, explicitly reviewed R4 derivatives; preserve all originals."""
import argparse
from collections import Counter
import json
from pathlib import Path

import training_data_inputs as data
from prepare_understanding_r3 import fields, protected_groups, verified_directory
from prepare_understanding_run import HoldoutIndex

SOURCE_SHA = "fc4d7023d0a0b42d06378ea3762d43e723ad18c17f04a6a68f37e3d8fff97dbc"


def materialize(proposal, decision, source, references):
    data.require(decision["source_id"] == proposal["source_id"] == source["id"], "Source identity differs")
    data.require(decision["proposal_record_sha256"] == data.digest(proposal), "Reviewed proposal changed")
    data.require(decision["decision"] in {"accepted", "repair", "reject"}, "Unknown review decision")
    if decision["decision"] == "reject" or decision.get("findings") or source.get("input_flags"):
        return None
    text = decision.get("corrected_twi") if decision["decision"] == "repair" else proposal["twi"]
    original = source["messages"]
    data.require(isinstance(text, list) and len(text) == len(original)
                 and all(isinstance(s, str) and s.strip() for s in text), "Incomplete translated history")
    turns = decision["per_turn"]
    data.require([r["index"] for r in turns] == list(range(len(original))), "Every turn needs explicit review")
    data.require(all(r["back_translation"].strip() and r["note"].strip() and not r["findings"] for r in turns),
                 "Unresolved or incomplete per-turn meaning review")
    reference = [dict(m) for m in original]
    amendments = proposal.get("english_overrides", {})
    data.require(not amendments or proposal.get("amendment_reason", "").strip(), "Unexplained source amendment")
    for key, value in amendments.items():
        data.require(str(int(key)) == key and 0 <= int(key) < len(original) and value.strip(), "Invalid reference amendment")
        reference[int(key)]["content"] = value
    messages = [{"role": m["role"], "content": t} for m, t in zip(original, text, strict=True)]
    data.require([m["role"] for m in messages] == ["user", "assistant"] * (len(messages) // 2), "Changed source roles")
    return {"id": "r4-derivative:" + source["id"], "source_id": source["source_id"],
        "source_hash": source["source_hash"], "group_id": source["group_id"], "source_ids": source["source_ids"],
        "messages": messages, "english_messages": reference, "original_english_messages": original,
        "english_amendments": amendments, "amendment_reason": proposal.get("amendment_reason"),
        "origin": "project_authored_agent_translated_and_agent_corrected", "language": "tw",
        "tags": source["tags"], "original_facts": source["facts"], "split": "train",
        "training_eligible": True, "private_research_only": True, "production_eligible": False,
        "human_reviewed": False,
        "review": {"status": "agent_checked", "findings": [], "per_turn": turns,
                   "reviewer": references["reviewer"], "human_reviewed": False},
        "original_ref": {**source["original_ref"], **references,
                         "proposal_record_sha256": data.digest(proposal), "review_decision_sha256": data.digest(decision)}}


def protected_derivative_groups(candidates, index):
    # Original wording must protect siblings even when the reviewed wording changed.
    expanded = [{**r, "english_messages": r.get("english_messages", []) + r.get("original_english_messages", [])}
                for r in candidates]
    return protected_groups(expanded, index)


def admit(pairs, r3_pack, out):
    data.require(not out.exists(), "Use a fresh immutable admission directory")
    verified_directory(r3_pack)
    held = [r for _, r in data.rows(r3_pack / "held-content.jsonl")]
    index = HoldoutIndex([(r["id"], r["text"]) for r in held])
    candidates, exclusions, evidence, seen = [], [], [], set()
    for proposal_path, review_path in pairs:
        proposed, review = data.read(proposal_path), data.read(review_path)
        data.require(proposed["human_reviewed"] is False and proposed["training_eligible"] is False,
                     "Proposal must not pre-authorize itself")
        data.require(review["kind"] == "agent_review" and review["human_reviewed"] is False
                     and review["reviewer"] and review["proposal_sha256"] == data.sha(proposal_path), "Review binding differs")
        source_path = Path(proposed["source_file"])
        data.require(data.sha(source_path) == SOURCE_SHA, "Original reviewed English source changed")
        sources = {r["id"]: r for _, r in data.rows(source_path)}
        decisions = {r["source_id"]: r for r in review["records"]}
        data.require(len(decisions) == len(review["records"]) and set(decisions) == {r["source_id"] for r in proposed["records"]},
                     "Every proposal needs one review")
        for p in proposed["records"]:
            data.require(p["source_id"] not in seen, "Duplicate proposal source")
            seen.add(p["source_id"])
            s, d = sources[p["source_id"]], decisions[p["source_id"]]
            refs = {"source_file": str(source_path.resolve()), "source_file_sha256": data.sha(source_path),
                "proposal_file": str(proposal_path.resolve()), "proposal_sha256": data.sha(proposal_path),
                "review_file": str(review_path.resolve()), "review_sha256": data.sha(review_path), "reviewer": review["reviewer"]}
            candidate = materialize(p, d, s, refs)
            evidence.append({"source": s, "proposal": p, "review": d, "references": refs})
            if candidate:
                candidates.append(candidate)
            else:
                exclusions.append({"id": p["source_id"], "reason": "unresolved_or_protected_source",
                    "source_flags": s.get("input_flags", []), "decision": d})
    blocked = protected_derivative_groups(candidates, index)
    accepted = []
    for candidate in candidates:
        if candidate["id"] in blocked:
            exclusions.append({"id": candidate["id"], "reason": "protected_content", "hit": blocked[candidate["id"]]})
        else:
            accepted.append(candidate)
    data.require(accepted, "No eligible derivatives")
    out.mkdir(parents=True)
    data.write_rows(out / "conversations.jsonl", accepted)
    data.write_rows(out / "excluded.jsonl", exclusions)
    data.write_rows(out / "source-review-evidence.jsonl", evidence)
    data.save(out / "report.json", {"proposed": len(seen), "admitted": len(accepted), "excluded": len(exclusions),
        "assistant_occurrences": sum(m["role"] == "assistant" for r in accepted for m in r["messages"]),
        "origin_counts": dict(Counter(r["origin"] for r in accepted)),
        "human_reviewed": 0, "clinical_certification": False, "training_executed": False,
        "note": "Existing project-authored frames, newly agent-translated; not natural recorded conversations or independent human review."})
    manifest = {"dataset_id": "r4-source-bound-conversation-derivatives", "dataset_version": "v1",
        "admission_status": "admitted", "private_research_only": True, "production_eligible": False,
        "r3_protection_manifest_sha256": data.sha(r3_pack / "manifest.json"),
        "artifacts": {p.name: data.sha(p) for p in out.iterdir() if p.is_file()}, "code_sha256": data.sha(__file__)}
    data.save(out / "manifest.json", manifest)
    data.save(out / "READY.json", {"manifest_sha256": data.sha(out / "manifest.json")})
    return data.read(out / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", type=Path, required=True)
    parser.add_argument("--r3-pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.pair, args.r3_pack, args.out), indent=2))
