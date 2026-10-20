"""Build a private review projection from pinned D/E checkpoints, without admission."""
import argparse
import hashlib
import json
from datetime import date
from collections import Counter
from html import escape
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(folder, expected=None):
    if expected:
        require(sha(folder / "manifest.json") == expected, "Manifest changed")
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest["training_ready"] is False, "Review input must not claim admission")
    for name, checksum in manifest["artifacts"].items():
        file = folder / name
        require(Path(name).name == name and not file.is_symlink(), "Unsafe artifact path")
        require(sha(file) == checksum, f"Artifact changed: {name}")
    return manifest


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def unique(values):
    result = {value["id"]: value for value in values}
    require(len(result) == len(values), "Duplicate source")
    return result


def project(handoffs, pending, assessments, originals):
    done, waiting, assessed = map(unique, (handoffs, pending, assessments))
    require(not done.keys() & waiting.keys(), "D/E overlap")
    require(done.keys() == assessed.keys(), "Missing or extra meaning assessments")
    require((done.keys() | waiting.keys()) == originals.keys(), "Source inventory mismatch")
    result = []
    for key, item in [*done.items(), *waiting.items()]:
        source = originals[key]
        require(source["split"] == "train" and source["parent_disposition"] != "excluded",
                "Protected source")
        require(digest(source["original_twi"]) == source["source_hash"] == item["source_hash"],
                "Changed original")
        require(digest(item["reference_english"]) == item["reference_hash"] == source["reference_hash"],
                "Changed English reference")
        require(item["training_eligible"] is False and item["human_reviewed"] is False,
                "Unsupported approval")
        decision = item.get("decision", {})
        proposal = item["after"] if key in done else None
        review = assessed.get(key)
        if review:
            require(digest(proposal) == item["after_hash"] == review["target_sha256"], "Stale E target")
            require(review["reference_sha256"] == item["reference_hash"] and
                    review["source_hash"] == item["source_hash"], "Stale E source")
            require(review["full_example"] == item, "Changed meaning-check handoff")
            require(review["human_reviewed"] is False and review["training_eligible"] is False,
                    "Unsupported meaning approval")
        adjudication = decision.get("source_adjudication")
        derivative = adjudication.get("reference_derivative") if adjudication else None
        if derivative:
            require(digest(derivative) == item["reference_derivative_sha256"] and
                    digest(adjudication) == item["source_adjudication_sha256"], "Changed adjudication")
        response = {"original": source["original_twi"], "english": item["reference_english"],
            "corrected_english": derivative, "proposed": proposal,
            "resolved_scope": decision.get("scope", "Correction remains pending."),
            "remaining_review": [decision.get("remaining", item.get("source_block") or "Complete repair needed.")],
            "issues": [decision.get("source_caveat", "Source preserved; not admitted to training.")],
            "meaning_review": review["review"] if review else None,
            "evidence": adjudication.get("verification_sources", []) if adjudication else []}
        result.append({"id": key, "source_hash": source["source_hash"], "source": "Existing instruction corpus",
            "language": "tw", "split": "train", "group_id": source["group_id"], "training_eligible": False,
            "proposal_set_id": digest([key, source["source_hash"], response]), "response": response,
            "original": source["original_twi"]["instruction"],
            "alternatives": [{"model": "Original Twi", "text": ""}] +
                ([{"model": "Proposed correction", "text": ""}] if proposal else []),
            "annotation_status": ("Agent checked; training admission pending" if review["review"]["outcome"] == "agent_pass_to_F"
                                  else "Meaning check: " + review["review"]["outcome"].replace("_", " ")) if review else "Correction pending"})
    return result


def progress_html(summary):
    d, e = summary["D"], summary["E"]
    counts = summary["decisions"]
    replaced = counts.get("replace", 0)
    retained = sum(counts.get(k, 0) for k in ("retain_current", "retain"))
    outcomes = e["outcomes"]
    nodes = [
        ("A", "Working copy", "Complete", "52,002 records accounted for. Originals and protected records preserved."),
        ("B", "Consolidate findings", "Complete", "674 initial work items. Source identities and related records retained."),
        ("C", "Diagnostic inspection", "Complete", "29 inspected examples added 25 work items. This is not an accuracy estimate."),
        ("D", "Complete-example corrections", "In progress", f"{replaced} replaced, {retained} retained, {d['pending_work_items']} pending. {d['cumulative_source_adjudicated_derivatives']} corrected English derivatives; original references unchanged."),
        ("E", "Meaning checks", "Current handoff checked", f"{e['agent_assessments']} assessed; {e['pending_assessments']} unassessed. {outcomes.get('agent_pass_to_F', 0)} scoped same-agent passes, {outcomes.get('source_hold', 0)} source holds, {outcomes.get('focused_twi_check', 0)} focused Twi check. Not independent native or clinical review."),
        ("F", "Admission decision", "Pending", "Decide which exact targets have adequate evidence. Correction is not admission. Resolve style and naturalness concerns."),
        ("H", "Focused review / quarantine", "Open", "Resolve specific uncertain phrases. Keep unresolved or unsupported examples outside training."),
        ("G", "Record admissions", "Pending", "Record the evidence, scope and exact target hash for every admission; keep human and automated statuses distinct."),
        ("I", "Final export validation", "Pending", "Check schemas, source hashes, duplicate families, cross-collection leakage, holdouts and coverage."),
        ("J", "Training dataset", "Pending", "Versioned exports, dataset cards and checksums. No training-ready seal yet."),
        ("K", "Training and evaluation", "Not started", "Evaluate meaning, naturalness, repetition, English retention and held-out tasks after admission.")]
    items = "".join(f'<li id="node-{key}"><details><summary><b>{key}</b><span>{escape(title)}</span><em>{escape(state)}</em></summary><p>{escape(detail)}</p>' +
                    ('<a href="/research/corpus/?collection=responses">Open complete examples</a>' if key in ("D", "E", "H") else "") +
                    '</details></li>' for key, title, state, detail in nodes)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corpus execution map</title><link rel="stylesheet" href="/research/corpus/progress.css"></head><body><main>
<header><a href="/research/corpus/?collection=responses">Back to corpus review</a><p>{escape(summary['as_of'])}</p><h1>Corpus execution map</h1><p>Instruction collection · {escape(summary['checkpoints'])}</p></header>
<p class="summary">{replaced} replaced · {retained} retained · {d['pending_work_items']} pending · 0 training admissions</p>
<ol>{items}</ol><p>E can return a target to D. F can request focused review at H or send an evidenced admission to G. Unresolved examples remain quarantined.</p>
<footer><p>This is the instruction-correction queue, not certification of the entire corpus. Language learning, semantic alignment, conversations and specialised actions retain separate readiness requirements.</p>
<p>D manifest: <code>{escape(summary['D_manifest_sha256'])}</code><br>E manifest: <code>{escape(summary['E_manifest_sha256'])}</code></p></footer></main></body></html>'''


def build(args):
    require(not args.out.exists(), "Use a new versioned output directory")
    checked(args.corrections, args.corrections_sha256)
    e_manifest = checked(args.meaning, args.meaning_sha256)
    require(e_manifest["parent"]["manifest_sha256"] == args.corrections_sha256, "E belongs to another D checkpoint")
    originals = []
    for item in args.originals:
        file = Path(item)
        manifest = checked(file.parent)
        require(file.name in manifest["artifacts"], "Original file not sealed")
        originals.extend(rows(file))
    handoffs = rows(args.corrections / "meaning-check-handoff.jsonl")
    pending = rows(args.corrections / "pending-worklist.jsonl")
    projected = project(handoffs, pending, rows(args.meaning / "assessments.jsonl"), unique(originals))
    report = json.loads((args.corrections / "report.json").read_text())
    require(len(handoffs) == report["cumulative_E_handoffs"] and len(pending) == report["pending_work_items"], "Wrong D counts")
    summary = {"as_of": args.as_of, "checkpoints": args.checkpoints,
        "D_manifest_sha256": args.corrections_sha256, "E_manifest_sha256": args.meaning_sha256,
        "D": report, "E": json.loads((args.meaning / "report.json").read_text()),
        "decisions": dict(Counter(row["decision"]["action"] for row in handoffs))}
    args.out.mkdir(parents=True)
    for name, value in (("queue.json", projected), ("progress.json", summary)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (args.out / "progress.html").write_text(progress_html(summary))
    manifest = {"policy": "instruction-review-projection-v1", "parents": {
        "D": args.corrections_sha256, "E": args.meaning_sha256},
        "queue_sha256": sha(args.out / "queue.json"), "sources": len(projected),
        "proposals": len(handoffs), "pending": len(pending), "training_ready": False,
        "artifacts": {name: sha(args.out / name) for name in ("queue.json", "progress.json", "progress.html")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corrections", "meaning", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("corrections-sha256", "meaning-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--originals", action="append", required=True)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--checkpoints", required=True)
    build(parser.parse_args())
