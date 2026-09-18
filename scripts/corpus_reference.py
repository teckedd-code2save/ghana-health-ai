"""Reference-backed annotation sidecar; sealed sources and reviews stay unchanged."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from corpus_release import ROOT, digest, jsonl, save, sha
from corpus_reference_core import MODEL, REVISION, PROMPT_VERSION, SYSTEM, METHODS, CLARIFICATIONS, messages, validate, control_score, evidence_offsets

APP = "ghana-corpus-reference-annotation"
DEFAULT = ROOT / "tmp/corpus-reference/v1-20260915"
PARENT = ROOT / "tmp/corpus-releases/twi-stage-v1-20260911"


def blocked_source_ids(reviews):
    latest = {}
    for row in reviews:
        field = row.get("correction_field", "meaning_english")
        if row.get("reference_analysis_id"):
            if field not in ("meaning_english", "suggested_normalization"):
                continue
            if row["decision"] == "reviewed" and not row["correction"].strip() and row.get("issue", "none") == "none":
                continue
        elif field not in ("meaning_english", "suggested_normalization") and row["decision"] != "exclude":
            continue
        latest[(row["id"], field)] = row
    return {row["id"] for row in latest.values() if row["decision"] != "reviewed" or
            row["correction"].strip() or row["selected"] >= 0}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def controls():
    values = [
        ("negation", "I do not have money.", ["narrative"], {"negation": ["not"]}),
        ("experiencer", "My child has a fever.", ["narrative"], {"experiencer": ["My child"]}),
        ("quantity", "I want to buy 2 kg of tomatoes.", ["request"], {"quantities": ["2 kg"]}),
        ("location", "Bring it to me at Adenta.", ["request"], {}),
        ("time", "I will go tomorrow.", ["narrative"], {"time": ["tomorrow"]}),
        ("greeting", "How are you?", ["question"], {}),
        ("uncertainty", "Maybe they will come tomorrow.", ["narrative"], {"uncertainty": ["Maybe"], "time": ["tomorrow"]}),
        ("identity", "I am a 25-year-old teacher, not a doctor.", ["narrative"], {"negation": ["not"], "quantities": ["25"]}),
        ("glossary", "heart attack", ["dictionary", "fragment"], {}),
        ("quoted_request", "He said, 'Buy two tomatoes tomorrow', but she did not agree.", ["narrative"], {"negation": ["not"], "quantities": ["two"], "time": ["tomorrow"]}),
    ]
    return [{"id": "control:" + name, "reference_english": text, "kind": "development_control_not_training",
             "expected": {"function": function, "features": features, "intent": bool(set(function) & {"request", "question"})}}
            for name, text, function, features in values]


def sync_reviews(parent, out):
    command = "test ! -f /opt/ghana-corpus-review/reviews/twi-stage-v1-20260911/decisions.jsonl || cat /opt/ghana-corpus-review/reviews/twi-stage-v1-20260911/decisions.jsonl"
    raw = subprocess.check_output(["ssh", "-i", str(Path.home() / ".ssh/optimi_github_actions"),
        "-o", "BatchMode=yes", "root@128.140.12.62", command])
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    db = sqlite3.connect(f"file:{parent / 'ledger.sqlite3'}?mode=ro", uri=True)
    seen = set()
    for row in rows:
        if row["release"] != parent.name or row["request_id"] in seen:
            raise ValueError("Invalid review identity")
        seen.add(row["request_id"])
        record = db.execute("SELECT payload FROM records WHERE id=?", (row["id"],)).fetchone()
        if not record or json.loads(zlib.decompress(record[0]))["record_hash"] != row["source_hash"]:
            raise ValueError("Review source hash does not match frozen source")
    db.close()
    (out / "reviews.jsonl").write_bytes(raw)
    save(out / "review-snapshot.json", {"journal_sha256": sha(out / "reviews.jsonl"),
        "actions": len(rows), "distinct_sources": len({r["id"] for r in rows}),
        "fetched_at": datetime.now(timezone.utc).isoformat(), "remote_journal_modified": False,
        "owner_feedback": "Data generally looks good; proceed with next steps.",
        "blanket_row_approval": False, "clinical_certification": False})
    return rows


def prepare(parent, out):
    if out.exists():
        raise ValueError("Existing work directory; resume instead")
    manifest = json.loads((parent / "manifest.json").read_text())
    files = ["annotation/work-plan.jsonl", "collections/meaning/alignment/train.jsonl",
             "collections/meaning/alignment/validation.jsonl"]
    for name in files:
        if sha(parent / name) != manifest["artifacts"][name]:
            raise ValueError("Frozen parent changed: " + name)
    out.mkdir(parents=True)
    reviews = sync_reviews(parent, out)
    blocked = blocked_source_ids(reviews)
    accepted = {r["source_id"]: r for name in files[1:] for r in jsonl(parent / name)
                if r["record_type"] in ("sentence", "document")}
    planned, accounted = [], []
    for row in jsonl(parent / files[0]):
        why = ("no_existing_english_reference" if not row.get("reference_english") else
               "outside_screened_sentence_alignment" if row["source_id"] not in accepted else
               "review_requires_resolution" if row["source_id"] in blocked else None)
        if why:
            accounted.append({"source_id": row["source_id"], "source_hash": row["source_hash"], "status": "pending_other_work", "reason": why})
            continue
        source = accepted[row["source_id"]]
        if (source["source_hash"], source["twi"], source["english"]) != (row["source_hash"], row["source_text"], row["reference_english"]):
            raise ValueError("Alignment/reference mismatch")
        reference_hash = digest(row["reference_english"])
        planned.append({**row, "reference_hash": reference_hash,
            "request_id": digest([row["source_hash"], reference_hash, MODEL, REVISION, PROMPT_VERSION]),
            "review_actions": [r for r in reviews if r["id"] == row["source_id"]]})
        accounted.append({"source_id": row["source_id"], "source_hash": row["source_hash"], "status": "planned_reference_analysis"})
    planned.sort(key=lambda r: r["source_hash"])
    validation = [r for r in planned if r["split"] == "validation"]
    # Sample both suppliers plus negatives, times, numbers and vocabulary fragments.
    selected = {}
    selectors = [lambda r: r["source_id"].startswith("ghana_nlp"), lambda r: r["source_id"].startswith("community"),
                 lambda r: len(r["reference_english"].split()) < 4,
                 lambda r: any(x in r["reference_english"].lower().split() for x in ("not", "never", "no")),
                 lambda r: any(c.isdigit() for c in r["reference_english"]),
                 lambda r: any(x in r["reference_english"].lower() for x in ("year", "when", "during", "before"))]
    for predicate in selectors:
        for row in [r for r in validation if predicate(r)][:5]:
            selected[row["source_id"]] = row
    for row in validation:
        if len(selected) >= 32:
            break
        selected[row["source_id"]] = row
    qualification = [{"id": r["request_id"], "kind": "source_validation", "source_id": r["source_id"],
        "source_hash": r["source_hash"], "reference_english": r["reference_english"], "source_text": r["source_text"],
        "split": r["split"], "group_id": r["group_id"]} for r in selected.values()] + controls()
    write_rows(out / "plan.jsonl", planned)
    write_rows(out / "source-accounting.jsonl", accounted)
    save(out / "qualification.json", qualification)
    save(out / "manifest.json", {"parent": str(parent), "parent_manifest_sha256": sha(parent / "manifest.json"),
        "source_artifacts": {name: sha(parent / name) for name in files}, "model": MODEL, "revision": REVISION,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(SYSTEM),
        "plan_sha256": sha(out / "plan.jsonl"), "qualification_sha256": sha(out / "qualification.json"),
        "source_accounting_sha256": sha(out / "source-accounting.jsonl"), "planned": len(planned),
        "splits": dict(Counter(r["split"] for r in planned)), "accounted": len(accounted),
        "statuses": dict(Counter(r["status"] for r in accounted)), "no_new_training": True,
        "pending_tasks": ["Twi source entity offsets", "missing English translations", "Twi multi-turn responses", "medical certification"]})
    print(json.dumps(json.loads((out / "manifest.json").read_text()), indent=2))


def verify_plan(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for file, key in (("plan.jsonl", "plan_sha256"), ("qualification.json", "qualification_sha256"),
                      ("source-accounting.jsonl", "source_accounting_sha256")):
        if sha(out / file) != manifest[key]:
            raise ValueError("Changed immutable annotation input: " + file)
    if (MODEL, REVISION, PROMPT_VERSION, digest(SYSTEM)) != (manifest["model"], manifest["revision"], manifest["prompt_version"], manifest["prompt_sha256"]):
        raise ValueError("Changed annotation method; create a new version")
    if sha(Path(manifest["parent"]) / "manifest.json") != manifest["parent_manifest_sha256"]:
        raise ValueError("Changed sealed parent")
    return manifest


def invoke(out, name, inputs=None, wait=0):
    import modal
    folder = out / "calls"; folder.mkdir(exist_ok=True)
    receipt_path, results_path = folder / (name + ".receipt.json"), folder / (name + ".results.json")
    request_hash = digest(inputs)
    if results_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request_hash"] != request_hash or receipt["result_sha256"] != sha(results_path):
            raise ValueError("Changed saved result/request")
        return json.loads(results_path.read_text())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request_hash"] != request_hash or not receipt.get("call_id"):
            raise ValueError("Uncertain or changed submission; inspect before retry")
        if receipt.get("state") == "failed":
            raise ValueError("Previous call failed; inspect without resubmission")
    else:
        run_id = "reference_" + digest([str(out), name, request_hash])[:24]
        receipt = {"state": "submitting", "run_id": run_id, "request_hash": request_hash,
                   "model": MODEL, "revision": REVISION}
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle)
        fn = modal.Function.from_name(APP, "prepare" if name == "prepare" else "annotate")
        call = fn.spawn() if name == "prepare" else fn.spawn(run_id, inputs)
        receipt.update(state="submitted", call_id=call.object_id)
        save(receipt_path, receipt)
        print(json.dumps({"submitted": name, "call_id": call.object_id}), flush=True)
    try:
        report = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=wait)
    except TimeoutError:
        print(json.dumps({"pending": name, "call_id": receipt["call_id"]}), flush=True)
        return None
    except Exception as exc:
        save(receipt_path, {**receipt, "state": "failed", "error": str(exc)})
        raise
    if (report["model"], report["revision"]) != (MODEL, REVISION):
        raise ValueError("Unexpected remote model")
    save(results_path, report)
    save(receipt_path, {**receipt, "state": "completed", "result_sha256": sha(results_path)})
    return report


def requests(rows):
    return [{"id": r.get("request_id", r.get("id")), "messages": messages(r["reference_english"], SYSTEM)} for r in rows]


def inspect_outputs(rows, report):
    outputs = {r["id"]: r for r in report["results"]}
    if len(outputs) != len(report["results"]) or set(outputs) - {r.get("request_id", r.get("id")) for r in rows}:
        raise ValueError("Unexpected or duplicate output")
    results = []
    for row in rows:
        identity = row.get("request_id", row.get("id"))
        output = outputs.get(identity)
        flags, analysis = [], None
        try:
            if not output:
                raise ValueError("Missing remote result; recover original call")
            if output["finish_reason"] != "stop":
                raise ValueError("Truncated annotation")
            analysis, flags = validate(row["reference_english"], json.loads(output["text"]))
            if row.get("expected"):
                flags += control_score(row["expected"], analysis)
        except (ValueError, TypeError, KeyError) as exc:
            flags.append(str(exc))
        results.append({"id": identity, "source_id": row.get("source_id"), "source_hash": row.get("source_hash"),
            "split": row.get("split"), "reference_english": row["reference_english"], "kind": row.get("kind", "source"),
            "analysis": analysis, "flags": flags, "status": "needs_correction" if flags else "automated_reference_checked",
            "human_reviewed": False, "semantic_certification": False,
            "reference_evidence_offsets": evidence_offsets(row["reference_english"], analysis) if analysis else [],
            "raw_output": output})
    return results


def qualification(out, wait):
    verify_plan(out)
    if not (out / "calls/prepare.results.json").exists():
        raise ValueError("CPU preparation must finish first")
    rows = json.loads((out / "qualification.json").read_text())
    report = invoke(out, "qualification", requests(rows), wait)
    if report is None:
        return
    results = inspect_outputs(rows, report)
    save(out / "qualification-review.json", results)
    save(out / "qualification-checks.json", {"results_sha256": sha(out / "calls/qualification.results.json"),
        "review_sha256": sha(out / "qualification-review.json"), "rows": len(rows),
        "failures": [{"id": r["id"], "flags": r["flags"]} for r in results if r["flags"]],
        "passed_automatic_checks": sum(not r["flags"] for r in results),
        "semantic_review_required": True, "qualified": False})
    print(json.dumps(json.loads((out / "qualification-checks.json").read_text()), indent=2))


def diagnose(out, wait):
    verify_plan(out)
    if MODEL != METHODS["oss120"][0]:
        raise ValueError("This diagnostic is only for the missing Harmony final channels")
    report = json.loads((out / "calls/qualification.results.json").read_text())
    receipt = json.loads((out / "calls/qualification.receipt.json").read_text())
    if sha(out / "calls/qualification.results.json") != receipt["result_sha256"]:
        raise ValueError("Changed qualification results")
    missing = {r["id"] for r in report["results"] if not r["text"].strip()}
    if not 1 <= len(missing) <= 3:
        raise ValueError("Inspect other failures separately; at most three missing final channels")
    inputs = [r for r in json.loads((out / "qualification.json").read_text())
              if r["id"] in missing or r["id"] == "control:experiencer"]
    result = invoke(out, "diagnostic-final-channel", requests(inputs), wait)
    if result:
        print(json.dumps({"diagnostic_only": True, "does_not_replace_qualification": True,
            "outputs": [{"id": r["id"], "parsed_fields": r.get("parsed_fields"),
                "last_token_id": r.get("last_token_id"), "stop_reason": r.get("stop_reason"),
                "final_channel_missing": r.get("final_channel_missing")} for r in result["results"]]}, indent=2))


def gate(out):
    checks = json.loads((out / "qualification-checks.json").read_text())
    review = json.loads((out / "semantic-audit.json").read_text())
    rows = json.loads((out / "qualification-review.json").read_text())
    if checks["results_sha256"] != sha(out / "calls/qualification.results.json") or checks["review_sha256"] != sha(out / "qualification-review.json"):
        raise ValueError("Changed qualification results")
    if review["review_sha256"] != checks["review_sha256"] or review["critical_failures"] or checks["failures"]:
        raise ValueError("Reference-analysis qualification failed")
    if set(review["checked_ids"]) != {r["id"] for r in rows} or not review["reviewer"]:
        raise ValueError("Incomplete semantic inspection")
    return {"qualified_task": "english_reference_analysis_only", "translation_qualified": False,
        "model": MODEL, "revision": REVISION, "prompt_version": PROMPT_VERSION,
        "human_reviewed": review["human_reviewed"], "audit_sha256": sha(out / "semantic-audit.json"),
        "qualifies_clinical_advice": False, "scope": "Screened source references only; exact evidence checks and correction queue remain required"}


def run(out, max_batches):
    verify_plan(out)
    if not max_batches or max_batches < 1:
        raise ValueError("Explicit positive batch bound required")
    qualification_gate = gate(out)
    save(out / "gate.json", qualification_gate)
    rows = list(jsonl(out / "plan.jsonl"))
    selected = {r["id"] for r in json.loads((out / "qualification.json").read_text()) if r["kind"] == "source_validation"}
    pending = [r for r in rows if r["request_id"] not in selected]
    size = 256
    layout = {"size": size, "request_ids_sha256": digest([r["request_id"] for r in pending])}
    layout_path = out / "batching.json"
    if layout_path.exists() and json.loads(layout_path.read_text()) != layout:
        raise ValueError("Batch layout changed; do not repeat completed requests")
    save(layout_path, layout)
    for index in range(min(max_batches, (len(pending) + size - 1) // size)):
        batch = pending[index * size:(index + 1) * size]
        name = "batch-" + digest([r["request_id"] for r in batch])[:16]
        report = invoke(out, name, requests(batch), wait=920)
        if report is None or len(report["results"]) != len(batch):
            raise ValueError("Batch incomplete; inspect saved call before continuing")
        inspected = inspect_outputs(batch, report)
        write_rows(out / "inspected" / (name + ".jsonl"), inspected)
        print(json.dumps({"batch": index + 1, "rows": len(batch), "flagged": sum(bool(r["flags"]) for r in inspected),
                          "seconds": report["last_attempt_seconds"]}), flush=True)
        if sum(bool(r["flags"]) for r in inspected) / len(batch) > .10:
            raise ValueError("Over 10% require correction; pause scaling and inspect")


def export(out):
    manifest = verify_plan(out)
    rows = {r["request_id"]: r for r in jsonl(out / "plan.jsonl")}
    generated = {}
    if (out / "qualification-review.json").exists():
        qualification_rows = json.loads((out / "qualification.json").read_text())
        report = invoke(out, "qualification", requests(qualification_rows))
        generated.update({r["id"]: r for r in inspect_outputs(qualification_rows, report) if r["kind"] == "source_validation"})
    for path in sorted((out / "calls").glob("batch-*.results.json")):
        report = json.loads(path.read_text())
        receipt = json.loads(path.with_name(path.name.replace(".results.json", ".receipt.json")).read_text())
        if sha(path) != receipt["result_sha256"] or (report["model"], report["revision"]) != (MODEL, REVISION):
            raise ValueError("Changed remote batch output")
        batch = [rows[r["id"]] for r in report["results"]]
        for row in inspect_outputs(batch, report):
            if row["id"] in generated:
                raise ValueError("Duplicated annotation result")
            generated[row["id"]] = row
    exports, corrections, pending = [], [], []
    for identity, source in rows.items():
        result = generated.get(identity)
        if not result:
            pending.append({"source_id": source["source_id"], "request_id": identity, "status": "pending"})
            continue
        if (result["source_hash"], result["reference_english"], result["split"]) != (source["source_hash"], source["reference_english"], source["split"]):
            raise ValueError("Annotation/source boundary mismatch")
        value = {**result, "group_id": source["group_id"], "twi": source["source_text"], "reference_preserved": True,
                 "source_file": source["source_file"], "source_file_sha256": source["source_file_sha256"],
                 "source_revision": source["source_revision"], "license": source["license"],
                 "model": MODEL, "revision": REVISION, "prompt_version": PROMPT_VERSION,
                 "review_actions": source["review_actions"], "source_twi_entity_offsets": None,
                 "response_training_eligible": False, "annotation_language": "en", "derived_from": "existing_english_reference"}
        (corrections if result["flags"] else exports).append(value)
    write_rows(out / "exports/reference-analysis.jsonl", exports)
    write_rows(out / "exports/corrections.jsonl", corrections)
    write_rows(out / "exports/pending.jsonl", pending)
    training = []
    if (out / "gate.json").exists():
        gate(out)
        for row in exports:
            if row["analysis"]["record_function"] in ("dictionary", "fragment", "uncertain"):
                continue
            target = {"meaning_english": row["reference_english"], **row["analysis"]}
            training.append({"id": row["id"], "source_id": row["source_id"], "source_hash": row["source_hash"],
                "group_id": row["group_id"], "split": row["split"], "task": "reference_grounded_structured_understanding",
                "language": "tw", "target_language": "en", "annotation_origin": "model_analysis_of_existing_english_reference",
                "human_reviewed": False, "semantic_certification": False, "response_training_eligible": False,
                "license": row["license"], "source_file": row["source_file"], "source_file_sha256": row["source_file_sha256"],
                "source_revision": row["source_revision"], "messages": [
                    {"role": "system", "content": "Interpret the Twi source. Return its faithful English meaning and structured analysis as JSON. Evidence quotes must refer to that English meaning, not Twi character offsets. Do not answer the source question or give advice."},
                    {"role": "user", "content": row["twi"]},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]})
        for split in ("train", "validation"):
            write_rows(out / "exports" / ("structured-" + split + ".jsonl"), [r for r in training if r["split"] == split])
    group_splits = {}
    for row in exports:
        previous = group_splits.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Cross-collection source-group split conflict")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from corpus_release import REGISTRY
    token_spec = json.loads(REGISTRY.read_text())["tokenizers"][0]
    tokenizer = Tokenizer.from_file(hf_hub_download(token_spec["repo"], "tokenizer.json", revision=token_spec["revision"], local_files_only=True))
    token_counts = Counter()
    for row in training:
        token_counts[row["split"]] += sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) for m in row["messages"])
    (out / "exports/DATASET_CARD.md").write_text(
        "# Reference-Grounded Understanding Annotations\n\n"
        "Private research derivative of twi-stage-v1-20260911. Original Twi and English references are unchanged. "
        "The model analyzed English references only; this is not a new translation model or proof of Twi fluency.\n\n"
        "Structured train/validation views contain source Twi inputs and English meaning/analysis targets. "
        "Evidence quotes refer to English, NOT Twi offsets. Dictionary, fragment and uncertain records remain outside these structured SFT views. "
        "Original source groups and validation boundaries are preserved. These are automatically checked research targets, not human-gold labels or direct-response targets.\n\n"
        "Reference faithfulness remains a separate assumption. Exact-span checks do not prove completeness, correct semantic categories, or clinical safety. "
        "Generated entity/intent fields are not covered by earlier approvals of source translations. "
        "Licenses and pinned source identifiers remain attached to every record; no public redistribution or commercial clearance is granted here.\n\n"
        "See ../coverage.json for measured processing coverage and ../semantic-audit.json for qualification inspection. "
        "corrections.jsonl contains flagged outputs; pending.jsonl is unfinished work. No source corrections or training jobs are applied automatically.\n",
        encoding="utf-8")
    save(out / "coverage.json", {"planned_sources": len(rows), "completed": len(exports) + len(corrections),
        "automated_reference_checked": len(exports), "needs_correction": len(corrections), "pending": len(pending),
        "function_counts": dict(Counter(r["analysis"]["record_function"] for r in exports)),
        "splits": dict(Counter(r["split"] for r in exports)), "new_human_reviews": 0,
        "new_twi_responses": 0, "new_twi_translations": 0, "synthetic_source_rows": 0,
        "synthetic_controls_excluded_from_training": 10,
        "structured_research_views": dict(Counter(r["split"] for r in training)),
        "structured_content_tokens": dict(token_counts), "counting_tokenizer": token_spec,
        "generated_annotation_share": 1 if exports else 0,
        "annotation_counts_are_not_clinical_or_translation_accuracy": True,
        "parent_manifest_sha256": manifest["parent_manifest_sha256"],
        "artifacts": {str(p.relative_to(out)): sha(p) for p in (out / "exports").glob("*.jsonl")}})
    print(json.dumps(json.loads((out / "coverage.json").read_text()), indent=2))


def projection(out):
    manifest = verify_plan(out)
    coverage = json.loads((out / "coverage.json").read_text())
    source = out / "exports/reference-analysis.jsonl"
    if sha(source) != coverage["artifacts"]["exports/reference-analysis.jsonl"]:
        raise ValueError("Changed screened annotation export")
    target = out / "review-projection.sqlite3"
    if target.exists():
        raise ValueError("Projection already exists; use a new version to publish changes")
    with sqlite3.connect(target) as db:
        db.execute("CREATE TABLE annotations(source_id TEXT PRIMARY KEY,source_hash TEXT,payload TEXT)")
        for row in jsonl(source):
            payload = {"id": digest([row["id"], row["analysis"]]), "source_id": row["source_id"],
                "source_hash": row["source_hash"], "reference_english": row["reference_english"],
                "analysis": row["analysis"], "status": row["status"], "model": MODEL,
                "evidence_language": "en", "human_reviewed": False, "semantic_certification": False}
            db.execute("INSERT INTO annotations VALUES(?,?,?)", (row["source_id"], row["source_hash"], json.dumps(payload, ensure_ascii=False)))
    save(out / "review-projection.json", {"parent_manifest_sha256": manifest["parent_manifest_sha256"],
        "projection_sha256": sha(target), "sources": coverage["automated_reference_checked"],
        "source_release_modified": False, "reviews_modified": False})


def retry_loading(source, out):
    import shutil
    verify_plan(source)
    receipt = json.loads((source / "calls/qualification.receipt.json").read_text())
    if receipt["state"] != "failed" or "IncompleteSnapshotError" not in receipt.get("error", ""):
        raise ValueError("Only the confirmed pre-generation snapshot failure can use this repair")
    if out.exists():
        raise ValueError("Preserve the failed attempt; use a new directory")
    out.mkdir(parents=True)
    for name in ("manifest.json", "plan.jsonl", "source-accounting.jsonl", "qualification.json", "reviews.jsonl", "review-snapshot.json"):
        shutil.copyfile(source / name, out / name)
    (out / "calls").mkdir()
    for name in ("prepare.receipt.json", "prepare.results.json"):
        shutil.copyfile(source / "calls" / name, out / "calls" / name)
    save(out / "parent-attempt.json", {"source": str(source), "receipt_sha256": sha(source / "calls/qualification.receipt.json"),
        "changed_variable": "Request the same cached artifact patterns as CPU preparation, not the missing irrelevant .gitattributes file",
        "inputs_unchanged": True, "previous_generated_rows": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "prepare", "qualify", "diagnose", "run", "export", "projection", "retry-loading"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--from-existing", type=Path, default=DEFAULT)
    parser.add_argument("--teacher", choices=tuple(METHODS), default="qwen9")
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    MODEL, REVISION, PROMPT_VERSION, APP = METHODS[args.teacher]
    if args.teacher != "qwen9": SYSTEM += CLARIFICATIONS
    if args.action not in ("plan", "retry-loading"):
        import fcntl
        writer_lock = (args.out / ".writer.lock").open("a")
        fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.action == "plan": prepare(args.parent.resolve(), args.out.resolve())
    elif args.action == "prepare": verify_plan(args.out); invoke(args.out, "prepare", wait=args.wait)
    elif args.action == "qualify": qualification(args.out, args.wait)
    elif args.action == "diagnose": diagnose(args.out, args.wait)
    elif args.action == "run": run(args.out, args.max_batches)
    elif args.action == "projection": projection(args.out)
    elif args.action == "retry-loading": retry_loading(args.from_existing.resolve(), args.out.resolve())
    else: export(args.out)
