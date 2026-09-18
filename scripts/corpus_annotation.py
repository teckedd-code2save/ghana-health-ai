"""Full-source field plan and bounded Modal-only, resumable annotation batches."""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from corpus_release import ROOT, digest, jsonl, save, sha

PROMPT_VERSION = "source-grounded-fields-v2"
FIELDS = {"meaning_english", "record_function", "source_grounded_entities", "ambiguity",
          "education_or_individual_assessment", "source_answer_support", "medical_contradictions",
          "grounded_twi_derivative", "conversational_intent"}


def existing_annotations():
    """Recover prior evidence, without upgrading model consensus to human review."""
    source_path = ROOT / "data/medical-response-corpus/afrihealth-akan-source.v1.jsonl"
    sources = {r["id"]: r for r in jsonl(source_path)}
    records, input_files, rejected = {}, {str(source_path.relative_to(ROOT)): sha(source_path)}, []
    for name in ("afrihealth-akan-annotations.v1.jsonl", "afrihealth-teacher-v3-train.jsonl"):
        path = source_path.with_name(name)
        input_files[str(path.relative_to(ROOT))] = sha(path)
        for annotation in jsonl(path):
            source = sources.get(annotation["row_id"])
            if not source or annotation["source_record_hash"] != source["record_source_hash"]:
                rejected.append({"row_id": annotation["row_id"], "reason": "stale_source_hash"})
                continue
            proposals = annotation.get("proposals", [])
            selected = next((r for r in proposals if r["proposal_id"] == annotation["recommended_proposal_id"]), None)
            if not selected:
                rejected.append({"row_id": annotation["row_id"], "reason": "missing_selected_proposal"})
                continue
            question, answer = source["question_twi_source"], source["answer_twi_source"]
            evidence_ok = all(span in question or span in answer for span in selected.get("evidence_spans_twi", []))
            entities_ok = all(text in question for texts in selected.get("entities", {}).values() for text in texts)
            key = digest([question, answer])
            records[key] = {"source_record_id": source["source_record_id"], "prior_annotation": annotation,
                            "selected": selected, "source_hash_verified": True,
                            "evidence_spans_valid": evidence_ok, "entities_valid": entities_ok,
                            "human_reviewed": False, "auto_accept": False}
    return records, input_files, rejected


def prepare(release):
    folder = release / "annotation"
    original = folder / "pending.jsonl"
    destination = folder / "work-plan.jsonl"
    if destination.exists() and (folder / "work-plan.json").exists():
        info = json.loads((folder / "work-plan.json").read_text())
        if sha(original) != info["parent_sha256"] or sha(destination) != info["work_plan_sha256"]:
            raise ValueError("Changed parent plan; create a new release")
        for name, expected in info["existing_annotation_files"].items():
            if sha(ROOT / name) != expected: raise ValueError("Changed previous annotations; create a new release")
        print(json.dumps(info)); return
    existing, files, rejected = existing_annotations()
    counts, reused = {}, 0
    plan_temp = destination.with_suffix(".tmp")
    preserved = folder / "preserved-annotations.jsonl"
    preserved_temp = preserved.with_suffix(".tmp")
    with plan_temp.open("w") as handle, preserved_temp.open("w") as recovered:
        for row in jsonl(original):
            previous = existing.get(digest([row["source_text"], row.get("reference_answer")]))
            retained = {}
            missing = list(row["missing_fields"])
            if "record_function" in missing and "conversational_intent" not in missing:
                missing.append("conversational_intent")
            if row.get("reference_english"):
                retained["meaning_english"] = {"value": row["reference_english"], "status": "existing_source_reference"}
                missing = [f for f in missing if f != "meaning_english"]
            if previous:
                reused += 1
                recovered.write(json.dumps({"source_id": row["source_id"], "source_hash": row["source_hash"], **previous}, ensure_ascii=False) + "\n")
                selected = previous["selected"]
                if selected.get("question_english") and "meaning_english" not in retained:
                    retained["meaning_english"] = {"value": selected["question_english"], "status": "existing_model_annotation_unreviewed"}
                if previous["entities_valid"]:
                    retained["source_grounded_entities"] = {"value": selected["entities"], "status": "existing_exact_span_checked"}
                # Medical correctness is intentionally NOT inherited from consensus.
                missing = [f for f in missing if f not in retained]
            for field in missing:
                counts[field] = counts.get(field, 0) + 1
            planned = {**row, "missing_fields": missing, "retained_fields": retained,
                       "prompt_revision": PROMPT_VERSION, "provider": "modal",
                       "request_id": digest([row["source_hash"], sorted(missing), PROMPT_VERSION])}
            handle.write(json.dumps(planned, ensure_ascii=False) + "\n")
    plan_temp.replace(destination)
    preserved_temp.replace(preserved)
    info = {"provider": "modal", "row_limit": None, "parent_sha256": sha(original),
            "work_plan_sha256": sha(destination), "existing_annotation_files": files,
            "preserved_annotation_records": reused, "missing_by_field": counts,
            "stale_or_invalid_records": rejected, "no_api_fallback": True, "qualified_teacher_required": True}
    save(folder / "work-plan.json", info)
    print(json.dumps(info, indent=2))


def prompt(row):
    if not set(row["missing_fields"]) <= FIELDS:
        raise ValueError("Unknown requested fields")
    system = """Annotate ONLY the requested missing fields from the supplied source. Return one JSON object, with exactly those field names. Never rewrite existing references or retained fields. Do not invent an answer to a narrative or fragment. Preserve negation, experiencer, time, quantities, uncertainty and speaker roles. Do not change a source author's identity into the assistant's identity.
meaning_english: faithful English meaning, or null if uncertain.
record_function: narrative, request, question, dictionary, fragment, or uncertain.
conversational_intent: a concise source-grounded action such as greeting, asking for an explanation, or searching for a product; null for narratives, dictionary entries, fragments, or unresolved intent. A broad HEALTH or COMMERCE topic is not an intent. Never invent a patient request from a narrative.
source_grounded_entities: list of {type, text, start, end}; exact source_text character offsets, no inferred entities.
ambiguity: list of unresolved readings or missing context, not invented facts.
education_or_individual_assessment: education, individual_assessment, or uncertain.
source_answer_support: {status: needs_expert_review|unsupported|source_consistent, evidence: [exact reference_answer spans], issues: [strings]}. Textual consistency is NOT clinical certification. If unsupported treatment claims, contraindications, or contradictions cannot be resolved, require expert review.
medical_contradictions: list of {claim: exact reference_answer span, concern: explanation}; [] does NOT mean clinically approved.
grounded_twi_derivative: translate the COMPLETE supplied original messages into natural Twi, preserving every turn and role. Return {messages:[{role,content}], uncertainties:[strings]}. Keep the meaning, reasoning, quantities, units and uncertainty. No invented history or facts. Source-conditioned generation is not a new independent source. If translation is uncertain say so in uncertainties; do not fabricate fluency.
No self-confidence score. No external tools or actions."""
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps({k: row.get(k) for k in
        ("source_text", "reference_english", "reference_answer", "messages", "missing_fields", "retained_fields")}, ensure_ascii=False)}]


def validate_fields(row, values):
    if not isinstance(values, dict) or set(values) != set(row["missing_fields"]):
        raise ValueError("Missing or unexpected generated fields")

    def strings(value):
        return isinstance(value, list) and all(isinstance(t, str) and t.strip() for t in value)

    if "meaning_english" in values and values["meaning_english"] is not None and (not isinstance(values["meaning_english"], str) or not values["meaning_english"].strip()):
        raise ValueError("Meaning must be nonempty text or null")
    if "record_function" in values and values["record_function"] not in ("narrative", "request", "question", "dictionary", "fragment", "uncertain"):
        raise ValueError("Invalid record function")
    intent = values.get("conversational_intent")
    if intent is not None and (not isinstance(intent, str) or not intent.strip() or intent.upper() in ("HEALTH", "COMMERCE")):
        raise ValueError("Intent must be a specific action or null")
    if intent is not None and values.get("record_function") in ("narrative", "dictionary", "fragment", "uncertain"):
        raise ValueError("Do not invent conversational intent for non-conversational text")
    if "education_or_individual_assessment" in values and values["education_or_individual_assessment"] not in ("education", "individual_assessment", "uncertain"):
        raise ValueError("Invalid health classification")
    if "ambiguity" in values and not strings(values["ambiguity"]):
        raise ValueError("Ambiguity must be a list of readings")
    if "source_grounded_entities" in values and not isinstance(values["source_grounded_entities"], list):
        raise ValueError("Entities must be a list")
    for entity in values.get("source_grounded_entities", []):
        if not isinstance(entity, dict) or set(entity) != {"type", "text", "start", "end"}:
            raise ValueError("Invalid entity schema")
        if not isinstance(entity["type"], str) or not entity["type"].strip():
            raise ValueError("Entity type is required")
        start, end = entity["start"], entity["end"]
        if type(start) != int or type(end) != int or not 0 <= start < end <= len(row["source_text"]) or row["source_text"][start:end] != entity["text"]:
            raise ValueError("Entity not an exact source span")
    support = values.get("source_answer_support")
    if "source_answer_support" in values:
        if (not isinstance(support, dict) or set(support) != {"status", "evidence", "issues"}
                or support["status"] not in ("needs_expert_review", "unsupported", "source_consistent")
                or not strings(support["evidence"]) or not strings(support["issues"])
                or any(t not in (row.get("reference_answer") or "") for t in support["evidence"])
                or (support["status"] == "source_consistent" and not support["evidence"])):
            raise ValueError("Unsupported evidence")
    if "medical_contradictions" in values:
        if not isinstance(values["medical_contradictions"], list):
            raise ValueError("Contradictions must be a list")
        for issue in values["medical_contradictions"]:
            if (not isinstance(issue, dict) or set(issue) != {"claim", "concern"}
                    or not all(isinstance(issue[k], str) and issue[k].strip() for k in issue)
                    or issue["claim"] not in (row.get("reference_answer") or "")):
                raise ValueError("Medical concern must refer to an exact source claim")
    derivative = values.get("grounded_twi_derivative")
    if "grounded_twi_derivative" in values:
        if not isinstance(derivative, dict) or set(derivative) != {"messages", "uncertainties"} or not strings(derivative["uncertainties"]):
            raise ValueError("Invalid conversation derivative")
        original = row.get("messages") or []
        messages = derivative.get("messages", [])
        if not isinstance(messages, list) or any(not isinstance(m, dict) or set(m) != {"role", "content"} for m in messages):
            raise ValueError("Invalid message schema")
        if not original or [m["role"] for m in messages] != [m["role"] for m in original]:
            raise ValueError("Changed conversation roles/history")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError("Empty conversation target")
    return values


def run(release, gate_path, max_batches):
    import modal
    from corpus_teacher import MODEL, REVISION
    if type(max_batches) != int or max_batches < 1:
        raise ValueError("A positive explicit batch bound is required")
    folder = release / "annotation"
    info = json.loads((folder / "work-plan.json").read_text())
    if sha(folder / "work-plan.jsonl") != info["work_plan_sha256"]:
        raise ValueError("Changed immutable plan")
    gate = json.loads(gate_path.read_text())
    if (gate.get("qualified") is not True or gate.get("bulk_annotation_authorized_by_this_gate") is not True
            or not gate.get("review_evidence") or gate.get("structural_failures")):
        raise ValueError("Teacher qualification has not passed; no GPU call made")
    if (gate.get("model"), gate.get("revision")) != (MODEL, REVISION):
        raise ValueError("Qualification does not cover the configured teacher")
    if sha(gate_path.parent / "qualify.results.json") != gate.get("results_sha256"):
        raise ValueError("Changed qualification evidence")
    requests_dir = folder / "requests"; requests_dir.mkdir(exist_ok=True)
    batches = 0
    rows = (r for r in jsonl(folder / "work-plan.jsonl") if r["missing_fields"])
    batch = []
    for row in rows:
        if not set(row["missing_fields"]) <= set(gate.get("validated_tasks", [])):
            raise ValueError("Qualification does not cover all requested tasks; no unqualified batch is submitted")
        # Batch membership is immutable, even if a crash saved only some rows.
        batch.append(row)
        if len(batch) < 32:
            continue
        batches += int(execute_batch(batch, requests_dir, modal))
        batch = []
        if batches >= max_batches:
            break
    if batch and batches < max_batches:
        execute_batch(batch, requests_dir, modal)


def execute_batch(batch, directory, modal):
    complete = set()
    for row in batch:
        path = directory / (row["request_id"] + ".json")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["source_hash"] != row["source_hash"] or previous["request_id"] != row["request_id"]:
                raise ValueError("Stale result; never overwrite")
            complete.add(row["request_id"])
    if len(complete) == len(batch): return False
    receipt_path = directory / ("batch-" + digest([r["request_id"] for r in batch]) + ".receipt.json")
    report_path = receipt_path.with_suffix(".results.json")
    inputs = [{"id": r["request_id"], "messages": prompt(r)} for r in batch]
    if report_path.exists():
        report = json.loads(report_path.read_text())
    elif receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request_ids"] != [r["request_id"] for r in batch]:
            raise ValueError("Changed batch membership")
        if not receipt.get("call_id"):
            raise ValueError("Uncertain request submission; inspect before retry")
        call = modal.FunctionCall.from_id(receipt["call_id"])
        report = call.get(timeout=1250)
        save(report_path, report)
    else:
        if complete: raise ValueError("Partial results without the original batch receipt; inspect before submitting")
        run_id = "corpus_teacher_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        save(receipt_path, {"status": "submitting", "run_id": run_id, "request_ids": [r["request_id"] for r in batch]})
        call = modal.Function.from_name("ghana-corpus-teacher-qualification", "qualify").spawn(run_id, inputs)
        save(receipt_path, {"status": "submitted", "run_id": run_id, "call_id": call.object_id,
                            "request_ids": [r["request_id"] for r in batch]})
        report = call.get(timeout=1250)
        save(report_path, report)
    by_id = {r["id"]: r for r in report["results"]}
    if len(by_id) != len(report["results"]) or set(by_id) != {r["request_id"] for r in batch}:
        raise ValueError("Incomplete or duplicate batch output; recover the original request")
    for row in batch:
        if row["request_id"] in complete: continue
        output = by_id.get(row["request_id"])
        if not output:
            raise ValueError("Incomplete batch; retain receipt and recover")
        try:
            if output["finish_reason"] != "stop": raise ValueError("Truncated output")
            values = validate_fields(row, json.loads(output["text"]))
            state, error = "machine_screened_needs_semantic_review", None
        except (ValueError, TypeError, KeyError) as exc:
            state, values, error = "rejected", None, str(exc)
        save(directory / (row["request_id"] + ".json"), {"source_id": row["source_id"], "source_hash": row["source_hash"],
            "request_id": row["request_id"], "fields": values, "status": state, "error": error,
            "raw_output": output, "model": report["model"], "revision": report["revision"],
            "human_reviewed": False, "training_eligible": False})
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if args.action == "plan": prepare(args.release)
    else:
        if not args.gate or not args.max_batches or args.max_batches < 1:
            parser.error("Execution requires --gate and an explicitly bounded --max-batches; the plan covers all sources")
        run(args.release, args.gate, args.max_batches)
