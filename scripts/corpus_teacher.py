"""Modal-only qualification receipts. Never silently fall back to a paid API."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from corpus_release import ROOT, digest, save, sha

APP = "ghana-corpus-teacher-qualification"
OUT = ROOT / "tmp/corpus-teacher/qwen235-v1"
PROMPT = "faithful-context-translation-v1"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
REVISION = "e156cb4efae43fbee1a1ab073f946a1377e6b969"


def prepare():
    if OUT.exists():
        raise ValueError("Qualification snapshot exists; resume it instead")
    folder = ROOT / "tmp/dialogue-translation-calibration/v1"
    sources = json.loads((folder / "inputs.json").read_text())
    requests, references = [], []
    for row in sources:
        for turn, message in enumerate(row["messages"]):
            identity = row["id"] + ":" + str(turn)
            requests.append({"id": identity, "messages": [
                {"role": "system", "content": "Translate the selected English turn into natural Twi (Akan), using the original conversation only to resolve references. Preserve meaning, negation, quantities, units, uncertainty, and speaker role. Translate the complete selected turn. Do not answer the question, add a diagnosis or instruction, or invent facts. Return only the Twi translation."},
                {"role": "user", "content": json.dumps({"original_conversation": row["messages"], "selected_turn": turn}, ensure_ascii=False)}]})
            references.append({"id": identity, "source_id": row["id"], "turn": turn, "original_en": message["content"], "role": message["role"], "kind": "saved_dialogue_comparison"})
    controls = [
        ("negation", "Menni sika.", "I do not have money.", ["not", "no", "don't"]),
        ("experiencer", "Me ba no ho ayɛ hyew.", "My child has a fever / feels hot.", ["child", "baby", "son", "daughter"]),
        ("quantity", "Mepɛ sɛ metɔ tomato 2 kg.", "I want to buy 2 kg of tomatoes.", ["2"]),
        ("location", "Fa brɛ me wɔ Adenta.", "Bring it to me at Adenta.", ["adenta"]),
        ("time", "Mɛkɔ ɔkyena.", "I will go tomorrow.", ["tomorrow"]),
        ("identity", "Wo ho te sɛn?", "How are you?", ["how"]),
        ("eye", "M'ani yɛ me yaw.", "My eye hurts.", ["eye"]),
        ("uncertainty", "Ebia ɔbɛba ɔkyena.", "Maybe they will come tomorrow.", ["maybe", "perhaps", "might", "may"]),
    ]
    for identity, text, meaning, required in controls:
        identity = "control:" + identity
        requests.append({"id": identity, "messages": [
            {"role": "system", "content": "Translate this Twi utterance faithfully into English. Preserve negation, person, time, quantities and uncertainty. Return only its meaning, not advice or an answer. Do not infer facts not stated."},
            {"role": "user", "content": text}]})
        references.append({"id": identity, "original_tw": text, "reference_en": meaning,
                           "required_any": required, "kind": "project_development_control_not_training"})
    OUT.mkdir(parents=True)
    save(OUT / "requests.json", requests)
    save(OUT / "references.json", references)
    save(OUT / "manifest.json", {"prompt_revision": PROMPT, "requests_sha256": sha(OUT / "requests.json"),
         "reference_sha256": sha(OUT / "references.json"), "count": len(requests), "training_eligible": False,
         "comparison_sources": {name: sha(folder / name) for name in ("inputs.json", "review.json")},
         "max_gpu_seconds": 1200, "gpu_count": 2, "provider": "modal", "proprietary_fallback": False})
    print(json.dumps({"prepared": len(requests), "out": str(OUT)}))


def status(kind, wait):
    import modal
    path = OUT / (kind + ".receipt.json")
    receipt = json.loads(path.read_text())
    if receipt["state"] in ("completed", "failed"):
        print(json.dumps(receipt)); return
    if not receipt.get("call_id"):
        raise ValueError("Uncertain submission; inspect Modal before resubmission")
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=wait)
    except TimeoutError:
        print(json.dumps({"state": "pending", "kind": kind, "call_id": receipt["call_id"]})); return
    except Exception as exc:
        save(path, {**receipt, "state": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    save(OUT / (kind + ".results.json"), result)
    save(path, {**receipt, "state": "completed", "result_sha256": sha(OUT / (kind + ".results.json"))})
    print(json.dumps({"state": "completed", "kind": kind}))


def submit(kind):
    import modal
    path = OUT / (kind + ".receipt.json")
    if path.exists():
        raise ValueError("Existing receipt: use status, never duplicate a request")
    manifest = json.loads((OUT / "manifest.json").read_text())
    if sha(OUT / "requests.json") != manifest["requests_sha256"]:
        raise ValueError("Changed requests")
    run_id = "corpus_teacher_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = {"state": "submitting", "kind": kind, "run_id": run_id}
    with path.open("x") as handle:
        json.dump(receipt, handle)
    if kind == "prepare":
        call = modal.Function.from_name(APP, "prepare").spawn()
    else:
        cached = json.loads((OUT / "prepare.receipt.json").read_text())
        if cached["state"] != "completed":
            raise ValueError("CPU weight preparation must complete before allocating GPUs")
        call = modal.Function.from_name(APP, "qualify").spawn(run_id, json.loads((OUT / "requests.json").read_text()))
    save(path, {**receipt, "state": "submitted", "call_id": call.object_id})
    print(json.dumps({"kind": kind, "call_id": call.object_id, "run_id": run_id}))


def review():
    if not (OUT / "qualify.results.json").exists():
        receipt = json.loads((OUT / "qualify.receipt.json").read_text())
        if receipt.get("state") not in ("failed", "cancelled_by_agent"):
            raise ValueError("Qualification is still running")
        gate = {"qualified": False, "status": "infrastructure_failure_no_quality_result",
                "receipt_sha256": sha(OUT / "qualify.receipt.json"), "outputs_measured": 0,
                "reason": receipt.get("error", receipt.get("reason")),
                "bulk_annotation_authorized_by_this_gate": False,
                "confidence_or_agreement_is_acceptance": False}
        save(OUT / "gate.json", gate)
        print(json.dumps(gate))
        return
    refs = json.loads((OUT / "references.json").read_text())
    report = json.loads((OUT / "qualify.results.json").read_text())
    if (report["model"], report["revision"]) != (MODEL, REVISION):
        raise ValueError("Unexpected teacher model/revision")
    results = {r["id"]: r for r in report["results"]}
    if set(results) != {r["id"] for r in refs}:
        raise ValueError("Incomplete qualification")
    previous = json.loads((ROOT / "tmp/dialogue-translation-calibration/v1/review.json").read_text())
    previous_by_id = {r["source"]["id"]: r for r in previous}
    rows, failures = [], []
    for r in refs:
        result = results[r["id"]]
        flags = []
        if result["finish_reason"] != "stop":
            flags.append("incomplete")
        if r.get("required_any") and not any(t in result["text"].lower() for t in r["required_any"]):
            flags.append("meaning_control_failed")
        alternatives = [{"model": report["model"], "revision": report["revision"], "text": result["text"]}]
        if r.get("source_id"):
            old = next(a for a in previous_by_id[r["source_id"]]["alternatives"] if a["variant"] == "gemma" and "context" in a["method"])
            alternatives.append({"model": old["model"], "revision": old["revision"], "text": old["turns"][r["turn"]]["translation"]})
        rows.append({**r, "alternatives": alternatives, "flags": flags, "human_decision": None, "training_eligible": False})
        failures.extend({"id": r["id"], "flag": flag} for flag in flags)
    save(OUT / "review.json", rows)
    save(OUT / "gate.json", {"qualified": False, "model": MODEL, "revision": REVISION,
         "results_sha256": sha(OUT / "qualify.results.json"), "validated_tasks": [], "structural_failures": failures,
         "status": "failed_controls" if failures else "awaiting_semantic_and_native_review",
         "confidence_or_agreement_is_acceptance": False,
         "required": "All meaning-critical controls plus representative dialogue meaning and native naturalness review",
         "bulk_annotation_authorized_by_this_gate": False})
    print(json.dumps({"review_rows": len(rows), "automatic_failures": len(failures), "qualified": False}))


def retry_loading():
    """Preserve the failed loader attempt and reuse CPU-prepared weights/inputs."""
    import modal
    import shutil
    source = ROOT / "tmp/corpus-teacher/qwen235-v1"
    path = source / "qualify.receipt.json"
    receipt = json.loads(path.read_text())
    call = modal.FunctionCall.from_id(receipt["call_id"])
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        call.cancel(terminate_containers=True)
        save(path, {**receipt, "state": "cancelled_by_agent", "reason": "Lazy mmap over Modal volume projected beyond 20-minute limit; replacing with eager shard reads", "cancelled_at": datetime.now(timezone.utc).isoformat()})
    else:
        save(source / "qualify.results.json", result)
        raise ValueError("Result already completed; inspect it instead of retrying")
    if OUT == source or OUT.exists():
        raise ValueError("Retry requires a new --out directory")
    OUT.mkdir(parents=True)
    for name in ("manifest.json", "references.json", "requests.json", "prepare.receipt.json", "prepare.results.json"):
        shutil.copyfile(source / name, OUT / name)
    save(OUT / "parent-attempt.json", {"source": str(source), "receipt": receipt, "changed_variable": "lazy to eager checkpoint loading; identical weights and requests"})
    print(json.dumps({"retry_prepared": str(OUT), "previous_call_cancelled": receipt["call_id"]}))


def staged_retry():
    import shutil
    source = ROOT / "tmp/corpus-teacher/qwen235-v2-eager"
    receipt = json.loads((source / "qualify.receipt.json").read_text())
    if receipt.get("state") != "failed": raise ValueError("Previous attempt must be terminal; no cancellation here")
    if OUT.exists(): raise ValueError("A new attempt directory is required")
    OUT.mkdir(parents=True)
    for name in ("manifest.json", "references.json", "requests.json", "prepare.receipt.json", "prepare.results.json"):
        shutil.copyfile(source / name, OUT / name)
    save(OUT / "parent-attempt.json", {"source": str(source), "receipt_sha256": sha(source / "qualify.receipt.json"),
        "changed_variable": "Verified four-worker ephemeral local staging; lazy local reads; warm engine reuse; same weights and 40 requests",
        "io_probe_call": "fc-01M2ETK68C8WWWT5YJ1NK6YQ2M", "estimated_staging_seconds": 119.35415246394783,
        "probe_is_not_gpu_loading_measurement": True, "max_gpu_seconds": 1200,
        "implementation_sha256": {name: sha(ROOT / "modal/train" / name) for name in ("corpus_checkpoint.py", "qualify_corpus_teacher.py")}})
    print(json.dumps({"retry_prepared": str(OUT), "training": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "submit", "status", "review", "retry-loading", "staged-retry"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--kind", choices=("prepare", "qualify"), default="prepare")
    parser.add_argument("--wait", type=int, default=2)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if args.action == "snapshot": prepare()
    elif args.action == "submit": submit(args.kind)
    elif args.action == "status": status(args.kind, args.wait)
    elif args.action == "review": review()
    elif args.action == "staged-retry": staged_retry()
    else: retry_loading()
