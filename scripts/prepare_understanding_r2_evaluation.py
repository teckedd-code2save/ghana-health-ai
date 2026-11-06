#!/usr/bin/env python3
"""Freeze existing development cases and protected language-loss inputs. CPU/offline only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata

VERSION = "understanding-r2-evaluation-v1"
R1_VARIANTS = ("base", "meaning-adapter", "response-adapter")
CHECKPOINTS = ("base", "r1_meaning", "r1_response", "r2_language", "r2_meaning", "r2_response")
LIVE_IDS = {"reported-informal", "standard", "negative", "third-person", "name", "english-control"}
DECODING = {"do_sample": False, "enable_thinking": False, "max_new_tokens": 256,
            "num_beams": 1, "repetition_penalty": 1.0, "seed": 17, "truncate": False}
LIMITS = [
    "Development diagnostics already inspected by researchers; not population accuracy or a blind final test.",
    "No new gold labels, native-speaker approval, semantic or clinical certification is inferred.",
    "Whole-field exact/normalized checks do not certify semantic or near-duplicate independence.",
    "Prior base pretraining exposure is unknown. A true blind final test remains future work.",
    "Saved comparisons require identical messages, prompt token IDs, tokenizer, template, decoding and runtime.",
    "No new system prompts, repetition guards, postprocessing, fallback models or tools in the paired protocol.",
    "Language loss is separate from chat response quality; protected validation text is never training or demonstration material.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse(text):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, "Non-finite JSON: " + value))


def read(path):
    return parse(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            require(bool(text.strip()), "Blank JSONL row")
            yield line, parse(text)


def safe(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path != root, "Unsafe artifact path")
    return path


def checked_manifest(folder):
    manifest = read(Path(folder) / "manifest.json")
    for name, checksum in manifest["artifacts"].items():
        require(sha(safe(folder, name)) == checksum, "Changed artifact: " + name)
    return manifest


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text).split())


def strings(value, path="$", depth=0):
    if isinstance(value, str):
        if value.strip():
            yield path, value
        if depth < 3 and value.lstrip().startswith(("{", "[")):
            try:
                yield from strings(parse(value), path + ".decoded", depth + 1)
            except (ValueError, RecursionError):
                pass
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, path + "." + key, depth)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f"{path}[{index}]", depth)


def language_holdouts(snapshot):
    """Use the existing snapshot pointers, not a fresh sample from the raw release."""
    checked_manifest(snapshot)
    batches = {b["batch_id"]: b for b in read(snapshot / "batches.json")}
    selected = defaultdict(dict)
    for _, entry in rows(snapshot / "records.jsonl"):
        if entry["collection"] == "language" and entry["split"] == "validation":
            selected[entry["batch_id"], entry["file"]][entry["line"]] = entry
    accepted, accounting, count = [], Counter(), 0
    for (batch_id, filename), entries in sorted(selected.items()):
        batch = batches[batch_id]
        source = safe(batch["release_path"], filename)
        require(sha(source) == batch["files"][filename], "Changed validation source")
        found = set()
        for line, row in rows(source):
            if line not in entries:
                continue
            entry = entries[line]
            require(digest(row) == entry["row_sha256"] and row["id"] == entry["id"]
                    and row["source_id"] == entry["source_id"] and row["split"] == "validation", "Changed validation row")
            found.add(line)
            count += 1
            eligible = (row.get("record_type") in {"sentence", "utterance"}
                        and row.get("origin") in {"compiled_parallel_text", "speech_transcript", "source_preserved"}
                        and row.get("generated_target") is False and row.get("language") == "tw"
                        and isinstance(row.get("text"), str) and bool(row["text"].strip()))
            accounting["source_backed_sentence_or_utterance" if eligible else "outside_loss_subset"] += 1
            if eligible:
                accepted.append({"id": row["id"], "text": row["text"], "source_id": row["source_id"],
                    "metadata": {k: v for k, v in row.items() if k != "text"},
                    "original_ref": {**entry, "file_sha256": batch["files"][filename],
                                     "release_manifest_sha256": batch["release_manifest_sha256"]}})
        require(found == set(entries), "Missing held-out source rows")
    require(accepted and len({r["id"] for r in accepted}) == len(accepted), "Empty or duplicated language holdouts")
    return accepted, {"existing_validation_rows": count, "selection": dict(accounting)}


def source_cases(r1_pack, r1_results, live_results):
    execution = checked_manifest(r1_pack)
    suite = checked_manifest(r1_pack / "evaluation")
    require(suite["protocol"]["decoding"] == execution["decoding"] == DECODING, "Unexpected R1 decoding")
    original = read(r1_pack / "evaluation/cases.json")
    require(len(original) == 30 and len({c["id"] for c in original}) == 30, "Expected original 30 R1 cases")
    cases = [{**c, "source_split": c["split"], "split": "development", "cohort": "r1_reused",
              "human_review": {"status": "pending"}, "training_eligible": False} for c in original]
    proof = read(r1_pack / "training-proof.json")
    execution_sha = sha(r1_pack / "manifest.json")
    results_manifest = read(r1_results / "results-manifest.json")
    require(results_manifest["execution_manifest_sha256"] == execution_sha, "R1 result execution mismatch")
    prompts = {p["id"]: p for p in read(r1_pack / "prompts.json")}
    runtime = {"environment_sha256": digest(execution["runtime"]),
        "tokenizer_sha256": execution["artifacts"]["tokenizer/tokenizer.json"],
        "chat_template_sha256": execution["artifacts"]["tokenizer/chat_template.jinja"]}
    references = []
    for variant in R1_VARIANTS:
        path = r1_results / (variant + ".json")
        require(sha(path) == results_manifest["artifacts"][path.name], "Changed R1 results")
        result = read(path)
        adapter = None if variant == "base" else proof["adapters"][variant]["identity_sha256"]
        require(result["load_verified"] is True and result["decoding"] == DECODING and result["runtime"] == runtime
                and result["suite_manifest_sha256"] == sha(r1_pack / "evaluation/manifest.json")
                and result["model"] == {"id": suite["model"], "revision": suite["revision"],
                    "weights_sha256": execution["weight"]["sha256"], "adapter_sha256": adapter}, "Unmatched saved R1 protocol/identity")
        require(len(result["records"]) == 30 and {r["case_id"] for r in result["records"]} == set(prompts), "Incomplete R1 references")
        for record in result["records"]:
            prompt = prompts[record["case_id"]]
            case = next(c for c in cases if c["id"] == record["case_id"])
            require(record["messages_sha256"] == prompt["messages_sha256"]
                    == digest(case["messages"])
                    and record["prompt_token_ids_sha256"] == digest(prompt["prompt_ids"]), "Saved prompt mismatch")
            references.append({"case_id": record["case_id"], "variant": variant, "record": record,
                "source_path": str(path.resolve()), "source_sha256": sha(path), "runtime": runtime, "decoding": DECODING})
    live = {}
    for path in sorted(live_results.glob("*.json")):
        saved = read(path)
        spec, result = saved["spec"], saved["result"]
        if spec["profile"] != "raw":
            continue
        cid, variant, messages = spec["case"], spec["variant"], spec["messages"]
        require(cid in LIVE_IDS and variant in R1_VARIANTS and len(messages) == 1
                and messages[0]["role"] == "user", "Noncanonical raw regression")
        identity = result["identity"]
        require(spec["execution_sha"] == identity["execution_manifest_sha256"] == execution_sha
                and result["load_verified"] is True and result["decoding"] == DECODING
                and identity["model"] == suite["model"] and identity["revision"] == suite["revision"]
                and identity["variant"] == result["variant"] == variant
                and identity["base_weights_sha256"] == execution["weight"]["sha256"]
                and identity["adapter_weights_sha256"] == (None if variant == "base" else proof["adapters"][variant]["weights"]["sha256"])
                and identity["runtime_sha256"] == runtime["environment_sha256"]
                and identity["tokenizer_sha256"] == runtime["tokenizer_sha256"]
                and identity["chat_template_sha256"] == runtime["chat_template_sha256"]
                and result["messages_sha256"] == digest(messages), "Unmatched live regression protocol/identity")
        require(cid not in live or live[cid] == messages, "Conflicting canonical raw prompt")
        live[cid] = messages
        references.append({"case_id": "live-" + cid, "variant": variant, "record": result,
            "source_path": str(path.resolve()), "source_sha256": sha(path), "runtime": runtime, "decoding": DECODING})
    require(set(live) == LIVE_IDS, "Missing canonical raw cases")
    require(len({(r["case_id"], r["variant"]) for r in references}) == len(references), "Repeated reference slot")
    for reference in references:
        reference["checkpoint_id"] = {"base": "base", "meaning-adapter": "r1_meaning", "response-adapter": "r1_response"}[reference["variant"]]
    for cid, messages in sorted(live.items()):
        cases.append({"id": "live-" + cid, "cohort": "live_raw", "split": "development", "messages": messages,
            "profile": "raw", "objective": None, "human_review": {"status": "pending"}, "training_eligible": False,
            "provenance": {"origin": "previously_observed_live_regression", "source_case_id": cid}})
    return cases, references, suite, execution, proof


def seal(out, files, metadata, copies=None):
    require(not out.exists(), "Output exists: verify it or choose a new version; never overwrite frozen packs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        staged = Path(temp) / "pack"
        staged.mkdir()
        for name, value in files.items():
            safe(staged, name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, path in (copies or {}).items():
            shutil.copyfile(path, safe(staged, name))
        manifest = {"version": VERSION, "builder_sha256": sha(__file__), "frozen_at": datetime.now(timezone.utc).isoformat(),
            "training_eligible": False, "final_test": False, "native_review": "pending", "limits": LIMITS, **metadata,
            "artifacts": {p.name: sha(p) for p in sorted(staged.iterdir())}}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staged / "SEALED.json").write_text(json.dumps({"manifest_sha256": sha(staged / "manifest.json")}) + "\n")
        staged.rename(out)
    return manifest


def freeze_sources(r1_pack, r1_results, live_results, snapshot, out):
    cases, references, suite, execution, proof = source_cases(r1_pack, r1_results, live_results)
    heldout, counts = language_holdouts(snapshot)
    loss = {"split": "validation", "purpose": "development_language_loss", "training_eligible": False,
        "expose_as_examples": False, "objective": "native_causal_lm", "chat_template": False,
        "encoding": "encode original text with add_special_tokens=False; append EOS=248046; no BOS",
        "labels": "all original token IDs plus EOS; causal shift excludes first token; padded positions=-100",
        "aggregation": "sum cross-entropy over logits[:-1],labels[1:] / total nonmasked shifted target tokens",
        "report": ["rows", "predicted_tokens", "summed_nll", "token_weighted_mean_nll", "exp_mean_nll"],
        "max_sequence_tokens": 4096, "truncate": False, "concatenate_unrelated_rows": False,
        "chat_loss": "Report separately using final-assistant masks; never pool with causal-language loss.",
        "model_mode": "eval; no gradients or optimizer updates", **counts}
    plan = {"checkpoints": [
        {"id": "base", "origin": "unadapted_base", "weights_sha256": execution["weight"]["sha256"]},
        *[{"id": "r1_" + name, "origin": "r1", "saved_variant": variant,
           "adapter_identity_sha256": proof["adapters"][variant]["identity_sha256"],
           "adapter_weights_sha256": proof["adapters"][variant]["weights"]["sha256"]}
          for name, variant in (("meaning", "meaning-adapter"), ("response", "response-adapter"))],
        *[{"id": "r2_" + name, "origin": "r2", "training_stage": stage,
           "identity_status": "pending_verified_training_export"}
          for name, stage in (("language", "language"), ("meaning", "meaning"), ("response", "responses"))]],
        "primary_pairs": [["base", "r2_language"], ["r1_meaning", "r2_meaning"], ["r1_response", "r2_response"]],
        "adapter_loading": "Each stage export is standalone cumulative LoRA; load one adapter onto the pinned base, never stack stages.",
        "chat_case_ids": [c["id"] for c in cases], "language_loss_checkpoint_ids": list(CHECKPOINTS),
        "saved_reference_counts": dict(Counter(r["checkpoint_id"] for r in references)),
        "missing_saved_slots": [{"checkpoint_id": checkpoint, "case_id": c["id"]}
            for checkpoint in CHECKPOINTS[:3] for c in cases
            if not any(r["checkpoint_id"] == checkpoint and r["case_id"] == c["id"] for r in references)],
        "missing_policy": "Generate missing slots under the exact frozen protocol; never substitute prompted profiles or a different checkpoint.",
        "reporting": "Per-case development diagnostics and specified JSON task scores; not general accuracy. Native review pending."}
    return seal(out, {"cases.json": cases, "reference-records.json": references,
        "language-heldout.json": heldout, "language-loss.json": loss, "comparison-plan.json": plan}, {
        "kind": "protected_sources", "protocol": suite["protocol"], "model": suite["model"], "revision": suite["revision"],
        "runtime": execution["runtime"], "counts": {"r1_reused": 30, "live_raw": 6, "language_loss_rows": len(heldout)},
        "bindings": {"r1_execution_sha256": sha(r1_pack / "manifest.json"),
            "r1_cases_sha256": sha(r1_pack / "evaluation/cases.json"), "snapshot_manifest_sha256": sha(snapshot / "manifest.json")}},
        {"r1-original-cases.json": r1_pack / "evaluation/cases.json", "r1-execution.json": r1_pack / "manifest.json"})


def audit_overlap(cases, heldout, witnesses):
    index, identities, hits = defaultdict(list), defaultdict(list), []
    for item in cases:
        units = [(f"messages[{i}].content", m["content"]) for i, m in enumerate(item["messages"])]
        if item.get("objective"):
            units.append(("objective.expected", json.dumps(item["objective"]["expected"], ensure_ascii=False, sort_keys=True)))
        for field, text in units:
            for method, value in (("exact", text), ("normalized", normalize(text))):
                index[method, value].append((item["id"], field))
    for item in heldout:
        for method, value in (("exact", item["text"]), ("normalized", normalize(item["text"]))):
            index[method, value].append((item["id"], "language.text"))
        for key in ("source_id", "group_id", "source_hash"):
            value = item["metadata"].get(key, item.get(key))
            if value:
                identities[key, value].append(item["id"])
    count = 0
    for line, row in witnesses:
        count += 1
        for field, text in strings(row):
            for method, value in (("exact", text), ("normalized", normalize(text))):
                for cid, protected_field in index.get((method, value), []):
                    hits.append({"protected_id": cid, "protected_field": protected_field, "method": method,
                        "training_line": line, "training_id": row["id"], "training_field": field})
        metadata = row.get("metadata", {})
        for key in ("source_id", "group_id", "source_hash"):
            value = metadata.get(key, row.get(key))
            for cid in identities.get((key, value), []):
                hits.append({"protected_id": cid, "method": key, "training_line": line, "training_id": row["id"]})
    return {"training_witness_rows": count, "matches": hits, "overlap_free": not hits,
            "near_duplicate_certification": False, "scope": "all selected pack witnesses, including language.text and embedded English/Twi fields"}


def verify_sources(pack):
    manifest = checked_manifest(pack)
    require(read(pack / "SEALED.json") == {"manifest_sha256": sha(pack / "manifest.json")}, "Changed evaluation seal")
    require(manifest["version"] == VERSION and manifest["builder_sha256"] == sha(__file__)
            and manifest["final_test"] is False and manifest["training_eligible"] is False, "Changed evaluation contract")
    cases = read(pack / "cases.json")
    original = read(pack / "r1-original-cases.json")
    require(sha(pack / "r1-original-cases.json") == manifest["bindings"]["r1_cases_sha256"], "Changed R1 case source")
    require(sha(pack / "r1-execution.json") == manifest["bindings"]["r1_execution_sha256"], "Changed R1 execution source")
    execution = read(pack / "r1-execution.json")
    require(execution["artifacts"]["evaluation/cases.json"] == manifest["bindings"]["r1_cases_sha256"]
            and execution["decoding"] == DECODING and execution["runtime"] == manifest["runtime"], "R1 protocol/source binding differs")
    require(len(cases) == 36 and len({c["id"] for c in cases}) == 36, "Expected 30 + 6 distinct cases")
    expected = [{**c, "source_split": c["split"], "split": "development", "cohort": "r1_reused",
                 "human_review": {"status": "pending"}, "training_eligible": False} for c in original]
    require(cases[:30] == expected and manifest["protocol"]["decoding"] == DECODING, "R1 cases or protocol changed")
    require({c["id"] for c in cases[30:]} == {"live-" + c for c in LIVE_IDS}
            and all(c["profile"] == "raw" and c["split"] == "development" and len(c["messages"]) == 1
                    and c["messages"][0]["role"] == "user" and c["objective"] is None for c in cases[30:]), "Changed canonical live cases")
    references = read(pack / "reference-records.json")
    require(len({(r["case_id"], r["checkpoint_id"]) for r in references}) == len(references), "Repeated saved reference slot")
    by_id = {c["id"]: c for c in cases}
    for reference in references:
        require(reference["checkpoint_id"] == {"base": "base", "meaning-adapter": "r1_meaning", "response-adapter": "r1_response"}[reference["variant"]]
                and reference["decoding"] == DECODING
                and reference["record"]["messages_sha256"] == digest(by_id[reference["case_id"]]["messages"]), "Saved reference protocol or checkpoint differs")
    plan = read(pack / "comparison-plan.json")
    require([c["id"] for c in plan["checkpoints"]] == list(CHECKPOINTS)
            and plan["chat_case_ids"] == [c["id"] for c in cases], "Ambiguous checkpoint comparison plan")
    heldout = read(pack / "language-heldout.json")
    require(len(heldout) == manifest["counts"]["language_loss_rows"] and all(
        r["metadata"]["split"] == "validation" and r["metadata"]["record_type"] in {"sentence", "utterance"}
        and r["metadata"]["generated_target"] is False and r["metadata"]["language"] == "tw" for r in heldout), "Invalid protected language subset")
    return manifest, cases


def prepare(training_pack, sources, out):
    source_manifest, cases = verify_sources(sources)
    training = checked_manifest(training_pack)
    require(training["model"] == source_manifest["model"] and training["revision"] == source_manifest["revision"], "R2 base differs")
    require(training["snapshot_manifest_sha256"] == source_manifest["bindings"]["snapshot_manifest_sha256"], "Different validation snapshot")
    heldout = read(sources / "language-heldout.json")
    audit = audit_overlap(cases, heldout, rows(training_pack / "source-witnesses.jsonl"))
    require(audit["training_witness_rows"] == training["selected_rows"], "Training witness count mismatch")
    copies = {name: sources / name for name in source_manifest["artifacts"]}
    copies["training-manifest.json"] = training_pack / "manifest.json"
    metadata = {k: source_manifest[k] for k in ("protocol", "model", "revision", "runtime", "counts")}
    return seal(out, {"overlap-audit.json": audit}, {**metadata, "kind": "training_bound",
        "ready_for_development_evaluation": audit["overlap_free"], "bindings": {**source_manifest["bindings"],
            "sources_manifest_sha256": sha(sources / "manifest.json"), "training_manifest_sha256": sha(training_pack / "manifest.json"),
            "training_witnesses_sha256": sha(training_pack / "source-witnesses.jsonl")}}, copies)


def verify_pack(pack, training_pack=None):
    pack = Path(pack)
    manifest, cases = verify_sources(pack)
    require(manifest["kind"] == "training_bound", "Training binding is pending")
    require(sha(pack / "training-manifest.json") == manifest["bindings"]["training_manifest_sha256"], "Changed training binding")
    audit = read(pack / "overlap-audit.json")
    require(manifest["ready_for_development_evaluation"] is True and audit["overlap_free"] is True
            and audit["matches"] == [], "Training overlap blocks this pack; inspect overlap-audit.json")
    if training_pack is not None:
        training_pack = Path(training_pack)
        checked_manifest(training_pack)
        require(sha(training_pack / "manifest.json") == manifest["bindings"]["training_manifest_sha256"]
                and sha(training_pack / "source-witnesses.jsonl") == manifest["bindings"]["training_witnesses_sha256"], "Wrong R2 training pack")
        require(audit == audit_overlap(cases, read(pack / "language-heldout.json"), rows(training_pack / "source-witnesses.jsonl")), "Overlap audit cannot be reproduced")
    return manifest, cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources")
    for name in ("r1-pack", "r1-results", "live-results", "snapshot", "out"):
        sources.add_argument("--" + name, type=Path, required=True)
    bind = sub.add_parser("prepare")
    for name in ("training-pack", "sources", "out"):
        bind.add_argument("--" + name, type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--pack", type=Path, required=True)
    check.add_argument("--training-pack", type=Path)
    args = parser.parse_args()
    if args.command == "sources":
        result = freeze_sources(args.r1_pack, args.r1_results, args.live_results, args.snapshot, args.out)
    elif args.command == "prepare":
        result = prepare(args.training_pack, args.sources, args.out)
    else:
        result, _ = verify_pack(args.pack, args.training_pack)
    print(json.dumps(result, indent=2))
