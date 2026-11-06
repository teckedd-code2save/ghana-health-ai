#!/usr/bin/env python3
"""Offline mechanical preflight, NOT a training admission or model-quality result."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal/train"))
from afrique_response_core import encode as encode_chat

MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
POLICY = "afrique-instruct-preflight-v1"
ASSET_HASHES = {
    "config.json": "d9e8e6f6712e871947fbfa7b29a63cfdd8105550ba02213f8641ffbeb19c9647",
    "generation_config.json": "c27f87e67b78f2d75ef30922e1eb1a7cd2254959f85848561829cf3c15d6f724",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
}
EOS, PAD = 248046, 248044
KINDS = {"twi_prose", "language_exercise", "code_exercise", "literal_label", "literal_structure"}
SYSTEM = "Answer the user's actual request in the requested language. Preserve supplied context and uncertainty. Ask for missing information rather than inventing facts or claiming actions you have not performed."
BLOCKERS = [
    "No current-instruction baseline-only runner or strict real-weight load receipt verified.",
    "Freeze source-disjoint evaluation inputs, decoding, budgets and review criteria before baseline.",
    "Training needs a versioned data bridge, refreshed admission/protection evidence and explicit mixture.",
    "Existing runners do not implement tested full optimizer/scheduler/RNG/sampler resume.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: require(False, "Nonfinite JSON: " + value))


def read(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL row: {path}:{number}")
            row = loads(line)
            require(isinstance(row, dict), f"Expected object: {path}:{number}")
            yield row


def indexed(values, key):
    result = {}
    for row in values:
        identity = row[key]
        require(isinstance(identity, str) and bool(identity.strip()) and identity not in result,
                "Empty or duplicate identity: " + str(identity))
        result[identity] = row
    return result


def bound_file(folder, name, checksum):
    path = (folder / name).resolve()
    require(path.is_relative_to(folder.resolve()), "Escaping artifact path")
    require(isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum), "Invalid SHA-256")
    require(sha(path) == checksum, "Changed artifact: " + name)
    return path


def check_identity(model, revision):
    require((model, revision) == (MODEL, REVISION),
            "Wrong foundation: require the exact pinned 9B 50Langs Instruct checkpoint, not CPT/Qwen/4B")


def load_tokenizer(folder):
    from transformers import TokenizersBackend
    for name, checksum in ASSET_HASHES.items():
        bound_file(folder, name, checksum)
    # Text-only projection of the pinned tokenizer_config.json. The older local
    # evidence snapshot lacks that file; never let AutoTokenizer select a fallback.
    tokenizer = TokenizersBackend(
        tokenizer_file=str(folder / "tokenizer.json"),
        chat_template=(folder / "chat_template.jinja").read_text(encoding="utf-8"),
        eos_token="<|im_end|>", pad_token="<|endoftext|>", bos_token=None, unk_token=None,
        model_max_length=262144, clean_up_tokenization_spaces=False, padding_side="right",
    )
    require((len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id) == (248077, EOS, PAD),
            "Unexpected tokenizer vocabulary or stop IDs")
    # Backend all_special_ids need not include every reserved ID in tokenizer.json.
    tokenizer.contract_special_ids = {t["id"] for t in read(folder / "tokenizer.json")["added_tokens"] if t["special"]}
    return tokenizer


def encode_complete(tokenizer, row, max_length):
    require(type(max_length) is int and 0 < max_length <= tokenizer.model_max_length, "Invalid context length")
    messages = row["messages"]
    require(isinstance(messages, list) and bool(messages) and all(isinstance(m, dict) for m in messages), "Missing/invalid messages")
    # This data bridge deliberately supports text conversations only. Tools get
    # separate serialization probes, not implicit admission as executed actions.
    require(not row.get("tools"), "Tool data needs a separately admitted bridge")
    start = int(messages[0].get("role") == "system")
    turns = messages[start:]
    require(bool(turns) and len(turns) % 2 == 0, "Incomplete conversation")
    for i, message in enumerate(messages):
        require(set(message) == {"role", "content"}, "Unsupported message fields")
        role = "system" if start and i == 0 else ("user" if (i - start) % 2 == 0 else "assistant")
        require(message["role"] == role, "Invalid conversation roles")
        content = message["content"]
        require(isinstance(content, str) and bool(content.strip()), "Empty message")
        require(not any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
                        (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in content),
                "Invalid Unicode in message")
        require(not tokenizer.contract_special_ids.intersection(tokenizer.encode(content, add_special_tokens=False)),
                "Reserved source token")
    encoded = encode_chat(tokenizer, row, max_length=max_length)
    require(encoded is not None, "overlength_no_truncation: " + row.get("id", "probe"))
    target = [v for v in encoded["labels"] if v != -100]
    ids = encoded["input_ids"]
    require(len(target) > 1 and target[-1] == EOS and target.count(EOS) == 1, "Invalid native target EOS")
    require(encoded["labels"] == [-100] * (len(ids) - len(target)) + target, "Invalid assistant-only mask")
    require(encoded["attention_mask"] == [1] * len(ids), "Invalid attention mask")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                              add_generation_prompt=False).rstrip("\n")
    require(tokenizer.encode(rendered, add_special_tokens=False) == ids, "Token boundary mismatch")
    decoded = tokenizer.decode(target[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    require(decoded == messages[-1]["content"].strip(), "Assistant answer changed beyond native boundary trim")
    return encoded


def check_export(folder, tokenizer, max_length):
    from jsonschema import Draft202012Validator
    manifest, seal = read(folder / "manifest.json"), read(folder / "SEALED.json")
    require(seal["manifest_sha256"] == sha(folder / "manifest.json") and seal["source_reconstruction"] is True,
            "Missing or changed historical export seal")
    require(manifest["policy"] == "instruction-training-export-v3" and manifest["training_ready"] is True
            and manifest["production_eligible"] is False, "Unsupported export policy")
    required = {"schema.json", "data-config.json", "admission-ledger.jsonl", "target-evidence.jsonl"}
    required |= {kind + suffix for kind in KINDS for suffix in (".train.jsonl", ".tokens.jsonl")}
    require(required <= set(manifest["artifacts"]), "Unbound required export artifacts")
    for name, checksum in manifest["artifacts"].items():
        bound_file(folder, name, checksum)
    config = read(folder / "data-config.json")
    require(config["partitions"] == {kind: kind + ".train.jsonl" for kind in KINDS}
            and config["automatic_concat"] is False, "Changed partition routing")
    validator = Draft202012Validator(read(folder / "schema.json"))
    ledger = indexed(rows(folder / "admission-ledger.jsonl"), "source_id")
    evidence = indexed(rows(folder / "target-evidence.jsonl"), "id")
    identities, sources, contents, stats = set(), set(), set(), {}
    for kind in sorted(KINDS):
        measured = Counter()
        data = rows(folder / (kind + ".train.jsonl"))
        tokens = rows(folder / (kind + ".tokens.jsonl"))
        for row, prior in zip(data, tokens, strict=True):
            require(not list(validator.iter_errors(row)), "Instruction schema validation failed")
            require(row["split"] == "train" and row["task_kind"] == kind and row["language"] == "tw",
                    "Wrong inherited split or task partition")
            require(row["training_eligible"] is True and row["quality_status"] == "scoped_agent_checked_research"
                    and row["production_eligible"] is False and row["semantic_certification"] is False
                    and row["human_reviewed"] is False and row["clinical_certification"] is False
                    and row["use_restriction"] == "private_noncommercial_research",
                    "Changed admission or certification")
            require(row["id"] not in identities and row["source_id"] not in sources, "Duplicate row/source")
            require(all(isinstance(row[k], str) and row[k].strip() for k in
                        ("id", "source_id", "canonical_source_id", "group_id", "family_id")), "Empty identity")
            require(all(isinstance(a, str) and a.strip() for a in row["group_aliases"]), "Empty group alias")
            identities.add(row["id"]); sources.add(row["source_id"])
            admission, proof = ledger[row["source_id"]], evidence[row["source_id"]]
            require(admission["final_export_eligible"] is True and admission["research_admitted"] is True
                    and admission["final_decision"] == "export" and not admission["final_reasons"], "Excluded source")
            for key in ("source_hash", "target_sha256", "reference_sha256", "split", "task_kind",
                        "group_id", "group_aliases", "family_id", "E_assessment_sha256"):
                require(row[key] == admission[key], "Admission binding differs: " + key)
            target = proof["after"]
            require(digest(target) == proof["after_hash"] == row["target_sha256"]
                    and proof["source_hash"] == row["source_hash"]
                    and proof["reference_hash"] == row["reference_sha256"], "Changed target/source evidence")
            expected = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": target["instruction"] +
                        ("\n\n" + target["input"] if target["input"] else "")},
                        {"role": "assistant", "content": target["output"]}]
            require(row["messages"] == expected, "Messages differ from complete admitted target")
            content = digest([unicodedata.normalize("NFC", m["content"].strip()) for m in expected])
            require(content not in contents, "Duplicate complete messages")
            contents.add(content)
            encoded = encode_complete(tokenizer, row, max_length)
            # Never trust old .tokens.jsonl just because the vocabulary is shared.
            require(all(isinstance(prior.get(k), list) and all(type(v) is int for v in prior[k])
                        for k in ("input_ids", "labels", "attention_mask")), "Invalid token-array types")
            require(prior == {"id": row["id"], **encoded}, "Legacy token file differs; a new versioned export is required")
            n = sum(v != -100 for v in encoded["labels"])
            require((row["assistant_tokens"], row["sequence_tokens"]) == (n, len(encoded["input_ids"])), "Changed token counts")
            measured.update(rows=1, sources=1, assistant_tokens=n, sequence_tokens=len(encoded["input_ids"]))
            measured["max_sequence_tokens"] = max(measured["max_sequence_tokens"], len(encoded["input_ids"]))
        stats[kind] = dict(measured)
    require(bool(sources), "Empty instruction export")
    require(sources == set(evidence) == {k for k, r in ledger.items() if r["final_export_eligible"] is True},
            "Incomplete exported source coverage")
    # Detect input edits during the check without writing or refreshing the source seal.
    require(sha(folder / "manifest.json") == seal["manifest_sha256"], "Manifest changed during check")
    for name, checksum in manifest["artifacts"].items():
        bound_file(folder, name, checksum)
    return {"path": str(folder.resolve()), "manifest_sha256": seal["manifest_sha256"], "split": "train",
            "partitions": stats, "current_tokens_equal_legacy": True, "fresh_source_reconstruction": False,
            "historical_seal_only": True, "new_admission": False}


def native_probes(tokenizer):
    from medical_pilot_core import PilotCollator
    samples = [{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."}]},
               {"messages": [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Earlier"},
                             {"role": "user", "content": "Correction"}, {"role": "assistant", "content": "Final answer."}]}]
    encoded = [encode_complete(tokenizer, r, tokenizer.model_max_length) for r in samples]
    for row, token in zip(samples, encoded, strict=True):
        target = tokenizer.decode([i for i in token["labels"] if i != -100], skip_special_tokens=False)
        require(target == row["messages"][-1]["content"] + "<|im_end|>", "Incorrect final-answer supervision")
    batch = PilotCollator(PAD)(encoded)
    for i, token in enumerate(encoded):
        n = len(token["input_ids"])
        require(batch["labels"][i, n:].tolist() == [-100] * (batch["labels"].shape[1] - n), "Padding supervised")
        require(not batch["attention_mask"][i, n:].any().item(), "Padding attended")
        require(batch["input_ids"][i, n:].tolist() == [PAD] * (batch["labels"].shape[1] - n), "Wrong padding token")
    tools = [{"type": "function", "function": {"name": "calculate_total", "parameters": {
        "type": "object", "properties": {"quantity": {"type": "number"}}, "required": ["quantity"]}}}]
    call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "calculate_total", "arguments": {"quantity": 2}}}]}
    messages = [{"role": "user", "content": "Calculate."}, call]
    serialized = encode_chat(tokenizer, {"messages": messages, "tools": tools}, max_length=tokenizer.model_max_length)
    call_target = tokenizer.decode([i for i in serialized["labels"] if i != -100], skip_special_tokens=False)
    require(call_target == "<tool_call>\n<function=calculate_total>\n<parameter=quantity>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>",
            "Native tool-call serialization changed")
    messages += [{"role": "tool", "content": "4"}, {"role": "assistant", "content": "Total: 4."}]
    reply = encode_chat(tokenizer, {"messages": messages, "tools": tools}, max_length=tokenizer.model_max_length)
    require(tokenizer.decode([i for i in reply["labels"] if i != -100], skip_special_tokens=False) == "Total: 4.<|im_end|>",
            "Tool-result or previous-call tokens supervised")
    return {"final_assistant_only": True, "padding_masked": True, "native_tool_serialization": True,
            "tools_executed": False, "tool_data_admitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--instruction-export", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True, help="Explicit capacity, not a quality threshold")
    parser.add_argument("--require-run-ready", action="store_true", help="Fail until baseline/training blockers are resolved")
    args = parser.parse_args(argv)
    try:
        check_identity(args.model, args.revision)
        dependencies = {name: sha(ROOT / name) for name in (
            "scripts/check_afrique_training_contract.py", "modal/train/afrique_response_core.py", "modal/train/medical_pilot_core.py")}
        tokenizer = load_tokenizer(args.assets)
        probes = native_probes(tokenizer)
        data = check_export(args.instruction_export, tokenizer, args.max_length)
        for name, checksum in ASSET_HASHES.items():
            bound_file(args.assets, name, checksum)
        require(all(sha(ROOT / name) == value for name, value in dependencies.items()), "Dependency changed during check")
        result = {"policy": POLICY, "checked_at": datetime.now(timezone.utc).isoformat(),
                  "mechanical_preflight_passed": True, "model": MODEL, "revision": REVISION,
                  "asset_sha256": ASSET_HASHES, "code_sha256": dependencies,
                  "runtime": {"python": sys.version, **{n: importlib.metadata.version(n) for n in
                              ("transformers", "tokenizers", "jinja2", "jsonschema", "torch")}},
                  "max_length": args.max_length, "truncated_targets": 0, "data": data, "probes": probes,
                  "stopping": {"generation_eos_token_id": EOS, "pad_token_id": PAD,
                               "publisher_generation_eos_ids": [EOS, PAD, PAD], "explicit_override_required": True},
                  "foundation_weights_loaded": False, "gpu_used": False, "training_started": False,
                  "baseline_run_ready": False, "training_run_ready": False, "full_resume_verified": False,
                  "blockers": BLOCKERS}
        print(json.dumps(result, indent=2))
        return 2 if args.require_run_ready else 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"policy": POLICY, "mechanical_preflight_passed": False,
                          "baseline_run_ready": False, "training_run_ready": False,
                          "error": type(error).__name__ + ": " + str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
