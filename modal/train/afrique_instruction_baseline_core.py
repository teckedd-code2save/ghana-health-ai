"""CPU-testable, per-record inference journal for the current Afrique instruction pin."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import time
import unicodedata

from check_afrique_training_contract import ASSET_HASHES, MODEL, REVISION, EOS, PAD, digest, encode_complete, read, require, sha

VERSION = "afrique-instruction-baseline-v1"
RUNTIME = {"torch": "2.13.0", "transformers": "5.17.0", "accelerate": "1.15.0",
           "flash-linear-attention": "0.5.2", "triton": "3.7.1"}
WEIGHT = {"filename": "model.safetensors", "bytes": 17907663008,
          "sha256": "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6",
          "pointer_blob_id": "d1dce72a64a5b10bc761c7a2b7871b1a405e3eab"}
SOURCES = {
    "development": ("tmp/general-foundation-comparison/cases-81a7aad880d9.json", "81a7aad880d9878d0c58a0de272a3552e01be655f777d798124eadec36f169fe"),
    "greetings": ("data/response-adaptation/greeting-regressions.v1.json", "22a0857e9ed8b928a51fdd41f46458a119c97b3990fd5fefa59d000a25faea04"),
    "conversations": ("data/response-adaptation/conversation-checks.v6.json", "c77883cdaac40de670326e1c904d28e7b3b866813cfa769d0a7f963be22cd7f7"),
}
SELECTED = {
    "negation-tw": "Negation and time in a Twi-to-English meaning request.",
    "negation-en": "English negation scope control, not a translation-equivalent pair.",
    "grounded-tw": "Actor and future/past referent resolution.",
    "count-tw": "Twi arithmetic and response language.",
    "count-en": "Matched arithmetic in English.",
    "search-tw-no-tools": "Commerce intent without invented tool execution.",
}
CODE = ("modal/train/afrique_instruction_baseline_core.py", "modal/train/evaluate_afrique_instruction_baseline.py",
        "scripts/run_afrique_instruction_baseline.py", "scripts/check_afrique_training_contract.py",
        "modal/train/afrique_response_core.py", "modal/train/medical_pilot_core.py",
        "modal/train/native_adaptation_run_core.py")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selection(root, frozen=False):
    source = {}
    for kind, (name, checksum) in SOURCES.items():
        path = root / "sources" / (kind + ".json") if frozen else root / name
        require(sha(path) == checksum, "Saved candidate source changed: " + kind)
        source[kind] = read(path)
    cases, review = [], []
    for row in source["development"]:
        if row["id"] in SELECTED:
            require(not row.get("tools"), "Tool execution is outside this diagnostic")
            cases.append({"id": row["id"], "messages": row["messages"]})
            review.append({**{k: v for k, v in row.items() if k != "messages"},
                           "selection_reason": SELECTED[row["id"]], "source": SOURCES["development"][0]})
    require({r["id"] for r in cases} == set(SELECTED), "Missing selected saved cases")
    for row in source["greetings"]["cases"]:
        cases.append({"id": row["id"], "messages": row["messages"]})
        review.append({"id": row["id"], "rubric": source["greetings"]["rubric"], "source": SOURCES["greetings"][0],
                       "selection_reason": "Retain the complete existing greeting regression family."})
    for row in source["conversations"]:
        cases.append({"id": row["id"], "turns": row["turns"]})
        review.append({"id": row["id"], "rubric": row["rubric"], "source": SOURCES["conversations"][0],
                       "selection_reason": "Retain the existing two-turn context/correction family; use actual generated history."})
    validate_cases(cases)
    return cases, review


def validate_cases(cases):
    require(isinstance(cases, list) and bool(cases), "Empty evaluation")
    seen = set()
    for case in cases:
        require(isinstance(case, dict) and set(case) in ({"id", "messages"}, {"id", "turns"}), "Only model-visible case fields")
        require(isinstance(case["id"], str) and case["id"] and case["id"] not in seen, "Duplicate/empty case ID")
        seen.add(case["id"])
        if "turns" in case:
            require(isinstance(case["turns"], list) and bool(case["turns"])
                    and all(isinstance(t, str) and t.strip() for t in case["turns"]), "Invalid rollout turns")
        else:
            require(isinstance(case["messages"], list) and bool(case["messages"]), "Empty prompt")


def validate_settings(settings):
    require(set(settings) == {"context_tokens", "max_new_tokens", "case_seconds", "run_seconds"}, "Unexpected generation settings")
    require(all(type(v) is int and v > 0 for v in settings.values()), "Positive integer limits required")
    require(settings["max_new_tokens"] < settings["context_tokens"] <= 262144, "Invalid token budget")
    require(settings["case_seconds"] <= settings["run_seconds"] <= 1100, "Exceeds the approved inner deadline")


def prompt_ids(tokenizer, messages, settings):
    require(isinstance(messages, list) and bool(messages) and messages[-1].get("role") == "user", "Prompt must end in user")
    # Reuse the verified mask/prefix contract with a diagnostic-only sentinel.
    token = encode_complete(tokenizer, {"messages": messages + [{"role": "assistant", "content": "probe"}]}, tokenizer.model_max_length)
    ids = [v for v, label in zip(token["input_ids"], token["labels"], strict=True) if label == -100]
    native = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                            enable_thinking=False, add_generation_prompt=True)
    require(ids == native, "Native generation prefix differs from the verified training prefix")
    require(len(ids) + settings["max_new_tokens"] <= settings["context_tokens"], "Prompt plus complete output budget exceeds context; no truncation")
    return ids


def verify_loading(info, checkpoint_shapes, loaded_shapes):
    required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    require(required <= set(info) and not any(info[k] for k in required), "Incomplete or unexpected loaded weights")
    mapped = {}
    for key, shape in checkpoint_shapes.items():
        name = re.sub(r"^model\.language_model\.", "model.", key)
        require(name not in mapped and not any(x in name for x in ("visual", "vision", "lora_")), "Unexpected/colliding checkpoint key")
        mapped[name] = list(shape)
    require(bool(mapped) and mapped == {k: list(v) for k, v in loaded_shapes.items()}, "Checkpoint and loaded state shapes differ")
    return {"strict_text_weights_complete": True, "tensors": len(mapped), "shape_sha256": digest(mapped)}


def verify_weight(path, expected):
    require(path.stat().st_size == expected["bytes"], "Checkpoint byte count differs from published LFS metadata")
    checksum = sha(path)
    require(checksum == expected["sha256"], "Checkpoint SHA-256 differs from published LFS metadata")
    return checksum


def verify_cache_receipt(value):
    require((value["model"], value["revision"]) == (MODEL, REVISION) and value["state"] == "ready"
            and value["weight_bytes_hash_verified"] is True and value["gpu_used"] is False, "Invalid CPU cache readiness")
    for name, checksum in {**ASSET_HASHES, WEIGHT["filename"]: WEIGHT["sha256"]}.items():
        require(value["files"][name]["sha256"] == checksum, "Cache readiness asset differs: " + name)
    require(value["files"][WEIGHT["filename"]]["bytes"] == WEIGHT["bytes"], "Cache readiness weight length differs")
    return True


def cache_files_present(files, blobs):
    required = set(ASSET_HASHES) | {WEIGHT["filename"]}
    if not required <= set(files):
        return False
    weight = files[WEIGHT["filename"]]
    if weight["type"] == "FILE":
        return weight["bytes"] == WEIGHT["bytes"]
    return weight["type"] == "SYMLINK" and blobs.get(WEIGHT["sha256"], {}).get("bytes") == WEIGHT["bytes"]


def repetition_metrics(ids, text):
    result = {}
    words = re.findall(r"\w+", text.casefold())
    for width in (4, 8):
        counts = Counter(tuple(words[i:i + width]) for i in range(max(0, len(words) - width + 1)))
        positions = sum(counts.values())
        repeated = sum(n - 1 for n in counts.values())
        result[str(width) + "_word_ngrams"] = {"positions": positions, "repeated_positions": repeated,
            "repeated_fraction": repeated / positions if positions else 0., "max_occurrences": max(counts.values(), default=0)}
    longest = streak = 0
    previous = None
    for token in ids:
        streak = streak + 1 if token == previous else 1
        previous, longest = token, max(longest, streak)
    return {"longest_identical_token_run": longest, "word_count": len(words), **result}


def completion(tokenizer, ids, ceiling, timed_out, seconds):
    require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), "Invalid generation IDs")
    require(len(ids) <= ceiling and math.isfinite(seconds) and seconds >= 0, "Invalid generation length/timing")
    reason = "eos" if ids and ids[-1] == EOS else "time_limit" if timed_out else "max_new_tokens" if len(ids) == ceiling else "unexpected_stop"
    require(EOS not in ids[:-1], "Unexpected tokens after EOS")
    body = ids[:-1] if reason == "eos" else ids
    prediction = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    controls = sorted(set(body) & tokenizer.contract_special_ids)
    control_text = bool(re.search(r"<\||</?(?:think|tool_call|tool_response)>", prediction))
    unmapped = sorted({i for i in body if i >= len(tokenizer)})
    invalid_unicode = any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
                          (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in prediction)
    return {"output_ids": ids, "raw_output": raw, "prediction": prediction, "generated_tokens": len(ids),
            "finish_reason": reason, "complete": reason == "eos" and bool(prediction.strip())
                and not (controls or control_text or unmapped or invalid_unicode),
            "unexpected_control_ids": controls, "empty": not bool(prediction.strip()), "seconds": seconds,
            "unexpected_control_text": control_text, "unmapped_token_ids": unmapped, "invalid_unicode": invalid_unicode,
            "repetition": repetition_metrics(body, prediction)}


class Journal:
    """Immutable finalized records; in-flight markers never masquerade as outputs."""
    def __init__(self, folder, binding, commit=lambda: None):
        self.folder, self.binding, self.commit = Path(folder), binding, commit
        self.folder.mkdir(parents=True, exist_ok=True)
        identity = self.folder / "binding.json"
        if identity.exists():
            require(read(identity) == binding, "Resume identity differs")
        else:
            require(not any(self.folder.iterdir()), "Unbound existing output directory")
            save(identity, binding); self.commit()

    def path(self, key):
        return self.folder / "records" / (digest(key) + ".json")

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        envelope = read(path)
        value = envelope["record"]
        require(envelope["sha256"] == digest(value) and value["id"] == key
                and value["binding_sha256"] == digest(self.binding), "Corrupt finalized record")
        return value

    def begin(self, key, attempt, inputs_sha):
        path = self.folder / "attempts" / str(attempt) / (digest(key) + ".json")
        require(not path.exists(), "Record attempt already exists; use the next terminally verified call attempt")
        save(path, {"id": key, "inputs_sha256": inputs_sha, "state": "started", "attempt": attempt})
        self.commit()

    def finish(self, key, value):
        require(not self.path(key).exists(), "Finalized record cannot be overwritten")
        record = {"id": key, "binding_sha256": digest(self.binding), **value}
        save(self.path(key), {"record": record, "sha256": digest(record)})
        self.commit()
        return record


def evaluate_records(cases, settings, tokenizer, generate, journal, attempt, deadline, clock=time.monotonic):
    validate_cases(cases); validate_settings(settings)
    results = []
    for case in cases:
        history, previous = [], None
        turns = case.get("turns", [None])
        for turn, content in enumerate(turns):
            key = case["id"] + ":" + str(turn)
            messages = case["messages"] if content is None else history + [{"role": "user", "content": content}]
            inputs_sha = digest(messages)
            record = journal.get(key)
            parent_sha = digest(previous) if previous is not None else None
            if record:
                require(record["inputs_sha256"] == inputs_sha and record["parent_sha256"] == parent_sha, "Saved prompt/history differs")
            else:
                if clock() >= deadline:
                    return {"state": "paused_budget", "finalized_records": len(results), "expected_records": sum(len(c.get("turns", [None])) for c in cases)}
                journal.begin(key, attempt, inputs_sha)
                if previous is not None and not previous["complete"]:
                    value = {"finish_reason": "blocked_previous_incomplete", "complete": False,
                             "prediction": "", "raw_output": "", "output_ids": [], "generated_tokens": 0, "seconds": 0.}
                else:
                    ids = prompt_ids(tokenizer, messages, settings)
                    value = generate(ids, min(deadline, clock() + settings["case_seconds"]))
                    # Recompute diagnostics instead of accepting caller-provided completion claims.
                    expected = completion(tokenizer, value["output_ids"], settings["max_new_tokens"],
                                          value["finish_reason"] == "time_limit", value["seconds"])
                    require(value == expected, "Generation result contract differs")
                    value["prompt_tokens"] = len(ids)
                    value["prompt_ids"] = ids
                record = journal.finish(key, {**value, "case_id": case["id"], "turn": turn, "attempt": attempt,
                    "messages": messages, "inputs_sha256": inputs_sha, "parent_sha256": parent_sha})
            results.append(record)
            if content is not None and record["complete"]:
                history = messages + [{"role": "assistant", "content": record["prediction"]}]
            previous = record
    return {"state": "completed", "finalized_records": len(results),
            "complete_outputs": sum(r["complete"] for r in results),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in results)),
            "semantic_quality_established": False}


def claim_attempt(store, run_key, attempt, call_id, terminal):
    require(type(attempt) is int and attempt >= 1, "Invalid attempt")
    if attempt > 1:
        previous = store.get(run_key + ":" + str(attempt - 1))
        require(previous is not None and terminal(previous), "Previous call is missing, active or ambiguous")
    require(store.put(run_key + ":" + str(attempt), call_id, skip_if_exists=True), "Attempt already claimed")


def load_pack(folder, code_root):
    manifest = read(folder / "manifest.json")
    require(read(folder / "PREPARED.json")["manifest_sha256"] == sha(folder / "manifest.json"), "Changed baseline pack seal")
    require(manifest["version"] == VERSION and (manifest["model"], manifest["revision"]) == (MODEL, REVISION)
            and manifest["runtime"] == RUNTIME and manifest["research_status"] == "development_diagnostic_proposal"
            and manifest["weight"] == WEIGHT and manifest["training"] is False
            and manifest["no_new_holdout"] is True, "Wrong baseline identity")
    validate_settings(manifest["settings"])
    for name, checksum in manifest["artifacts"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and sha(path) == checksum, "Changed baseline artifact")
    required = {"cases.json", "review.json"} | {"tokenizer/" + n for n in ASSET_HASHES} | {"sources/" + k + ".json" for k in SOURCES}
    require(set(manifest["artifacts"]) == required, "Unbound or unexpected baseline inputs")
    require(set(manifest["code_hashes"]) == set(CODE), "Incomplete runtime binding")
    for name, checksum in manifest["code_hashes"].items():
        require(sha(code_root / name) == checksum, "Baseline implementation changed: " + name)
    cases = read(folder / "cases.json")
    validate_cases(cases)
    expected, review = selection(folder, frozen=True)
    require(cases == expected and read(folder / "review.json") == review, "Selected prompts/review differ from pinned saved sources")
    require(manifest["cases"] == len(cases) and manifest["planned_generation_records"] == sum(len(c.get("turns", [None])) for c in cases), "Changed evaluation accounting")
    return manifest, cases
