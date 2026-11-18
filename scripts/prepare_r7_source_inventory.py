#!/usr/bin/env python3
"""Read-only source audit and source-preserved English R7 review queues.

No translation, teacher requests, training, source edits or inferred admission.
Raw OASST contributor IDs are never loaded. Pattern flags are review selectors,
not judgments of semantic correctness, clinical safety, or native fluency.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from prepare_understanding_run import HoldoutIndex

VERSION = "r7-source-inventory-v1"
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai")
HISTORIES = Path("/Users/welcome/ghana-health-twi-r6/work/responses-r6/records.jsonl")
OASST_COLUMNS = ("message_id", "parent_id", "message_tree_id", "text", "role", "lang", "rank",
                 "review_count", "review_result", "deleted", "synthetic", "labels", "tree_state")
HEALTH_COLUMNS = ("id", "question", "answer", "language_country")
IDENTITIES = {"source_id", "source_ids", "canonical_source_id", "group_id", "group_aliases",
              "family_id", "source_family_id", "original_group_id", "tree_id", "message_tree_id"}
TEXT_FIELDS = {"text", "question", "answer"}
MESSAGE_FIELDS = {"messages", "english_messages", "original_english_messages"}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]){2,3}\d{3,4}(?!\w)")
IDENTIFIER = re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{1,3}(?:\.\d{1,3}){3})\b")
CONTROL = re.compile(r"<\||</?(?:think|tool_call|tool_response)(?:>|\s)")
CORRECTION = re.compile(r"\b(?:actually|instead|i meant|i said|you said|not what|wrong|incorrect|correction|"
                        r"correct that|change (?:it|that|the)|replace|daabi)\b", re.I)
LIST_REQUEST = re.compile(r"\b(?:list|examples?|categories|types|name|enumerate|nhw\u025bso|ahorow)\b", re.I)
SYMPATHY = {"i am sorry to hear that", "i m sorry to hear that", "sorry to hear that",
            "\u025by\u025b me yaw s\u025b wote nka saa", "mepa wo ky\u025bw", "that sounds difficult"}
GENERIC = SYMPATHY | {"yes", "no", "okay", "ok", "sure", "thanks", "thank you", "you re welcome",
                       "i understand", "i can help", "yiw", "daabi", "mate", "meda wo ase"}
LIST_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.+)$", re.M)
PAST_SEARCH = re.compile(r"\b(?:sources? i (?:could )?(?:find|found) online|"
                         r"i (?:searched|looked|checked|browsed).{0,45}(?:online|web|internet)|"
                         r"i found.{0,45}(?:online|web)|intan\u025bt so ns\u025bm a mihui)\b", re.I)
MAX_HISTORY_CHARS = 100000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def strict_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


class Inputs:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(path.is_file() and path.resolve() == path, "Missing/linked input: " + str(path))
        checksum = sha(path)
        require(expected is None or checksum == expected, "Source hash differs: " + str(path))
        require(str(path) not in self.files or self.files[str(path)]["sha256"] == checksum,
                "Source changed during inventory: " + str(path))
        self.files[str(path)] = {"sha256": checksum, "bytes": path.stat().st_size}
        return checksum

    def read(self, path):
        self.bind(path)
        return strict_json(Path(path).read_text(encoding="utf-8"))

    def rows(self, path):
        self.bind(path)
        result = []
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                require(line.strip(), "Blank JSONL row")
                row = strict_json(line)
                require(isinstance(row, dict), "Non-object JSONL row")
                result.append(row)
        return result

    def parquet(self, path, columns, split):
        import pyarrow.parquet as pq
        checksum = self.bind(path)
        source = pq.ParquetFile(path)
        require(set(columns) <= set(source.schema_arrow.names), "Missing parquet columns")
        result = []
        for batch in source.iter_batches(batch_size=4096, columns=list(columns)):
            for row in batch.to_pylist():
                result.append({**row, "source_split": split,
                               "source_ref": {"file": str(Path(path).absolute()), "file_sha256": checksum,
                                              "row_index": len(result), "projection_sha256": digest(row)}})
        return result

    def recheck(self):
        for path in list(self.files):
            self.bind(path)


def identity_keys(value):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in IDENTITIES:
                for item in child if isinstance(child, list) else [child]:
                    if isinstance(item, str) and item:
                        result.add(item)
                        found = re.search(r"oasst:([a-f0-9-]{36})", item)
                        if found:
                            result.add(found.group(1))
            elif isinstance(child, (dict, list)):
                result.update(identity_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(identity_keys(child))
    return result


class Families:
    def __init__(self):
        self.parents = {}

    def root(self, key):
        self.parents.setdefault(key, key)
        while key != self.parents[key]:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def connect(self, keys):
        keys = sorted(set(keys))
        for key in keys[1:]:
            left, right = self.root(keys[0]), self.root(key)
            self.parents[max(left, right)] = min(left, right)
        return self.root(keys[0]) if keys else None


def text_views(value):
    """Scan source text and every full assistant prefix, without rewriting it."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in TEXT_FIELDS and isinstance(child, str) and child.strip():
                yield child
            elif key in MESSAGE_FIELDS and isinstance(child, list):
                history = []
                for message in child:
                    if not isinstance(message, dict) or message.get("role") == "system":
                        continue
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        yield content
                        history.append(content)
                        if message.get("role") == "assistant":
                            yield "\n".join(history)
            elif isinstance(child, (dict, list)):
                yield from text_views(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_views(child)


def contact_flags(text):
    return [name for name, pattern in (("possible_email", EMAIL), ("possible_phone", PHONE),
                                       ("possible_personal_identifier", IDENTIFIER)) if pattern.search(text)]


def history_flags(messages):
    findings, previous, assistant_seen = [], set(), False
    for turn, message in enumerate(messages):
        text = message.get("content", "")
        if not isinstance(text, str):
            findings.append({"code": "invalid_content", "turn": turn})
            continue
        if message.get("role") == "user":
            if assistant_seen:
                findings.append({"code": "followup_turn", "turn": turn})
                if CORRECTION.search(text):
                    findings.append({"code": "correction_cue", "turn": turn})
            if LIST_REQUEST.search(text):
                findings.append({"code": "category_or_list_request_cue", "turn": turn})
            continue
        if message.get("role") != "assistant":
            continue
        assistant_seen = True
        words, norm = normalize(text).split(), normalize(text)
        if len(words) <= 8:
            findings.append({"code": "short_target_le8_words", "turn": turn, "words": len(words)})
        if norm in GENERIC:
            findings.append({"code": "generic_exact_phrase", "turn": turn})
        if norm in SYMPATHY:
            findings.append({"code": "sympathy_only_exact_phrase", "turn": turn})
        if norm in previous:
            findings.append({"code": "duplicate_target_within_history", "turn": turn})
        previous.add(norm)
        sentences = [normalize(x) for x in re.split(r"[.!?\n]+", text) if len(normalize(x).split()) >= 4]
        repeats = sum(n - 1 for n in Counter(sentences).values() if n > 1)
        if repeats:
            findings.append({"code": "repeated_sentence_or_line", "turn": turn, "duplicate_occurrences": repeats})
        grams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
        if any(n >= 3 for n in grams.values()):
            findings.append({"code": "repeated_8gram_at_least_three_times", "turn": turn,
                             "distinct_repeated_8grams": sum(n >= 3 for n in grams.values())})
        items = LIST_LINE.findall(text)
        if items:
            unique = {normalize(x) for x in items}
            findings.append({"code": "list_form_target", "turn": turn, "items": len(items), "distinct_items": len(unique)})
            if len(unique) < len(items):
                findings.append({"code": "repeated_list_items", "turn": turn, "duplicate_items": len(items) - len(unique)})
            examples = defaultdict(set)
            for item in items:
                if ":" in item:
                    label, value = item.split(":", 1)
                    for example in re.split(r"[,;]", value):
                        example = normalize(example.strip(" *`"))
                        if example:
                            examples[example].add(normalize(label))
            reused = sum(len(labels) > 1 for labels in examples.values())
            if reused:
                findings.append({"code": "same_literal_example_across_category_labels", "turn": turn,
                                 "reused_distinct_literals": reused})
        if CONTROL.search(text):
            findings.append({"code": "literal_control_token", "turn": turn})
        if PAST_SEARCH.search(text) and not any(m.get("role") == "tool" for m in messages[:turn]):
            findings.append({"code": "past_search_claim_without_tool_evidence", "turn": turn,
                             "repair_review_required": True, "tool_execution_evidence_present": False})
    return findings


def pattern_summary(records):
    occurrences, normalized_targets, raw_targets, counts, flagged = 0, Counter(), set(), Counter(), 0
    for row in records:
        messages = row.get("messages", [])
        flags = history_flags(messages)
        counts.update(f["code"] for f in flags)
        flagged += bool(flags)
        for message in messages:
            if message.get("role") == "assistant":
                text = message["content"]
                occurrences += 1
                raw_targets.add(digest(text))
                normalized_targets[digest(normalize(text))] += 1
    return {"records": len(records), "assistant_occurrences": occurrences,
            "distinct_assistant_targets_exact": len(raw_targets),
            "distinct_assistant_targets_normalized": len(normalized_targets),
            "duplicate_normalized_target_occurrences": sum(n - 1 for n in normalized_targets.values()),
            "flags_by_occurrence": dict(sorted(counts.items())), "records_with_any_pattern": flagged,
            "pattern_flags_are_semantic_certification": False}


def load_pack(path, inputs):
    path = Path(path).absolute()
    manifest = inputs.read(path / "manifest.json")
    require(inputs.read(path / "PREPARED.json")["manifest_sha256"] == inputs.bind(path / "manifest.json"), "Pack seal differs")
    for name, checksum in manifest["artifacts"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe pack artifact")
        inputs.bind(path / name, checksum)
    for name in ("source-witnesses.jsonl", "source-manifests.json", "rows.jsonl", "stages.json", "evaluation-manifest.json"):
        require(name in manifest["artifacts"], "Required unbound pack input: " + name)
    records = []
    for ref in inputs.read(path / "source-manifests.json"):
        require(manifest["artifacts"].get(ref["manifest_path"]) == ref["manifest_sha256"], "Dataset manifest unbound")
        dataset = inputs.read(path / ref["manifest_path"])
        require(manifest["artifacts"].get(dataset["records_path"]) == dataset["records_sha256"], "Dataset rows unbound")
        values = inputs.rows(path / dataset["records_path"])
        require(len(values) == dataset["record_count"], "Dataset count differs")
        collection = Path(dataset["records_path"]).parts[-2]
        records.extend({"collection": collection, "record": row,
                        "ref": {"file": str(path / dataset["records_path"]), "line": index + 1,
                                "file_sha256": dataset["records_sha256"], "record_sha256": digest(row)}}
                       for index, row in enumerate(values))
    witnesses = inputs.rows(path / "source-witnesses.jsonl")
    encoded = inputs.rows(path / "rows.jsonl")
    require(len(witnesses) == len(encoded), "Packed rows/witnesses differ")
    stages = inputs.read(path / "stages.json")
    visits = Counter(i for stage in stages for i in stage["schedule"])
    require(all(type(i) is int and 0 <= i < len(encoded) for i in visits), "Invalid source schedule")
    collections = {}
    for collection in sorted({row["collection"] for row in witnesses}):
        subset = [w for w in witnesses if w["collection"] == collection]
        actual = [w["record"] for w in records if w["collection"] == collection]
        indices = [i for i, row in enumerate(encoded) if row["collection"] == collection]
        targets = [w["messages"][-1]["content"] for w in subset if w.get("messages")]
        collections[collection] = {
            "registered_records": len(actual), "packed_views": len(subset),
            "unique_source_ids": len({w["source_id"] for w in subset}),
            "witness_origins": dict(Counter(w.get("metadata", {}).get("origin", "missing") for w in subset)),
            "record_origins": dict(Counter(w.get("origin", w.get("metadata", {}).get("origin", "missing")) for w in actual)),
            "source_prefixes": dict(Counter(w["source_id"].split(":", 1)[0] for w in subset)),
            "preparation_origins": dict(Counter(encoded[i].get("preparation_origin", "missing") for i in indices)),
            "scheduled_exposures": sum(visits[i] for i in indices),
            "scheduled_input_tokens": sum(len(encoded[i]["input_ids"]) * visits[i] for i in indices),
            "scheduled_effective_target_tokens": sum(sum(v != -100 for v in encoded[i]["labels"][1:]) * visits[i] for i in indices),
            "final_target_occurrences": len(targets), "distinct_final_targets_exact": len({digest(t) for t in targets}),
            "distinct_final_targets_normalized": len({digest(normalize(t)) for t in targets}),
            "complete_record_patterns": pattern_summary(actual),
        }
    held = inputs.rows(path / "held-content.jsonl") if "held-content.jsonl" in manifest["artifacts"] else []
    decisions = inputs.read(path / "additional-protection.json").get("decisions", []) if "additional-protection.json" in manifest["artifacts"] else []
    return {"path": str(path), "manifest_sha256": inputs.bind(path / "manifest.json"), "collections": collections,
            "held": held, "decisions": decisions, "records": records, "witnesses": witnesses,
            "evaluation": inputs.read(path / "evaluation-manifest.json"),
            "overlap_audit": inputs.read(path / "overlap-audit.json") if "overlap-audit.json" in manifest["artifacts"] else {}}


def oasst_path(node, nodes):
    chain, seen = [], set()
    current = node
    while current is not None:
        identity = current["message_id"]
        require(identity not in seen, "cycle")
        seen.add(identity)
        require(current["message_tree_id"] == node["message_tree_id"], "cross_tree_parent")
        require(current["source_split"] == node["source_split"], "cross_split_parent")
        chain.append(current)
        parent = current["parent_id"]
        if parent is None:
            break
        require(parent in nodes, "missing_parent")
        current = nodes[parent]
    chain.reverse()
    require(chain[0]["message_id"] == node["message_tree_id"], "root_tree_identity_differs")
    require(len(chain) % 2 == 0 and all(n["role"] == ("prompter" if i % 2 == 0 else "assistant")
                                      for i, n in enumerate(chain)), "invalid_role_sequence")
    return chain


def queue_order(candidates, limit):
    """Deterministic round-robin facets; never a first-N file-order sample."""
    bins = defaultdict(list)
    for candidate in candidates:
        codes = {f["code"] for f in candidate["review_flags"]}
        bucket = ("correction_cue" if "correction_cue" in codes else
                  "followup" if "followup_turn" in codes else
                  "category_list" if "category_or_list_request_cue" in codes else "other")
        candidate["coverage_bucket"] = bucket
        bins[bucket].append(candidate)
    for values in bins.values():
        values.sort(key=lambda row: digest(row["id"]))
    ordered = []
    for index in range(max((len(values) for values in bins.values()), default=0)):
        for name in sorted(bins):
            if index < len(bins[name]):
                ordered.append(bins[name][index])
    return ordered[:limit], ordered[limit:]


def oasst_candidates(train, validation, families, blocked, prior_trees, limit):
    nodes = {n["message_id"]: n for n in train + validation}
    require(len(nodes) == len(train) + len(validation), "Duplicate original OASST message ID")
    candidates, ledger = [], []
    for node in train + validation:
        if node["role"] != "assistant":
            continue
        reasons, chain = [], []
        if node["source_split"] != "train":
            reasons.append("original_validation_split")
        root = families.root(node["message_tree_id"])
        if root in blocked:
            reasons.append("protected_source_family")
        try:
            chain = oasst_path(node, nodes)
        except ValueError as error:
            reasons.append(str(error))
        if chain:
            if any(n["lang"] != "en" for n in chain):
                reasons.append("history_not_entirely_original_english")
            if any(n["deleted"] is not False or not isinstance(n["text"], str) or not n["text"].strip() for n in chain):
                reasons.append("deleted_or_empty_ancestry")
            if any(n["role"] == "assistant" and n["rank"] != 0 for n in chain):
                reasons.append("history_not_all_rank_zero")
            if any(n["review_result"] is not True for n in chain):
                reasons.append("upstream_review_not_positive")
            if any(n["synthetic"] is not False for n in chain):
                reasons.append("synthetic_source_requires_separate_review")
            for n in chain:
                labels = dict(zip((n.get("labels") or {}).get("name", []), (n.get("labels") or {}).get("value", [])))
                if labels.get("pii") != 0:
                    reasons.append("pii_annotation_positive_or_missing")
                if isinstance(n["text"], str):
                    reasons.extend(contact_flags(n["text"]))
                    if CONTROL.search(n["text"]):
                        reasons.append("literal_control_token")
            if sum(len(n["text"] or "") for n in chain) > MAX_HISTORY_CHARS:
                reasons.append("history_exceeds_review_capacity_no_truncation")
        reasons = sorted(set(reasons))
        hard = {"original_validation_split", "protected_source_family", "history_not_entirely_original_english",
                "deleted_or_empty_ancestry", "history_not_all_rank_zero", "cycle", "cross_tree_parent",
                "cross_split_parent", "missing_parent", "root_tree_identity_differs", "invalid_role_sequence"}
        state = "excluded" if set(reasons) & hard else "pending" if reasons else "eligible"
        entry = {"id": "oasst:" + node["message_id"], "source_split": node["source_split"],
                 "tree_id": node["message_tree_id"], "status": state,
                 "reasons": reasons or ["original_train_complete_rank_zero_english_history"],
                 "training_eligible": False, "source_ref": node["source_ref"]}
        if root in blocked:
            entry["protection_evidence"] = blocked[root]
        ledger.append(entry)
        if state != "eligible":
            continue
        messages = [{"role": "user" if n["role"] == "prompter" else "assistant", "content": n["text"]} for n in chain]
        candidates.append({**entry, "queue_kind": "source_review_before_translation_or_admission", "language": "en",
            "source_id": "oasst:" + node["message_tree_id"], "origin": "original_oasst_public_train_source_preserved",
            "messages": messages, "messages_sha256": digest(messages), "path_message_ids": [n["message_id"] for n in chain],
            "ancestry": [{"message_id": n["message_id"], "parent_id": n["parent_id"], "rank": n["rank"],
                          "source_ref": n["source_ref"]} for n in chain],
            "review_flags": history_flags(messages), "already_in_r6_source_family": root in prior_trees,
            "license": "apache-2.0", "human_reviewed_for_r7": False, "semantic_certification": False,
            "privacy_screen": "upstream pii=0 on every ancestor plus contact-pattern exclusion; not a PII guarantee"})
    selected, deferred = queue_order(candidates, limit)
    deferred_ids = {row["id"] for row in deferred}
    for row in ledger:
        if row["id"] in deferred_ids:
            row.update(status="pending", reasons=["outside_explicit_queue_budget"])
    return selected, ledger


def health_candidates(rows, index, limit):
    groups, suffixes, ledger, candidates = defaultdict(list), defaultdict(set), [], []
    for row in rows:
        groups[digest(normalize(row["question"]))].append(row)
        suffixes[row["source_split"]].add(row["id"].split("_")[-1])
    held_groups = {key for key, group in groups.items() if any(r["source_split"] != "train" for r in group)}
    for row in rows:
        key, reasons = digest(normalize(row["question"])), []
        if row["source_split"] != "train":
            reasons.append("original_" + row["source_split"] + "_split")
        if key in held_groups:
            reasons.append("normalized_question_family_crosses_holdout")
        match = next((hit for text in (row["question"], row["answer"], row["question"] + "\n" + row["answer"])
                      if text.strip() and (hit := index.match(text))), None)
        if match:
            reasons.append("protected_content_overlap")
        if not row["language_country"].startswith("Eng_"):
            reasons.append("not_original_english_candidate")
        if not row["answer"].strip():
            reasons.append("missing_source_answer")
        privacy = contact_flags(row["question"] + "\n" + row["answer"])
        reasons.extend(privacy)
        if CONTROL.search(row["question"] + row["answer"]):
            reasons.append("literal_control_token")
        state = "excluded" if reasons else "pending"
        if not reasons:
            reasons = ["cross_language_family_alignment_unverified", "primary_clinical_grounding_and_terms_review_required"]
        entry = {"id": "afrihealth:" + row["id"], "source_split": row["source_split"],
                 "language_country": row["language_country"], "question_group_id": "question:" + key,
                 "status": state, "reasons": reasons, "source_ref": row["source_ref"], "training_eligible": False}
        if match:
            entry["protection_evidence"] = match
        ledger.append(entry)
        if state == "pending":
            messages = [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]
            candidates.append({**entry, "source_id": "afrihealth:" + row["id"], "language": "en",
                "origin": "original_afrihealth_public_train_source_preserved", "messages": messages,
                "messages_sha256": digest(messages), "review_flags": history_flags(messages),
                "cross_language_grouping": "unresolved; neither suffix nor row position proves alignment",
                "source_backing": "original dataset QA only; not an independently verified clinical reference",
                "clinical_certification": False, "semantic_certification": False})
    selected, deferred = queue_order(candidates, limit)
    deferred_ids = {r["id"] for r in deferred}
    for row in ledger:
        if row["id"] in deferred_ids:
            row["reasons"].append("outside_explicit_queue_budget")
    split_names = sorted(suffixes)
    return selected, ledger, {"normalized_question_groups": len(groups), "question_groups_crossing_holdout": len(held_groups),
        "suffix_cross_split_collisions": {a + ":" + b: len(suffixes[a] & suffixes[b])
                                          for i, a in enumerate(split_names) for b in split_names[i + 1:]},
        "suffix_is_not_family_identity": True, "cross_language_pairs_verified": 0}


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def dispositions(rows):
    return {"rows": len(rows), "status": dict(Counter(r["status"] for r in rows)),
            "reasons_nonexclusive": dict(Counter(reason for row in rows for reason in row["reasons"]))}


def raw_assistant_patterns(nodes):
    assistants = [{"messages": [{"role": "assistant", "content": row["text"]}]}
                  for row in nodes if row["role"] == "assistant" and row["lang"] == "en" and isinstance(row["text"], str)]
    return {**pattern_summary(assistants), "counting_unit": "each_original_english_assistant_node_once_not_expanded_paths",
            "english_followup_user_nodes": sum(r["role"] == "prompter" and r["lang"] == "en" and r["parent_id"] is not None for r in nodes),
            "english_followup_correction_cue_nodes": sum(r["role"] == "prompter" and r["lang"] == "en" and
                r["parent_id"] is not None and bool(CORRECTION.search(r["text"] or "")) for r in nodes)}


def replay_indexes(witnesses, families=None):
    def key(row):
        return families.root(row["source_id"]) if families is not None else row["source_id"]
    return {"histories": {(w["collection"], digest(w.get("messages", w.get("text")))) for w in witnesses},
            "targets": {(w["collection"], key(w), digest(w["messages"][-1]["content"]))
                        for w in witnesses if w.get("messages")},
            "sources": {(w["collection"], key(w)) for w in witnesses},
            "raw_sources": {(w["collection"], w["source_id"]) for w in witnesses}}


def compare_replay(row, collection, current, families=None):
    messages = row.get("messages", [])
    targets = [m["content"] for m in messages if m["role"] == "assistant"]
    payloads = [messages[:i + 1] for i, m in enumerate(messages) if m["role"] == "assistant"] if messages else [row.get("text")]
    key = families.root(row["source_id"]) if families is not None else row["source_id"]
    return {"missing_exact_history_views": sum((collection, digest(p)) not in current["histories"] for p in payloads),
            "total_history_views": len(payloads), "source_present_in_r6": (collection, key) in current["sources"],
            "raw_source_id_present_in_r6": (collection, row["source_id"]) in current["raw_sources"],
            "assistant_target_occurrences_missing_for_same_source": sum((collection, key, digest(t))
                                                                        not in current["targets"] for t in targets),
            "assistant_target_occurrences": len(targets)}


def parallel_sources(corpus, release, inputs):
    """Keep raw pairs unchanged; inherit splits only from the sealed source ledger."""
    import corpus_record_types as types
    inputs.bind(Path(types.__file__))
    corpus, release = Path(corpus), Path(release)
    manifest = inputs.read(release / "manifest.json")
    require(inputs.read(release / "READY.json")["manifest_sha256"] == inputs.bind(release / "manifest.json"), "Alignment release seal differs")
    require(manifest["record_type_policy"] == types.POLICY, "Alignment routing policy differs")
    for name, checksum in manifest["artifacts"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe alignment artifact")
        inputs.bind(release / name, checksum)
    accounts = {r["source_id"]: r for r in inputs.rows(release / "source-accounting.jsonl")}
    assessments = {r["source_id"]: r for r in inputs.rows(release / "type-assessments.jsonl")}
    lineage = defaultdict(list)
    for name in manifest["artifacts"]:
        if name.startswith(("sentences/", "lexicon/", "utterances/", "review/record-types")) and name.endswith(".jsonl"):
            for line, row in enumerate(inputs.rows(release / name), 1):
                lineage[row["source_id"]].append({"group_id": row.get("group_id"), "split": row["split"],
                    "source_file_sha256": row["source_file_sha256"], "type_assessment": row.get("type_assessment"),
                    "license": row.get("license"), "source_ref": {"file": str(release / name), "line": line,
                    "file_sha256": manifest["artifacts"][name], "record_sha256": digest(row)}})
    result = []
    csvs = sorted((corpus / "ghana_nlp_parallel").glob("*.csv"))
    parquets = sorted((corpus / "community_parallel").glob("*.parquet"))
    require(csvs and parquets, "Both original parallel corpora required")
    for family in ("ghana_nlp_parallel", "community_parallel"):
        if (corpus / family / "README.md").is_file():
            inputs.bind(corpus / family / "README.md")
    for path in csvs:
        checksum = inputs.bind(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            require({"id", "text", "label"} <= set(reader.fieldnames or []), "Unexpected GhanaNLP CSV schema")
            for index, row in enumerate(reader):
                result.append({"source_id": "ghana_nlp_parallel:" + row["id"], "family": "ghana_nlp_parallel",
                    "english": row["text"] or "", "twi": row["label"] or "", "genre": "professional_parallel",
                    "source_ref": {"file": str(path), "file_sha256": checksum, "row_index": index, "csv_record": index + 2,
                                   "record_sha256": digest(row)}})
    for path in parquets:
        values = inputs.parquet(path, ("eng", "twi", "count", "source_files"), "original_file_train_not_release_split")
        for index, row in enumerate(values):
            result.append({"source_id": "community_parallel:" + str(index), "family": "community_parallel",
                "english": row["eng"] or "", "twi": row["twi"] or "", "genre": row["source_files"] or "",
                "original_duplicate_count": row["count"], "source_ref": row["source_ref"]})
    require(len({r["source_id"] for r in result}) == len(result), "Ambiguous original parallel source IDs")
    for row in result:
        account, assessment = accounts.get(row["source_id"]), assessments.get(row["source_id"])
        row["inherited_account"] = account
        row["inherited_assessment"] = assessment
        row["lineage"] = lineage.get(row["source_id"], [])
        row["source_split"] = account["split"] if account else "unassigned"
        row["type_assessment"] = types.classify(row["english"], row["twi"], row["family"], row["genre"],
                                                 "sentence", "compiled_parallel_text")
        prose = row["type_assessment"]["effective_type"] != "dictionary_entry"
        row["document_like_structure"] = prose and (len(row["english"].split()) > 100 or
            len(re.findall(r"[.!?](?:\s|$)", row["english"])) >= 4 or row["english"].count("\n") >= 2)
    return result, {"path": str(release), "manifest_sha256": inputs.bind(release / "manifest.json"),
                    "routing_policy": types.POLICY, "source_accounting_rows": len(accounts)}


def parallel_candidates(rows, families, blocked, index, r6_sources):
    ledger = []
    for row in rows:
        reasons, assessment = [], row["type_assessment"]
        aliases = identity_keys(row)
        protected = next((blocked[families.root(k)] for k in sorted(aliases) if families.root(k) in blocked), None)
        hit = next((h for text in (row["english"], row["twi"], row["english"] + "\n" + row["twi"])
                    if text.strip() and (h := index.match(text))), None)
        if protected or hit:
            reasons.append("protected_source_family_or_content")
        if row["source_split"] not in ("train", "unassigned"):
            reasons.append("inherited_" + row["source_split"] + "_split")
        if row["source_split"] == "unassigned":
            reasons.append("no_inherited_split_assignment")
        old = row["inherited_assessment"]
        if old and old["content_sha256"] != assessment["content_sha256"]:
            reasons.append("original_pair_differs_from_inherited_projection")
        if any(r["source_file_sha256"] != row["source_ref"]["file_sha256"] for r in row["lineage"]):
            reasons.append("inherited_original_file_hash_differs")
        if len({r["split"] for r in row["lineage"]} | {row["source_split"]}) > 1:
            reasons.append("inherited_split_conflict")
        if assessment["status"] != "automatically_screened":
            reasons.append(assessment["reason"])
        if row["document_like_structure"]:
            reasons.append("document_like_structure_requires_separate_review")
        reasons.extend(contact_flags(row["english"] + "\n" + row["twi"]))
        if CONTROL.search(row["english"] + row["twi"]):
            reasons.append("literal_control_token")
        state = "excluded" if any(reason.startswith(("protected_", "inherited_validation", "inherited_test"))
                                   for reason in reasons) else "pending" if reasons else "eligible"
        route = ("lexicon" if assessment["effective_type"] == "dictionary_entry" else
                 "documents_review" if row["document_like_structure"] else assessment["view"] or "type_review")
        ledger.append({"id": "parallel:" + row["source_id"], "source_id": row["source_id"],
            "source_family": row["family"], "source_genre": row["genre"], "source_split": row["source_split"],
            "status": state, "reasons": sorted(set(reasons)) or ["unchanged_source_pair_inherited_train_scoped_routing"],
            "routing": route, "type_assessment": assessment, "document_like_structure": row["document_like_structure"],
            "source_ref": row["source_ref"], "inherited_refs": [r["source_ref"] for r in row["lineage"]],
            "source_family_keys": sorted(aliases), "english_sha256": digest(row["english"]), "twi_sha256": digest(row["twi"]),
            "semantic_reference": {"kind": "original_bilingual_pair_not_certified_gold",
                "english_field": "text" if row["family"] == "ghana_nlp_parallel" else "eng",
                "twi_field": "label" if row["family"] == "ghana_nlp_parallel" else "twi",
                "original_pair_content_sha256": assessment["content_sha256"],
                "inherited_pair_content_sha256": old["content_sha256"] if old else None,
                "inherited_source_hash": (row["inherited_account"] or {}).get("source_hash"),
                "meaning_correctness_certified": False, "directions": ["en_to_tw", "tw_to_en"]},
            "already_in_r6_source": row["source_id"] in r6_sources,
            "target_policy": "original bilingual pair only; no retranslation or free-response target",
            "origin": "compiled_parallel_text", "generated_target": False, "text_exported": False,
            "training_eligible": False, "semantic_certification": False,
            **({"protection_evidence": protected or hit} if protected or hit else {})})
    return [r for r in ledger if r["status"] == "eligible"], ledger


def prepare(r6_pack, r5_pack, histories, oasst_train, oasst_validation, health, out,
            oasst_limit=20000, health_limit=20000, parallel_corpus=None, alignment_release=None):
    out = Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "Use a new inventory directory; never overwrite evidence")
    require(all(type(n) is int and 1000 <= n <= 50000 for n in (oasst_limit, health_limit)), "Queue budgets must be 1000..50000")
    inputs = Inputs()
    inputs.bind(__file__)
    inputs.bind(Path(__file__).with_name("prepare_understanding_run.py"))
    packs = {"r5": load_pack(r5_pack, inputs), "r6": load_pack(r6_pack, inputs)}
    originals = inputs.rows(histories)
    train_files, validation_files = sorted(Path(oasst_train).glob("*.parquet")), sorted(Path(oasst_validation).glob("*.parquet"))
    require(train_files and validation_files, "Both original OASST partitions are required")
    for folder in (Path(oasst_train), Path(oasst_validation)):
        if (folder / "README.md").is_file():
            inputs.bind(folder / "README.md")
    train = [row for path in train_files for row in inputs.parquet(path, OASST_COLUMNS, "train")]
    validation = [row for path in validation_files for row in inputs.parquet(path, OASST_COLUMNS, "validation")]
    health_rows = [row for split in ("train", "validation", "test_holdout")
                   for row in inputs.parquet(Path(health) / (split + ".parquet"), HEALTH_COLUMNS, split)]
    require((parallel_corpus is None) == (alignment_release is None), "Parallel corpus and inherited alignment release required together")
    parallel, parallel_release = parallel_sources(parallel_corpus, alignment_release, inputs) if parallel_corpus else ([], None)
    require(len({r["id"] for r in health_rows}) == len(health_rows), "Duplicate AfriHealth ID")
    print("Loaded and hashed source files; building protection and ancestry", file=sys.stderr, flush=True)
    families, held, explicit_blocked = Families(), [], []
    all_records = [w["record"] for pack in packs.values() for w in pack["records"]]
    all_records += [w for pack in packs.values() for w in pack["witnesses"]] + originals
    for row in all_records:
        families.connect(identity_keys(row))
    for row in parallel:
        families.connect(identity_keys(row))
    for row in train + validation:
        families.connect([row["message_id"], row["message_tree_id"]])
    for name, pack in packs.items():
        held.extend({**row, "id": name + ":" + row["id"]} for row in pack["held"])
        for decision in pack["decisions"]:
            keys = identity_keys(decision.get("record", {})) | identity_keys(decision)
            families.connect(keys)
            explicit_blocked.extend((key, {"kind": "prior_pack_explicit_exclusion", "pack": name,
                                           "record_id": decision.get("record_id")}) for key in keys)
        fresh = pack["evaluation"].get("fresh_path")
        if fresh:
            folder = Path(fresh)
            inputs.bind(folder / "manifest.json", pack["evaluation"]["fresh_manifest_sha256"])
            manifest = inputs.read(folder / "manifest.json")
            require(inputs.read(folder / "FROZEN.json")["manifest_sha256"] == inputs.bind(folder / "manifest.json"), "Fresh holdout seal differs")
            checksum = manifest["artifacts"]["held-content.jsonl"]
            inputs.bind(folder / "held-content.jsonl", checksum)
            held.extend({**row, "id": name + ":fresh:" + row["id"]} for row in inputs.rows(folder / "held-content.jsonl"))
    for row in validation:
        explicit_blocked.append((row["message_tree_id"], {"kind": "original_oasst_validation_tree"}))
        if isinstance(row["text"], str) and row["text"].strip():
            held.append({"id": "oasst-validation:" + row["message_id"], "text": row["text"]})
    for row in health_rows:
        if row["source_split"] != "train":
            for field in ("question", "answer"):
                if row[field].strip():
                    held.append({"id": "afrihealth-" + row["source_split"] + ":" + row["id"] + ":" + field, "text": row[field]})
    for row in parallel:
        if row["source_split"] not in ("train", "unassigned"):
            explicit_blocked.extend((key, {"kind": "inherited_parallel_heldout", "source_id": row["source_id"]})
                                    for key in identity_keys(row))
            for field in ("english", "twi"):
                if row[field].strip():
                    held.append({"id": "parallel-heldout:" + row["source_id"] + ":" + field, "text": row[field]})
    require(held and all(isinstance(row.get("text"), str) and row["text"].strip() for row in held), "Missing/empty protection")
    for row in held:
        keys = identity_keys(row)
        families.connect(keys)
        explicit_blocked.extend((key, {"kind": "held_source_identity", "held_id": row["id"]}) for key in keys)
    index = HoldoutIndex((row["id"], row["text"]) for row in held)
    cache = {}
    def match(text):
        key = digest(text)
        if key not in cache:
            cache[key] = index.match(text)
        return cache[key]
    blocked = {families.root(key): proof for key, proof in explicit_blocked}
    for row in train:
        hit = match(row["text"]) if isinstance(row["text"], str) else None
        if hit:
            blocked.setdefault(families.root(row["message_tree_id"]), {**hit, "matched_source_message": row["message_id"]})
    for row in all_records:
        hit = next((h for text in text_views(row) if (h := match(text))), None)
        if hit:
            for key in identity_keys(row):
                blocked.setdefault(families.root(key), {**hit, "matched_record": row.get("id")})
    prior_trees = {families.root(key) for row in packs["r6"]["records"] for key in identity_keys(row["record"])}
    prior_trees |= {families.root(key) for row in originals for key in identity_keys(row)}
    candidates, oasst_ledger = oasst_candidates(train, validation, families, blocked, prior_trees, oasst_limit)
    print("OASST candidates selected; auditing health and inherited targets", file=sys.stderr, flush=True)
    health_queue, health_ledger, health_grouping = health_candidates(health_rows, index, health_limit)
    r6_records = {r["record"]["id"]: r["record"] for r in packs["r6"]["records"] if r["collection"] == "responses"}
    require(len({r["id"] for r in originals}) == len(originals), "Duplicate original R6 history")
    require(set(r6_records) == {r["id"] for r in originals}, "R6 original history coverage differs from pack")
    flags, duplicate_targets = [], defaultdict(set)
    for row in originals:
        require(row["messages"] == r6_records[row["id"]]["messages"] and row["source_id"] == r6_records[row["id"]]["source_id"],
                "R6 original history differs from packed source")
        for turn, message in enumerate(row["messages"]):
            if message["role"] == "assistant":
                duplicate_targets[digest(normalize(message["content"]))].add((row["id"], turn))
    histories_sha = inputs.files[str(Path(histories).absolute())]["sha256"]
    for line, row in enumerate(originals, 1):
        findings = history_flags(row["messages"])
        findings.extend({**flag, "view": "preserved_english_source"} for flag in history_flags(row.get("english_messages", []))
                        if flag["code"] == "past_search_claim_without_tool_evidence")
        for turn, message in enumerate(row["messages"]):
            if message["role"] == "assistant":
                occurrences = duplicate_targets[digest(normalize(message["content"]))]
                if len(occurrences) > 1:
                    findings.append({"code": "duplicate_normalized_target_across_occurrences", "turn": turn,
                                     "occurrences": len(occurrences)})
        for finding in findings:
            flags.append({"id": row["id"], "source_id": row["source_id"], "source_ref": {"file": str(histories), "line": line,
                "file_sha256": histories_sha,
                "record_sha256": digest(row)}, "status": "pending_review", "semantic_error_confirmed": False, **finding})
    replay, current = [], replay_indexes(packs["r6"]["witnesses"], families)
    for wrapper in packs["r5"]["records"]:
        row, collection = wrapper["record"], wrapper["collection"]
        keys = identity_keys(row)
        protected = next((blocked[families.root(k)] for k in sorted(keys) if families.root(k) in blocked), None)
        messages = row.get("messages", [])
        comparison = compare_replay(row, collection, current, families)
        missing = comparison["missing_exact_history_views"]
        reasons = (["protected_source_family"] if protected else
                   ["all_exact_target_views_present_in_r6"] if not missing else
                   ["r5_exact_history_views_missing_from_r6", "prior_review_preserved_new_admission_required"])
        if missing and not comparison["assistant_target_occurrences_missing_for_same_source"] and messages:
            reasons.append("source_targets_retained_but_context_or_prompt_changed")
        replay.append({"id": row["id"], "collection": collection, "source_id": row["source_id"],
                       "status": "excluded" if protected or not missing else "pending", "reasons": reasons,
                       **comparison, "source_ref": wrapper["ref"],
                       "source_family_keys": sorted(keys), "review_flags": history_flags(messages),
                       "prior_origin": row.get("origin", row.get("metadata", {}).get("origin", "missing")),
                       "prior_review_sha256": digest(row.get("review", row.get("admission_provenance"))),
                       "training_eligible": False, "text_exported": False,
                       **({"protection_evidence": protected} if protected else {})})
    print("Auditing original parallel pairs and inherited routing", file=sys.stderr, flush=True)
    parallel_queue, parallel_ledger = parallel_candidates(parallel, families, blocked, index,
        {r["record"]["source_id"] for r in packs["r6"]["records"]})
    summary = {"version": VERSION, "scope": "source_inventory_and_review_queues_not_training_admission",
        "packs": {name: {k: pack[k] for k in ("path", "manifest_sha256", "collections", "overlap_audit")}
                  for name, pack in packs.items()},
        "oasst": {split: {"messages": len(values), "trees": len({r["message_tree_id"] for r in values}),
                           "by_language": dict(Counter(r["lang"] for r in values)),
                           "by_role": dict(Counter(r["role"] for r in values)),
                           "original_english_assistant_patterns": raw_assistant_patterns(values),
                           "english_assistant_rank_counts": dict(Counter(str(r["rank"]) for r in values
                                                                          if r["lang"] == "en" and r["role"] == "assistant"))}
                  for split, values in (("train", train), ("validation", validation))},
        "oasst_decisions": dispositions(oasst_ledger), "oasst_queue": {"histories": len(candidates),
            "trees": len({r["tree_id"] for r in candidates}), "fresh_vs_r6_family_histories": sum(not r["already_in_r6_source_family"] for r in candidates),
            "coverage_buckets": dict(Counter(r["coverage_bucket"] for r in candidates)), **pattern_summary(candidates)},
        "afrihealth": {split: {"rows": sum(r["source_split"] == split for r in health_rows),
                              "by_language_country": dict(Counter(r["language_country"] for r in health_rows if r["source_split"] == split))}
                       for split in ("train", "validation", "test_holdout")},
        "afrihealth_grouping": health_grouping, "health_decisions": dispositions(health_ledger),
        "health_queue": {"rows": len(health_queue), "by_language_country": dict(Counter(r["language_country"] for r in health_queue)),
                         "coverage_buckets": dict(Counter(r["coverage_bucket"] for r in health_queue))},
        "r6_original": {**pattern_summary(originals), "origin_counts": dict(Counter(r.get("origin", "missing") for r in originals)),
                        "flags_by_occurrence": dict(Counter(f["code"] for f in flags)), "source_pack_exactly_matched": True},
        "r5_replay": {**dispositions(replay), "by_collection": {c: {**dispositions([r for r in replay if r["collection"] == c]),
            "registered_records_source_family_absent_from_r6": sum(not r["source_present_in_r6"] for r in replay if r["collection"] == c),
            "registered_records_raw_source_id_absent_from_r6": sum(not r["raw_source_id_present_in_r6"] for r in replay if r["collection"] == c),
            "assistant_target_occurrences_missing_for_same_source": sum(r["assistant_target_occurrences_missing_for_same_source"]
                                                                        for r in replay if r["collection"] == c),
            "pending_source_targets_retained_context_changed": sum(r["status"] == "pending" and
                 "source_targets_retained_but_context_or_prompt_changed" in r["reasons"] for r in replay if r["collection"] == c)}
            for c in sorted({r["collection"] for r in replay})}},
        "parallel_meaning": {"release": parallel_release, "decisions": dispositions(parallel_ledger),
            "candidate_pairs": len(parallel_queue), "candidate_routes": dict(Counter(r["routing"] for r in parallel_queue)),
            "fresh_vs_r6_candidate_pairs": sum(not r["already_in_r6_source"] for r in parallel_queue),
            "by_source": {name: {"original_rows": len(values),
                "origin": "compiled_parallel_text",
                "source_schema": ["id", "text", "label", "Unnamed: 3"] if name == "ghana_nlp_parallel"
                                 else ["eng", "twi", "count", "source_files"],
                "original_pair_distinct_hashes": len({digest([r["english"], r["twi"]]) for r in values}),
                "distinct_original_english": len({digest(r["english"]) for r in values}),
                "distinct_original_twi": len({digest(r["twi"]) for r in values}),
                "genre_counts": dict(Counter(r["genre"] for r in values)),
                "inherited_splits": dict(Counter(r["source_split"] for r in values)),
                "recomputed_types": dict(Counter(r["type_assessment"]["effective_type"] or "unresolved" for r in values)),
                "document_like_structures": sum(r["document_like_structure"] for r in values),
                "decisions": dispositions([r for r in parallel_ledger if r["source_family"] == name])}
                for name, values in ((family, [r for r in parallel if r["family"] == family])
                                     for family in sorted({r["family"] for r in parallel}))}},
        "protection": {"held_units_including_references": len(held), "distinct_normalized_held_texts": len(index.exact),
                       "blocked_connected_families": len(blocked), "original_oasst_validation_trees": len({r["message_tree_id"] for r in validation}),
                       "r6_has_pack_local_held_content": bool(packs["r6"]["held"]),
                       "matcher": "NFC/casefold word-normalized exact; eight-word shingle Jaccard >=0.85; connected identity propagation",
                       "semantic_independence_certified": False},
        "limits": ["Short/generic/list/correction/repetition flags are lexical review selectors, not confirmed semantic mistakes.",
                   "Raw word mentions are not category-example coverage; source-faithful translation does not validate a claimed past search.",
                   "Parallel meaning candidates are references to existing pairs with inherited splits; dictionary entries are not sentences or dialogue.",
                   "Upstream PII labels and contact screens cannot prove all personal information absent; contributor user IDs are not loaded/exported.",
                   "AfriHealth cross-language alignment and primary clinical grounding remain unresolved; no health candidate is admitted.",
                   "Counts describe available records, prefixes and planned historical exposures, not independent examples or new consumption.",
                   "Prior validation/test and pack holdouts are excluded; unknown pretraining overlap and unregistered holdouts are not certified."],
        "training_eligible": False, "synthetic_twi_created": 0, "api_calls": 0, "gpu_runs": 0}
    inputs.recheck()
    out.mkdir(parents=True, mode=0o700)
    outputs = {"oasst-candidates.jsonl": candidates, "oasst-decisions.jsonl": oasst_ledger,
               "health-candidates.jsonl": health_queue, "health-decisions.jsonl": health_ledger,
               "r5-replay-candidates.jsonl": [r for r in replay if r["status"] == "pending"],
               "r5-replay-decisions.jsonl": replay, "r6-review-flags.jsonl": flags,
               "parallel-meaning-candidates.jsonl": parallel_queue, "parallel-meaning-decisions.jsonl": parallel_ledger,
               "oasst-tree-index.jsonl": [{k: row[k] for k in ("message_id", "parent_id", "message_tree_id", "role", "lang", "rank", "source_split", "source_ref")}
                                          for row in train + validation],
               "protection-index.jsonl": [{"id": row["id"], "text_sha256": digest(row["text"]),
                                           "normalized_text_sha256": digest(normalize(row["text"])),
                                           "identity_keys": sorted(identity_keys(row))} for row in held]}
    for name, rows in outputs.items():
        write_rows(out / name, rows)
    write_json(out / "summary.json", summary)
    inputs.recheck()
    write_json(out / "manifest.json", {"version": VERSION, "input_files": inputs.files,
        "configuration": {"oasst_limit": oasst_limit, "health_limit": health_limit, "max_history_characters": MAX_HISTORY_CHARS,
                          "ranking": "all assistant ancestors rank 0; deterministic round-robin lexical coverage buckets",
                          "history_truncated": False, "user_ids_loaded": False},
        "artifacts": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "training_eligible": False, "production_eligible": False})
    write_json(out / "SEALED.json", {"manifest_sha256": sha(out / "manifest.json")})
    verify(out)
    return summary


def verify(out):
    out = Path(out)
    manifest = strict_json((out / "manifest.json").read_text())
    require(strict_json((out / "SEALED.json").read_text()) == {"manifest_sha256": sha(out / "manifest.json")}, "Inventory seal differs")
    require(manifest["version"] == VERSION and manifest["training_eligible"] is False, "Wrong inventory identity")
    require({p.name for p in out.iterdir()} == set(manifest["artifacts"]) | {"manifest.json", "SEALED.json"}, "Unbound inventory artifact")
    for name, checksum in manifest["artifacts"].items():
        require(Path(name).name == name and (out / name).is_file() and not (out / name).is_symlink()
                and sha(out / name) == checksum, "Inventory artifact changed: " + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--r6-pack", type=Path, default=PROJECT / "tmp/understanding-r6-20261113/pack-v1")
    parser.add_argument("--r5-pack", type=Path, default=PROJECT / "tmp/understanding-r5-20261109/pack-v1")
    parser.add_argument("--histories", type=Path, default=HISTORIES)
    parser.add_argument("--oasst-train", type=Path, default=PROJECT / "tmp/general-language-corpus/oasst_train")
    parser.add_argument("--oasst-validation", type=Path, default=PROJECT / "tmp/general-language-corpus/oasst_validation")
    parser.add_argument("--health", type=Path, default=PROJECT / "tmp/external/afrihealth-qa")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/r7-20261117/source-inventory/v4")
    parser.add_argument("--parallel-corpus", type=Path, default=PROJECT / "tmp/general-language-corpus")
    parser.add_argument("--alignment-release", type=Path, default=PROJECT / "tmp/corpus-ready/twi-alignment-v3-20260929")
    parser.add_argument("--oasst-limit", type=int, default=20000)
    parser.add_argument("--health-limit", type=int, default=20000)
    args = parser.parse_args()
    if args.action == "verify":
        value = verify(args.out)
        print(json.dumps({"verified": True, "artifacts": len(value["artifacts"])}))
    else:
        value = prepare(args.r6_pack, args.r5_pack, args.histories, args.oasst_train, args.oasst_validation,
                        args.health, args.out, args.oasst_limit, args.health_limit, args.parallel_corpus, args.alignment_release)
        print(json.dumps({"directory": str(args.out), "oasst_histories": value["oasst_queue"]["histories"],
                          "health_pending": value["health_queue"]["rows"], "r5_replay": value["r5_replay"]["status"],
                          "parallel_candidate_pairs": value["parallel_meaning"]["candidate_pairs"],
                          "parallel_candidate_routes": value["parallel_meaning"]["candidate_routes"],
                          "r6_flags": value["r6_original"]["flags_by_occurrence"]}, indent=2))


if __name__ == "__main__":
    main()
