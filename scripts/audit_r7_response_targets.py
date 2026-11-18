"""Read-only, whole-corpus R6 response-target audit for R7; not admission.

Join every preserved English history to its registered source and original OASST
message IDs. Reproduce all inventory flags, then inspect repeated sentence/list/
8-gram/cross-label spans with code/quotation/creative-structure protection.
No translator, model, network, GPU or observed evaluation prompt is consulted.

No source is edited. Exact adjacent deletion proposals require a separately
hash-bound agent-reviewed source alignment; lexical detection alone never edits.
All candidates remain pending bilingual review and fresh training admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import prepare_r7_source_inventory as inventory


ROOT = Path(__file__).resolve().parents[1]
RESTORED = Path("/Users/welcome/ghana-health-twi-r6")
DEFAULT_INVENTORY = ROOT / "tmp/r7-20261117/source-inventory/v3"
DEFAULT_OUT = ROOT / "tmp/r7-20261117/response-target-audit-v1"
HISTORIES = RESTORED / "work/responses-r6/records.jsonl"
ENGLISH_QUEUE = RESTORED / "twi-r5-resume/source-review-v2/source-review-queue.jsonl"
VERSION = "r7-response-target-audit-v1"
SELECTED = {"repeated_sentence_or_line", "repeated_list_items",
            "repeated_8gram_at_least_three_times", "same_literal_example_across_category_labels",
            "past_search_claim_without_tool_evidence"}
PENDING = {"training_ready": False, "training_eligible": False, "production_eligible": False,
           "human_reviewed": False, "native_certified": False}
INTERNET_CAPABILITY = re.compile(r"(?:my programming allows me to access any information on the internet|"
                                 r"I (?:can|am able to) (?:browse|access) (?:any|all) (?:information|websites))", re.I)
CODE_LINE = re.compile(r"^\s*(?:(?:def|class|function|import|from|const|let|var|return|for|while|if|elif|else|"
                       r"public|private|void|typedef|#include)\b|[{}]|[\w.\[\]]+\s*(?:=|\+=|\-=)|"
                       r"[\w.]+\([^\n]*\)\s*;?\s*$)")
CREATIVE = re.compile(r"\b(?:chorus|verse|lyrics|refrain|poem|screenplay|song|script dialogue)\b", re.I)
PLACEHOLDER = re.compile(r"\[[^\]\n]{1,90}\]")

# These notes follow direct inspection of paired source text, not heuristic labels.
REVIEWED_HISTORY_SHA = "cbac18d36bd33742f5e2b37084e0d6deb0e3ff6a9cdce3df52449f7b64a986ee"
AGENT_REVIEW_NOTES = [
    (132, 1, "nonadjacent_exact_duplicate_review", "The same Elijah entry occurs twice in both languages, separated by other names. This reduces list diversity but is not an adjacent deletion.",
     "Review removal of the later duplicate against both histories; no revised target supplied."),
    (179, 1, "preserve_section_specific_repetition", "Act 1: Setup and midpoint recur in different named story-structure lists in the English source. Global list deduplication would remove a step from a separate structure.", None),
    (188, 1, "unsupported_search_claim_review", "The English source says sources were found online and the Twi repeats that claim. Neither history contains a tool result. The two inventory flags are language views of this one turn, not two independent claims.",
     "Re-review the unsupported browsing assertion and factual ranking; do not admit a rewritten Twi target without grounding and bilingual review."),
    (221, 3, "preserve_section_specific_repetition", "Repeated quantities of coffee, sugar, paprika and seasoning belong to separate recipes in both languages. Removing them globally would change an individual recipe.", None),
    (254, 1, "confirmed_normalization_collision", "The source lists C, C++ and C# as different programming languages. Word-only normalization collapses all three to c; the raw entries are not duplicates.", None),
    (294, 1, "preserve_section_specific_repetition", "The calendar schedules similar tasks on different named weekdays, as does the English source. These are separate scheduled occurrences, not adjacent duplicate prose.", None),
    (344, 3, "preserve_shared_attribute", "The same cannot-expand/contract property is assigned to solid and liquid subsections in the English notes. Repetition screening alone does not establish a category error or certify the science.", None),
    (403, 3, "preserve_algorithm_structure", "The failure-function construction and pattern-search phases each contain the same mismatch recovery instruction. Both source phases and code must remain separate.", None),
    (440, 1, "confirmed_normalization_collision", "C# and C++ are distinct raw list entries in the English and Twi histories; both normalize to c. No duplicate item should be deleted.", None),
    (464, 1, "preserve_code_coordinates", "Adjacent repeated random.uniform expressions are coordinate components in the source Blender code. They are code, not sentences to remove.", None),
    (464, 3, "preserve_code_coordinates", "The follow-up Bezier code retains repeated random.uniform coordinate expressions from the source. Deduplication would alter the program.", None),
    (661, 1, "preserve_definition_summary_structure", "The deep-learning definition appears in a main bullet and again in the English conclusion, and is similarly repeated in Twi. This is source summary redundancy, not an adjacent duplicate.", None),
    (784, 1, "preserve_template_slots", "Two identical adjacent project placeholder lines are present in both versions of the resume template. They represent two fillable project slots; deleting one changes the template.", None),
    (842, 1, "unsupported_internet_capability_review", "The English source claims unrestricted internet access regardless of harmfulness; the Twi preserves it. This is an unsupported capability assertion, distinct from the two past-search flag views.",
     "Re-review the capability assertion and unverified ranked list; no blind translation or replacement claims supplied."),
    (875, 1, "nonadjacent_exact_duplicate_prose_review", "The English essay repeats complete sentences about the importance of work and its place in a larger plan across later paragraphs; the Twi retains these repetitions. They are real prose redundancy, not code, but are non-adjacent.",
     "Queue a source-grounded concision review of repeated prose; non-adjacent editing exceeds the authorized automatic rule."),
]
APPROVED_DELETIONS = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        result = [inventory.strict_json(line) for line in stream]
    require(all(isinstance(row, dict) for row in result), "Invalid JSONL row")
    return result


def normalize(text):
    return inventory.normalize(text)


def protected_spans(text):
    """Conservative syntax protection, not a code correctness judgment."""
    spans, active = [], None
    for match in re.finditer(r".*(?:\n|$)", text):
        line = match.group()
        if not line:
            continue
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if active:
            if marker and marker[1][0] == active[1] and len(marker[1]) >= active[2]:
                spans.append({"start": active[0], "end": match.end(), "kind": "fenced_code"})
                active = None
        elif marker:
            active = (match.start(), marker[1][0], len(marker[1]))
        elif CODE_LINE.search(line):
            spans.append({"start": match.start(), "end": match.end(), "kind": "code_like_line"})
        elif re.match(r"^\s*>", line):
            spans.append({"start": match.start(), "end": match.end(), "kind": "blockquote"})
    if active:
        spans.append({"start": active[0], "end": len(text), "kind": "unclosed_code_fence"})
    for pattern, kind in [(r"`[^`\n]+`", "inline_code"), (r'"[^"\n]+"', "quotation"),
                          (r"\u201c[^\u201d\n]+\u201d", "quotation"),
                          (r"\[[^\]\n]{1,90}\]", "placeholder_or_stage_direction")]:
        spans.extend({"start": m.start(), "end": m.end(), "kind": kind} for m in re.finditer(pattern, text))
    return sorted(spans, key=lambda value: (value["start"], value["end"], value["kind"]))


def _span(text, start, end, protected):
    return {"start": start, "end": end, "text": text[start:end],
            "protected_by": sorted({p["kind"] for p in protected if p["start"] < end and start < p["end"]})}


def pattern_evidence(text):
    """Reproduce the inventory's lexical units; offsets address NFC codepoints."""
    text = unicodedata.normalize("NFC", text)
    protected = protected_spans(text)
    sentences, lists, grams, labels = (defaultdict(list) for _ in range(4))
    for match in re.finditer(r"[^.!?\n]+", text):
        value = normalize(match.group())
        if len(value.split()) >= 4:
            sentences[value].append(_span(text, match.start(), match.end(), protected))
    for match in inventory.LIST_LINE.finditer(text):
        value = match.group(1)
        lists[normalize(value)].append(_span(text, *match.span(1), protected))
        if ":" in value:
            label, tail = value.split(":", 1)
            base = match.start(1) + value.index(":") + 1
            for part in re.finditer(r"[^,;]+", tail):
                literal = normalize(part.group().strip(" *`"))
                if literal:
                    labels[literal].append({**_span(text, base + part.start(), base + part.end(), protected),
                                            "label": label, "normalized_label": normalize(label)})
    words = list(re.finditer(r"\w+", text))
    for index in range(max(0, len(words) - 7)):
        window = words[index:index + 8]
        key = " ".join(m.group().casefold() for m in window)
        grams[key].append(_span(text, window[0].start(), window[-1].end(), protected))

    def repeated(groups, minimum=2, cross=False):
        return [{"normalized_unit": key, "occurrences": value,
                 "distinct_exact_forms": len({s["text"].strip() for s in value})}
                for key, value in sorted(groups.items()) if len(value) >= minimum and
                (not cross or len({v["normalized_label"] for v in value}) > 1)]

    return {"offset_basis": "NFC_unicode_codepoints", "protected_spans": protected,
            "repeated_sentence_or_line": repeated(sentences), "repeated_list_items": repeated(lists),
            "repeated_8gram_at_least_three_times": repeated(grams, 3),
            "same_literal_example_across_category_labels": repeated(labels, cross=True)}


def adjacent_duplicates(text):
    """Find exact adjacent units only; finding one is not permission to delete."""
    units = []
    for match in re.finditer(r"[^.!?\n]+[.!?]+|[^.!?\n]+(?=\n|$)", text):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        units.append((start, end, text[start:end]))
    found = []
    for left, right in zip(units, units[1:]):
        if left[2] == right[2] and len(normalize(left[2]).split()) >= 4 and not text[left[1]:right[0]].strip():
            found.append({"kind": "sentence_or_line", "kept_span": list(left[:2]),
                          "duplicate_span": list(right[:2]), "text": left[2]})
    matches = list(inventory.LIST_LINE.finditer(text))
    for left, right in zip(matches, matches[1:]):
        if left.group(1) == right.group(1) and not text[left.end():right.start()].strip():
            found.append({"kind": "list_item", "kept_span": list(left.span()),
                          "duplicate_span": list(right.span()), "text": right.group()})
    return found


def safe_adjacent_deletion(target, english, proof):
    """Validate a manually aligned deletion; reject all structural ambiguity."""
    require(proof.get("review_method") == "agent_source_alignment" and proof.get("unchanged_meaning") is True,
            "Explicit source-alignment review required")
    require(proof.get("target_sha256") == digest(target) and proof.get("english_sha256") == digest(english),
            "Deletion source changed")
    require(not protected_spans(target) and not protected_spans(english)
            and not re.search(r"[\"'\u2018\u2019\u201c\u201d]", target + english)
            and not CREATIVE.search(english), "Code, quoted, creative or template ambiguity")
    candidates = adjacent_duplicates(target)
    pair = next((p for p in candidates if p["kind"] == proof.get("kind", "sentence_or_line")
                 and p["kept_span"] == proof.get("kept_span")
                 and p["duplicate_span"] == proof.get("duplicate_span")), None)
    require(pair is not None, "Not an exact adjacent duplicate")
    require(bool(proof.get("reason", "").strip()), "Missing alignment rationale")
    source_span = proof["english_span"]
    require(len(source_span) == 2 and all(type(x) is int for x in source_span)
            and 0 <= source_span[0] < source_span[1] <= len(english), "Invalid English alignment span")
    evidence = english[source_span[0]:source_span[1]]
    require(len(normalize(evidence).split()) >= 4 and english.count(evidence) == 1,
            "English alignment must identify one unique source unit")
    start, end = pair["duplicate_span"]
    require(target[pair["kept_span"][1]:start].isspace(), "Ambiguous duplicate boundary")
    if pair["kind"] == "list_item":
        require(re.match(r"^\s*[-*\u2022]\s", pair["text"]), "Numbered lists require review, not renumbering")
    else:
        require(pair["text"][-1] in ".!?", "Unpunctuated prose line requires review")
    return {"start": pair["kept_span"][1], "end": end,
            "deleted_text": target[pair["kept_span"][1]:end],
            "replacement": "", "proof": copy.deepcopy(proof)}


def classify_flag(flag, tw, en, english_context):
    code = flag["code"]
    if code == "past_search_claim_without_tool_evidence":
        return {"classification": "unsupported_execution_claim_review", "action": "review_no_rewrite",
                "reason": "Source-faithful translation does not establish a browsing event.",
                "target_matches": [], "english_matches": []}
    target, source = tw[code], en[code]
    require(target, "Inventory flag has no reproducible span evidence: " + code)
    if code == "repeated_sentence_or_line":
        require(sum(len(g["occurrences"]) - 1 for g in target) == flag["duplicate_occurrences"], "Sentence flag count differs")
    elif code == "repeated_list_items":
        require(sum(len(g["occurrences"]) - 1 for g in target) == flag["duplicate_items"], "List flag count differs")
    elif code == "repeated_8gram_at_least_three_times":
        require(len(target) == flag["distinct_repeated_8grams"], "Eight-gram flag count differs")
    elif code == "same_literal_example_across_category_labels":
        require(len(target) == flag["reused_distinct_literals"], "Cross-label flag count differs")
    occurrences = [span for group in target for span in group["occurrences"]]
    code_kinds = {"fenced_code", "unclosed_code_fence", "inline_code", "code_like_line"}
    if code == "repeated_list_items" and all(g["distinct_exact_forms"] > 1 for g in target):
        kind, reason = "normalization_collision_review", "Word normalization erased distinctions between literal list entries."
    elif all(code_kinds & set(span["protected_by"]) for span in occurrences):
        kind, reason = "protected_code_repetition", "All flagged spans overlap code; removing them could change execution."
    elif CREATIVE.search(english_context):
        kind, reason = "creative_repetition_review", "A song, verse or creative request can intentionally repeat text."
    elif code == "same_literal_example_across_category_labels":
        kind, reason = "cross_label_shared_literal_review", "Shared fragments or attributes do not prove category confusion."
    elif any(span["protected_by"] for span in occurrences):
        kind, reason = "mixed_structure_review", "Some repeated spans are quoted, code-like or placeholders."
    elif code == "repeated_8gram_at_least_three_times":
        kind, reason = "repeated_phrase_review", "A repeated eight-word phrase is not necessarily a duplicated proposition."
    else:
        kind, reason = "prose_or_section_repetition_review", "Check source section boundaries and meaning; no blind deduplication."
    return {"classification": kind, "reason": reason, "action": "preserve_pending_review",
            "target_matches": target, "english_matches": source,
            "same_pattern_present_in_english": bool(source), "semantic_equivalence_certified": False}


def materialize_agent_reviews(records, original_sha):
    require(original_sha == REVIEWED_HISTORY_SHA, "Agent notes apply only to the inspected original snapshot")
    reviews = []
    for line, turn, decision, reason, plan in AGENT_REVIEW_NOTES:
        row = records[line - 1]
        review = {"id": row["id"], "source_id": row["source_id"], "source_group_id": row["group_id"],
                  "record_sha256": digest(row), "original_file_sha256": original_sha,
                  "original_line": line, "turn": turn,
                  "target_sha256": digest(row["messages"][turn]["content"]),
                  "english_sha256": digest(row["english_messages"][turn]["content"]),
                  "reviewer": "codex_agent", "method": "direct_paired_source_inspection",
                  "scope": "specified_repetition_or_claim_only_not_full_translation_certification",
                  "decision": decision, "rationale": reason, "candidate_correction_plan": plan,
                  "revised_target_supplied": False, **PENDING}
        review["review_id"] = "r7-agent-review:" + digest(review)
        reviews.append(review)
    return reviews


def audit_records(records, flags, english_sources, *, agent_reviews=(), deletion_proofs=()):
    """Account every record; preserve all flags and separately inspect selected ones."""
    by_id = {row["id"]: row for row in records}
    require(len(by_id) == len(records), "Duplicate response record ID")
    by_flags, reviewed = defaultdict(list), defaultdict(list)
    for flag in flags:
        require(flag["id"] in by_id and flag["source_id"] == by_id[flag["id"]]["source_id"], "Orphan inventory flag")
        require(flag["source_ref"]["record_sha256"] == digest(by_id[flag["id"]]), "Flag source hash differs")
        by_flags[flag["id"]].append(flag)
    for review in agent_reviews:
        require(review["id"] in by_id and digest(by_id[review["id"]]) == review["record_sha256"], "Agent review source differs")
        row = by_id[review["id"]]
        require(digest(row["messages"][review["turn"]]["content"]) == review["target_sha256"]
                and digest(row["english_messages"][review["turn"]]["content"]) == review["english_sha256"],
                "Agent reviewed turn differs")
        reviewed[review["id"]].append(review)
    audited, queue, candidates, registry = [], [], [], []
    for line, row in enumerate(records, 1):
        english = english_sources[row["tree_id"]]
        require(row["english_messages"] == english["messages"], "English source history differs")
        require([m["role"] for m in row["messages"]] == [m["role"] for m in english["messages"]], "Unaligned history roles")
        require(row["final_history_sha256"] == digest(row["messages"]), "Original final-history hash differs")
        findings, turns = [], {}
        for turn, message in enumerate(row["messages"]):
            if message["role"] != "assistant":
                continue
            tw = pattern_evidence(message["content"])
            en = pattern_evidence(english["messages"][turn]["content"])
            context = "\n".join(m["content"] for m in english["messages"][max(0, turn - 1):turn + 1])
            adj = adjacent_duplicates(message["content"])
            turns[turn] = {"tw": tw, "en": en, "context": context, "adjacent": adj}
        for flag in by_flags[row["id"]]:
            if flag["code"] not in SELECTED:
                continue
            turn = flag["turn"]
            evidence = turns[turn]
            detail = classify_flag(flag, evidence["tw"], evidence["en"], evidence["context"])
            findings.append({"id": "r7-target-finding:" + digest(flag), "turn": turn, "inventory_flag": flag,
                             "method": "automatic_span_inspection", "agent_reviewed": False,
                             "exact_adjacent_units": evidence["adjacent"], **detail})
        for turn, evidence in turns.items():
            source_text = english["messages"][turn]["content"]
            capability = INTERNET_CAPABILITY.search(source_text)
            if capability:
                findings.append({"id": "r7-target-finding:" + digest([row["id"], turn, "internet_capability"]),
                    "turn": turn, "method": "automatic_source_claim_inspection", "agent_reviewed": False,
                    "classification": "unsupported_internet_capability_review", "action": "review_no_rewrite",
                    "english_claim": capability.group(), "target_matches": [], "english_matches": [],
                    "reason": "The source asserts unrestricted internet access without an execution receipt."})
        for finding in findings:
            finding["linked_agent_review_ids"] = [r["review_id"] for r in reviewed[row["id"]] if r["turn"] == finding["turn"]]
            finding["agent_reviewed"] = bool(finding["linked_agent_review_ids"])
            turn = finding["turn"]
            queue.append({"id": finding["id"], "record_id": row["id"], "source_id": row["source_id"],
                          "source_group_id": row["group_id"], "original_record_sha256": digest(row),
                          "turn": turn, "twi_target": row["messages"][turn]["content"],
                          "english_source": english["messages"][turn]["content"],
                          "english_original_ref": english["original_ref"], "finding": finding,
                          "status": "pending_independent_review", **PENDING})
        report = {"id": row["id"], "source_id": row["source_id"], "source_group_id": row["group_id"],
                  "original_line": line, "original_record_sha256": digest(row), "original_record": row,
                  "english_original_ref": english["original_ref"], "english_join_exact": True,
                  "all_inventory_flags": by_flags[row["id"]], "findings": findings,
                  "assistant_turns_inspected": len(turns),
                  "adjacent_duplicate_inspection": [{"turn": t, "pairs": value["adjacent"],
                      "protected_spans": value["tw"]["protected_spans"]} for t, value in turns.items() if value["adjacent"]],
                  "automatic_status": "review_required" if findings else "no_selected_pattern_not_semantic_clearance",
                  "agent_review_ids": [r["review_id"] for r in reviewed[row["id"]]], **PENDING}
        audited.append(report)
    for proposal in deletion_proofs:
        row = by_id[proposal["id"]]
        require(digest(row) == proposal["record_sha256"], "Correction source differs")
        turn = proposal["turn"]
        require(any(r["review_id"] == proposal.get("review_id") and r["turn"] == turn
                    and r["decision"] == "approve_exact_adjacent_deletion" for r in reviewed[row["id"]]),
                "Deletion requires a registered explicit approval")
        edit = safe_adjacent_deletion(row["messages"][turn]["content"], row["english_messages"][turn]["content"], proposal["proof"])
        corrected = copy.deepcopy(row)
        text = corrected["messages"][turn]["content"]
        corrected["messages"][turn]["content"] = text[:edit["start"]] + text[edit["end"]:]
        corrected["final_history_sha256"] = digest(corrected["messages"])
        corrected.update(PENDING)
        corrected["admission"] = "pending_new_bilingual_review"
        correction = {"id": "r7-exact-deletion:" + digest([row["id"], turn, edit]),
                      "source_id": row["source_id"], "source_group_id": row["group_id"],
                      "original_record_sha256": digest(row), "original_id": row["id"], "turn": turn,
                      "edit": edit, "candidate_record": corrected, "candidate_record_sha256": digest(corrected),
                      "status": "candidate_not_admitted", **PENDING}
        candidates.append(correction)
        registry.append({"id": correction["id"], "source_id": row["source_id"], "source_group_id": row["group_id"],
                         "original_record_sha256": digest(row), "candidate_sha256": digest(correction),
                         "edit_policy": "agent_aligned_exact_adjacent_deletion_only", **PENDING})
    return {"records": audited, "review_queue": queue, "candidate_corrections": candidates,
            "correction_registry": registry, "agent_reviews": list(agent_reviews)}


def _project(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def join_english(records, queue, raw_messages, queue_ref, raw_ref):
    source = {row["tree_id"]: (line, row) for line, row in enumerate(queue, 1)}
    require(len(source) == len(queue), "Duplicate English source tree")
    joined = {}
    for row in records:
        require(row["tree_id"] in source, "Missing original English tree")
        line, original = source[row["tree_id"]]
        require(_project(original["messages"]) == row["english_messages"], "English queue join differs")
        require(original["source_file_sha256"] == raw_ref["sha256"]
                and Path(original["source_file"]).name == Path(raw_ref["file"]).name, "Original English file reference differs")
        require(len(original["message_ids"]) == len(original["messages"]), "English message ID count differs")
        for identity, message in zip(original["message_ids"], original["messages"], strict=True):
            require(identity in raw_messages, "Missing raw English message")
            raw = raw_messages[identity]
            require(raw["message_tree_id"] == row["tree_id"] and raw["lang"] == "en"
                    and raw["text"] == message["content"]
                    and {"prompter": "user", "assistant": "assistant"}[raw["role"]] == message["role"],
                    "Original English message differs")
        joined[row["tree_id"]] = {"messages": row["english_messages"],
            "original_ref": {"queue": queue_ref, "queue_line": line, "queue_record_sha256": digest(original),
                             "original_parquet": raw_ref, "message_ids": original["message_ids"],
                             "english_messages_sha256": digest(row["english_messages"]),
                             "raw_projection_sha256": digest([raw_messages[i] for i in original["message_ids"]])}}
    return joined


def load_inputs(folder=DEFAULT_INVENTORY, histories=HISTORIES, english_queue=ENGLISH_QUEUE, expected_count=997):
    folder, histories, english_queue = Path(folder), Path(histories), Path(english_queue)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).absolute()
        require(path.is_file() and path.resolve() == path, "Missing or linked input: " + str(path))
        value = sha(path)
        require(expected is None or value == expected, "Input hash differs: " + str(path))
        inputs[str(path)] = value
        return {"file": str(path), "sha256": value}

    inv_ref = bind(folder / "manifest.json")
    seal = inventory.strict_json((folder / "SEALED.json").read_text())
    require(seal["manifest_sha256"] == inv_ref["sha256"], "Inventory seal differs")
    bind(folder / "SEALED.json")
    manifest = inventory.strict_json((folder / "manifest.json").read_text())
    flags_ref = bind(folder / "r6-review-flags.jsonl", manifest["artifacts"]["r6-review-flags.jsonl"])
    bind(folder / "summary.json", manifest["artifacts"]["summary.json"])
    histories_ref = bind(histories, manifest["input_files"][str(histories)]["sha256"])
    original_manifest_ref = bind(histories.with_name("manifest.json"))
    original_manifest = inventory.strict_json(histories.with_name("manifest.json").read_text())
    require(original_manifest["records_sha256"] == histories_ref["sha256"], "Original registration differs")
    records, flags = rows(histories), rows(folder / "r6-review-flags.jsonl")
    require(len(records) == expected_count == original_manifest["registered"], "Incomplete response corpus")
    for flag in flags:
        require(flag["source_ref"]["file_sha256"] == histories_ref["sha256"]
                and records[flag["source_ref"]["line"] - 1]["id"] == flag["id"], "Flag original reference differs")
    packed = [Path(path) for path in manifest["input_files"] if path.endswith(
        "understanding-r6-20261113/pack-v1/datasets/responses/records.jsonl")]
    require(len(packed) == 1, "Missing exact R6 response pack source")
    bind(packed[0], manifest["input_files"][str(packed[0])]["sha256"])
    pack_rows = {row["id"]: row for row in rows(packed[0])}
    require(set(pack_rows) == {r["id"] for r in records}, "Pack corpus coverage differs")
    for row in records:
        require(pack_rows[row["id"]]["messages"] == row["messages"]
                and pack_rows[row["id"]]["source_id"] == row["source_id"], "Packed target differs")
    queue_ref = bind(english_queue)
    queue = rows(english_queue)
    raw_paths = [Path(path) for path in manifest["input_files"] if "/oasst_train/" in path and path.endswith(".parquet")]
    require(len(raw_paths) == 1, "Expected one pinned original English parquet")
    raw_ref = bind(raw_paths[0], manifest["input_files"][str(raw_paths[0])]["sha256"])
    trees = {r["tree_id"] for r in records}
    needed = {i for q in queue if q["tree_id"] in trees for i in q["message_ids"]}
    import pyarrow.parquet as pq
    raw = {}
    for batch in pq.ParquetFile(raw_paths[0]).iter_batches(columns=["message_id", "message_tree_id", "role", "lang", "text"]):
        for row in batch.to_pylist():
            if row["message_id"] in needed:
                require(row["message_id"] not in raw, "Duplicate raw English message ID")
                raw[row["message_id"]] = row
    require(set(raw) == needed, "Incomplete raw English join")
    joined = join_english(records, queue, raw, queue_ref, raw_ref)
    for filename, expected in inputs.items():
        require(sha(filename) == expected, "Input changed during audit")
    return records, flags, joined, {"files": inputs, "inventory": inv_ref, "flags": flags_ref,
                                   "histories": histories_ref, "original_manifest": original_manifest_ref}


def summary(audit):
    records, queue = audit["records"], audit["review_queue"]
    flags = [flag for row in records for flag in row["all_inventory_flags"]]
    selected = [flag for flag in flags if flag["code"] in SELECTED]
    claims = [flag for flag in selected if flag["code"] == "past_search_claim_without_tool_evidence"]
    return {"records_accounted": len(records), "english_original_joins": sum(r["english_join_exact"] for r in records),
            "assistant_turns_inspected": sum(r["assistant_turns_inspected"] for r in records),
            "inventory_flags_preserved": len(flags), "selected_inventory_flags_inspected": len(selected),
            "selected_flags_by_code": dict(sorted(Counter(f["code"] for f in selected).items())),
            "records_with_selected_flags": len({f["id"] for f in selected}),
            "classifications": dict(sorted(Counter(r["finding"]["classification"] for r in queue).items())),
            "review_queue_findings": len(queue), "review_queue_records": len({r["record_id"] for r in queue}),
            "agent_reviewed_turns": len({(r["id"], r["turn"]) for r in audit["agent_reviews"]}),
            "agent_reviewed_records": len({r["id"] for r in audit["agent_reviews"]}),
            "agent_reviewed_findings": sum(r["finding"]["agent_reviewed"] for r in queue),
            "automatic_only_findings": sum(not r["finding"]["agent_reviewed"] for r in queue),
            "candidate_correction_plans_pending_review": sum(bool(r.get("candidate_correction_plan")) for r in audit["agent_reviews"]),
            "automatically_semantically_certified_records": 0,
            "past_search_flag_views": len(claims), "past_search_unique_turns": len({(f["id"], f["turn"]) for f in claims}),
            "past_search_unique_records": len({f["id"] for f in claims}),
            "exact_adjacent_duplicate_pairs": sum(len(t["pairs"]) for r in records for t in r["adjacent_duplicate_inspection"]),
            "candidate_corrections": len(audit["candidate_corrections"]), "original_records_modified": 0,
            "blind_translations": 0, "observed_eval_prompts_imported": 0, "gpu_calls": 0, "api_calls": 0,
            "root_cause_limits": ["Lexical normalization and section-insensitive matching produce false positives.",
                                  "Repeated code, lyrics, templates and shared attributes need structural preservation.",
                                  "Faithful English-to-Twi transfer can preserve unsupported source claims.",
                                  "Corpus findings do not prove causal attribution for any model behavior.",
                                  "Automatic triage is not a native-language or whole-corpus factual certification."], **PENDING}


def _json_bytes(value):
    return (canonical(value) + "\n").encode()


def _rows_bytes(values):
    return b"".join(_json_bytes(value) for value in values)


def build_package(records, flags, joined, inputs, agent_reviews=None, deletion_proofs=None):
    if agent_reviews is None:
        agent_reviews = materialize_agent_reviews(records, inputs["histories"]["sha256"])
    audit = audit_records(records, flags, joined,
                          agent_reviews=agent_reviews,
                          deletion_proofs=APPROVED_DELETIONS if deletion_proofs is None else deletion_proofs)
    by_id = {r["id"]: r for r in audit["records"]}
    proof = [{"review": review, "original_ref": by_id[review["id"]]["english_original_ref"],
              "twi_target": by_id[review["id"]]["original_record"]["messages"][review["turn"]]["content"],
              "english_source": by_id[review["id"]]["original_record"]["english_messages"][review["turn"]]["content"],
              "finding_ids": [f["id"] for f in by_id[review["id"]]["findings"] if f["turn"] == review["turn"]]}
             for review in audit["agent_reviews"]]
    files = {"record-audit.jsonl": _rows_bytes(audit["records"]),
             "review-queue.jsonl": _rows_bytes(audit["review_queue"]),
             "candidate-corrections.jsonl": _rows_bytes(audit["candidate_corrections"]),
             "correction-registry.jsonl": _rows_bytes(audit["correction_registry"]),
             "agent-reviews.jsonl": _rows_bytes(audit["agent_reviews"]),
             "root-cause-evidence.jsonl": _rows_bytes(proof),
             "reviewed-correction-plans.jsonl": _rows_bytes(r for r in proof if r["review"]["candidate_correction_plan"]),
             "summary.json": _json_bytes(summary(audit))}
    manifest = {"version": VERSION, "scope": "whole_r6_response_target_audit_not_training_admission",
                "inputs": inputs, "source_registry": "record-audit.jsonl", "correction_registry": "correction-registry.jsonl",
                "artifacts": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
                "code_sha256": {"scripts/audit_r7_response_targets.py": sha(__file__),
                                "scripts/prepare_r7_source_inventory.py": sha(inventory.__file__)},
                "originals_preserved": True, "automatic_edits_enabled": False, **PENDING}
    files["manifest.json"] = _json_bytes(manifest)
    files["SEALED.json"] = _json_bytes({"manifest_sha256": hashlib.sha256(files["manifest.json"]).hexdigest(),
                                       "audit_only": True, "training_ready": False})
    return files


def prepare(out=DEFAULT_OUT, **input_options):
    out = Path(out)
    require(not out.exists() and not out.is_symlink(), "Use a fresh audit directory; refusing overwrite")
    loaded = load_inputs(**input_options)
    files = build_package(*loaded)
    out.mkdir(parents=True, mode=0o700)
    for name, content in files.items():
        with (out / name).open("xb") as stream:
            stream.write(content)
    return verify_package(out, **input_options)


def verify_package(out=DEFAULT_OUT, **input_options):
    out = Path(out)
    require(out.is_dir() and not out.is_symlink() and not any(p.is_symlink() for p in out.rglob("*")), "Linked audit artifact")
    expected = build_package(*load_inputs(**input_options))
    require({p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == set(expected), "Unregistered audit artifact")
    for name, content in expected.items():
        require((out / name).read_bytes() == content, "Audit replay differs: " + name)
    return {"verified": True, "audit_only": True, "summary": json.loads(expected["summary.json"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    action = verify_package if args.verify else prepare
    print(json.dumps(action(args.out, folder=args.inventory), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
