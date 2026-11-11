"""Offline proposal validation; no private examples, inference, or file writes.

Run without arguments for synthetic unit tests. Explicit --corpus and --prior
paths produce an IDs/hashes/counts-only JSON report on stdout. Keep that report
private too. Structural/overlap checks are not semantic or native certification.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import unittest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def byte_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(_):
    raise ValueError("Non-finite JSON number")


def read_json(path):
    require(path.is_file() and not path.is_symlink(), "Use a regular input file")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Invalid JSON input; contents withheld") from error


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    # Fold the spelling perturbation only next to letters, not the number 3.
    text = re.sub(r"(?<=[^\W\d_])3|3(?=[^\W\d_])", "\u025b", text)
    return " ".join(re.findall(r"[^\W_]+", text))


def nonempty_strings(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value)


def variants(group):
    for language, stimulus in group["inputs"].items():
        yield language, stimulus, False
    if group.get("informal3", False):
        stimulus = deepcopy(group["inputs"]["tw"])
        changes = 0
        if group["mode"] == "fixed_context":
            for message in stimulus["messages"]:
                if message["role"] == "user":
                    changes += message["content"].count("\u025b")
                    message["content"] = message["content"].replace("\u025b", "3")
        else:
            changes = sum(turn.count("\u025b") for turn in stimulus["user_turns"])
            stimulus["user_turns"] = [turn.replace("\u025b", "3")
                                      for turn in stimulus["user_turns"]]
        require(changes > 0, "Spelling variant has no eligible character")
        yield "tw-i3", stimulus, True


def validate(corpus):
    require(isinstance(corpus, dict), "Expected corpus object")
    for key in ("training_eligible", "execution_authorized", "native_certification",
                "clinical_certification"):
        require(corpus.get(key) is False, "Proposal authorization boundary changed")
    require(corpus.get("private") is True and corpus.get("origin") == "project_authored",
            "Expected private project-authored proposal")
    require(type(corpus.get("model_calls")) is int and corpus["model_calls"] == 0
            and corpus.get("results", "missing") is None, "Proposal contains execution")
    groups = corpus.get("groups")
    require(isinstance(groups, list) and groups, "No source groups")
    seen_ids, seen_sources, seen_stimuli, cases = set(), set(), {}, []
    for group in groups:
        require(isinstance(group, dict), "Invalid source group")
        group_id, source = group.get("id"), group.get("source_group_id")
        require(isinstance(group_id, str) and re.fullmatch(r"[a-z0-9-]+", group_id),
                "Invalid group ID")
        require(isinstance(source, str) and source and source not in seen_sources,
                "Duplicate or missing source identity; merge connected scenarios")
        require(group_id not in seen_ids, "Duplicate group ID")
        seen_ids.add(group_id)
        seen_sources.add(source)
        require(group.get("origin") == "project_authored"
                and group.get("split") == "heldout_proposal", "Source split/origin drift")
        require(nonempty_strings(group.get("phenomena")), "Missing phenomenon coverage")
        require(isinstance(group.get("review_notes"), str) and group["review_notes"].strip(),
                "Missing review uncertainty notes")
        require(type(group.get("informal3", False)) is bool, "Invalid spelling flag")
        mode = group.get("mode")
        require(mode in {"fixed_context", "own_history_rollout"}, "Invalid history mode")
        inputs = group.get("inputs")
        require(isinstance(inputs, dict) and set(inputs) == {"en", "tw"},
                "A source group needs an English/Twi pair")
        slots = []
        for stimulus in inputs.values():
            require(isinstance(stimulus, dict), "Invalid stimulus")
            if mode == "fixed_context":
                require(set(stimulus) == {"messages", "output_languages"},
                        "Unexpected fixed-context fields")
                messages = stimulus["messages"]
                require(isinstance(messages, list) and messages and len(messages) % 2 == 1,
                        "Fixed history must end on a user turn")
                for index, message in enumerate(messages):
                    require(isinstance(message, dict) and set(message) == {"role", "content"},
                            "Invalid message fields")
                    require(message["role"] == ("user" if index % 2 == 0 else "assistant")
                            and isinstance(message["content"], str) and message["content"].strip(),
                            "Invalid alternating history")
                slots.append(1)
            else:
                require(set(stimulus) == {"user_turns", "output_languages"},
                        "Rollouts must contain user turns only, never teacher-forced replies")
                require(nonempty_strings(stimulus["user_turns"])
                        and len(stimulus["user_turns"]) >= 2, "Invalid rollout")
                slots.append(len(stimulus["user_turns"]))
            require(isinstance(stimulus["output_languages"], list)
                    and len(stimulus["output_languages"]) == slots[-1]
                    and all(lang in {"en", "tw", "mixed"} for lang in stimulus["output_languages"]),
                    "Each score slot requires an explicit output language")
        require(slots[0] == slots[1], "Bilingual rollout length differs")
        if mode == "fixed_context":
            require(len(inputs["en"]["messages"]) == len(inputs["tw"]["messages"]),
                    "Bilingual fixed history length differs")
        reviews = group.get("reviews")
        require(isinstance(reviews, list) and len(reviews) == slots[0],
                "Every score slot needs its own review")
        for index, review in enumerate(reviews, 1):
            require(isinstance(review, dict) and type(review.get("slot")) is int
                    and review["slot"] == index, "Review slots are not contiguous")
            facts, claims, rubric = review.get("facts"), review.get("expected_claims"), review.get("rubric")
            require(isinstance(facts, dict) and facts and all(
                isinstance(key, str) and isinstance(value, str) and value.strip()
                for key, value in facts.items()), "Missing review evidence")
            require(isinstance(claims, list) and claims, "Missing structured claims")
            for claim in claims:
                require(isinstance(claim, dict) and all(
                    isinstance(claim.get(key), str) and claim[key].strip()
                    for key in ("subject", "predicate")) and "value" in claim,
                    "Invalid structured claim")
                require(nonempty_strings(claim.get("support"))
                        and set(claim["support"]) <= set(facts), "Unsupported claim reference")
                canonical(claim["value"])
            require(isinstance(rubric, dict) and nonempty_strings(rubric.get("must"))
                    and nonempty_strings(rubric.get("must_not")), "Incomplete explicit rubric")
        for suffix, stimulus, spelling in variants(group):
            case_id = group_id + "-" + suffix
            payload = {"mode": mode, **stimulus}
            user_texts = stimulus.get("user_turns") or [
                row["content"] for row in stimulus["messages"] if row["role"] == "user"]
            normalized = digest([normalize(text) for text in user_texts])
            require(seen_stimuli.get(normalized, source) == source,
                    "Duplicate normalized stimulus across source groups")
            seen_stimuli[normalized] = source
            cases.append({"id": case_id, "source_group_id": source,
                          "origin": group["origin"], "split": group["split"],
                          "training_eligible": False, "mode": mode,
                          "language": "tw" if spelling else suffix,
                          "synthetic_informal3": spelling, "stimulus": stimulus,
                          "reviews": reviews, "phenomena": group["phenomena"],
                          "input_sha256": digest(payload),
                          "normalized_user_input_sha256": normalized,
                          "case_sha256": digest({"source_group_id": source,
                                                "input": payload, "reviews": reviews})})
    return cases


def text_entries(value, location="root"):
    """Extract input/target text for overlap, never returning it in reports."""
    if isinstance(value, dict):
        for key, item in value.items():
            child = location + "/" + key
            if key in {"content", "text", "prompt", "response", "answer", "completion"} and isinstance(item, str):
                yield child, item
            elif key in {"user_turns", "rubric"} and isinstance(item, list):
                for index, text in enumerate(item):
                    if isinstance(text, str):
                        yield child + "/" + str(index), text
            else:
                yield from text_entries(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from text_entries(item, location + "/" + str(index))


def shingles(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def overlap(cases, prior_entries):
    prepared = []
    for ref, text in prior_entries:
        norm = normalize(text)
        prepared.append((ref, text, norm, shingles(norm.split())))
    exact, normalized, candidates = [], [], []
    for case in cases:
        for position, text in text_entries(case["stimulus"]):
            norm = normalize(text)
            tokens = norm.split()
            chunks = shingles(tokens)
            for ref, old, old_norm, old_chunks in prepared:
                match = {"case_id": case["id"], "input_position": position,
                         "prior_ref": ref, "input_text_sha256": digest(text),
                         "prior_text_sha256": digest(old)}
                if text == old:
                    exact.append(match)
                elif norm and norm == old_norm:
                    normalized.append(match)
                elif len(tokens) >= 12 and len(old_norm.split()) >= 12:
                    shared = len(chunks & old_chunks)
                    union = len(chunks | old_chunks)
                    jaccard = shared / union if union else 0
                    containment = shared / min(len(chunks), len(old_chunks))
                    if jaccard >= 0.30 or containment >= 0.80:
                        candidates.append({**match, "trigram_jaccard": round(jaccard, 4),
                                           "trigram_containment": round(containment, 4)})
    return {"prior_text_entries": len(prepared), "exact_matches": exact,
            "normalized_matches": normalized, "near_overlap_candidates": candidates,
            "semantic_independence_certified": False,
            "policy": "NFKC/case/whitespace/punctuation and letter-adjacent 3 folding; near screen uses >=12-token inputs, trigram Jaccard >=0.30 or containment >=0.80. Review candidates, do not auto-clear semantic overlap."}


def inspect_prior(root):
    require(root.is_dir() and not root.is_symlink(), "Prior input must be a regular directory")
    entries, files, missing = [], {}, []
    for path in sorted(root.iterdir()):
        if path.suffix != ".json":
            continue
        before = byte_hash(path)
        value = read_json(path)
        require(byte_hash(path) == before, "Prior input changed during read")
        files[path.name] = before
        entries.extend((path.name + ":" + ref, text) for ref, text in text_entries(value))
        if path.name == "manifest.json" and isinstance(value, dict):
            for name, checksum in value.get("artifacts", {}).items():
                relative = Path(name)
                require(not relative.is_absolute() and ".." not in relative.parts,
                        "Unsafe manifest path")
                artifact = root / relative
                if not artifact.exists():
                    missing.append(name)
                else:
                    require(not artifact.is_symlink() and byte_hash(artifact) == checksum,
                            "Prior manifest artifact hash mismatch")
    require(entries, "No prior text available for overlap audit")
    require(all(byte_hash(root / name) == checksum for name, checksum in files.items()),
            "Prior input changed during audit")
    return entries, {"files_sha256": files, "missing_manifest_artifacts": sorted(missing),
                     "complete_manifest_coverage": not missing}


def summary(corpus, cases):
    counts = Counter(case["mode"] for case in cases)
    slot_count = sum(len(case["reviews"]) for case in cases)
    return {"version": "private-proposal-audit-v1", "private": True,
            "execution_authorized": False, "model_calls": 0,
            "counts": {"source_groups": len(corpus["groups"]), "cases": len(cases),
                       "fixed_context_cases": counts["fixed_context"],
                       "rollout_conversations": counts["own_history_rollout"],
                       "rollout_user_turns": sum(len(case["reviews"]) for case in cases
                                                 if case["mode"] == "own_history_rollout"),
                       "score_slots_per_model": slot_count,
                       "synthetic_informal3_cases": sum(case["synthetic_informal3"] for case in cases),
                       "fixed_multiturn_cases": sum(case["mode"] == "fixed_context"
                                                    and len(case["stimulus"]["messages"]) > 1
                                                    for case in cases)},
            "language_case_counts": dict(Counter(case["language"] for case in cases)),
            "phenomenon_group_counts": dict(Counter(tag for group in corpus["groups"]
                                                    for tag in set(group["phenomena"]))),
            "inventory_sha256": digest([{key: case[key] for key in
                                         ("id", "source_group_id", "case_sha256")}
                                        for case in cases]),
            "groups": [{"id": group["id"], "source_group_id": group["source_group_id"],
                        "split": group["split"], "group_sha256": digest(group)}
                       for group in corpus["groups"]],
            "cases": [{key: case[key] for key in
                       ("id", "source_group_id", "origin", "split", "training_eligible",
                        "mode", "language", "synthetic_informal3", "input_sha256",
                        "normalized_user_input_sha256", "case_sha256")} for case in cases]}


def fixture():
    # Opaque stimuli test the schema only, not bilingual or semantic performance.
    return {"private": True, "origin": "project_authored", "training_eligible": False,
            "execution_authorized": False, "native_certification": False,
            "clinical_certification": False, "model_calls": 0, "results": None,
            "groups": [{"id": "fixture", "source_group_id": "synthetic:fixture",
                        "origin": "project_authored", "split": "heldout_proposal",
                        "mode": "fixed_context", "phenomena": ["schema_only"],
                        "review_notes": "Not a semantic test.",
                        "inputs": {lang: {"messages": [{"role": "user", "content": text}],
                                          "output_languages": [lang]}
                                   for lang, text in (("en", "X?"), ("tw", "Z?"))},
                        "reviews": [{"slot": 1, "facts": {"f": "Opaque fixture."},
                                     "expected_claims": [{"subject": "fixture", "predicate": "value",
                                                          "value": None, "support": ["f"]}],
                                     "rubric": {"must": ["Schema only."],
                                                "must_not": ["Treat as a quality test."]}}]}]}


class FreshEvaluationSchemaTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.group = self.data["groups"][0]

    def test_pair_inherits_origin_split_and_review(self):
        cases = validate(self.data)
        self.assertEqual(len(cases), 2)
        self.assertEqual(len({case["source_group_id"] for case in cases}), 1)
        self.assertTrue(all(case["reviews"] == self.group["reviews"] for case in cases))
        self.assertTrue(all(case["training_eligible"] is False for case in cases))

    def test_execution_and_certification_fail_closed(self):
        for field in ("training_eligible", "execution_authorized", "native_certification",
                      "clinical_certification", "model_calls", "results"):
            with self.subTest(field=field):
                changed = deepcopy(self.data)
                changed[field] = True
                with self.assertRaises(ValueError):
                    validate(changed)

    def test_development_cannot_be_relabelled_by_validator(self):
        self.group["split"] = "development_not_blind_final"
        with self.assertRaisesRegex(ValueError, "split/origin"):
            validate(self.data)

    def test_duplicate_source_or_group_rejected(self):
        self.data["groups"].append(deepcopy(self.group))
        with self.assertRaises(ValueError):
            validate(self.data)

    def test_same_text_in_different_group_rejected(self):
        other = deepcopy(self.group)
        other.update(id="another", source_group_id="synthetic:another")
        self.data["groups"].append(other)
        with self.assertRaisesRegex(ValueError, "normalized stimulus"):
            validate(self.data)

    def test_missing_pair_or_language_rejected(self):
        del self.group["inputs"]["tw"]
        with self.assertRaisesRegex(ValueError, "pair"):
            validate(self.data)
        self.group["inputs"]["tw"] = deepcopy(self.group["inputs"]["en"])
        self.group["inputs"]["tw"]["output_languages"] = []
        with self.assertRaisesRegex(ValueError, "output language"):
            validate(self.data)

    def test_evidence_and_rubric_required(self):
        review = self.group["reviews"][0]
        review["expected_claims"][0]["support"] = ["absent"]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate(self.data)
        review["expected_claims"][0]["support"] = ["f"]
        review["rubric"]["must_not"] = []
        with self.assertRaisesRegex(ValueError, "rubric"):
            validate(self.data)

    def test_fixed_context_cannot_end_on_assistant(self):
        self.group["inputs"]["en"]["messages"].append({"role": "assistant", "content": "Y."})
        with self.assertRaisesRegex(ValueError, "user turn"):
            validate(self.data)

    def rollout(self):
        self.group["mode"] = "own_history_rollout"
        self.group["inputs"] = {lang: {"user_turns": [lang + "?", lang + "!"],
                                      "output_languages": [lang, lang]} for lang in ("en", "tw")}
        second = deepcopy(self.group["reviews"][0])
        second["slot"] = 2
        self.group["reviews"].append(second)

    def test_rollout_requires_per_turn_reviews_and_equal_lengths(self):
        self.rollout()
        self.assertEqual(len(validate(self.data)[0]["reviews"]), 2)
        self.group["reviews"].pop()
        with self.assertRaisesRegex(ValueError, "own review"):
            validate(self.data)

    def test_rollout_rejects_teacher_forcing(self):
        self.rollout()
        self.group["inputs"]["en"]["messages"] = [{"role": "assistant", "content": "Y."}]
        with self.assertRaisesRegex(ValueError, "teacher-forced"):
            validate(self.data)

    def test_spelling_changes_user_only_and_does_not_add_source(self):
        self.group["informal3"] = True
        for lang in ("en", "tw"):
            self.group["inputs"][lang]["messages"] = [
                {"role": "user", "content": lang + " \u025b?"},
                {"role": "assistant", "content": "\u025b."},
                {"role": "user", "content": lang + " q\u025b 3"}]
        cases = validate(self.data)
        parent, child = cases[1], cases[2]
        self.assertEqual(child["stimulus"]["messages"][1], parent["stimulus"]["messages"][1])
        self.assertEqual(child["stimulus"]["messages"][2]["content"], "tw q3 3")
        self.assertEqual(child["source_group_id"], parent["source_group_id"])
        self.assertEqual(len(self.data["groups"]), 1)

    def test_spelling_noop_rejected(self):
        self.group["informal3"] = True
        with self.assertRaisesRegex(ValueError, "eligible character"):
            validate(self.data)

    def test_normalization_preserves_numbers(self):
        self.assertEqual(normalize("Q3, 3 30 3q"), "q\u025b 3 30 \u025bq")
        self.assertEqual(normalize(" Q\u025b! "), "q\u025b")

    def test_report_never_contains_stimuli_or_rubrics(self):
        marker = "PRIVATE_SYNTHETIC_SENTINEL"
        self.group["inputs"]["en"]["messages"][0]["content"] = marker
        self.group["reviews"][0]["facts"]["f"] = marker
        cases = validate(self.data)
        report = summary(self.data, cases)
        report["overlap"] = overlap(cases, [("fixture", marker)])
        self.assertNotIn(marker, json.dumps(report))
        self.assertEqual(len(report["overlap"]["exact_matches"]), 1)

    def test_normalized_and_near_overlap_detected(self):
        cases = validate(self.data)
        self.assertEqual(len(overlap(cases, [("old", " x! ")])["normalized_matches"]), 1)
        text = "one two three four five six seven eight nine ten eleven twelve thirteen"
        cases[0]["stimulus"]["messages"][0]["content"] = text
        result = overlap(cases, [("old", text + " fourteen")])
        self.assertEqual(len(result["near_overlap_candidates"]), 1)

    def test_claim_edit_changes_case_seal_not_input_hash(self):
        before = validate(self.data)[0]
        self.group["reviews"][0]["expected_claims"][0]["value"] = False
        after = validate(self.data)[0]
        self.assertEqual(before["input_sha256"], after["input_sha256"])
        self.assertNotEqual(before["case_sha256"], after["case_sha256"])

    def test_duplicate_keys_and_nonfinite_numbers_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"x":1,"x":2}', object_pairs_hook=unique_object)
        with self.assertRaises(ValueError):
            json.loads('{"x":NaN}', parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    if args.corpus is None:
        require(args.prior is None, "--prior requires --corpus")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FreshEvaluationSchemaTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    before = byte_hash(args.corpus)
    corpus = read_json(args.corpus)
    cases = validate(corpus)
    report = summary(corpus, cases)
    report["corpus_file_sha256"] = before
    report["validator_file_sha256"] = byte_hash(Path(__file__))
    if args.prior:
        entries, inventory = inspect_prior(args.prior)
        report["prior_inventory"] = inventory
        report["overlap"] = overlap(cases, entries)
    else:
        report["overlap"] = {"status": "not_checked"}
    require(byte_hash(args.corpus) == before, "Corpus changed during audit")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 2 if (report["overlap"].get("exact_matches")
                 or report["overlap"].get("normalized_matches")
                 or report["overlap"].get("near_overlap_candidates")) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, TypeError, KeyError) as error:
        # No parser snippets, input values, or model-visible content on failure.
        print("Proposal audit failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(1)
