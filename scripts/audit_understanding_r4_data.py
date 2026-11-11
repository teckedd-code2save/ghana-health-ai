"""Audit sealed R3 supervision without relabeling sources or granting admission."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def sentence_repeats(text):
    parts = [normalized(s) for s in re.split(r"[.!?\n]+", text)]
    counts = Counter(s for s in parts if len(s.split()) >= 5)
    return {s: n for s, n in counts.items() if n > 1}


def audit(pack):
    pack = Path(pack)
    manifest = json.loads((pack / "manifest.json").read_text())
    seal = json.loads((pack / "PREPARED.json").read_text())
    if seal != {"manifest_sha256": sha(pack / "manifest.json")}:
        raise ValueError("R3 pack seal changed")
    for name, checksum in manifest["artifacts"].items():
        path = (pack / name).resolve()
        if not path.is_relative_to(pack.resolve()) or sha(path) != checksum:
            raise ValueError("Changed or escaping source artifact: " + name)
    records_path = pack / "datasets/responses/records.jsonl"
    records = [json.loads(line) for line in records_path.open()]
    witnesses = [json.loads(line) for line in (pack / "source-witnesses.jsonl").open()]
    views = [w for w in witnesses if w["collection"] == "responses"]
    targets, histories, prompt_targets = [], [], defaultdict(list)
    for record in records:
        messages = record["messages"]
        seen = {}
        english = record.get("english_messages", [])
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            text, flags = message["content"], []
            norm = normalized(text)
            prompt = messages[index - 1]["content"]
            repeats = sentence_repeats(text)
            if repeats:
                flags.append("repeated_sentence_inspect_in_context")
            if norm in seen and normalized(prompt) != normalized(messages[seen[norm] - 1]["content"]):
                flags.append("same_answer_to_changed_user_turn")
            if norm == normalized(prompt):
                flags.append("verbatim_user_echo")
            if re.search(r"\b(OpenAI|ChatGPT)\b", text, re.I):
                flags.append("identity_or_attribution_reference_inspect")
            if len(norm.split()) <= 6:
                flags.append("short_target_inspect_not_automatic_rejection")
            previous = seen.get(norm)
            seen[norm] = index
            target = {"record_id": record["id"], "source_id": record["source_id"],
                "record_sha256": digest(record), "assistant_index": index,
                "history": messages[:index + 1], "english_history": english[:index + 1],
                "origin": record.get("origin"), "review": record.get("review", {}),
                "word_count": len(norm.split()),
                "is_followup": any(m["role"] == "assistant" for m in messages[:index]),
                "flags": flags, "repeated_sentences": repeats,
                "previous_same_answer_index": previous, "source_ref": record.get("original_ref")}
            targets.append(target)
            prompt_targets[normalized(prompt)].append({"record_id": record["id"],
                "assistant_index": index, "answer_hash": digest(norm)})
        histories.append({"record_id": record["id"], "source_id": record["source_id"],
            "source_group": record.get("group_id", record.get("metadata", {}).get("group_id")),
            "assistant_turns": sum(m["role"] == "assistant" for m in messages),
            "origin": record.get("origin"), "record_sha256": digest(record)})
    collisions = [{"normalized_prompt": prompt, "occurrences": values}
                  for prompt, values in prompt_targets.items()
                  if len({v["answer_hash"] for v in values}) > 1]
    distinct_targets = {(digest(t["history"]), t["assistant_index"]): t for t in targets}
    lengths = sorted(t["word_count"] for t in distinct_targets.values())
    result = {"schema": "r4-supervision-audit-v1", "r3_pack_sha256": sha(pack / "manifest.json"),
        "source_path": str(records_path.resolve()), "source_sha256": sha(records_path),
        "response_records": len(records), "actual_training_views": len(views),
        "actual_followup_training_views": sum(sum(m["role"] == "assistant" for m in w["messages"]) > 1 for w in views),
        "history_assistant_occurrences": len(targets), "distinct_history_prefixes": len(distinct_targets),
        "multi_turn_records": sum(h["assistant_turns"] > 1 for h in histories),
        "origin_counts": dict(Counter(r.get("origin", "unspecified") for r in records)),
        "flag_counts_occurrences": dict(Counter(f for t in targets for f in t["flags"])),
        "distinct_prefix_word_lengths": {"min": min(lengths), "median": statistics.median(lengths),
                                         "p90": lengths[int((len(lengths) - 1) * .9)], "max": max(lengths)},
        "prompts_with_multiple_targets": len(collisions),
        "counts_note": "Occurrences in full source histories include shared prefixes; actual views come from the sealed pack.",
        "limitations": ["Flags are inspection cues, not semantic scores or automatic exclusions.",
            "Multiple valid responses to one prompt need not be contradictory.",
            "A correctly encoded target can still be unhelpful or mistranslated.",
            "No source, original review, training flag or model weight has been changed."],
        "training_executed": False, "new_admissions": 0}
    return result, targets, histories, collisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report, targets, histories, collisions = audit(args.pack)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", report), ("prompt-targets.json", collisions)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, values in (("targets.jsonl", targets), ("histories.jsonl", histories)):
        (args.out / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))
    evidence = {"script_sha256": sha(__file__), "input_manifest_sha256": report["r3_pack_sha256"],
        "artifacts": {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}}
    (args.out / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
