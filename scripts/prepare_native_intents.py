"""Preserve native INJONGO utterances and labels for auxiliary understanding SFT."""
import hashlib
import argparse
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/native-intent-v2"
REVISION = "1f8be590da6699aee3dc23de6f63e801e2352eff"
BASE = f"https://raw.githubusercontent.com/masakhane-io/masakhane-nlu/{REVISION}"


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def entities(row):
    # The upstream names say byte, but all five inspected splits use Python
    # character offsets. Require exact agreement with the published target.
    text, result, previous = row["text"], [], 0
    for span in row["spans"]:
        start, end = span["start_byte"], span["limit_byte"]
        if not isinstance(start, int) or not isinstance(end, int) or not previous <= start < end <= len(text):
            raise ValueError("Invalid or overlapping source span")
        result.append({"type": span["label"], "text": text[start:end]})
        previous = end
    if " $$ ".join(f"{r['type']}: {r['text']}" for r in result) != row["target"]:
        raise ValueError("Source offsets and entity target disagree")
    return result


def annotation_instruction(intents, entity_types=None):
    if entity_types is None:
        return ("Annotate this request, without carrying it out. Return a JSON object with intent and entities. "
                "Copy entity text exactly from the request; do not invent missing details. "
                "Intent choices: " + ", ".join(intents) + ".")
    return (
        'Annotate this request, without carrying it out. Return only JSON, without markdown or prose, '
        'in this exact structure: {"intent":"one intent choice","entities":[{"type":"one entity type","text":"exact request span"}]}. '
        'Return entities: [] when none are present. Copy each entity text exactly from the request; '
        'do not translate it or invent missing details. Intent choices: ' + ', '.join(intents) + '. '
        'Entity types: ' + ', '.join(entity_types) + '.')


def convert(row, language, split, intents, entity_types=None):
    if row["intent"] not in intents or not row["text"].strip():
        raise ValueError("Missing utterance or unknown intent")
    target = {"intent": row["intent"], "entities": entities(row)}
    source_id = f"injongo:{language}:{row['intent']}:{row['example_id']}"
    return {"id": source_id + ":understanding", "source_id": source_id,
            "source": "masakhane-io/masakhane-nlu/InjongoIntent", "revision": REVISION,
            "language": language, "split": split, "task": "native_intent_entities",
            "source_record_hash": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "source_evidence": row, "source_license": "apache-2.0", "project_human_reviewed": False,
            "messages": [{"role": "system", "content": annotation_instruction(intents, entity_types)},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--schema-v2", action="store_true")
    args = parser.parse_args()
    out = args.output
    if (out / "manifest.json").exists():
        raise ValueError("Prepared source already exists; do not overwrite")
    raw_dir = out / "sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, grouped = [], {}
    for language, source_split in (("twi", "train"), ("twi", "dev"), ("twi", "test"), ("eng", "train"), ("eng", "test")):
        relative = f"InjongoIntent/{language}/{source_split}.jsonl"
        raw = urllib.request.urlopen(BASE + "/" + relative, timeout=60).read()
        filename = f"{language}-{source_split}.jsonl"
        path = raw_dir / filename
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Pinned source changed")
        path.write_bytes(raw)
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        grouped[("tw" if language == "twi" else "en", "validation" if source_split == "dev" else source_split)] = rows
        sources.append({"url": BASE + "/" + relative, "file": "sources/" + filename, "rows": len(rows),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    intents = sorted({r["intent"] for (lang, split), rows in grouped.items() if split == "train" for r in rows})
    if len(intents) != 40:
        raise ValueError("Unexpected source intent inventory")
    entity_types = sorted({span["label"] for (lang, split), rows in grouped.items() if split == "train"
                           for row in rows for span in row["spans"]}) if args.schema_v2 else None
    heldout = {normalize(row["text"]) for (lang, split), rows in grouped.items() if split != "train" for row in rows}
    counts, seen, outputs, rejected = Counter(), set(), {"train": [], "validation": [], "test": []}, []
    for (language, split), rows in grouped.items():
        for row in rows:
            key = (language, split, normalize(row["text"]))
            if key in seen or (split == "train" and key[-1] in heldout):
                rejected.append({"language": language, "split": split, "id": row["example_id"], "reason": "duplicate_or_reserved_overlap"})
                continue
            try:
                converted = convert(row, language, split, intents, entity_types)
            except ValueError as exc:
                rejected.append({"language": language, "split": split, "id": row["example_id"], "reason": str(exc)})
                continue
            outputs[split].append(converted)
            seen.add(key)
            counts[f"{split}:{language}"] += 1
    artifacts = []
    for split, rows in outputs.items():
        if len(rows) != len({row["id"] for row in rows}):
            raise ValueError("Source IDs are not unique within their intent namespace")
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (out / f"{split}.jsonl").write_bytes(raw)
        artifacts.append({"file": f"{split}.jsonl", "rows": len(rows), "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"experiment": "native_intent_source_v3" if args.schema_v2 else "native_intent_source_v2",
                "prompt_version": "schema-v2" if args.schema_v2 else "legacy", "entity_types": entity_types,
                "revision": REVISION, "license": "apache-2.0",
                "sources": sources, "artifacts": artifacts, "counts": dict(counts), "intents": intents,
                "exclusions": rejected, "training_started": False, "synthetic_responses_added": 0,
                "limitations": ["Auxiliary intent/entity supervision, not direct response or clinical supervision.",
                    "Source-created native-speaker requests and labels, not project-certified gold.",
                    "Intent labels do not fully specify negation, action authorization, missing details or multi-turn context.",
                    "Entity offsets retain original source fields; extraction uses character offsets validated against source targets.",
                    "Official test texts inspected for split isolation only, not used for training or checkpoint selection."]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"folder": str(out), "counts": counts, "excluded": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
