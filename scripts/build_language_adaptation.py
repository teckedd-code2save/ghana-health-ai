"""Stage-one language adaptation, not an accepted medical-response corpus.

Keep original public source pairs, isolate duplicate groups before splitting,
and replay reviewed English conversations. Never synthesize answers for speech.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/general-language-corpus"
OUT = BASE / "adaptation-v1"


def canonical(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def quality_reason(english: str, twi: str) -> str | None:
    if not english.strip() or not twi.strip():
        return "empty"
    if canonical(english) == canonical(twi):
        return "identical_languages"
    if max(len(english), len(twi)) > 1600:
        return "long_pair"
    if min(len(canonical(english)), len(canonical(twi))) < 2:
        return "fragment"
    if max(len(english), len(twi)) / max(1, min(len(english), len(twi))) > 6:
        return "length_alignment"
    if any(x in english + twi for x in ("\ufffd", "<|", "</", "http://", "https://")):
        return "markup_or_encoding"
    if re.search(r"\b(?:pl\.|inf\.|pr\.\s*\d|s\.\s+\w)", english + " " + twi):
        return "dictionary_notation"
    if re.findall(r"\d+(?:[.,]\d+)*", english) != re.findall(r"\d+(?:[.,]\d+)*", twi):
        return "numeric_alignment_review"
    return None


class OverlapIndex:
    def __init__(self, texts: list[str]):
        self.exact = {canonical(t) for t in texts if canonical(t)}
        self.grams = set()
        for t in self.exact:
            words = t.split()
            self.grams.update(tuple(words[i:i + 8]) for i in range(len(words) - 7))

    def contains(self, text: str) -> bool:
        value = canonical(text)
        if value in self.exact:
            return True
        words = value.split()
        return any(tuple(words[i:i + 8]) in self.grams for i in range(len(words) - 7))


def group_pairs(rows: list[dict]) -> list[dict]:
    """Shared English, Twi, or source block stays in one connected component."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("en:" + canonical(row["en"]), "tw:" + canonical(row["tw"]), "source:" + row["block"]):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row["id"])
    for i, row in enumerate(rows):
        row["group"] = digest(sorted(groups[root(i)]))
        row["split"] = "validation" if int(row["group"][:8], 16) % 10 == 0 else "train"
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def content_for_overlap(row: dict) -> list[str]:
    if row["task"] == "translation":
        return [row["messages"][0]["content"].split("\n\n", 1)[1], row["messages"][-1]["content"]]
    return [m["content"] for m in row["messages"] if m["role"] != "system"]


def protected_texts() -> list[str]:
    texts = []
    for row in read_jsonl(ROOT / "data/medical-response-corpus/afrihealth-ghana-response-eval.v1.jsonl"):
        texts.extend([row["question"], row["answer"]])
    for row in read_jsonl(ROOT / "data/annotated-source-corpus/sources.v1.jsonl"):
        if row["split"] != "train":
            texts.append(row["utterance"])
    for row in read_jsonl(ROOT / "data/medical-response-corpus/response-product-eval.v1.jsonl"):
        texts.extend(m["content"] for m in row["messages"])
    # Saved independent foundation checks also remain excluded from training.
    for path in (ROOT / "tmp/general-foundation-comparison").glob("cases-*.json"):
        obj = json.loads(path.read_text())
        for row in obj:
            texts.extend(m["content"] for m in row.get("messages", []) if m["role"] != "system")
    regression = ROOT / "data/response-adaptation/greeting-regressions.v1.json"
    if regression.exists():
        # Protect the tested query, not generic Hello/Hello history or rubrics.
        for row in json.loads(regression.read_text())["cases"]:
            final = row["messages"][-1]
            if final["role"] != "user":
                raise ValueError("Regression query must end with the user")
            texts.append(final["content"])
    return texts


def english_replay(folder: Path, split: str, limit: int, blocked: OverlapIndex) -> list[dict]:
    rows = pq.read_table(next(folder.glob("*.parquet"))).to_pylist()
    by_id = {r["message_id"]: r for r in rows}
    chosen = {}
    for row in sorted(rows, key=lambda r: digest(r["message_id"])):
        if row["role"] != "assistant" or row["rank"] != 0:
            continue
        chain = [row]
        while chain[0]["parent_id"] in by_id and len(chain) <= 6:
            chain.insert(0, by_id[chain[0]["parent_id"]])
        if chain[0]["parent_id"] is not None or len(chain) > 6:
            continue
        if any(r["lang"] != "en" or r["deleted"] or r["synthetic"] or r["review_result"] is not True for r in chain):
            continue
        if any((r.get("detoxify") or {}).get("toxicity", 0) > 0.1 for r in chain):
            continue
        labels = [dict(zip((r.get("labels") or {}).get("name", []), (r.get("labels") or {}).get("value", []))) for r in chain]
        if labels[-1].get("quality", 0) < 0.6 or any(any(label.get(k, 0) > 0 for k in ("pii", "spam", "not_appropriate", "lang_mismatch")) for label in labels):
            continue
        if any(blocked.contains(r["text"]) for r in chain):
            continue
        if any(len(r["text"]) > 2200 or "<|" in r["text"] for r in chain) or sum(len(r["text"]) for r in chain) > 4200:
            continue
        if any(r["role"] != ("prompter" if i % 2 == 0 else "assistant") for i, r in enumerate(chain)):
            continue
        tree = row["message_tree_id"]
        if tree in chosen:
            continue
        chosen[tree] = {"id": "oasst:" + row["message_id"], "group": "oasst:" + tree,
                        "task": "general_conversation_replay", "language": "en", "split": split,
                        "messages": [{"role": "user" if r["role"] == "prompter" else "assistant", "content": r["text"]} for r in chain],
                        "source": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
                        "source_record_hash": digest(chain), "upstream_reviewed": True,
                        "native_reviewed_by_project": False}
        if len(chosen) >= limit:
            break
    return list(chosen.values())


def main() -> None:
    audit = json.loads((BASE / "public-source-audit.json").read_text())
    sources = {r["name"]: r for r in audit}
    for name, source in sources.items():
        for file, sha in source["hashes"].items():
            if hashlib.sha256((BASE / name / Path(file).name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Source hash mismatch: {name}/{file}")
    blocked = OverlapIndex(protected_texts())
    pairs, excluded, seen = [], [], set()
    for name in ("ghana_nlp_parallel", "community_parallel"):
        source = sources[name]
        path = BASE / name / Path(source["file"]).name
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig") as handle:
                raw = list(csv.DictReader(handle))
        else:
            raw = pq.read_table(path).to_pylist()
        for index, original in enumerate(raw):
            en = str(original.get("text", original.get("eng")) or "").strip()
            tw = str(original.get("label", original.get("twi")) or "").strip()
            source_id = str(original.get("id", index))
            source_file = original.get("source_files", "GhanaNLP professional translation")
            row = {"id": f"{name}:{source_id}", "en": en, "tw": tw,
                   "source": source["repo"], "revision": source["revision"],
                   "source_file": source_file, "source_record_id": source_id,
                   "source_record_hash": digest(original), "original": original,
                   "block": f"{name}:{int(source_id) // 100}" if name == "ghana_nlp_parallel" else f"{name}:{digest(canonical(en))}",
                   "native_reviewed_by_project": False}
            reason = quality_reason(en, tw)
            if name == "community_parallel" and not any(s in source_file.split("; ") for s in ("14_Twi-Language-Guide_parallel", "15_twi-sample_parallel", "16_Twi-census-Glossary_parallel")):
                reason = "out_of_stage_one_source_scope"
            if name == "community_parallel" and "01_medical_glossary" in source_file:
                reason = "medical_glossary_semantic_ambiguity_review"
            if blocked.contains(en) or blocked.contains(tw):
                reason = "locked_evaluation_overlap"
            key = (canonical(en), canonical(tw))
            if key in seen:
                reason = "duplicate_pair"
            seen.add(key)
            if reason:
                excluded.append({"id": row["id"], "source_record_hash": row["source_record_hash"], "reason": reason})
            else:
                pairs.append(row)
    group_pairs(pairs)
    outputs = {"train": [], "validation": []}
    for row in pairs:
        for source_lang, target_lang in (("tw", "en"), ("en", "tw")):
            names = {"en": "English", "tw": "Twi (Akan)"}
            item = {k: v for k, v in row.items() if k not in ("en", "tw", "original", "block")}
            item.update({"id": row["id"] + ":" + target_lang, "task": "translation", "language": target_lang,
                         "messages": [{"role": "user", "content": f"Translate from {names[source_lang]} to {names[target_lang]}. Return only the translation.\n\n{row[source_lang]}"},
                                      {"role": "assistant", "content": row[target_lang]}]})
            outputs[row["split"]].append(item)
    for split, limit in (("train", 1000), ("validation", 128)):
        outputs[split] += english_replay(BASE / ("oasst_" + split), split, limit, blocked)
    # Protect against duplicate conversations crossing the official tree split.
    evaluation_index = OverlapIndex([text for row in outputs["validation"] for text in content_for_overlap(row)])
    retained = []
    for row in outputs["train"]:
        if any(evaluation_index.contains(text) for text in content_for_overlap(row)):
            excluded.append({"id": row["id"], "source_record_hash": row["source_record_hash"], "reason": "new_validation_text_overlap"})
        else:
            retained.append(row)
    outputs["train"] = retained
    if {r["group"] for r in outputs["train"]} & {r["group"] for r in outputs["validation"]}:
        raise ValueError("Training/validation group leakage")
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, rows in {**outputs, "source_pairs": pairs, "excluded": excluded}.items():
        path = OUT / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        artifacts.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)})
    manifest = {"experiment": "general_language_adaptation_v1", "base_model": "google/gemma-4-31B-it",
                "base_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475", "sources": [{k: v for k, v in r.items() if k not in ("sample", "source_distribution")} for r in audit],
                "artifacts": artifacts, "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
                "splits": {s: {"examples": len(rs), "distinct_sources": len({r["source_record_hash"] for r in rs}), "distinct_groups": len({r["group"] for r in rs}), "tasks": dict(Counter(r["task"] + ":" + r["language"] for r in rs))} for s, rs in outputs.items()},
                "ready_for_production": False, "experimental_only": True,
                "limitations": ["Stage-one language alignment and English replay, not full response/tool/health SFT.", "Published translations can contain errors; project native-speaker review not completed.", "Research-only source terms retained; no commercial or public redistribution clearance inferred.", "Medical glossary quarantined: includes symptom-to-disease conflation and contradictory translation variants.", "Numeric mismatches quarantined, including legitimate digit-to-word translations.", "Original document IDs unavailable for GhanaNLP: adjacent ID blocks and shared text grouped; document independence not guaranteed.", "Both translation directions are two views of one source pair, not independent records.", "Validation groups and protected existing evaluation texts excluded from training."]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
