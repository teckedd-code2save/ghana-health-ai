"""Measure full public Twi text sources without generating labels or accepting gold.

Run with --download to cache pinned public Pristine shards. Existing parallel
sources and speech manifests are read in place. No audio or private chat inputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/response-adaptation/twi-text-audit-sources.v1.json"
BASE = ROOT / "tmp/general-language-corpus"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def text_flags(text, limits):
    flags = []
    if not isinstance(text, str) or not text.strip():
        return ["empty_or_nontext"]
    words = normalized(text).split()
    if len(words) < limits["minimum_words_per_candidate"]:
        flags.append("short_fragment")
    if len(text) > limits["max_characters_per_candidate"]:
        flags.append("overlength_review")
    if any(x in text for x in ("\ufffd", "<|", "</", "http://", "https://")):
        flags.append("encoding_or_markup_review")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text):
        flags.append("control_characters")
    if words and sum(c.isalpha() for c in text) / max(1, len(text)) < 0.45:
        flags.append("low_letter_fraction")
    grams = [tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))]
    if grams and 1 - len(set(grams)) / len(grams) > limits["repeated_eight_word_fraction"]:
        flags.append("within_text_repetition")
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        flags.append("contact_information_review")
    return flags


def pair_flags(english, twi):
    """Alignment warnings never change the independently assessed Twi text."""
    if not english:
        return ["missing_english"]
    flags = []
    if normalized(english) == normalized(twi):
        flags.append("identical_languages")
    if re.findall(r"\d+(?:[.,]\d+)*", english) != re.findall(r"\d+(?:[.,]\d+)*", twi):
        flags.append("numeric_alignment_review")
    if max(len(english), len(twi)) / max(1, min(len(english), len(twi))) > 6:
        flags.append("length_alignment_review")
    return flags


class Sampler:
    def __init__(self, limit=6):
        self.limit = limit
        self.groups = defaultdict(list)

    def add(self, key, row):
        priority = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16)
        heap = self.groups[key]
        item = (-priority, row["id"], row)
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def rows(self):
        return [entry[2] for key in sorted(self.groups) for entry in sorted(self.groups[key], reverse=True)]


def sources(registry, download):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    audit = json.loads((BASE / "public-source-audit.json").read_text())
    for source in audit:
        if source["name"] not in ("ghana_nlp_parallel", "community_parallel"):
            continue
        for file, expected in source["hashes"].items():
            if sha(BASE / source["name"] / Path(file).name) != expected:
                raise ValueError("Existing source hash mismatch: " + source["name"])
        path = BASE / source["name"] / Path(source["file"]).name
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = pq.read_table(path).to_pylist()
        for index, row in enumerate(rows):
            yield {"id": source["name"] + ":" + str(row.get("id", index)),
                   "source": source["name"], "repo": source["repo"], "revision": source["revision"],
                   "origin": "compiled_parallel_text", "terms": source["terms"],
                   "file": str(path.relative_to(ROOT)), "file_sha256": source["hashes"][source["file"]],
                   "row_index": index, "upstream_split": "unspecified",
                   "genre": row.get("source_files", "professional_parallel"),
                   "text": row.get("label", row.get("twi")) or "",
                   "english": row.get("text", row.get("eng")) or ""}

    path = ROOT / "data/annotated-source-corpus/sources.v1.jsonl"
    file_hash = sha(path)
    for index, row in enumerate(read_jsonl(path)):
        if row["source"] not in ("waxal", "ghana_nlp_speech") or row["language"] != "tw":
            continue
        yield {"id": row["id"], "source": row["source"], "repo": row["source_url"],
               "revision": row["source_revision"], "origin": "speech_transcript", "terms": row["license"],
               "file": str(path.relative_to(ROOT)), "file_sha256": file_hash, "row_index": index,
               "upstream_split": row["split"], "genre": "speech_transcript", "text": row["utterance"],
               "speaker_id": row.get("speaker_id")}

    for source in registry["sources"]:
        folder = ROOT / "tmp/twi-pretraining-sources" / source["name"]
        for file in ("README.md", *source["files"]):
            target = folder / file
            if not target.exists():
                if not download:
                    raise FileNotFoundError(f"Missing {target}; pass --download for pinned public sources")
                print("Fetching public source " + source["name"] + "/" + file, flush=True)
                hf_hub_download(source["repo"], file, repo_type="dataset", revision=source["revision"],
                                local_dir=folder, token=False)
            if file == "README.md":
                continue
            expected = source["files"][file]
            if sha(target) != expected:
                raise ValueError("Downloaded source hash mismatch: " + file)
            index = 0
            for batch in pq.ParquetFile(target).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    yield {"id": f"{source['name']}:{Path(file).stem}:{index}", "source": source["name"],
                           "repo": source["repo"], "revision": source["revision"], "origin": source["origin"],
                           "terms": source["license"], "file": str(target.relative_to(ROOT)),
                           "file_sha256": expected, "row_index": index, "upstream_split": "train",
                           "genre": row["style"], "text": row["twi"]}
                    index += 1


def protected_texts():
    from build_language_adaptation import protected_texts as previous_protected, content_for_overlap
    texts = previous_protected()
    for row in read_jsonl(BASE / "adaptation-v1/validation.jsonl"):
        texts.extend(content_for_overlap(row))
    for split in ("validation", "test"):
        for row in read_jsonl(ROOT / f"tmp/native-intent-v3/{split}.jsonl"):
            texts.extend(m["content"] for m in row["messages"] if m["role"] == "user")
    for language in ("twi", "eng"):
        with (ROOT / f"tmp/semantic-response-v4/afrixnli/{language}-dev.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                texts.extend((row["premise"], row["hypothesis"]))
    return texts


def run(output, download):
    from build_language_adaptation import OverlapIndex
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    if output.exists():
        raise ValueError("Choose a new output folder; prior audits are immutable")
    registry = json.loads(REGISTRY.read_text())
    tokenizers, tokenizer_evidence = {}, []
    for spec in registry["tokenizers"]:
        path = hf_hub_download(spec["repo"], "tokenizer.json", revision=spec["revision"], token=False)
        tokenizers[spec["name"]] = Tokenizer.from_file(path)
        tokenizer_evidence.append({**spec, "sha256": sha(path)})
    protected = protected_texts()
    blocked = OverlapIndex(protected)
    output.mkdir(parents=True)
    sampler, stats, inputs = Sampler(), defaultdict(Counter), {}
    db = sqlite3.connect(output / "dedup.sqlite3")
    db.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY, id TEXT NOT NULL)")
    pending = []

    def count_tokens():
        if not pending:
            return
        for name, tokenizer in tokenizers.items():
            lengths = [len(x.ids) for x in tokenizer.encode_batch([r["text"] for r in pending], add_special_tokens=False)]
            for row, length in zip(pending, lengths):
                key = row["source"] + ":" + row["disposition"]
                stats[key][name + "_tokens"] += length
                stats[key][name + "_rows_above_2048_tokens"] += int(length > 2048)
        pending.clear()

    try:
        with gzip.open(output / "index.jsonl.gz", "wt", encoding="utf-8") as ledger:
            for number, original in enumerate(sources(registry, download), 1):
                row = dict(original)
                text = row["text"]
                row["flags"] = text_flags(text, registry["limits"])
                row["alignment_flags"] = pair_flags(row["english"], text) if "english" in row else []
                key = hashlib.sha256(normalized(text).encode()).hexdigest()
                row["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                row["normalized_sha256"] = key
                row["disposition"] = "structural_candidate"
                if row["flags"]:
                    row["disposition"] = "text_review"
                elif row["upstream_split"] in ("test", "validation") or blocked.contains(text):
                    row["disposition"] = "protected_evaluation"
                else:
                    existing = db.execute("SELECT id FROM seen WHERE hash=?", (key,)).fetchone()
                    if existing:
                        row["disposition"] = "duplicate"
                        row["duplicate_of"] = existing[0]
                    else:
                        db.execute("INSERT INTO seen VALUES (?, ?)", (key, row["id"]))
                row["training_eligible"] = False
                inputs[row["file"]] = row["file_sha256"]
                source_stats = stats[row["source"]]
                source_stats["rows"] += 1
                source_stats[row["disposition"]] += 1
                source_stats["characters"] += len(text)
                source_stats["words"] += len(normalized(text).split())
                for flag in row["flags"]:
                    source_stats["flag:" + flag] += 1
                for flag in row["alignment_flags"]:
                    source_stats["alignment:" + flag] += 1
                source_stats["genre:" + row["genre"]] += 1
                if row["disposition"] == "structural_candidate":
                    if row["alignment_flags"]:
                        source_stats["text_candidates_with_alignment_warnings"] += 1
                    stats[row["source"] + ":structural_candidate"]["words"] += len(normalized(text).split())
                    pending.append(row)
                    if len(pending) >= 256:
                        count_tokens()
                sample_key = (row["source"], row["genre"], row["disposition"])
                sampler.add(sample_key, row)
                ledger.write(json.dumps({k: v for k, v in row.items() if k not in ("text", "english", "terms")}, ensure_ascii=False) + "\n")
                if number % 10000 == 0:
                    db.commit()
                    print(json.dumps({"processed": number, "source": row["source"], "unique_candidates": db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]}), flush=True)
        count_tokens()
        db.commit()
    finally:
        db.close()
    with (output / "review-samples.jsonl").open("w") as handle:
        for row in sampler.rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"stage": "completed_audit_not_training_export", "registry_sha256": sha(REGISTRY),
              "sources": registry, "input_hashes": inputs, "tokenizers": tokenizer_evidence,
              "statistics": dict(stats), "protected_text_count": len(protected),
              "protected_text_sha256": hashlib.sha256(json.dumps(sorted(set(protected)), ensure_ascii=False).encode()).hexdigest(),
              "artifacts": {p.name: sha(p) for p in (output / "index.jsonl.gz", output / "review-samples.jsonl")},
              "limitations": ["No model annotation, audio, private chats, training, or upload.",
                  "Structural candidates are not language-verified, fact-verified or approved training rows.",
                  "Token counts use native tokenizer.json without chat wrappers or EOS tokens.",
                  "Exact normalized dedup only; near-duplicate/topic grouping still required before splitting.",
                  "Existing protected evaluation overlap checked; not proof of zero semantic contamination.",
                  "WAXAL/GhanaNLP speech coverage is the existing text manifest, not the complete upstream audio corpus.",
                  "Dictionary, religious, educational and synthetic subdomains must be mixture-balanced; row count is not dialogue count."]}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": str(output), "statistics": dict(stats)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/twi-pretraining-audit/v1")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.output, args.download)
