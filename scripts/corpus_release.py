"""Immutable, reference-first corpus releases. No model calls or training.

Each source row is accounted for before views are exported. Text is never
rewritten; normalized text is used only for deduplication and split protection.
"""
from __future__ import annotations

import argparse
import array
import functools
import gzip
import hashlib
import json
import re
import sqlite3
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from audit_twi_pretraining import ROOT, REGISTRY, Sampler, normalized, pair_flags, protected_texts, sha, sources, text_flags

VERSION = "stage-corpus-v1"
SPEECH = ROOT / "tmp/twi-pretraining-sources/full-public-speech-v2/sources.jsonl"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def envelope(original, path, index, source, revision, text, language, kind, **extra):
    return dict(id=f"{source}:{revision}:{index}", source=source, revision=revision,
                file=str(path.relative_to(ROOT)), file_sha256=sha_cached(str(path)), row_index=index,
                original_sha256=digest(original), text=text, language=language, record_type=kind,
                origin="source_preserved", upstream_split="train", topic="unclassified",
                human_reviewed=False, **extra)


sha_cached = functools.lru_cache(maxsize=None)(sha)


def release_holdouts(output=None):
    protected = protected_texts()
    path = ROOT / "tmp/balanced-afrique-v6/corpus/validation.jsonl"
    protected.extend(t for r in jsonl(path) for t in row_texts({"original_messages": r["messages"]}))
    snapshot = {"text_hashes": sorted(digest(normalized(t)) for t in protected),
                "count": len(protected), "policy": "exact or shared eight-word spans; all related records inherit protection"}
    if output is not None:
        path = output / "protected.json"
        if path.exists() and json.loads(path.read_text()) != snapshot:
            raise ValueError("Frozen holdouts changed; do not resume with different evaluation data")
        if not path.exists(): save(path, snapshot)
    return protected


class DiskNearIndex:
    """Restartable bounded-memory version of NearIndex; no corpus-size RAM index."""
    def __init__(self, db):
        self.db = db
        db.execute("CREATE TABLE IF NOT EXISTS near_anchors (anchor INTEGER,n INTEGER,PRIMARY KEY(anchor,n)) WITHOUT ROWID")
        db.execute("CREATE TABLE IF NOT EXISTS near_sketches (n INTEGER PRIMARY KEY, sketch BLOB)")

    @functools.lru_cache(maxsize=2048)
    def sketch(self, n):
        result = self.db.execute("SELECT sketch FROM near_sketches WHERE n=?", (n,)).fetchone()
        values = array.array("I")
        values.frombytes(result[0])
        return frozenset(values)

    def add(self, n, text):
        grams = shingle_hashes(text)
        if not grams:
            return [], False
        sketch = frozenset(sorted(grams)[:64])
        anchors = sorted(grams)[:8]
        candidates, overflow, available = set(), False, []
        for anchor in anchors:
            bucket = [r[0] for r in self.db.execute("SELECT n FROM near_anchors WHERE anchor=? LIMIT 64", (anchor,))]
            overflow |= len(bucket) == 64
            candidates.update(bucket)
            if len(bucket) < 64:
                available.append(anchor)
        if len(candidates) > 256:
            overflow = True
        matches = []
        for other_n in sorted(candidates)[:256]:
            other = self.sketch(other_n)
            if len(sketch & other) / len(sketch | other) >= .82:
                matches.append(other_n)
        self.db.execute("INSERT INTO near_sketches VALUES(?,?)", (n, array.array("I", sorted(sketch)).tobytes()))
        self.db.executemany("INSERT INTO near_anchors VALUES(?,?)", ((anchor, n) for anchor in available))
        return matches, overflow


def recover(output):
    """Resume ingestion and dedup checks without discarding completed source work."""
    from build_language_adaptation import OverlapIndex
    if (output / "manifest.json").exists():
        verify(output)
        return
    if (output / "collections").exists():
        archive = output.with_name(output.name + ".interrupted-export-" + str(time.time_ns()))
        archive.mkdir()
        for name in ("collections", "annotation", "review", "reports", "training", "exclusions.jsonl"):
            path = output / name
            if path.exists(): path.rename(archive / name)
        print(json.dumps({"preserved_partial_export": str(archive)}), flush=True)
    db = sqlite3.connect(output / "ledger.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA cache_size=-65536")
    saved = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if db.execute("SELECT COALESCE(MAX(n),-1)+1 FROM records").fetchone()[0] != saved:
        raise ValueError("Non-contiguous source checkpoint")
    known = {(r["source"], r["revision"], str(r["source_record_id"])) for r in jsonl(ROOT / "data/response-adaptation/source-review-findings.jsonl")}
    blocked = OverlapIndex(release_holdouts(output))
    old_files = {}
    # File hashes validate complete saved source shards; boundary ID detects a
    # changed source order without re-running completed normalization checks.
    for n, payload in db.execute("SELECT n,payload FROM records ORDER BY n"):
        row = json.loads(zlib.decompress(payload))
        old_files[row["file"]] = row["file_sha256"]
        if row.get("companion_file"):
            old_files[row["companion_file"]] = row["companion_sha256"]
        # Recheck the same frozen holdouts for saved and newly appended rows.
        if row["upstream_split"] in ("test", "validation") or any(blocked.contains(t) for t in row_texts(row)):
            db.execute("UPDATE records SET locked=1 WHERE n=?", (n,))
    for path, expected in old_files.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Changed frozen source: " + path)
    db.commit()
    last_id = db.execute("SELECT id FROM records ORDER BY n DESC LIMIT 1").fetchone()[0] if saved else None
    print(json.dumps({"phase": "recover_sources", "saved_rows": saved}), flush=True)
    for n, row in enumerate(all_sources()):
        if n < saved:
            if n == saved - 1 and row["id"] != last_id:
                raise ValueError("Source order changed at recovery boundary")
            continue
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        row["record_hash"] = digest(row)
        locked = int(row["upstream_split"] in ("test", "validation") or any(blocked.contains(t) for t in row_texts(row)))
        db.execute("INSERT INTO records(n,id,payload,flags,locked) VALUES(?,?,?,?,?)", (n, row["id"],
            zlib.compress(json.dumps(row, ensure_ascii=False).encode(), 1), json.dumps(flags_for(row, known)), locked))
        if n % 1000 == 0:
            db.commit()
    db.commit()
    total = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    db.execute("CREATE TABLE IF NOT EXISTS edges(n INTEGER,other INTEGER,PRIMARY KEY(n,other)) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS aliases(key TEXT PRIMARY KEY,n INTEGER) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS checks(n INTEGER PRIMARY KEY,duplicate INTEGER,near INTEGER,overflow INTEGER)")
    groups = Groups()
    for _ in range(total): groups.add()
    for a, b in db.execute("SELECT n,other FROM edges"):
        groups.join(a, b)
    done = db.execute("SELECT COALESCE(MAX(n),-1) FROM checks").fetchone()[0]
    near = DiskNearIndex(db)
    started = time.monotonic()
    print(json.dumps({"phase": "recover_dedup", "total": total, "checks_saved": done + 1}), flush=True)
    for n, payload in db.execute("SELECT n,payload FROM records WHERE n>? ORDER BY n", (done,)):
        row = json.loads(zlib.decompress(payload))
        content_key = digest([row["record_type"], row["text"], row.get("english"), row.get("reference_answer"), row.get("original_messages"), row.get("structured")])
        keys = ["record:" + content_key]
        keys.extend("text:" + digest(normalized(row[k])) for k in ("text", "english") if row.get(k))
        keys.extend("text:" + digest(normalized(t)) for t in row_texts(row) if len(normalized(t).split()) >= 8)
        if row.get("source_group"):
            keys.append("group:" + row["source_group"])
        duplicate = False
        for key in set(keys):
            existing = db.execute("SELECT n FROM aliases WHERE key=?", (key,)).fetchone()
            if existing:
                groups.join(n, existing[0])
                db.execute("INSERT OR IGNORE INTO edges VALUES(?,?)", (n, existing[0]))
                duplicate |= key == keys[0]
            else:
                db.execute("INSERT INTO aliases VALUES(?,?)", (key, n))
        matches, overflow = near.add(n, row["text"])
        for other in matches:
            groups.join(n, other)
            db.execute("INSERT OR IGNORE INTO edges VALUES(?,?)", (n, other))
            db.execute("UPDATE checks SET near=1 WHERE n=?", (other,))
        db.execute("INSERT INTO checks VALUES(?,?,?,?)", (n, int(duplicate), int(bool(matches)), int(overflow)))
        if n % 5000 == 0:
            db.commit()
            print(json.dumps({"phase": "dedup", "rows": n + 1, "total": total,
                              "seconds_this_attempt": round(time.monotonic() - started)}), flush=True)
    db.commit()
    locked_roots = {groups.root(n) for n, in db.execute("SELECT n FROM records WHERE locked=1")}
    files, counts = {}, Counter()
    for n, payload in db.execute("SELECT n,payload FROM records ORDER BY n"):
        row = json.loads(zlib.decompress(payload))
        files[row["file"]] = row["file_sha256"]
        if row.get("companion_file"): files[row["companion_file"]] = row["companion_sha256"]
        counts[row["source"]] += 1
        root = groups.root(n)
        split = "protected" if root in locked_roots else "validation" if int(digest(root)[:8], 16) % 50 == 0 else "train"
        db.execute("UPDATE records SET root=?,split=? WHERE n=?", (root, split, n))
    db.commit()
    duplicate = {r[0] for r in db.execute("SELECT n FROM checks WHERE duplicate=1")}
    near_rows = {r[0] for r in db.execute("SELECT n FROM checks WHERE near=1")}
    overflow_rows = {r[0] for r in db.execute("SELECT n FROM checks WHERE overflow=1")}
    save(output / "inventory.json", {"version": VERSION, "input_files": files, "source_rows": dict(counts),
         "rows": total, "inventory_complete": True, "resumed_saved_rows": saved,
         "near_duplicates": {"method": "bottom-8 five-word anchors, bottom-64 similarity >= .82; bounded candidate overflow quarantined",
                             "review_rows": len(near_rows), "overflow_rows": len(overflow_rows), "paraphrase_recall_measured": False}})
    export(output, db, groups, duplicate, near_rows, overflow_rows)
    db.close()


def all_sources():
    from build_language_adaptation import OverlapIndex, english_replay
    registry = json.loads(REGISTRY.read_text())
    # Reuse the complete pinned iterator, replacing the earlier speech subset.
    for original in sources(registry, False):
        if original["origin"] == "speech_transcript":
            continue
        row = dict(original)
        genre = row["genre"].lower()
        kind = "document" if row["origin"].startswith("synthetic") else "sentence"
        if any(word in genre for word in ("dictionary", "glossary", "mono")):
            kind = "dictionary_entry"
        row.update(language="tw", record_type=kind, original_sha256=digest(original),
                   topic="unclassified", human_reviewed=False)
        # Related Bible versions share a partition; there is no reliable verse ID.
        if "bible" in genre or "youversion" in genre:
            row["source_group"] = "community:bible-versions"
            row["topic"] = "religion"
        yield row
    for i, r in enumerate(jsonl(SPEECH)):
        if hashlib.sha256(r["text"].encode()).hexdigest() != r["text_sha256"]:
            raise ValueError("Changed transcript " + r["id"])
        item = envelope(r, SPEECH, i, r["source"], r["revision"], r["text"], r["language"],
                        "utterance", terms=r["license"], repo=r["repo"], speaker_id=r.get("speaker_id"))
        item.update(id=r["id"], source_record_id=r["source_record_id"], upstream_split=r["split"], origin="speech_transcript")
        if r["source"].startswith("waxal"):
            item["source_group"] = "waxal:stimulus:" + r["source_record_id"].split("_u", 1)[0]
        yield item
    for split in ("train", "validation", "test"):
        path = ROOT / f"tmp/native-intent-v3/{split}.jsonl"
        for i, r in enumerate(jsonl(path)):
            evidence = r["source_evidence"]
            target = json.loads(r["messages"][-1]["content"])
            item = envelope(r, path, i, "injongo", r["revision"], evidence["text"], r["language"],
                            "intent_example", terms=r["source_license"], repo=r["source"],
                            structured=target, original_messages=r["messages"],
                            source_group="injongo:" + evidence["intent"] + ":" + evidence["example_id"])
            item.update(id=r["id"], upstream_split=split)
            yield item
    for split in ("train", "eval", "review"):
        path = ROOT / f"data/medical-response-corpus/afrihealth-ghana-response-{split}.v1.jsonl"
        for i, r in enumerate(jsonl(path)):
            item = envelope(r, path, i, "afrihealth", r["source_revision"], r["question"], r["language"],
                            "health_qa", terms=r["license"], repo=r["source_dataset"],
                            reference_answer=r["answer"], original_messages=r["messages"],
                            source_record_id=r["source_record_id"], previous_quality=r["quality_annotation"],
                            source_group="afrihealth:" + r["source_record_id"].split("_")[-1])
            item.update(id=r["id"], upstream_split="test" if split == "eval" else "train", topic="health")
            yield item
    base = ROOT / "tmp/semantic-response-v4/sources"
    from prepare_grounded_twi_response import valid_span, REVISION
    for split in ("train", "dev", "test"):
        path = base / f"gold_span_passages.afriqa.twi.en.{split}.json"
        query_path = base / f"queries.afriqa.twi.en.{split}.json"
        queries = {str(r["id"]): r for r in jsonl(query_path)}
        for i, r in enumerate(jsonl(path)):
            query = queries[str(r["id"])]
            okay = (all(isinstance(r.get(k), str) and r[k].strip() for k in ("title", "context", "question_lang", "question_translated", "answer_lang"))
                    and valid_span(r) and query["translation_type"] == "human_translation"
                    and query["question"] == r["question_lang"] and query["translated_question"] == r["question_translated"])
            item = envelope(r, path, i, "afriqa", REVISION, r["question_lang"] if isinstance(r["question_lang"], str) else "", "tw", "grounded_qa",
                            terms="cc-by-sa-4.0", repo="masakhane-io/afriqa", reference_answer=r["answer_lang"] if isinstance(r["answer_lang"], str) else None,
                            english=r["question_translated"] if isinstance(r["question_translated"], str) else None,
                            context=r["context"] if isinstance(r["context"], str) else None, grounded_span_valid=okay,
                            invalid_source_fields=[k for k in ("question_lang", "question_translated", "context", "title", "answer_lang") if not isinstance(r.get(k), str) or not r[k].strip()],
                            source_group="afriqa:title:" + normalized(r["title"]) if isinstance(r.get("title"), str) and r["title"].strip() else f"afriqa:unresolved:{split}:{r['id']}",
                            companion_file=str(query_path.relative_to(ROOT)), companion_sha256=sha_cached(str(query_path)))
            item.update(upstream_split="validation" if split == "dev" else split)
            if split != "train":
                item["id"] = f"afriqa:{REVISION}:{split}:{i}"
            yield item
    audit = json.loads((ROOT / "tmp/general-language-corpus/public-source-audit.json").read_text())
    for split in ("train", "validation"):
        folder = ROOT / f"tmp/general-language-corpus/oasst_{split}"
        path = next(folder.glob("*.parquet"))
        # The full underlying tree inventory is frozen separately. Only intact,
        # upstream-reviewed English paths pass the existing source quality gate.
        selected = english_replay(folder, split, 10**9, OverlapIndex([]))
        for i, r in enumerate(selected):
            item = envelope(r, path, i, "oasst", r["revision"], r["messages"][0]["content"], "en",
                            "conversation", terms="apache-2.0", repo=r["source"], original_messages=r["messages"],
                            source_group=r["group"], origin_detail="upstream_reviewed_not_project_reviewed")
            item.update(id=r["id"], upstream_split=split)
            yield item


class Groups:
    def __init__(self):
        self.parent = array.array("I")

    def add(self):
        i = len(self.parent)
        self.parent.append(i)
        return i

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def shingle_hashes(text):
    words = normalized(text).split()
    if len(words) < 12:
        return frozenset()
    return frozenset(zlib.crc32(" ".join(words[i:i + 5]).encode()) for i in range(len(words) - 4))


class NearIndex:
    """Bottom-8 shingle candidate search with bottom-64 Jaccard review routing.

    Approximate lexical recall, not paraphrase certification. Common-anchor
    overflow is explicitly flagged instead of silently calling text unique.
    """
    def __init__(self):
        self.anchors = {}
        self.signatures = {}

    def add(self, number, text):
        grams = shingle_hashes(text)
        if not grams:
            return [], False
        anchors = sorted(grams)[:8]
        candidates, overflow = set(), False
        for anchor in anchors:
            bucket = self.anchors.get(anchor, [])
            overflow |= len(bucket) >= 64
            candidates.update(bucket)
        matches = []
        sketch = frozenset(sorted(grams)[:64])
        for candidate in candidates:
            other = frozenset(self.signatures[candidate])
            # A compact bottom-64 sketch estimates similarity for routing to
            # review; it does not certify equivalence or remove a source row.
            score = len(sketch & other) / len(sketch | other)
            if score >= .82:
                matches.append(candidate)
        self.signatures[number] = array.array("I", sorted(sketch))
        for anchor in anchors:
            bucket = self.anchors.setdefault(anchor, array.array("I"))
            if len(bucket) < 64:
                bucket.append(number)
        return matches, overflow


def row_texts(row):
    if row.get("original_messages"):
        return [m["content"] for m in row["original_messages"] if m["role"] != "system"]
    return [row[k] for k in ("text", "english", "reference_answer", "context") if isinstance(row.get(k), str) and row[k]]


def flags_for(row, known):
    limits = {"minimum_words_per_candidate": 1, "max_characters_per_candidate": 20000, "repeated_eight_word_fraction": .3}
    flags = text_flags(row["text"], limits)
    if row.get("invalid_source_fields"):
        flags.append("missing_or_nontext_source_fields")
    if row["language"] not in ("tw", "en"):
        flags.append("dialect_not_resolved")
    if any(c in row["text"] for c in ("Ↄ", "ℇ", "ᴐ", "ε")):
        flags.append("orthographic_variant_review")
    if row["record_type"] == "dictionary_entry" and re.search(r"\b(pl\.|pr\.|inf\.)|ŋ", row["text"]):
        flags.append("dictionary_notation_review")
    if row["source"] == "community_parallel":
        if "Asante-Fante" in row.get("genre", ""):
            flags.append("dialect_not_resolved")
        if "medical_glossary" in row.get("genre", ""):
            flags.append("known_source_semantic_risk")
    source_id = row.get("source_record_id", row["id"].split(":")[-1])
    if (row.get("repo"), row["revision"], str(source_id)) in known:
        flags.append("prior_semantic_review_finding")
    return sorted(set(flags))


def views_for(row, flags):
    views = []
    kind = row["record_type"]
    if not flags and not row["origin"].startswith("synthetic") and kind in ("sentence", "document", "utterance", "dictionary_entry"):
        name = "lexicon" if kind == "dictionary_entry" else "short_utterances" if len(normalized(row["text"]).split()) < 8 else "text"
        if row["language"] == "en":
            name = "english_replay"
        views.append(("language/" + name, {"text": row["text"]}))
    if not flags and row.get("english") and not pair_flags(row["english"], row["text"]):
        views.append(("meaning/alignment", {"twi": row["text"], "english": row["english"], "reference_preserved": True}))
    if not flags and kind == "intent_example":
        target = row["structured"]
        if all(entity["text"] in row["text"] for entity in target["entities"]):
            views.append(("meaning/structured", {"text": row["text"], "understanding": target,
                          "messages": row["original_messages"], "meaning_english": None,
                          "missing_fields": ["meaning_english"] if row["language"] != "en" else []}))
    if not flags and kind == "conversation":
        views.append(("conversation/english_retention", {"messages": row["original_messages"]}))
    if not flags and kind == "grounded_qa" and row["grounded_span_valid"]:
        views.append(("conversation/grounded_twi_qa", {"messages": [
            {"role": "system", "content": "Answer in Twi using only the supplied source passage. Do not claim the passage is current."},
            {"role": "user", "content": "Source passage:\n" + row["context"] + "\n\nQuestion:\n" + row["text"]},
            {"role": "assistant", "content": row["reference_answer"]}], "context": row["context"],
            "reference_answer": row["reference_answer"], "upstream_human_translation": True}))
    # AfriHealth structural acceptance alone is not medical/semantic approval.
    return views


def build(output):
    if output.exists():
        raise ValueError("Release directories are immutable; choose a new --out")
    output.mkdir(parents=True)
    release_holdouts(output)
    db = sqlite3.connect(output / "ledger.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE records (n INTEGER PRIMARY KEY, id TEXT UNIQUE, payload BLOB, flags TEXT, locked INTEGER, root INTEGER, split TEXT, disposition TEXT)")
    db.commit()
    db.close()
    recover(output)


def export(output, db, groups, duplicate, near_rows, overflow_rows):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    spec = json.loads(REGISTRY.read_text())["tokenizers"][0]
    tokenizer_path = hf_hub_download(spec["repo"], "tokenizer.json", revision=spec["revision"], local_files_only=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    streams, stats = {}, defaultdict(Counter)
    sampler, review_count, disposition_counts = Sampler(limit=4), Counter(), Counter()
    known = {(r["source"], r["revision"], str(r["source_record_id"])) for r in jsonl(ROOT / "data/response-adaptation/source-review-findings.jsonl")}
    work = (output / "annotation"); work.mkdir()
    plan_stream = (work / "pending.jsonl").open("w")
    exclusions = (output / "exclusions.jsonl").open("w")
    corrections = (output / "review"); corrections.mkdir()
    queue = gzip.open(corrections / "queue.jsonl.gz", "wt", encoding="utf-8")

    def write(view, split, item):
        key = view + "/" + split
        if key not in streams:
            path = output / "collections" / (key + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[key] = path.open("w")
        texts = [m["content"] for m in item["messages"]] if "messages" in item else [str(item[k]) for k in ("text", "twi", "english") if item.get(k)]
        tokens = sum(len(tokenizer.encode(t, add_special_tokens=False).ids) for t in texts)
        item["tokens"] = tokens
        streams[key].write(json.dumps(item, ensure_ascii=False) + "\n")
        stats[key]["rows"] += 1
        stats[key]["tokens"] += tokens
        stats[key]["language:" + item["language"]] += 1
        stats[key]["topic:" + item["topic"]] += 1
        stats[key]["synthetic"] += item["origin"].startswith("synthetic")

    for n, payload, encoded_flags, root, split in db.execute("SELECT n,payload,flags,root,split FROM records ORDER BY n"):
        row = json.loads(zlib.decompress(payload))
        flags = flags_for(row, known)
        if n in near_rows:
            flags.append("near_duplicate_review")
        if n in overflow_rows:
            flags.append("near_index_common_anchor_review")
        extra = []
        if row["origin"].startswith("synthetic"):
            extra.append("synthetic_source_quality_unqualified")
        if row["record_type"] == "health_qa":
            extra.append("medical_semantic_review_required")
        pair_issues = pair_flags(row["english"], row["text"]) if row.get("english") is not None else []
        views = views_for(row, flags)
        disposition = "excluded" if split == "protected" or n in duplicate else "accepted" if views else "review_needed"
        reasons = (["protected_related_source"] if split == "protected" else ["exact_duplicate"] if n in duplicate else flags + extra + pair_issues)
        base = {"schema_version": 1, "source_id": row["id"], "source_hash": row["record_hash"],
                "source_file": row["file"], "source_file_sha256": row["file_sha256"],
                "source_row_index": row["row_index"], "source_revision": row["revision"],
                "group_id": digest([VERSION, root]), "split": split, "language": row["language"],
                "record_type": row["record_type"], "origin": row["origin"], "topic": row["topic"],
                "license": row.get("terms"), "human_reviewed": False, "production_eligible": False,
                "screening": "automated_source_preserved", "semantic_certification": False}
        disposition_counts[row["source"] + ":" + disposition] += 1
        db.execute("UPDATE records SET disposition=?,flags=? WHERE n=?", (disposition, json.dumps(reasons), n))
        if disposition == "accepted":
            for name, content in views:
                write(name, split, {**base, "id": row["id"] + ":" + name, **content})
        if disposition == "excluded":
            exclusions.write(json.dumps({"id": row["id"], "source_hash": row["record_hash"], "reasons": reasons}) + "\n")
            continue
        missing = []
        if row["record_type"] in ("sentence", "utterance", "intent_example") and row["language"] != "en":
            if not row.get("english"):
                missing.append("meaning_english")
            if not row.get("structured"):
                missing.extend(("record_function", "source_grounded_entities", "ambiguity"))
        if row["record_type"] == "health_qa":
            missing.extend(("education_or_individual_assessment", "source_answer_support", "medical_contradictions"))
        if row["record_type"] == "conversation":
            missing.append("grounded_twi_derivative")
        if missing:
            plan_stream.write(json.dumps({**base, "task_id": digest([row["record_hash"], missing]),
                "missing_fields": missing, "status": "pending_teacher_qualification", "provider": "modal",
                "existing_reference_preserved": bool(row.get("english")), "source_text": row["text"],
                "reference_answer": row.get("reference_answer"), "reference_english": row.get("english"),
                "messages": row.get("original_messages")}, ensure_ascii=False) + "\n")
        if reasons or missing:
            review = {**base, "id": row["id"], "original": row["text"], "reference_english": row.get("english"),
                      "reference_answer": row.get("reference_answer"), "messages": row.get("original_messages"),
                      "structured": row.get("structured"), "reasons": reasons, "missing_fields": missing,
                      "alternatives": [], "suggested_normalization": None, "decision": None}
            queue.write(json.dumps(review, ensure_ascii=False) + "\n")
            key = (row["source"], row["record_type"], ",".join(reasons) or "missing_annotation")
            sampler.add(key, review)
            review_count[row["source"]] += 1
    db.commit()
    for stream in streams.values():
        stream.close()
    queue.close(); exclusions.close(); plan_stream.close()
    selected = sampler.rows()
    # Prioritize source-backed corrections, not a thousand repetitive synthetic examples.
    selected.sort(key=lambda r: (r["origin"].startswith("synthetic"), not bool(r["reference_english"]), r["id"]))
    with (corrections / "session.jsonl").open("w") as handle:
        for r in selected[:40]:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (corrections / "representative.jsonl").open("w") as handle:
        for r in selected:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    save(work / "plan.json", {"provider": "modal", "scope": "all_eligible_missing_fields", "row_limit": None,
         "status": "pending_teacher_qualification", "pending_sha256": sha(work / "pending.jsonl"),
         "reuse_policy": "source_hash + field + prompt_revision; accepted existing fields are immutable",
         "no_api_fallback": True, "confidence_is_acceptance": False})
    save(output / "reports/coverage.json", {"views": stats, "source_dispositions": disposition_counts,
         "review_rows": review_count, "tokenizer": {**spec, "sha256": sha(tokenizer_path)},
         "raw_synthetic_tokens": "See frozen full-source tokenizer audit; not counted as accepted training tokens."})
    save(output / "reports/readiness.json", {"version": VERSION, "complete_source_processing": True,
         "all_four_collections_ready": False, "human_reviewed_by_this_run": 0,
         "language": "Source-preserved screened text; synthetic and uncertain text excluded",
         "meaning": "Existing alignment and source intent targets; missing meanings pending",
         "conversation": "English retention and source-grounded Twi QA; general multi-turn Twi pending qualified teacher",
         "domain_actions": "AfriHealth references retained in review; medical semantic approval pending",
         "new_training_started": False, "production_changed": False})
    save(output / "training/data-config.json", {"format": "jsonl", "files": sorted(streams),
         "mixture_weights": None, "base_model": None, "exclude": ["review", "protected", "pending"],
         "note": "Select explicit views and weights. Never combine every file by wildcard."})
    checksums = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and not p.name.startswith("ledger.sqlite3")}
    save(output / "manifest.json", {"version": VERSION, "artifacts": checksums,
         "raw_mutated": False, "public_uploaded": False, "training_started": False})


def verify(output):
    from build_language_adaptation import OverlapIndex
    from corpus_tools import verify_trajectory
    from corpus_schemas import validate as validate_schema
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if sha(output / name) != expected:
            raise ValueError("Changed artifact " + name)
    inventory = json.loads((output / "inventory.json").read_text())
    for name, expected in inventory["input_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed source " + name)
    sealed = output / "SEALED.json"
    if sealed.exists():
        seal = json.loads(sealed.read_text())
        if sha(output / "manifest.json") != seal["manifest_sha256"] or sha(output / "ledger.sqlite3") != seal["ledger_sha256"]:
            raise ValueError("Changed sealed release")
    blocked = OverlapIndex(release_holdouts(output))
    db = sqlite3.connect(f"file:{output / 'ledger.sqlite3'}?mode=ro", uri=True)
    groups, identities, counts = {}, {}, Counter()

    def require(condition, message):
        if not condition: raise ValueError(message)

    for path in sorted((output / "collections").rglob("*.jsonl")):
        for row in jsonl(path):
            view = path.parent.relative_to(output / "collections").as_posix()
            validate_schema(view, row)
            if "tools-english-simulated" in path.parts:
                verify_trajectory(row)
                require(row["catalogue_sha256"] == sha(path.parent / "catalogue.json"), "Changed catalogue")
                require(not any(blocked.contains(m["content"]) for m in row["messages"] if m.get("content") and m["role"] in ("user", "assistant")), "Tool fixture overlaps a holdout")
                require(groups.setdefault(row["group_id"], row["split"]) == row["split"], "Tool group split conflict")
                counts[str(path.relative_to(output))] += 1
                continue
            require(row.get("schema_version") == 1 and row.get("split") in ("train", "validation") and row.get("human_reviewed") is False, "Invalid export schema/status")
            require(row.get("production_eligible") is False and row.get("semantic_certification") is False, "Invalid promotion")
            require(type(row.get("tokens")) is int and row["tokens"] >= 0, "Invalid token count")
            source = db.execute("SELECT payload,split,disposition,root FROM records WHERE id=?", (row["source_id"],)).fetchone()
            require(source and source[1] == row["split"] and source[2] == "accepted", "Ineligible source exported")
            original = json.loads(zlib.decompress(source[0]))
            expected = original.pop("record_hash")
            require(digest(original) == expected == row["source_hash"], "Changed source record")
            require(row["source_file_sha256"] == original["file_sha256"] and row["source_revision"] == original["revision"], "Changed provenance")
            require(not any(blocked.contains(t) for t in row_texts(original)), "Held-out text entered an export")
            require(not original["origin"].startswith("synthetic"), "Unqualified synthetic export")
            group = row["group_id"]
            require(group == digest([VERSION, source[3]]), "Incorrect source group")
            require(groups.setdefault(group, row["split"]) == row["split"], "Cross-collection split conflict")
            identity = row["source_id"]
            require(identities.setdefault(identity, row["split"]) == row["split"], "Source identity split conflict")
            expected_views = dict(views_for(original, []))
            require(view in expected_views, "Record has no eligible task view")
            for field, value in expected_views[view].items():
                require(row.get(field) == value, "Changed training target: " + field)
            counts[str(path.relative_to(output))] += 1
    total = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    require(total == inventory["rows"], "Source inventory count mismatch")
    require(db.execute("SELECT COUNT(*) FROM records WHERE disposition IS NULL").fetchone()[0] == 0, "Unaccounted sources")
    require(not db.execute("SELECT root FROM records GROUP BY root HAVING COUNT(DISTINCT split)>1 LIMIT 1").fetchone(), "Related source partitions diverged")
    require(not db.execute("SELECT id FROM records WHERE locked=1 AND split!='protected' LIMIT 1").fetchone(), "Lost holdout protection")
    db.close()
    result = {"status": "passed", "source_rows": total, "unique_exported_sources": len(identities),
              "cross_collection_split_conflicts": 0, "exports": dict(counts)}
    save(output / "reports/verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "recover", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "recover": recover, "verify": verify}[args.action](args.out.resolve())
