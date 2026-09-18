"""Compile the release's reference views, correction index and final receipts."""
import argparse
import json
import shutil
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from corpus_release import ROOT, jsonl, save, sha, verify
from corpus_annotation import prepare


def collection_reports(release):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from corpus_release import REGISTRY
    spec = json.loads(REGISTRY.read_text())["tokenizers"][0]
    tokenizer = Tokenizer.from_file(hf_hub_download(spec["repo"], "tokenizer.json", revision=spec["revision"], local_files_only=True))
    summary = {}
    for collection in (release / "collections").iterdir():
        if not collection.is_dir(): continue
        counts, languages, topics, statuses = Counter(), Counter(), Counter(), Counter()
        sources, groups = set(), set()
        views = {}
        for path in sorted(collection.rglob("*.jsonl")):
            view_counts = Counter()
            for row in jsonl(path):
                sources.add(row.get("source_id", row["id"]))
                groups.add(row["group_id"])
                tokens = row.get("tokens")
                if tokens is None:
                    tokens = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) for m in row["messages"] if m.get("content"))
                counts["rows"] += 1; counts["tokens_across_views"] += tokens
                counts["synthetic_rows"] += row["origin"].startswith("synthetic")
                counts["human_reviewed_rows"] += row["human_reviewed"]
                languages[row["language"]] += 1; topics[row.get("topic", "commerce" if row.get("simulated") else "unclassified")] += 1
                statuses[row["screening"]] += 1
                if row["language"] == "tw" and sum(m["role"] == "assistant" for m in row.get("messages", [])) > 1:
                    counts["multi_turn_twi_response_rows"] += 1
                view_counts["rows"] += 1; view_counts["tokens"] += tokens
            views[str(path.relative_to(release))] = dict(view_counts)
        summary[collection.name] = {**counts, "unique_sources": len(sources), "related_source_groups": len(groups),
            "language": dict(languages), "topic": dict(topics), "annotation_status": dict(statuses),
            "synthetic_share": counts["synthetic_rows"] / max(1, counts["rows"]), "views": views,
            "tokens_note": "Pinned tokenizer, content tokens without chat-template overhead; shared sources can occur in different task views."}
        save(collection / "coverage.json", summary[collection.name])
        (collection / "DATASET_CARD.md").write_text(
            f"# {collection.name}\n\n{counts['rows']:,} task-view rows from {len(sources):,} unique source records.\n"
            "\nSee coverage.json for token, language, topic, split-view and screening counts. "
            "All original licences and provenance are retained per record. Automated source checks "
            "are not semantic certification. No project human reviews are invented.\n"
            "\nUse only the explicit files for this stage from training/data-config.json. "
            "Related source IDs share the same split across collections. "
            "Health source material is outside these training views. "
            "Simulated tools contain fixture prices and cannot place orders.\n", encoding="utf-8")
    save(release / "reports/collections.json", summary)
    return summary


def source_selection_report(release):
    import pyarrow.parquet as pq
    db = sqlite3.connect(f"file:{release / 'ledger.sqlite3'}?mode=ro", uri=True)
    selected = {}
    for payload, disposition in db.execute("SELECT r.payload,r.disposition FROM records r JOIN review_index i ON r.n=i.n WHERE i.source='oasst'"):
        row = json.loads(zlib.decompress(payload))
        selected[row["source_group"].removeprefix("oasst:")] = {"source_id": row["id"], "disposition": disposition}
    db.close()
    report, totals = [], Counter()
    for split in ("train", "validation"):
        path = next((ROOT / f"tmp/general-language-corpus/oasst_{split}").glob("*.parquet"))
        file_hash = sha(path)
        trees = defaultdict(list)
        for i, row in enumerate(pq.read_table(path, columns=["message_id", "message_tree_id"]).to_pylist()):
            trees[row["message_tree_id"]].append(i)
        for tree, indices in trees.items():
            chosen = selected.get(tree)
            status = chosen["disposition"] if chosen else "excluded"
            totals[status + "_trees"] += 1; totals["raw_messages"] += len(indices)
            report.append({"tree_id": tree, "source_file": str(path.relative_to(ROOT)), "source_file_sha256": file_hash,
                "upstream_split": split, "raw_message_indices": indices, "selected_path": chosen,
                "status": status, "reason": "One intact eligible path per tree; other branches not selected" if chosen else "No path meets the documented retention policy"})
    target = release / "reports/oasst-tree-accounting.jsonl"
    with target.open("w") as handle:
        for row in report: handle.write(json.dumps(row) + "\n")
    save(release / "reports/source-selection.json", {"oasst": dict(totals),
        "policy": "One complete root-to-assistant English path per eligible tree; reviewed nonsynthetic rank-zero target; quality >= .6; no upstream PII/spam/language-mismatch flags; max 6 turns, 2200 chars per turn, 4200 chars per path. Longer or ineligible paths are excluded, never truncated.",
        "other_sources": "Every row from the selected full-source files appears in the main ledger.",
        "accepted_definition": "A record supplies at least one automatically screened source-preserved training view; not every annotation or training stage is complete."})


def index(release):
    db = sqlite3.connect(release / "ledger.sqlite3", timeout=120)
    db.execute("CREATE TABLE IF NOT EXISTS review_index(n INTEGER PRIMARY KEY,record_type TEXT,source TEXT,synthetic INTEGER,split TEXT,disposition TEXT)")
    batch = []
    for n, payload, split, disposition, locked in db.execute("SELECT n,payload,split,disposition,locked FROM records ORDER BY n"):
        row = json.loads(zlib.decompress(payload))
        batch.append((n, row["record_type"], row["source"], int(row["origin"].startswith("synthetic")),
                      split or ("protected" if locked else "unassigned"), disposition or "pending"))
        if len(batch) >= 10000:
            db.executemany("INSERT OR REPLACE INTO review_index VALUES(?,?,?,?,?,?)", batch)
            db.commit(); batch.clear()
    db.executemany("INSERT OR REPLACE INTO review_index VALUES(?,?,?,?,?,?)", batch)
    db.execute("CREATE INDEX IF NOT EXISTS review_filter ON review_index(record_type,synthetic,split,disposition,n)")
    db.execute("CREATE TABLE IF NOT EXISTS review_annotations(source_id TEXT PRIMARY KEY, source_hash TEXT, annotation TEXT)")
    preserved = release / "annotation/preserved-annotations.jsonl"
    if preserved.exists():
        for row in jsonl(preserved):
            db.execute("INSERT OR REPLACE INTO review_annotations VALUES(?,?,?)", (row["source_id"], row["source_hash"], json.dumps(row["prior_annotation"], ensure_ascii=False)))
    db.commit(); db.close()


def finalize(release, teacher):
    if (release / "SEALED.json").exists():
        raise ValueError("Release is sealed; create another version for later corrections")
    checkpoint = release / "FINALIZING.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["finalizer_sha256"] != sha(__file__):
            raise ValueError("Finalizer changed during an unfinished release; inspect before resuming")
        for name, expected in state["immutable_inputs"].items():
            if sha(release / name) != expected: raise ValueError("Changed finalizer input: " + name)
    else:
        verify(release)
        core = json.loads((release / "manifest.json").read_text())
        save(checkpoint, {"finalizer_sha256": sha(__file__), "immutable_inputs": {
            name: checksum for name, checksum in core["artifacts"].items() if name.startswith(("collections/", "review/")) or name in ("inventory.json", "protected.json", "exclusions.jsonl", "annotation/pending.jsonl")}})
    prepare(release)
    index(release)
    source_selection_report(release)
    domain = release / "collections/domain-actions"
    domain.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(release / "ledger.sqlite3")
    health_counts = Counter()
    material = release / "source-material"; material.mkdir(exist_ok=True)
    with (material / "health.jsonl").open("w") as stream:
        for payload, split, disposition, flags in db.execute("SELECT r.payload,r.split,r.disposition,r.flags FROM records r JOIN review_index i ON r.n=i.n WHERE i.record_type='health_qa'"):
            row = json.loads(zlib.decompress(payload))
            health_counts[split + ":" + disposition] += 1
            if split == "protected": continue
            stream.write(json.dumps({"id": row["id"], "source_hash": row["record_hash"], "source_file": row["file"],
                "source_file_sha256": row["file_sha256"], "source_revision": row["revision"], "source_record_id": row["source_record_id"],
                "language": row["language"], "split": split, "question": row["text"], "reference_answer": row["reference_answer"],
                "messages": row["original_messages"], "review_reasons": json.loads(flags), "status": "source_material_not_approved_response",
                "research_training_eligible": False, "clinical_certification": False, "human_reviewed": False}, ensure_ascii=False) + "\n")
    db.close()
    shutil.copytree(ROOT / "tmp/corpus-tools/catalogue-v1", domain / "tools-english-simulated", dirs_exist_ok=True)
    if teacher and (teacher / "gate.json").exists():
        target = release / "teacher"; target.mkdir(exist_ok=True)
        for name in ("review.json", "gate.json", "manifest.json", "qualify.receipt.json", "qualify.results.json", "references.json"):
            if (teacher / name).exists(): shutil.copyfile(teacher / name, target / name)
    readiness = json.loads((release / "reports/readiness.json").read_text())
    readiness["health_source_counts"] = health_counts
    readiness["tool_fixture_rows"] = json.loads((domain / "tools-english-simulated/report.json").read_text())["rows"]
    readiness["teacher"] = json.loads((teacher / "gate.json").read_text()) if teacher and (teacher / "gate.json").exists() else {"qualified": False, "status": "qualification_not_completed"}
    readiness["missing"] = ["Qualified bulk translation/annotation method", "General multi-turn Twi conversation targets", "Medical semantic/clinical screening", "Twi tool trajectories"]
    save(release / "reports/readiness.json", readiness)
    coverage = json.loads((release / "reports/coverage.json").read_text())
    coverage["health_source_material"] = dict(health_counts)
    coverage["known_unknowns"] = ["Topic and dialect labels absent upstream remain unclassified.",
        "Synthetic Pristine tokens are measured in the prior pinned audit, not accepted CPT tokens.",
        "Lexical near-duplicate routing is approximate; semantic paraphrase recall is not certified."]
    save(release / "reports/coverage.json", coverage)
    config_path = release / "training/data-config.json"
    config = json.loads(config_path.read_text())
    config["files"] = sorted(set(config["files"]) | {"domain-actions/tools-english-simulated/" + split for split in ("train", "validation")})
    config["exclude"] = sorted(set(config["exclude"]) | {"source-material"})
    save(config_path, config)
    collection_reports(release)
    from corpus_schemas import SCHEMAS
    for name, schema in SCHEMAS.items():
        save(release / "schemas" / (name + ".json"), schema.model_json_schema())
    scripts = ("corpus_release.py", "corpus_annotation.py", "corpus_handoff.py", "corpus_teacher.py",
               "corpus_schemas.py", "corpus_tools.py", "corpus_review_row.py", "audit_twi_pretraining.py",
               "build_language_adaptation.py", "prepare_grounded_twi_response.py", "prepare_tool_context_repair.py")
    save(release / "reports/implementation.json", {"python": "3.13", "files": {
        "scripts/" + name: sha(ROOT / "scripts" / name) for name in scripts},
        "requirements_sha256": sha(ROOT / "scripts/requirements-corpus.txt"),
        "note": "Run the recorded research-script versions against the pinned source cache. No student model or mixture is selected."})
    save(release / "reports/research-reference.json", {
        "project": "Masakhane", "url": "https://github.com/masakhane-io/masakhane-community",
        "dataset_guidelines": "https://github.com/masakhane-io/masakhane-community/blob/master/dataset-creation-guidelines.md",
        "adopt_now": ["Explicit entity annotation boundaries and source spans", "Data sovereignty and source ownership", "Reproducible code/data/results and error analysis"],
        "limits": "Community translation guideline is marked TODO; do not claim it certifies our translations.",
        "future_contribution": "Twi benchmark/error taxonomy, reproducible scripts, and only distributable approved artifacts. No upload in this pass."})
    (release / "DATASET_CARD.md").write_text(card(release, readiness), encoding="utf-8")
    (release / "HANDOFF.md").write_text(handoff(release), encoding="utf-8")
    files = {str(p.relative_to(release)): sha(p) for p in release.rglob("*") if p.is_file() and not p.name.startswith("ledger.sqlite3") and p.name not in ("manifest.json", "SEALED.json")}
    save(release / "manifest.json", {"version": "stage-corpus-v1", "artifacts": files, "all_four_collections_ready": False,
         "new_training": False, "production_deployment": False, "public_upload": False})
    verify(release)
    files["reports/verification.json"] = sha(release / "reports/verification.json")
    save(release / "manifest.json", {"version": "stage-corpus-v1", "artifacts": files, "all_four_collections_ready": False,
         "new_training": False, "production_deployment": False, "public_upload": False})
    save(release / "SEALED.json", {"manifest_sha256": sha(release / "manifest.json"),
         "status": "source-preserved research release; generated gaps not concealed", "ledger_sha256": sha(release / "ledger.sqlite3")})
    print(json.dumps(readiness, indent=2))


def card(release, readiness):
    inventory = json.loads((release / "inventory.json").read_text())
    return f"""# Twi Stage Corpus V1

## Scope
{inventory['rows']:,} selected source records accounted for. This is a source-preserved
research release, not a clinically certified or fully human-reviewed training set.
It is not a finished general-purpose Twi conversation corpus.

## Sources And Rights
All source paths, hashes and pinned revisions are in inventory.json and row metadata.
WAXAL, GhanaNLP, community parallel, INJONGO, AfriQA, OASST and AfriHealth retain
their own source terms. Mixed licences are not replaced with one blanket licence.
Pristine is synthetic news-conditioned text, not native gold. It is quarantined.
Private derived artifacts remain private. No redistribution clearance is claimed.

## Views
- language: unchanged text, short utterances, dictionary entries kept distinct.
- meaning: preserved English references and INJONGO structured intent/entity targets.
- conversation: intact source English retention and grounded Twi QA, not fabricated dialogue.
- domain-actions: complete health references for correction and executable English
  simulated tool fixtures. These fixtures do not count as a large Twi commerce corpus.

## Quality
Automated screening, source grounding, source author review, project human review
and clinical certification are distinct. No new human reviews were invented.
Near duplicates route to review; similarity is not meaning equivalence. Related
source groups share splits. Existing held-out overlaps are excluded from training.
Medical answers retain their full references and are not admitted by a structural
pass or model agreement alone. Null means unknown or not applicable, not absence
of an entity or proof of safety.

## Limitations
General multi-turn Twi targets and large-scale qualified annotation remain gaps.
Missing topics, dialects and intent are not filled with guesses. Teacher failure
does not invalidate original source text, but blocks unreviewed generated derivatives.
Read reports/readiness.json, reports/coverage.json and reports/verification.json.
"""


def handoff(release):
    return f"""# Handoff

## What To Load
Use explicit JSONL views listed in training/data-config.json. Do not glob every
collection: source-material/health.jsonl is a correction corpus, not approved SFT.
No training base, mixture weights or new model run is selected here.

## Verify
From the repository root with Python 3.13, pyarrow 24.0.0, tokenizers 0.22.2,
huggingface-hub and Modal 1.4.3 available:

```sh
python scripts/corpus_release.py verify --out {release.relative_to(ROOT)}
python -m unittest discover -s scripts -p test_corpus_release.py
node --import tsx scripts/eval-source-corpus-runner.ts
```

To reproduce a new release, build into a new directory. Recover an unsealed
checkpoint with `corpus_release.py recover`; do not replace source files or reviews.
The old 12-row paid-API default is disabled. `corpus_annotation.py plan` covers all
eligible missing fields; execution requires the teacher's passed gate and an
explicit batch bound. Request receipts prevent duplicate completed calls.

## Review
Local page: http://localhost:3100/research/ase?release={release.name}
Original text and reference answers are read-only. Corrections are separate,
source-hash-bound append-only decisions under tmp/corpus-review/{release.name}/.
Postgres stores correction/reference metadata when available; never the corpus blobs.
Original source documents and prior annotation files remain unchanged.

## Next Training Decision
Language adaptation and existing alignment can be considered separately from
response training after reviewing representative accepted records. Do not train
a new direct-response model claiming the multi-turn Twi gap has been solved.
No ASR/TTS work, deployment, public upload or model training occurred in this pass.
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("index", "finalize"))
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--teacher", type=Path)
    args = p.parse_args()
    if args.action == "index": index(args.release.resolve())
    else: finalize(args.release.resolve(), args.teacher.resolve() if args.teacher else None)
