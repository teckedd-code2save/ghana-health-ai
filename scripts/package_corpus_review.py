"""Project only eligible review rows; leave the sealed training release untouched."""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def sha(path):
    with path.open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def package(release, output):
    seal = json.loads((release / "SEALED.json").read_text())
    if sha(release / "manifest.json") != seal["manifest_sha256"] or sha(release / "ledger.sqlite3") != seal["ledger_sha256"]:
        raise ValueError("Changed source release")
    if output.exists(): raise ValueError("Use a new package path; never replace saved reviews")
    folder = output / "tmp/corpus-releases" / release.name
    folder.mkdir(parents=True)
    source = sqlite3.connect(f"file:{release / 'ledger.sqlite3'}?mode=ro", uri=True)
    target = sqlite3.connect(folder / "ledger.sqlite3")
    target.execute("CREATE TABLE records(n INTEGER PRIMARY KEY,id TEXT UNIQUE,payload BLOB,flags TEXT)")
    target.execute("CREATE TABLE review_index(n INTEGER PRIMARY KEY,record_type TEXT,source TEXT,synthetic INTEGER,split TEXT,disposition TEXT)")
    target.execute("CREATE TABLE review_annotations(source_id TEXT PRIMARY KEY,source_hash TEXT,annotation TEXT)")
    count = 0
    for row in source.execute("SELECT r.n,r.id,r.payload,r.flags,i.record_type,i.source,i.synthetic,i.split,i.disposition FROM records r JOIN review_index i ON r.n=i.n WHERE i.synthetic=0 AND i.split!='protected' AND i.disposition!='excluded'"):
        target.execute("INSERT INTO records VALUES(?,?,?,?)", row[:4])
        target.execute("INSERT INTO review_index VALUES(?,?,?,?,?,?)", (row[0], *row[4:]))
        count += 1
    target.executemany("INSERT INTO review_annotations VALUES(?,?,?)", source.execute("SELECT * FROM review_annotations"))
    target.execute("CREATE INDEX review_filter ON review_index(record_type,synthetic,split,disposition,n)")
    target.commit(); target.close(); source.close()
    if (release / "teacher").exists(): shutil.copytree(release / "teacher", folder / "teacher")
    scripts = output / "scripts"; scripts.mkdir()
    shutil.copyfile("scripts/corpus_review_row.py", scripts / "corpus_review_row.py")
    reviews = release.parents[1] / "corpus-review" / release.name / "decisions.jsonl"
    saved = 0
    if reviews.exists():
        decisions = output / "tmp/corpus-review" / release.name
        decisions.mkdir(parents=True)
        shutil.copyfile(reviews, decisions / "decisions.jsonl")
        saved = sum(1 for line in reviews.read_text().splitlines() if line)
    report = {"parent_manifest_sha256": seal["manifest_sha256"], "review_rows": count,
              "copied_saved_decisions": saved, "protected_rows": 0, "synthetic_rows": 0,
              "projection_ledger_sha256": sha(folder / "ledger.sqlite3"),
              "raw_release_changed": False, "training_export": False}
    (output / "projection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.release.resolve(), args.out.resolve())
