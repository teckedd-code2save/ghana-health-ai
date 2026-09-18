"""Bounded read-only local UI projection of the source ledger; standard library."""
import argparse
import json
import sqlite3
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILTERS = {"language": ("sentence", "utterance", "dictionary_entry", "document"),
           "meaning": ("sentence", "intent_example"), "conversation": ("conversation", "grounded_qa"),
           "health": ("health_qa",)}


def read(folder, collection, offset):
    db = sqlite3.connect(f"file:{folder / 'ledger.sqlite3'}?mode=ro", uri=True)
    # Compact local projection is indexed once by the finalizer. While building,
    # the UI deliberately reports unavailable rather than reading a million blobs.
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='review_index'").fetchone():
        db.close()
        return {"ready": False, "rows": [], "total": 0}
    kinds = FILTERS[collection]
    placeholders = ",".join("?" for _ in kinds)
    where = f"record_type IN ({placeholders}) AND split!='protected' AND disposition!='excluded' AND synthetic=0"
    total = db.execute("SELECT COUNT(*) FROM review_index WHERE " + where, kinds).fetchone()[0]
    row = db.execute("SELECT r.payload,r.flags FROM review_index i JOIN records r ON r.n=i.n WHERE " +
                     where.replace("split", "i.split").replace("disposition", "i.disposition") +
                     " ORDER BY i.n LIMIT 1 OFFSET ?", (*kinds, offset)).fetchone()
    if not row:
        db.close()
        return {"ready": True, "rows": [], "total": total}
    source = json.loads(zlib.decompress(row[0]))
    annotation = None
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='review_annotations'").fetchone():
        item = db.execute("SELECT annotation FROM review_annotations WHERE source_id=? AND source_hash=?", (source["id"], source["record_hash"])).fetchone()
        if item: annotation = json.loads(item[0])
    db.close()
    reference_analysis = None
    sidecar = ROOT / "tmp/corpus-reference-review" / folder.name / "annotations.sqlite3"
    if sidecar.exists():
        with sqlite3.connect(f"file:{sidecar}?mode=ro", uri=True) as annotations:
            found = annotations.execute("SELECT payload FROM annotations WHERE source_id=? AND source_hash=?",
                                        (source["id"], source["record_hash"])).fetchone()
        if found:
            candidate = json.loads(found[0])
            if candidate["reference_english"] == source.get("english"):
                reference_analysis = candidate
    alternatives = []
    if annotation:
        for proposal in annotation.get("proposals", [])[:2]:
            alternatives.append({"model": proposal.get("model", proposal.get("proposal_id", "Prior annotation")),
                "text": proposal.get("question_english", ""), "fields": {k: proposal.get(k) for k in
                    ("intent", "entities", "answer_english", "reply_twi", "requires_clarification", "source_answer_issues")}})
    return {"ready": True, "total": total, "rows": [{"id": source["id"], "source_hash": source["record_hash"],
        "original": source["text"], "reference_english": source.get("english"),
        "reference_answer": source.get("reference_answer"), "messages": source.get("original_messages"),
        "structured": source.get("structured"), "source": source["source"], "language": source["language"],
        "record_type": source["record_type"], "context": source.get("context"),
        "reasons": json.loads(row[1]), "alternatives": alternatives, "reference_analysis": reference_analysis}]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True)
    p.add_argument("--collection", choices=tuple(FILTERS), required=True)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()
    if args.offset < 0: p.error("Invalid offset")
    folder = (ROOT / "tmp/corpus-releases" / args.release).resolve()
    if folder.parent != ROOT / "tmp/corpus-releases": p.error("Invalid release")
    print(json.dumps(read(folder, args.collection, args.offset), ensure_ascii=False))
