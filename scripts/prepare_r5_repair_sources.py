"""Register explicit bilingual repair proposals; this does not approve them."""
import argparse
from pathlib import Path
import training_data_inputs as data


def prepare(source, out):
    data.require(not out.exists(), "Use a fresh proposal directory")
    raw = data.read(source)
    data.require(raw["training_eligible"] is False and raw["human_reviewed"] is False, "Unreviewed sources required")
    sources, proposals, seen = [], [], set()
    for number, record in enumerate(raw["records"], 1000):
        data.require(record["id"] not in seen and data.valid_id(record["id"]), "Repeated or malformed source")
        seen.add(record["id"])
        data.require(len(record["en"]) == len(record["tw"]) and len(record["en"]) % 2 == 0, "Incomplete translation")
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
                    for i, text in enumerate(record["en"])]
        identity = "r5-development-repair:" + record["id"]
        witness = {"id": identity, "source_id": identity, "messages": messages,
            "metadata": {"origin": "project_authored_development_informed", "group_id": identity,
                         "source_hash": data.digest(record), "topic": record["topic"], "human_reviewed": False},
            "original_ref": {"file": str(source.resolve()), "file_sha256": data.sha(source),
                             "source_record_sha256": data.digest(record), "record_index": number - 1000}}
        sources.append({"index": number, "source_record": witness, "source_record_sha256": data.digest(witness),
                        "training_eligible": False, "review_status": "meaning_review_required"})
        proposals.append({"index": number, "source_id": identity, "source_record_sha256": data.digest(witness),
            "decision": "translate", "reason": "Explicit project-authored contrast for " + record["topic"],
            "twi": record["tw"], "english_overrides": {}, "amendment_reason": ""})
    out.mkdir(parents=True)
    data.write_rows(out / "sources.jsonl", sources)
    data.save(out / "proposals.json", {"kind": "agent_translation_proposals", "agent": "main-author",
        "human_reviewed": False, "training_eligible": False, "source_file": str((out / "sources.jsonl").resolve()),
        "source_file_sha256": data.sha(out / "sources.jsonl"), "records": proposals})
    return {"sources": len(sources), "assistant_occurrences": sum(len(s["source_record"]["messages"]) // 2 for s in sources)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.out))
