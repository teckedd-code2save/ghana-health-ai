"""Project pinned public speech Parquet text columns without exporting audio."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCES = [
    {"name": "waxal_akan_full", "repo": "google/WaxalNLP",
     "revision": "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580", "license": "cc-by-4.0",
     "prefix": "data/ASR/aka/aka-", "text_column": "transcription",
     "columns": ["id", "speaker_id", "transcription", "language"]},
    {"name": "ghana_nlp_speech_full", "repo": "ghanaopenai/twi-speech-text-multispeaker-16k",
     "revision": "410ef3e8ed3d16829247339e7f4fc75b519271a8", "license": "cc-by-nc-4.0",
     "prefix": "data/train-", "text_column": "text", "columns": ["text"]},
]


def split_for(source, filename):
    if not filename.startswith(source["prefix"]) or not filename.endswith(".parquet"):
        return None
    if source["name"] == "ghana_nlp_speech_full":
        return "train"
    for split in ("train", "validation", "test"):
        if filename.startswith(source["prefix"] + split + "-"):
            return split
    return None


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def project_language(source_name):
    return {"waxal_akan_full": "ak", "ghana_nlp_speech_full": "tw"}[source_name]


def run(output):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    if output.exists():
        raise ValueError("Choose a fresh output directory; source exports are immutable")
    output.mkdir(parents=True)
    fs, api = HfFileSystem(token=False), HfApi(token=False)
    files, counts = [], Counter()
    path = output / "sources.jsonl"
    with path.open("x", encoding="utf-8") as target:
        for source in SOURCES:
            info = api.dataset_info(source["repo"], revision=source["revision"], files_metadata=True)
            if info.sha != source["revision"]:
                raise ValueError("Hub revision changed")
            selected = [(item, split_for(source, item.rfilename)) for item in info.siblings]
            selected = [(item, split) for item, split in selected if split]
            if not selected:
                raise ValueError("No labelled source shards found")
            for item, split in sorted(selected, key=lambda value: value[0].rfilename):
                print(f"Reading text only: {source['name']} / {item.rfilename}", flush=True)
                remote = f"datasets/{source['repo']}@{source['revision']}/{item.rfilename}"
                with fs.open(remote, "rb", block_size=65536) as handle:
                    parquet = pq.ParquetFile(handle, pre_buffer=False)
                    if not set(source["columns"]).issubset(parquet.schema_arrow.names):
                        raise ValueError("Source columns changed")
                    row_index = 0
                    shard_hash = hashlib.sha256()
                    for batch in parquet.iter_batches(batch_size=1024, columns=source["columns"]):
                        for original in batch.to_pylist():
                            text = original[source["text_column"]]
                            if not isinstance(text, str):
                                raise ValueError("Non-text transcript; preserve partial export for investigation")
                            row = {"id": f"{source['name']}:{Path(item.rfilename).stem}:{row_index}",
                                "source": source["name"], "repo": source["repo"], "revision": source["revision"],
                                "source_file": item.rfilename, "row_index": row_index,
                                "source_record_id": original.get("id"), "speaker_id": original.get("speaker_id"),
                                "source_language": original.get("language"), "language": project_language(source["name"]),
                                "split": split, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "license": source["license"], "training_eligible": False}
                            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            shard_hash.update(serialized.encode())
                            target.write(serialized)
                            row_index += 1
                            counts[source["name"] + ":" + split] += 1
                    if row_index != parquet.metadata.num_rows:
                        raise ValueError("Incomplete text projection")
                files.append({"source": source["name"], "revision": source["revision"],
                    "file": item.rfilename, "source_bytes": item.size,
                    "source_lfs_sha256": item.lfs.sha256 if item.lfs else None,
                    "columns": source["columns"], "split": split, "rows": row_index,
                    "projected_rows_sha256": shard_hash.hexdigest()})
                target.flush()
                (output / "progress.json").write_text(json.dumps({"stage": "partial", "files": files, "counts": counts}, indent=2) + "\n")
    summary = {"stage": "completed_public_text_projection_not_training_export", "sources": SOURCES,
        "counts": counts, "rows": sum(counts.values()), "files": files,
        "output_sha256": digest(path), "script_sha256": digest(Path(__file__)),
        "limitations": [
            "Only labelled WAXAL Akan ASR and the three GhanaNLP source shards; no unlabelled audio or TTS aliases.",
            "Audio columns are not decoded or exported. HTTP range reads may include adjacent file bytes.",
            "Published transcripts are unchanged, not freshly transcribed or independently language-verified.",
            "Upstream splits retained; existing local holdouts must also be excluded before training.",
            "No deduplication, speaker/topic grouping, annotation or training acceptance in this projection.",
            "WAXAL is labelled ak (Akan), not verified tw (Twi); dialect curation remains necessary.",
            "This overlaps the earlier local speech manifest; never add their row counts as independent records.",
        ]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": str(output), "rows": summary["rows"], "counts": counts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tmp/twi-pretraining-sources/full-public-speech-v1"))
    run(parser.parse_args().output)
