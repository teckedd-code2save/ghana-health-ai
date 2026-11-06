#!/usr/bin/env python3
"""CPU-only, immutable snapshot-to-Parquet bridge; never authorizes training.

prepare() and verify_prepared() return the manifest. load_partition() returns a
pyarrow.Table (also usable as datasets.Dataset(table)), never a mixed objective.
Original sources and pinned tokenizer assets must remain available and unchanged.
"""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

import check_afrique_training_contract as contract
import afrique_response_core as chat_core
import training_data_inputs as inputs

POLICY = "notebook-training-data-v1"
MODEL, REVISION = contract.MODEL, contract.REVISION
BUFFER_ROWS = 64
CODE_FILES = (Path(__file__), Path(inputs.__file__), Path(contract.__file__), Path(chat_core.__file__))
IMPORTED_CODE = {p.name: inputs.sha(p) for p in CODE_FILES}
LIMITS = (
    "Format-ready only, not semantic approval, training authorization, licensing clearance, "
    "clinical certification or production approval. All original source restrictions still apply; "
    "original_ref and source_metadata preserve their bindings. No automatic mixture, concatenation, "
    "truncation, GPU, model weights, jobs, upload or training. Checksums bind bytes, not approval."
)
FLAGS = {"format_ready": True, "semantic_approval": False, "training_authorized": False,
         "production_eligible": False, "source_restrictions_inherited": True}
REF_TYPE = pa.struct([
    pa.field("snapshot_manifest_sha256", pa.string(), nullable=False),
    pa.field("release_path", pa.string(), nullable=False),
    pa.field("release_manifest_sha256", pa.string(), nullable=False),
    pa.field("file", pa.string(), nullable=False),
    pa.field("file_sha256", pa.string(), nullable=False),
    pa.field("line", pa.int64(), nullable=False),
    pa.field("row_sha256", pa.string(), nullable=False),
])
SCHEMA = pa.schema([
    *[pa.field(k, pa.string(), nullable=False) for k in
      ("id", "source_id", "batch_id", "collection", "split", "objective", "source_metadata")],
    pa.field("original_ref", REF_TYPE, nullable=False),
    *[pa.field(k, pa.list_(pa.int64()), nullable=False) for k in
      ("input_ids", "attention_mask", "labels")],
    pa.field("sequence_tokens", pa.int64(), nullable=False),
    pa.field("target_tokens", pa.int64(), nullable=False),
])


class PreparationError(ValueError):
    """A row failure with machine-readable diagnosis; no partial export is published."""

    def __init__(self, failure, completed_rows):
        self.report = {"policy": POLICY, "format_ready": False, "training_authorized": False,
                       "published": False, "completed_rows": completed_rows, "failure": failure}
        super().__init__(json.dumps(self.report, ensure_ascii=True, sort_keys=True))


def _config(max_length):
    inputs.require(type(max_length) is int and 0 < max_length <= 262144, "Invalid context length")
    return {"max_length": max_length, "automatic_concat": False, "truncation": False,
            "language_objective": "causal_lm", "chat_objective": "final_assistant",
            "eos_token_id": contract.EOS, "pad_token_id": contract.PAD,
            "parquet_compression": "zstd", "row_group_rows": BUFFER_ROWS}


def _identity(snapshot, assets, max_length):
    snapshot_manifest = inputs.verify_snapshot(snapshot)
    code = {p.name: inputs.sha(p) for p in CODE_FILES}
    inputs.require(code == IMPORTED_CODE, "Implementation changed since import; restart before preparing")
    assets_hashes = {name: inputs.sha(inputs.safe_path(assets, name)) for name in contract.ASSET_HASHES}
    inputs.require(assets_hashes == contract.ASSET_HASHES, "Changed pinned tokenizer assets")
    return {"snapshot": {"path": str(snapshot.resolve()),
                         "manifest_sha256": inputs.sha(snapshot / "manifest.json"),
                         "manifest": snapshot_manifest},
            "tokenizer": {"path": str(assets.resolve()), "model": MODEL, "revision": REVISION,
                          "artifacts": assets_hashes},
            "code": code, "config": _config(max_length),
            "runtime": {"python": platform.python_version(), "unicode": unicodedata.unidata_version,
                        **{name: importlib.metadata.version(name) for name in
                           ("pyarrow", "transformers", "tokenizers", "jinja2")}}}


def _paths(snapshot, assets, out):
    snapshot, assets, out = Path(snapshot), Path(assets), Path(out)
    inputs.require(not any(p.is_symlink() for p in (snapshot, assets, out)), "Symlink root is not immutable")
    snapshot, assets, out = snapshot.resolve(), assets.resolve(), out.resolve()
    inputs.verify_snapshot(snapshot)
    roots = [snapshot, assets, *[Path(b["release_path"]) for b in inputs.read(snapshot / "batches.json")]]
    inputs.require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in roots),
                   "Output must be disjoint from snapshot, tokenizer and original sources")
    return snapshot, assets, out


def _encode(tokenizer, collection, row, max_length):
    inputs.require(not any(row.get(k) for k in ("tools", "tool_calls", "functions", "function_call", "tool_choice")),
                   "Tool data needs a separately admitted bridge")
    if collection != "language":
        return contract.encode_complete(tokenizer, row, max_length)
    text = row["text"]
    inputs.require(isinstance(text, str) and bool(text.strip()), "Empty language text")
    inputs.require(not any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
                           (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text),
                   "Invalid Unicode in language text")
    ids = tokenizer.encode(text, add_special_tokens=False)
    inputs.require(not chat_core.CONTROL.search(text) and not tokenizer.contract_special_ids.intersection(ids),
                   "Reserved source token")
    inputs.require(ids and tokenizer.decode(ids, skip_special_tokens=False,
                   clean_up_tokenization_spaces=False) == text, "Original language text changed")
    ids = ids + [contract.EOS]
    inputs.require(len(ids) <= max_length, "overlength_no_truncation: " + row["id"])
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}


def _prepared_rows(snapshot, identity, tokenizer):
    batches = {b["batch_id"]: b for b in inputs.read(snapshot / "batches.json")}
    refs = {}
    for _, ref in inputs.rows(snapshot / "records.jsonl"):
        key = (ref["collection"], ref["batch_id"], ref["id"])
        inputs.require(key not in refs, "Repeated snapshot row identity")
        refs[key] = ref
    expected_count = len(refs)
    completed = 0
    for wrapper in inputs.iter_snapshot_records(snapshot):
        collection, batch_id, row = wrapper["collection"], wrapper["batch_id"], wrapper["row"]
        diagnosis = {"collection": collection, "batch_id": batch_id,
                     **{k: row.get(k) for k in ("id", "source_id", "split")}}
        try:
            ref = refs.pop((collection, batch_id, row["id"]))
            diagnosis["original_ref"] = ref
            batch = batches[batch_id]
            encoded = _encode(tokenizer, collection, row, identity["config"]["max_length"])
            sequence = len(encoded["input_ids"])
            target = sum(v != -100 for v in encoded["labels"])
            inputs.require(sequence > 0 and target > 1 and encoded["attention_mask"] == [1] * sequence
                           and len(encoded["labels"]) == sequence, "Invalid encoded arrays")
            yield {**diagnosis, "objective": "causal_lm" if collection == "language" else "final_assistant",
                   "source_metadata": json.dumps({k: v for k, v in row.items() if k not in ("text", "messages")},
                                                 ensure_ascii=True, sort_keys=True, allow_nan=False),
                   "original_ref": {"snapshot_manifest_sha256": identity["snapshot"]["manifest_sha256"],
                                    "release_path": batch["release_path"],
                                    "release_manifest_sha256": batch["release_manifest_sha256"],
                                    "file": ref["file"], "file_sha256": batch["files"][ref["file"]],
                                    "line": ref["line"], "row_sha256": ref["row_sha256"]},
                   **encoded, "sequence_tokens": sequence, "target_tokens": target}
        except Exception as error:
            raise PreparationError({**diagnosis, "reason": str(error), "error_type": type(error).__name__}, completed) from error
        completed += 1
    inputs.require(not refs and completed == expected_count and completed > 0, "Incomplete or empty snapshot export")


def _measure(partitions, row):
    key = (row["collection"], row["split"])
    count = partitions.setdefault(key, {"collection": key[0], "split": key[1],
        "path": key[0] + "/" + key[1] + ".parquet", "objective": row["objective"],
        "rows": 0, "sequence_tokens": 0, "target_tokens": 0, "max_sequence_tokens": 0})
    count["rows"] += 1
    for name in ("sequence_tokens", "target_tokens"):
        count[name] += row[name]
    count["max_sequence_tokens"] = max(count["max_sequence_tokens"], row["sequence_tokens"])
    return key


def _manifest(out):
    inputs.require(out.is_dir() and not out.is_symlink(), "Use a complete prepared directory")
    seal = inputs.read(inputs.safe_path(out, "READY.json"))
    inputs.require(seal == {"policy": POLICY, "manifest_sha256": inputs.sha(inputs.safe_path(out, "manifest.json")),
                            "training_authorized": False}, "Changed prepared manifest/checksum")
    manifest = inputs.read(out / "manifest.json")
    inputs.require(manifest["policy"] == POLICY and all(manifest.get(k) is v for k, v in FLAGS.items())
                   and manifest["limits"] == LIMITS, "Unsupported prepared policy or authorization claim")
    inputs.require(manifest["identity"]["config"] == _config(manifest["identity"]["config"]["max_length"]),
                   "Changed preparation config")
    partitions = manifest["partitions"]
    inputs.require(isinstance(partitions, list) and bool(partitions), "Missing partitions")
    paths = set()
    for part in partitions:
        collection, split = part["collection"], part["split"]
        inputs.require(collection in inputs.COLLECTIONS and split in inputs.SPLITS, "Unknown partition")
        path = collection + "/" + split + ".parquet"
        inputs.require(part["path"] == path and path not in paths, "Changed or repeated partition path")
        paths.add(path)
    artifacts = manifest["artifacts"]
    inputs.require(set(artifacts) == paths | {"snapshot-manifest.json"}, "Incomplete prepared artifacts")
    actual, directories = set(), set()
    for path in out.rglob("*"):
        inputs.require(not path.is_symlink(), "Symlink prepared artifact")
        name = path.relative_to(out).as_posix()
        if path.is_dir():
            directories.add(name)
        else:
            inputs.require(path.is_file(), "Not a regular prepared artifact")
            actual.add(name)
    inputs.require(actual == set(artifacts) | {"manifest.json", "READY.json"}
                   and directories == {p["collection"] for p in partitions}, "Unexpected or missing prepared files")
    for name, checksum in artifacts.items():
        inputs.require(inputs.valid_hash(checksum) and inputs.sha(inputs.safe_path(out, name)) == checksum,
                       "Changed prepared artifact: " + name)
    snapshot_copy = out / "snapshot-manifest.json"
    inputs.require(inputs.sha(snapshot_copy) == manifest["identity"]["snapshot"]["manifest_sha256"]
                   and inputs.read(snapshot_copy) == manifest["identity"]["snapshot"]["manifest"],
                   "Changed snapshot manifest copy")
    return manifest


def _partition_rows(path):
    with pq.ParquetFile(path) as file:
        inputs.require(file.schema_arrow.equals(SCHEMA), "Changed Parquet schema")
        for batch in file.iter_batches(batch_size=BUFFER_ROWS):
            yield from batch.to_pylist()


def verify_prepared(out: Path, snapshot: Path, assets: Path):
    """Strictly recheck identities, all hashes, every original row and its encoding."""
    snapshot, assets, out = _paths(snapshot, assets, out)
    manifest = _manifest(out)
    identity = _identity(snapshot, assets, manifest["identity"]["config"]["max_length"])
    inputs.require(identity == manifest["identity"], "Changed inputs/config/code/runtime; use a new output version")
    tokenizer = contract.load_tokenizer(assets)
    measured, readers = {}, {}
    with ExitStack() as stack:
        for part in manifest["partitions"]:
            reader = _partition_rows(out / part["path"])
            stack.callback(reader.close)
            readers[(part["collection"], part["split"])] = reader
        for expected in _prepared_rows(snapshot, identity, tokenizer):
            key = _measure(measured, expected)
            inputs.require(key in readers and next(readers[key], None) == expected,
                           "Prepared row differs from original encoding: " + expected["id"])
        inputs.require(all(next(reader, None) is None for reader in readers.values()), "Unexpected extra prepared rows")
    inputs.require([measured[k] for k in sorted(measured)] == manifest["partitions"], "Changed partition counts")
    inputs.require(manifest["rows"] == sum(p["rows"] for p in measured.values())
                   and manifest["rejected_rows"] == 0, "Changed total/coverage")
    inputs.require(_identity(snapshot, assets, identity["config"]["max_length"]) == identity,
                   "Inputs changed during verification")
    inputs.require(_manifest(out) == manifest, "Output changed during verification")
    return manifest


def prepare(snapshot: Path, assets: Path, out: Path, max_length=4096):
    """Export all selected rows, or raise without publishing a partial dataset.

    out names one immutable version. Existing versions are replay-verified before
    reuse; any mismatch requires a different directory, never in-place repair.
    """
    _config(max_length)
    snapshot, assets, out = _paths(snapshot, assets, out)
    identity = _identity(snapshot, assets, max_length)
    if out.exists():
        inputs.require(_manifest(out)["identity"] == identity,
                       "Output already binds different inputs/config/code; use a new output version")
        return verify_prepared(out, snapshot, assets)
    tokenizer = contract.load_tokenizer(assets)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".notebook-data-") as temp:
        staged = Path(temp)
        partitions, writers, buffers = {}, {}, {}
        with ExitStack() as stack:
            for row in _prepared_rows(snapshot, identity, tokenizer):
                key = _measure(partitions, row)
                if key not in writers:
                    path = staged / partitions[key]["path"]
                    path.parent.mkdir(exist_ok=True)
                    writers[key] = stack.enter_context(pq.ParquetWriter(path, SCHEMA, compression="zstd"))
                    buffers[key] = []
                buffers[key].append(row)
                if len(buffers[key]) == BUFFER_ROWS:
                    writers[key].write_table(pa.Table.from_pylist(buffers[key], schema=SCHEMA))
                    buffers[key].clear()
            for key, buffer in buffers.items():
                if buffer:
                    writers[key].write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        (staged / "snapshot-manifest.json").write_bytes((snapshot / "manifest.json").read_bytes())
        inputs.require(_identity(snapshot, assets, max_length) == identity, "Inputs changed during export")
        manifest = {"policy": POLICY, **FLAGS, "limits": LIMITS, "identity": identity,
                    "partitions": [partitions[k] for k in sorted(partitions)],
                    "rows": sum(p["rows"] for p in partitions.values()), "rejected_rows": 0,
                    "artifacts": {p.relative_to(staged).as_posix(): inputs.sha(p)
                                  for p in sorted(staged.rglob("*")) if p.is_file()}}
        inputs.save(staged / "manifest.json", manifest)
        inputs.save(staged / "READY.json", {"policy": POLICY, "manifest_sha256": inputs.sha(staged / "manifest.json"),
                                            "training_authorized": False})
        _manifest(staged)
        # Exclusive reservation cannot replace even an empty preexisting version.
        # READY is linked last; interrupted publication is incomplete and unreusable.
        try:
            out.mkdir()
        except FileExistsError:
            inputs.require(_manifest(out)["identity"] == identity, "Concurrent conflicting preparation")
            return verify_prepared(out, snapshot, assets)
        for name in sorted(manifest["artifacts"]) + ["manifest.json", "READY.json"]:
            dest = out / name
            dest.parent.mkdir(exist_ok=True)
            dest.hardlink_to(staged / name)
    return manifest


def load_partition(out: Path, collection, split):
    """Return one standard Arrow table after input/code/output checksum checks.

    This does not retokenize all partitions; verify_prepared() provides that audit.
    Use datasets.Dataset(table) when the notebook needs a Hugging Face Dataset.
    """
    inputs.require(collection in inputs.COLLECTIONS and split in inputs.SPLITS, "Unknown partition")
    out = Path(out)
    manifest = _manifest(out)
    identity = manifest["identity"]
    snapshot, assets, out = _paths(Path(identity["snapshot"]["path"]), Path(identity["tokenizer"]["path"]), out)
    inputs.require(_identity(snapshot, assets, identity["config"]["max_length"]) == identity,
                   "Changed prepared inputs/config/code/runtime")
    part = next((p for p in manifest["partitions"] if (p["collection"], p["split"]) == (collection, split)), None)
    if part is None:
        raise KeyError((collection, split))
    table = pq.read_table(out / part["path"])
    inputs.require(table.schema.equals(SCHEMA) and table.num_rows == part["rows"], "Changed Parquet partition")
    inputs.require(inputs.sha(out / part["path"]) == manifest["artifacts"][part["path"]], "Partition changed during load")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = (verify_prepared(args.out, args.snapshot, args.assets) if args.verify else
                  prepare(args.snapshot, args.assets, args.out, args.max_length))
    except Exception as error:
        report = error.report if isinstance(error, PreparationError) else {
            "format_ready": False, "training_authorized": False, "error": str(error)}
        print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
