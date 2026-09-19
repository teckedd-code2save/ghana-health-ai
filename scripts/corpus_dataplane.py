#!/usr/bin/env python3
"""Build and benchmark a versioned Arrow/Parquet view of Ghana Health AI corpus JSONL.

The benchmark intentionally measures real training-adjacent operations:
1. metadata scan/filter;
2. train-subset loading;
3. batched tokenization with the same pinned tokenizer.

It does not claim Parquet is globally faster. Results are workload-specific.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1
PARTITION_COLUMNS = ("split", "source")
TEXT_COLUMNS = ("normalized_twi", "natural_english", "literal_english")
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
DEFAULT_TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

STRING_FIELDS = (
    "id",
    "split",
    "domain",
    "language",
    "source",
    "source_record_id",
    "consent_scope",
    "training_lane",
    "quality_tier",
    "verification_status",
    "license_policy",
    "original_text",
    "normalized_twi",
    "natural_english",
    "literal_english",
    "intent",
    "ambiguities",
    "label_source",
)
JSON_FIELDS = ("entities", "messages")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_partition(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value or "unknown"


def normalize_row(raw: dict) -> dict:
    row = {field: str(raw.get(field) or "") for field in STRING_FIELDS}
    for field in JSON_FIELDS:
        row[field + "_json"] = json.dumps(raw.get(field), ensure_ascii=False, separators=(",", ":"))
    row["requires_clarification"] = bool(raw.get("requires_clarification", False))
    return row


def iter_jsonl(path: Path, limit: int = 0) -> Iterator[dict]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                return


def build_schema():
    import pyarrow as pa

    fields = [pa.field(name, pa.string(), nullable=False) for name in STRING_FIELDS]
    fields += [pa.field(field + "_json", pa.string(), nullable=False) for field in JSON_FIELDS]
    fields.append(pa.field("requires_clarification", pa.bool_(), nullable=False))
    return pa.schema(fields)


def rows_to_table(rows: list[dict], schema):
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=schema)


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("split=*/source=*/*.parquet"))


def build_dataset(input_path: Path, out_dir: Path, batch_size: int, limit: int = 0) -> dict:
    import pyarrow.parquet as pq

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = build_schema()
    writers = {}
    partition_rows = defaultdict(int)
    source_counts = defaultdict(int)
    split_counts = defaultdict(int)
    pending = defaultdict(list)
    total = 0

    def flush(key):
        nonlocal writers
        rows = pending[key]
        if not rows:
            return
        split, source = key
        partition = out_dir / f"split={safe_partition(split)}" / f"source={safe_partition(source)}"
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / "part-00000.parquet"
        writer = writers.get(key)
        if writer is None:
            writer = pq.ParquetWriter(
                path,
                schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                data_page_size=1024 * 1024,
            )
            writers[key] = writer
        writer.write_table(rows_to_table(rows, schema), row_group_size=batch_size)
        partition_rows[key] += len(rows)
        pending[key] = []

    try:
        for raw in iter_jsonl(input_path, limit=limit):
            row = normalize_row(raw)
            split = row["split"] or "unknown"
            source = row["source"] or "unknown"
            key = (split, source)
            pending[key].append(row)
            source_counts[source] += 1
            split_counts[split] += 1
            total += 1
            if len(pending[key]) >= batch_size:
                flush(key)
        for key in list(pending):
            flush(key)
    finally:
        for writer in writers.values():
            writer.close()

    files = []
    for path in parquet_files(out_dir):
        meta = pq.ParquetFile(path).metadata
        files.append({
            "path": str(path.relative_to(out_dir)),
            "bytes": path.stat().st_size,
            "rows": meta.num_rows,
            "row_groups": meta.num_row_groups,
            "sha256": sha256_file(path),
        })

    source_sha = sha256_file(input_path)
    build_material = json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha,
        "partition_columns": PARTITION_COLUMNS,
        "rows": total,
    }, sort_keys=True).encode()
    build_id = hashlib.sha256(build_material).hexdigest()[:16]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": source_sha,
        },
        "format": {
            "type": "parquet",
            "compression": "zstd",
            "partition_columns": list(PARTITION_COLUMNS),
            "batch_size": batch_size,
        },
        "rows": total,
        "splits": dict(sorted(split_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "schema": [{"name": field.name, "type": str(field.type)} for field in schema],
        "files": files,
        "limitations": [
            "This release is a derived view of the JSONL source, not a new corpus.",
            "Partitioning is selected for current training/evaluation access patterns and may change after measurement.",
            "File-size reduction or faster scans do not imply faster end-to-end training.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


class RssSampler:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        import psutil
        process = psutil.Process(os.getpid())

        def run():
            while not self._stop.is_set():
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


@contextmanager
def measured():
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    with RssSampler() as rss:
        yield_value = {}
        yield yield_value
    yield_value.update({
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": rss.peak,
    })


def jsonl_filter_rows(path: Path, split: str, source: str | None = None) -> Iterator[dict]:
    for row in iter_jsonl(path):
        if split and row.get("split") != split:
            continue
        if source and row.get("source") != source:
            continue
        yield row


def stable_values_sha(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def metadata_scan_jsonl(path: Path, split: str, source: str | None) -> dict:
    ids = []
    with measured() as metrics:
        for row in jsonl_filter_rows(path, split, source):
            ids.append(str(row.get("id") or ""))
    return {**metrics, "rows": len(ids), "result_sha256": stable_values_sha(ids)}


def metadata_scan_parquet(dataset_dir: Path, split: str, source: str | None) -> dict:
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive", exclude_invalid_files=True)
    filt = ds.field("split") == split
    if source:
        filt = filt & (ds.field("source") == source)
    ids = []
    with measured() as metrics:
        scanner = dataset.scanner(columns=["id"], filter=filt, batch_size=4096)
        for batch in scanner.to_batches():
            ids.extend(str(value or "") for value in batch.column(0).to_pylist())
    return {**metrics, "rows": len(ids), "result_sha256": stable_values_sha(ids)}


def load_subset_jsonl(path: Path, split: str, source: str | None) -> dict:
    rows = []
    with measured() as metrics:
        for row in jsonl_filter_rows(path, split, source):
            rows.append(tuple(str(row.get(col) or "") for col in ("id", *TEXT_COLUMNS)))
    rows.sort(key=lambda row: row[0])
    rows.sort(key=lambda row: row[0])
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {**metrics, "rows": len(rows), "result_sha256": digest}


def load_subset_parquet(dataset_dir: Path, split: str, source: str | None) -> dict:
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive", exclude_invalid_files=True)
    filt = ds.field("split") == split
    if source:
        filt = filt & (ds.field("source") == source)
    rows = []
    with measured() as metrics:
        table = dataset.scanner(columns=["id", *TEXT_COLUMNS], filter=filt, batch_size=4096).to_table()
        cols = [table.column(name).to_pylist() for name in ["id", *TEXT_COLUMNS]]
        rows = list(zip(*cols))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {**metrics, "rows": len(rows), "result_sha256": digest}


def load_tokenizer(repo: str, revision: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(repo, "tokenizer.json", revision=revision, token=False)
    return Tokenizer.from_file(path), path


def tokenize_batches(batches: Iterable[list[tuple[str, str]]], tokenizer) -> tuple[int, int, str]:
    rows = 0
    tokens = 0
    row_digests = []
    for batch in batches:
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        rows += len(encoded)
        for row_id, item in zip(ids, encoded):
            tokens += len(item.ids)
            material = row_id + ":" + ",".join(map(str, item.ids))
            row_digests.append(hashlib.sha256(material.encode()).hexdigest())
    return rows, tokens, stable_values_sha(row_digests)


def jsonl_text_batches(path: Path, split: str, source: str | None, batch_size: int) -> Iterator[list[tuple[str, str]]]:
    batch = []
    for row in jsonl_filter_rows(path, split, source):
        batch.append((
            str(row.get("id") or ""),
            str(row.get("normalized_twi") or row.get("original_text") or ""),
        ))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parquet_text_batches(dataset_dir: Path, split: str, source: str | None, batch_size: int) -> Iterator[list[tuple[str, str]]]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive", exclude_invalid_files=True)
    filt = ds.field("split") == split
    if source:
        filt = filt & (ds.field("source") == source)
    scanner = dataset.scanner(columns=["id", "normalized_twi"], filter=filt, batch_size=batch_size)
    for batch in scanner.to_batches():
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield [(str(row_id or ""), str(text or "")) for row_id, text in zip(ids, texts)]


def benchmark_tokenization(
    input_path: Path,
    dataset_dir: Path,
    split: str,
    source: str | None,
    batch_size: int,
    tokenizer_repo: str,
    tokenizer_revision: str,
) -> tuple[dict, dict, dict]:
    tokenizer, tokenizer_path = load_tokenizer(tokenizer_repo, tokenizer_revision)
    tokenizer_sha = sha256_file(Path(tokenizer_path))

    with measured() as json_metrics:
        j_rows, j_tokens, j_sha = tokenize_batches(
            jsonl_text_batches(input_path, split, source, batch_size), tokenizer
        )
    json_result = {
        **json_metrics,
        "rows": j_rows,
        "tokens": j_tokens,
        "tokens_per_second": j_tokens / json_metrics["wall_seconds"] if json_metrics["wall_seconds"] else None,
        "result_sha256": j_sha,
    }

    with measured() as pq_metrics:
        p_rows, p_tokens, p_sha = tokenize_batches(
            parquet_text_batches(dataset_dir, split, source, batch_size), tokenizer
        )
    parquet_result = {
        **pq_metrics,
        "rows": p_rows,
        "tokens": p_tokens,
        "tokens_per_second": p_tokens / pq_metrics["wall_seconds"] if pq_metrics["wall_seconds"] else None,
        "result_sha256": p_sha,
    }

    tokenizer_info = {
        "repo": tokenizer_repo,
        "revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha,
        "batch_size": batch_size,
    }
    return json_result, parquet_result, tokenizer_info


def add_rates(result: dict):
    wall = result.get("wall_seconds") or 0
    if wall and result.get("rows") is not None:
        result["rows_per_second"] = result["rows"] / wall


def compare(json_result: dict, parquet_result: dict) -> dict:
    same_rows = json_result.get("rows") == parquet_result.get("rows")
    same_result = json_result.get("result_sha256") == parquet_result.get("result_sha256")
    j = json_result.get("wall_seconds") or 0
    p = parquet_result.get("wall_seconds") or 0
    return {
        "same_rows": same_rows,
        "same_result_sha256": same_result,
        "parquet_vs_jsonl_wall_ratio": (p / j) if j else None,
        "speedup_jsonl_over_parquet": (j / p) if p else None,
    }


def run_benchmark(
    input_path: Path,
    dataset_dir: Path,
    output: Path,
    split: str,
    source: str | None,
    tokenizer_repo: str,
    tokenizer_revision: str,
    token_batch_size: int,
) -> dict:
    cases = {}

    j = metadata_scan_jsonl(input_path, split, source)
    p = metadata_scan_parquet(dataset_dir, split, source)
    add_rates(j); add_rates(p)
    cases["metadata_scan_filter"] = {"jsonl": j, "parquet": p, "comparison": compare(j, p)}

    j = load_subset_jsonl(input_path, split, source)
    p = load_subset_parquet(dataset_dir, split, source)
    add_rates(j); add_rates(p)
    cases["training_subset_load"] = {"jsonl": j, "parquet": p, "comparison": compare(j, p)}

    j, p, tokenizer = benchmark_tokenization(
        input_path, dataset_dir, split, source, token_batch_size, tokenizer_repo, tokenizer_revision
    )
    add_rates(j); add_rates(p)
    cases["batched_tokenization"] = {
        "jsonl": j,
        "parquet": p,
        "comparison": compare(j, p),
        "tokenizer": tokenizer,
    }

    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {"split": split, "source": source},
        "source_jsonl": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "parquet_release": {
            "path": str(dataset_dir),
            "build_id": manifest["build_id"],
            "bytes": sum(item["bytes"] for item in manifest["files"]),
            "files": len(manifest["files"]),
        },
        "cases": cases,
        "environment": {
            "python": os.sys.version,
            "platform": os.uname().sysname + " " + os.uname().release,
            "max_rss_note": "peak RSS is sampled for the whole benchmark process; native Arrow allocations are included.",
        },
        "interpretation_guardrail": (
            "Report each workload independently. A smaller file or faster metadata scan does not imply faster "
            "tokenization/training, and Parquet regressions should be retained."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case", "format", "rows", "wall_seconds", "cpu_seconds", "peak_rss_bytes",
            "rows_per_second", "tokens", "tokens_per_second",
        ])
        writer.writeheader()
        for case_name, case in cases.items():
            for fmt in ("jsonl", "parquet"):
                row = case[fmt]
                writer.writerow({
                    "case": case_name,
                    "format": fmt,
                    "rows": row.get("rows"),
                    "wall_seconds": row.get("wall_seconds"),
                    "cpu_seconds": row.get("cpu_seconds"),
                    "peak_rss_bytes": row.get("peak_rss_bytes"),
                    "rows_per_second": row.get("rows_per_second"),
                    "tokens": row.get("tokens"),
                    "tokens_per_second": row.get("tokens_per_second"),
                })

    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Create a versioned partitioned Parquet view and manifest.")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=1024)
    build.add_argument("--limit", type=int, default=0)

    bench = sub.add_parser("benchmark", help="Benchmark JSONL vs Parquet on identical workloads.")
    bench.add_argument("--input", type=Path, required=True)
    bench.add_argument("--dataset", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--split", default="train")
    bench.add_argument("--source")
    bench.add_argument("--token-batch-size", type=int, default=256)
    bench.add_argument("--tokenizer-repo", default=DEFAULT_TOKENIZER_REPO)
    bench.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)

    args = parser.parse_args()
    if args.command == "build":
        report = build_dataset(args.input, args.out, args.batch_size, args.limit)
    else:
        report = run_benchmark(
            args.input,
            args.dataset,
            args.output,
            args.split,
            args.source,
            args.tokenizer_repo,
            args.tokenizer_revision,
            args.token_batch_size,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
