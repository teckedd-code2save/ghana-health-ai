# ML systems experiment 01 — JSONL vs Arrow/Parquet corpus data plane

Date: 2026-09-19

## Question

For Ghana Health AI's real Twi understanding corpus, which operations benefit from a
partitioned Parquet/Arrow view compared with the current JSONL representation, and
what does it cost in memory, CPU, file size, and preprocessing complexity?

This is **not** a benchmark designed to prove that Parquet is faster.

## Initial workload

Use the committed 7,000-row source-paired silver corpus first:

```text
data/understanding-corpus/silver-medical-paired-v2/all.jsonl
```

It is useful because its source/split policy is already documented and its held-out
test set is protected from training.

After the harness is validated, repeat it against a larger source-backed alignment
release without changing the measurement contract.

## Representation

The derived data plane:
- preserves source IDs and the existing split;
- partitions by `split/source`;
- uses Zstandard-compressed Parquet;
- serializes nested entities/messages explicitly as JSON strings;
- writes a release manifest with source SHA-256, row counts, schema, partition
  policy, file sizes and per-file SHA-256 checksums.

Parquet is a derived view. JSONL remains the source of truth for this experiment.

## Workloads

### 1. Metadata scan / filter

Question: how expensive is it to find the training rows for one source/split without
materializing full training examples?

Measure:
- wall time;
- CPU time;
- peak RSS;
- rows/sec;
- result checksum.

### 2. Training-subset load

Load `id + normalized_twi + natural_english + literal_english` for the same split
and source.

Measure the same resource metrics and verify both representations produce the same
logical result checksum.

### 3. Batched tokenization

Use the same pinned tokenizer for both formats.

Default:
- `Qwen/Qwen3.5-4B`
- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

Measure:
- rows;
- token count;
- tokens/sec;
- CPU;
- peak RSS;
- token-ID checksum.

This tells us whether columnar/batched reads help the transformation path or whether
tokenization dominates enough that file format barely matters.

## Run

```bash
python -m venv .venv-data-plane
source .venv-data-plane/bin/activate
pip install -r scripts/requirements-ml-systems.txt

pnpm ml:data-plane:build
pnpm ml:data-plane:test
pnpm ml:data-plane:benchmark
```

The commands default to `tmp/ml-systems/data-plane/` and do not mutate the source corpus.

## Exit evidence

Commit or preserve:
- `manifest.json`
- benchmark `.json`
- benchmark `.csv`
- environment/hardware description
- 300–600 word interpretation
- regressions as well as improvements

## Interpretation questions

1. Did partition pruning materially improve metadata/filter scans?
2. Did Parquet lower source bytes enough to matter operationally?
3. Did converting Arrow batches to Python strings erase I/O gains during tokenization?
4. Is peak RSS better, worse, or simply shifted into Arrow/native allocations?
5. Is `split/source` the right partition design, or does it create too many tiny files on the larger corpus?
6. Would row-group size matter more than file format for our training reads?
7. What should become the canonical training release format after measurement?

## CV-worthy outcome only after measurement

Do not write:
> Migrated Ghana Health to Parquet for faster training.

Possible future evidence if supported:
> Built and benchmarked a versioned Arrow/Parquet data plane for the Twi corpus,
> reducing train-split scan cost by X while preserving source/split provenance and
> identical tokenized inputs; documented where tokenization remained the dominant
> bottleneck.

If Parquet loses on a workload, that is part of the result.
