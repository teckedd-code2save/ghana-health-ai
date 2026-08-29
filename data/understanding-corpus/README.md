# Understanding corpus candidates

`candidates.v0.jsonl` is a review queue, not a training corpus.

Each row is derived from project-owned prompt packs or existing local recordings
and keeps source identity, text, review status, draft model proposals, consent
scope, and training eligibility flags.

Important boundaries:

- `model_proposal` fields are drafts for human correction.
- `eligible_for_training=false` until reviewed and exported through a split-safe
  process.
- `eligible_for_final_evaluation=false`; these rows must not become promotion
  evidence without frozen split policy.
- `local-research://...` values are portable artifact identifiers, not embedded
  audio.

Next step: review rows in `/research/ase`, correct the Twi/English meaning and
semantic fields, then export only reviewed rows into a versioned training or
evaluation manifest.

## Training export gate

Use:

```bash
pnpm corpus:understanding:export
```

This writes split manifests under `tmp/understanding-corpus/exports/v0/`.
Rows are exported only when a saved review has:

- `decision=reviewed`
- non-empty normalized Twi
- non-empty faithful English meaning
- non-empty intent

Use the strict gate before training:

```bash
pnpm corpus:understanding:export:strict
```

Strict mode fails when no rows are eligible. That is intentional: model drafts
and unreviewed corpus candidates are not training data.

The deployed review workspace also exposes the same gate at:

```text
/api/research/understanding/export
```

That endpoint returns the current accepted row count, train/dev/test split
counts, readiness checks, and grouped rows. It should show `ready=false` until
the required checks pass:

- at least 20 reviewed rows
- non-empty train, dev, and test splits
- health-domain coverage
- no duplicate meaning keys in the export
- consent scope on every row

Commerce-domain coverage is tracked as a warning for the next product lane,
not as the first health-training blocker.

If the local database is not running and you intentionally want to export from
the local JSONL fallback, use:

```bash
node --import tsx scripts/export-understanding-training-corpus.ts --review-source=file
```

In the review UI, use **Accept and next** only after checking/correcting the
normalized Twi, faithful English meaning, and intent. Use **Second review** for
ambiguous health/commerce rows and **Exclude** for bad, duplicate, or unusable
records.

## Bulk review sheet

For spreadsheet review, download the sheet from the workbench or run:

```bash
pnpm corpus:understanding:review-sheet
```

For the fastest first pass toward a trainable corpus, download **Download 20-row
training pack** in the workbench or run:

```bash
pnpm corpus:understanding:review-sheet -- --scope minimum-training --out tmp/understanding-corpus/minimum-training-review.v0.csv
```

To prefill the review columns from the model draft and only correct the fields
that are wrong, add `--prefill draft`:

```bash
pnpm corpus:understanding:review-sheet -- --scope minimum-training --prefill draft --out tmp/understanding-corpus/minimum-training-assisted.v0.csv
```

Rows still import as training data only when `decision` is changed to
`reviewed`. Leaving a row as `unreviewed` skips it, even if draft text is
present in the review columns.

That pack is selected to cover the minimum row count, train/dev/test splits,
health rows, and commerce rows before falling back to the rest of the queue.

Fill only the `review_*`, `decision`, `review_notes`, and `reviewer` columns.
Keep `id`, `proposed_split`, source, consent, and draft columns unchanged so
provenance and split assignment remain stable. To pass the training gate, the
accepted rows must include at least one `train`, one `dev`, and one `test` row;
use `proposed_split` to choose the first review batch intentionally. In
production, upload the corrected CSV through the workbench with **Upload
reviewed CSV**; saved rows go to Postgres and immediately update the readiness
gate.

To import a completed local sheet into the JSONL review fallback:

```bash
pnpm corpus:understanding:import-review-sheet -- --input tmp/understanding-corpus/review-sheet.v0.csv
```

After import, run:

```bash
pnpm corpus:understanding:export:strict
```

If the gate fails, use the readiness checks in the output to decide which rows
or splits still need review.
