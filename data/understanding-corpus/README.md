# Understanding corpus candidates

`candidates.v0.jsonl` is a review queue, not a training corpus.

Each row is derived from licensed training-data manifests, project-owned prompt
packs, or existing local recordings and keeps source identity, text, review
status, draft model proposals, consent scope, and training eligibility flags.

Current committed queue:

- 6,811 candidates.
- 2,500 GhanaNLP Twi speech-text rows.
- 2,450 WAXAL Akan rows across train/dev/test manifests.
- 1,600 Ghana Health Symptoms Twi medical rows.
- 210 curated health, commerce, and code-switch text prompts.
- 35 local recording rows with audio artifact references.
- 1,922 rows currently have model/source draft proposals; these are not gold
  labels until reviewed.

The 50-row benchmark under `data/understanding-benchmark/` is only for comparing
candidate annotators. It is not the training corpus.

The 16-row medical response seed under `data/medical-response-corpus/` is also
not the corpus. It is only a schema/provenance smoke test for patient-facing
medical answer rows.

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

To rebuild the large candidate queue from the available local training-data
manifests:

```bash
pnpm corpus:understanding:candidates
cp tmp/understanding-corpus/candidates.v0.jsonl data/understanding-corpus/candidates.v0.jsonl
```

The builder defaults to a 5,000-row queue. Use `--limit` for a smaller or larger
queue, and use `--annotate --annotate-limit <n>` only when intentionally spending
model credits on a bounded proposal batch.

## Corpus-scale draft annotation

The real annotation target is the large candidate queue from WAXAL, GhanaNLP,
project-owned recordings/prompts, and approved external medical sources.

Use the standalone annotator to spend model credits against the existing queue
without rebuilding it:

```bash
pnpm corpus:understanding:annotate -- --source ghana_nlp_speech,waxal --max-new 500 --chunk-size 8 --in-place
```

This fills `model_proposal` only. It does not mark rows as reviewed or eligible
for training. Human review is still the promotion gate.

Useful variants:

```bash
# Annotate the next batch of WAXAL/GhanaNLP rows.
pnpm corpus:understanding:annotate -- --source ghana_nlp_speech,waxal --max-new 1000 --chunk-size 8 --in-place

# Annotate imported Twi health-symptom rows.
pnpm corpus:understanding:annotate -- --source ghana_health_symptoms --max-new 1000 --chunk-size 8 --in-place

# Write an annotated copy instead of mutating the committed queue.
pnpm corpus:understanding:annotate -- --source waxal --max-new 100 --out tmp/understanding-corpus/waxal.annotated.v0.jsonl
```

## External medical sources

The current source inventory is tracked at:

```text
data/medical-response-corpus/source-inventory.v0.json
```

The first large Twi health source is:

```text
ghananlpcommunity/ghana-health-symptoms
```

It contains about 98k Twi symptom descriptions with English triage tags, but it
is `cc-by-nc-4.0`, so keep it as non-commercial research data unless permission
or legal approval changes that status.

Import a bounded local copy with:

```bash
pnpm corpus:medical:ghana-health-symptoms -- --limit 5000
```

Then rebuild a larger candidate queue:

```bash
pnpm corpus:understanding:candidates -- --limit 10000
cp tmp/understanding-corpus/candidates.v0.jsonl data/understanding-corpus/candidates.v0.jsonl
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
