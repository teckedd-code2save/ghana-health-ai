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
counts, and grouped rows. It should show `ready=false` until at least one
corpus candidate has been accepted through review.

If the local database is not running and you intentionally want to export from
the local JSONL fallback, use:

```bash
node --import tsx scripts/export-understanding-training-corpus.ts --review-source=file
```

In the review UI, use **Accept and next** only after checking/correcting the
normalized Twi, faithful English meaning, and intent. Use **Second review** for
ambiguous health/commerce rows and **Exclude** for bad, duplicate, or unusable
records.
