# Understanding corpus candidates

`candidates.v0.jsonl` is a review queue, not a training corpus.

Each row is derived from licensed training-data manifests, project-owned prompt
packs, or existing local recordings and keeps source identity, text, review
status, draft model proposals, consent scope, and training eligibility flags.

Current committed queue:

- 12,235 candidates.
- 2,500 GhanaNLP Twi speech-text rows.
- 2,450 WAXAL Akan rows across train/dev/test manifests.
- 7,000 Ghana Health Symptoms Twi medical rows.
- 12 translated Twi medical QA draft rows.
- 210 curated health, commerce, and code-switch text prompts.
- 35 local recording rows with audio artifact references.
- 9,257 rows currently have model/source draft proposals; these are not gold
  labels until reviewed.
- Current draft coverage by large source:
  - Ghana Health Symptoms: 7,000 / 7,000
  - GhanaNLP speech: 1,852 / 2,500
  - WAXAL: 120 / 2,450

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

## Agentic synthesis

`model_proposal` remains the immutable first/source proposal. The synthesis
pipeline writes a separate, versioned sidecar so model output never overwrites
source evidence:

```bash
pnpm corpus:understanding:synthesize -- \
  --source ghana_health_symptoms \
  --max-new 20 \
  --chunk-size 1
```

For each row, the pipeline attempts three independently sourced alternatives:

- the source annotation;
- a translation-focused proposal;
- a semantic/entity-focused proposal.

An adjudicator selects one or creates a synthesis. Deterministic validators
score completeness, source alignment, Twi preservation, intent ontology,
entity structure, cross-agent agreement, and response grounding. Only
source-backed rows may receive `reply_twi` or `safety_level` values.

The output checkpoints after every chunk at
`tmp/understanding-corpus/synthesis.v1.jsonl`. Runs are resumable and skip rows
already produced by the current prompt version. After inspecting a batch,
validate source hashes and promote only the current synthesis version with:

```bash
pnpm corpus:understanding:synthesis:promote
```

That creates `data/understanding-corpus/synthesis.v1.jsonl`, which powers the
annotation selector in the chat review card. Do not promote a run that failed
semantic spot checks. The saved review retains the selected proposal ID and
synthesis version through the training export.

Current open-model synthesis finding (`v9`):

- AfriqueQwen 9B, NLLB Twi-English, and a Qwen 7B adjudicator all run on Modal
  with source hashes and private raw artifacts.
- The 20-row stratified `v7` checkpoint parsed all three stages, but only 3/20
  rows cleared the conservative gate.
- AfriqueQwen and NLLB both made material Twi meaning errors. The adjudicator
  also failed the known `aduro bi` (unspecified medicine) versus `herbal
  treatment` conflict after prompt hardening.
- These open-model proposals are therefore audit evidence only. They are not
  trusted labels and were not scaled across the 7,000 rows.
- The parser now rejects prompt echoes and schema placeholders as incomplete
  model output.

Re-run deterministic scoring without spending model credits:

```bash
pnpm corpus:understanding:synthesize -- --rescore-only
pnpm eval:understanding:synthesis
```

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

For a rate-limit-resistant import from the Hugging Face parquet shard:

```bash
python scripts/import_ghana_health_symptoms_parquet.py --limit 7000
```

Import patient-facing MIT English medical QA and translate a review batch to Twi:

```bash
pnpm corpus:medical:qa -- --limit 200
pnpm corpus:medical:qa:twi -- --limit 50 --chunk-size 2
```

Then rebuild a larger candidate queue:

```bash
pnpm corpus:understanding:candidates -- --limit 10000
cp tmp/understanding-corpus/candidates.v0.jsonl data/understanding-corpus/candidates.v0.jsonl
```

## Delivered silver corpus

The current source-grounded research corpus is:

```text
data/understanding-corpus/silver-medical-paired-v2/
```

It contains exactly the 7,000 source-paired Ghana Health Symptoms rows:

- `train.jsonl`: 5,659 rows
- `dev.jsonl`: 669 rows
- `test.jsonl`: 672 rows
- zero duplicate IDs or normalized utterances
- 17 body-system categories
- no WAXAL, GhanaNLP speech, local recordings, generated prompt seeds,
  translated QA pilots, or product-failure fixtures
- original punctuated Twi preserved as `normalized_twi`
- source URLs, licence notes, and body-system metadata removed from the model's
  ambiguity target

Every row is marked `quality_tier=paired_source_silver` and
`verification_status=source_paired_unreviewed`. This is a real research training
lane, but it is not human gold and must not be represented as such.

Regenerate and validate it with:

```bash
pnpm corpus:understanding:silver -- \
  --out-dir data/understanding-corpus/silver-medical-paired-v2
pnpm eval:understanding:silver
```

The validator requires exactly 7,000 rows, complete split coverage, no duplicate
IDs/text, valid intents and entities, matching chat targets, 10+ body-system
categories, no split leakage, and no metadata contamination.

The next adapter uses `Qwen/Qwen2.5-3B-Instruct` rather than the previous 1.5B
base and pushes to `teckedd/gha-understand-twi-medical-v4` with source, licence,
row counts, training loss, and held-out development loss in its model card:

```bash
pnpm train:understanding:modal
```

The full run completed in Modal app `ap-1nIwJgPPwgMaQbsmjlPtJK` at 1,200 steps:

- final training loss: `0.7776`
- final development loss: `0.6560`
- public adapter: `teckedd/gha-understand-twi-medical-v4`

The complete 672-row held-out test ran in Modal app
`ap-VBCFWCEnygrgqJbgxzdYma`:

- parseable JSON: `672/672` (`100%`)
- exact intent: `672/672` (`100%`), but every row has the same
  `health_symptom_report` label, so this does not measure intent generalization
- exact body system: `259/672` (`38.54%`)
- strict semantic pass: `79/672` (`11.76%`)
- mean natural-English token F1: `0.3425`

The corrected 11-row product fixture evaluation ran in Modal app
`ap-F2cPX9VDMcSyGdMtreG4GU`:

- overall: `1/11`
- health: `1/6`
- commerce: `0/5`

The unadapted base comparison ran against the same frozen inputs:

- product fixtures: `0/11`
- held-out parseable JSON: `278/672` (`41.37%`)
- held-out strict semantic pass: `0/672`
- held-out mean natural-English token F1: `0.0295`
- Modal product app: `ap-p8bh0KAnUk0kWoF0PWDqgm`
- Modal held-out app: `ap-SfY0cEY1VOLbvqTwJfGWXy`

The adapter therefore produced a measurable medical-domain improvement over
the base model, but the absolute quality remains unusable. Its schema gains
must not be confused with robust semantics.

**Decision: do not promote or route product traffic to v4.** It learned the
output schema and the repeated medical label, but it did not learn reliable
Twi semantics. It frequently hallucinated common symptom patterns and converted
commerce requests into medical complaints. More steps on this corpus are not a
valid next experiment.

This remains an understanding checkpoint; the 7,000 symptom rows do not contain
grounded Twi answers, so the model must not be described as response-capable.
The next corpus must add varied health and commerce intents, faithful semantic
targets, real product-failure paraphrases, context turns, and a separately
reviewed patient-response lane. Keep the same frozen base-versus-adapter test
design for the next promotion decision.

Previous v3 remains at `teckedd/gha-understand-twi-medical-plus-language-v3`.
It passed 7/11 product fixtures but was not promoted because it still
misunderstood `mani kum paa` and hospital-choice phrasing.

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
