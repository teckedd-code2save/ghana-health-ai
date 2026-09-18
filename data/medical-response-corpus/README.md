# Medical Twi response corpus

This folder contains source-linked medical voice-response rows for review.

## Current execution status (2026-09-07)

The full Twi training-source annotation run was launched and stopped on the
provider's `credit_balance_exhausted` response. Its saved v3 file contains 305
complete rows: 120 model-consensus candidates and 185 needing review. There are
4,102 Twi rows still pending. Consensus is not a passed semantic/clinical gate.
All 28 selected rows failing the question-entity span audit were routed to
review, with none admitted to consensus. Strict training export remains closed.

The local dataset review reader now prefers `afrihealth-teacher-v3-train.jsonl`.
This change is not deployed. Original pilot/reference files remain unchanged.
The English and WAXAL/GhanaNLP preparation is documented in
`../annotated-source-corpus/README.md`; it adds 8,451 source-training rows, not
8,451 completed annotations. No synthetic seeds are used to fill that count.

Stable request plans and raw request checkpoints preserve paid work on resume:

```bash
pnpm corpus:medical:afrihealth:annotate --train-only --limit 0 --chunk-size 4 \
  --out data/medical-response-corpus/afrihealth-teacher-v3-train.jsonl --dry-run
```

## AfriHealth Akan source

`afrihealth-akan-source.v1.jsonl` is a pinned import of the 5,569 answered
`Aka_Gha` rows in the upstream AfriHealth-QA train and validation files. The
question and answer are immutable source evidence, not verified labels. Every
row is explicitly ineligible for training until the separate annotation layer
passes translation, semantic, response-grounding, and safety gates.

The original challenge terms are CC-BY-SA 4.0, which is stricter than the
current Hugging Face repository metadata. The importer records the original
terms and attribution on every row and excludes the unlabeled test holdout.

Reproduce and audit the import:

```bash
pnpm corpus:medical:afrihealth:import --no-download
```

## Bilingual response training corpus

`afrihealth-ghana-response-train.v1.jsonl` contains the deterministic-pass
source-train rows for direct health response SFT: 4,407 Twi and 4,402 Ghanaian
English rows. Keeping the two languages balanced is deliberate protection
against the English regression seen in earlier Twi-only fine-tunes.

`afrihealth-ghana-response-eval.v1.jsonl` locks 1,102 Twi and 1,096 English
validation rows out of training. `afrihealth-ghana-response-review.v1.jsonl`
contains the 109 records isolated for short or malformed questions, truncation,
repetition, links, non-Ghana care assumptions, unbalanced delimiters, or
unverified emergency numbers. No model-generated translation is needed for
these direct question-to-answer examples.

Rebuild all three artifacts from the pinned upstream files:

```bash
pnpm corpus:medical:afrihealth:bilingual -- --no-download
```

These rows are eligible for research training only. The source dataset describes
its answers as clinical consensus, but the project has not medically reviewed
every answer. Consequently every row remains `eligible_for_production_training:
false` and `eligible_for_final_evaluation: false`.

The response adapter must be compared with its untouched base before it can be
published or exposed in the app. The gate uses a balanced hash-selected sample
from the locked validation artifact plus product failures in
`response-product-eval.v1.jsonl`. It checks reference similarity separately for
Twi and English, response language, repetition, forbidden claims, multi-turn
follow-ups, and critical health behavior without using a proprietary LLM judge:

```bash
pnpm eval:response:afrihealth:modal
```

The first full run is recorded in
`afrihealth-response-run.v1.summary.json`. It improved English retention and
response-language matching, but reduced Twi reference similarity and passed
none of the eight critical product cases. It is therefore retained as a failed
research checkpoint and must not be pushed or served.

Run a deterministic, hash-sampled annotation pilot through two independent
teachers and a separate adjudicator:

```bash
sec -- pnpm corpus:medical:afrihealth:annotate --limit 100 --chunk-size 2 --concurrency 4 \
  --out data/medical-response-corpus/afrihealth-teacher-v3-pilot.jsonl \
  --summary data/medical-response-corpus/afrihealth-teacher-v3-pilot.summary.json
```

Each annotation retains two selectable proposals and, when needed, a third
adjudicated synthesis. Only rows with high-confidence agreement, source-grounded
evidence, no unsupported numeric claims, and no unresolved source-answer or
safety concern are marked `silver_consensus`. Upstream validation rows remain
ineligible for final evaluation until a human verifies them.

The open proposal path can be checked on real reference rows with:

```bash
pnpm corpus:medical:afrihealth:annotate:pivot -- --reference-only --limit 4
modal volume get --force ghana-health-understanding-results \
  /response-annotations/pivot-v1/latest.raw.jsonl \
  tmp/understanding-corpus/afrihealth-pivot-reference.raw.jsonl
pnpm corpus:medical:afrihealth:import-modal -- \
  --input tmp/understanding-corpus/afrihealth-pivot-reference.raw.jsonl \
  --out tmp/understanding-corpus/afrihealth-pivot-reference.imported.jsonl \
  --summary tmp/understanding-corpus/afrihealth-pivot-reference.imported.summary.json
pnpm eval:medical:afrihealth:annotation-calibration -- \
  --candidate tmp/understanding-corpus/afrihealth-pivot-reference.imported.jsonl
```

The current Qwen3.5-35B plus NLLB pivot fails that calibration and must not be
run over the full corpus as accepted labels. On 58 comparable real rows it
reached 56.9% intent agreement, 72.4% safety agreement, and only 55.0% review
capture against the stronger model-assisted reference. The reference itself is
not human gold. The runner keeps complete raw attempts and retries malformed
JSON once so proposal artifacts remain auditable.

The versioned rejection record is
`afrihealth-annotation-calibration.v1.summary.json`.

## Annotated training export

The new source records have a dedicated export instead of passing through the
older seed/candidate exporter:

```bash
pnpm corpus:medical:afrihealth:export-annotated -- \
  --annotations data/medical-response-corpus/afrihealth-akan-annotations.v1.jsonl \
  --reviews tmp/understanding-review/reviews.v0.jsonl \
  --calibration data/medical-response-corpus/afrihealth-annotation-calibration.v1.summary.json \
  --strict
```

Use `--reviews-db` through the normal secrets-backed environment for live
Postgres decisions; database errors are not silently treated as zero reviews.
Exports appear in `tmp/afrihealth-annotated-corpus/v1/`, with per-row exclusions
and checksums. Reviewer exclusions override model recommendations. Silver
requires a passing report for the same annotation pipeline. Source hashes,
locked splits, selected proposal identity, and review version are checked.

Twi/English translated views retain the same source identity and split. The
manifest counts unique sources separately from training examples; paired
views cannot inflate the 7,000-unique-source readiness target. English replies
whose Twi text was changed by a reviewer wait for an updated translation.

The current corpus does not pass strict export. The 62 reference annotations
are model-assisted calibration material, and no AfriHealth human reviews have
been saved in production as of the latest checkpoint.

The research launcher uses the running application's correctly parsed provider
environment. The user replenished credits on September 6 and real annotation
requests now succeed. Per-stage `.jsonl.requests` directories store raw responses
and usage locally and are excluded from Git.

The funded teacher-v2 rerun completed 62/62 reference rows but failed consistency
calibration. Inspection also found entities copied from reference answers into
question interpretation. Teacher-v3 separates these roles explicitly, and both
annotation and training export now check that model entity values are grounded
in the question alone. The reference answer is not additional patient history.

The bounded v3 check completed 12 new source-train rows: 10 model consensus and
two needing review, with zero selected entity span violations. The v3 run is
not a semantic/clinical calibration pass. Do not use its per-row eligibility
flags as a substitute for the full export gates. The 62 v2 records repeat the
original reference; across these strong-teacher artifacts there are 74 unique
source IDs, not 136.

```bash
pnpm eval:medical:afrihealth:question-grounding \
  --annotations data/medical-response-corpus/afrihealth-teacher-v3-grounding.jsonl --strict
```

Keep annotation versions in separate output files. The runner refuses to mix
different prompt versions or teacher/adjudicator models in a resumed output.

## Translation benchmark

The first 62 hash-selected source rows have stronger dual-model reference
annotations. They are the fixed gate for cheaper open translation candidates;
the references are stronger model outputs, not human gold.

The September 2026 benchmark rejected Aya-101 and MADLAD-400-10B-MT as
automatic teachers. The Twi-specific `ninte/twi-en-nllb-v2` checkpoint was
materially better, but still changed important meanings and is CC-BY-NC, so its
outputs remain research-only candidates. The GhanaNLP translation-quality
classifier did not correlate with reference accuracy and is not an automatic
acceptance gate. These decisions are recorded in
`afrihealth-translation-benchmark.v3.summary.json`.

Recompute the fixed summary after retrieving the corresponding Modal artifact:

```bash
pnpm eval:medical:afrihealth:translation
```

Do not scale an open model to the 5,569 rows until it beats this gate. Valid JSON
or a high self-reported confidence score is not evidence of correct Twi meaning.

`seed.v0.jsonl` contains English source utterances, faithful English meanings,
intents, entities, safety level, safe answer text, and source URLs.

`twi-drafts.v0.jsonl` adds model-produced Twi user utterance and Twi answer
drafts generated in chunks. These rows are ready for review, not final gold
labels.

Current source anchors:

- CDC urgent maternal warning signs.
- WHO maternal and newborn counselling danger signs via NCBI Bookshelf.
- CDC flu emergency warning signs.
- WHO malaria fact sheet.
- NHS red-eye guidance.

Validation:

```bash
pnpm eval:medical:twi
```

Regeneration:

```bash
pnpm corpus:medical:twi -- --chunk-size 4
pnpm corpus:understanding:candidates
cp tmp/understanding-corpus/candidates.v0.jsonl data/understanding-corpus/candidates.v0.jsonl
```

Do not use any row for a production health model until a reviewer has checked
the language, intent, and answer safety. Research-only source-supervised runs
must retain their source and review status in the model card.
