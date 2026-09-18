# Stage Corpus: September 14, 2026

## Scope

The active work is the user's four-collection dataset plan, not ASR, TTS,
another model training run, a public upload, or production deployment.
The source freeze began on September 11; its local release directory is
`tmp/corpus-releases/twi-stage-v1-20260911`.

## Sources And Outputs

The ledger has 1,178,118 selected source records: complete selected Pristine
shards, GhanaNLP/community parallel text, full public WAXAL/GhanaNLP speech
transcript extraction, native INJONGO splits, AfriHealth partitions, cached
AfriQA passages, and qualifying intact English OASST paths. This number is
NOT the number of clean Twi training examples.

The release separates:

- `collections/language`: source-preserved text, short utterances, and lexicon.
- `collections/meaning`: existing bilingual references and structured intent data.
- `collections/conversation`: English retention and source-grounded Twi QA.
- `collections/domain-actions`: executed English simulated catalogue fixtures.
- `source-material/health.jsonl`: complete health references needing semantic
  and clinical assessment, explicitly not an approved response-training export.
- `annotation/work-plan.jsonl`: full missing-field plan, not a 12-row sample.
- `annotation/preserved-annotations.jsonl`: prior hash-matched annotations,
  retained with their original provenance, not upgraded to human approval.
- `review/session.jsonl`: a bounded representative correction session.
- `reports`: inventory, coverage, source accounting, readiness and verification.

Synthetic Pristine remains outside accepted language-learning exports. Unknown
language/dialect labels are not silently relabelled as Twi. English references
are preserved, not translated again. The source row is accepted if it supplies
at least one eligible task view; that does not complete its missing annotations.
OASST raw tree accounting documents excluded paths without inventing roles or
truncating histories. Separate shared-source groups prevent cross-stage leakage.

## Verified Release Counts

The release is sealed, and verification passed again after sealing. All
1,178,118 selected records have a disposition; 49,922 unique source records
contribute one or more screened training views. Counts below include train and
validation and overlap across collections; do not add them as unique sources.

| View | Rows | Interpretation |
| --- | ---: | --- |
| Language text | 10,582 | Source-backed text, not necessarily long documents |
| Short utterances | 12,883 | Kept separate from paragraphs |
| Lexicon | 21,408 | Dictionary entries, not assistant conversations |
| Existing bilingual alignment | 30,404 | Original English references preserved |
| Structured understanding | 2,751 | 1,748 Twi and 1,003 English source examples |
| Grounded Twi QA | 110 | Passage-grounded, single-turn answers |
| English conversation retention | 2,175 | Intact source paths |
| Simulated English tools | 45 | Executed fixtures, not Twi commerce coverage |
| Health correction material | 4,264 | Complete references, not approved SFT |

Language views contain 773,316 tokens under the pinned counting tokenizer,
including 223,803 lexicon tokens. This is not a million-row native-language CPT
corpus and does not justify claims of broad Twi fluency. All accepted topics
without upstream labels remain explicitly unclassified. There are zero general
multi-turn Twi response examples in this release.

The Modal-only work plan covers 14,276 missing English meanings, 24,461 records
needing function/entity/ambiguity/intent work, 4,264 health reference assessments,
and 2,192 candidate conversation derivatives. These field counts overlap.
Fifty-three prior source-hash-matched annotation records were retained in this
eligible correction partition; none was promoted to human review.

WAXAL was fully processed: 11,892 records are protected through upstream
holdouts or related stimulus groups, and 860 require dialect/orthography review.
Do not silently call all Akan text Twi to increase accepted counts. AfriHealth
has 6,852 protected related records and 4,264 correction records. Conservative
eight-word holdout overlap checks can over-exclude; changing that policy needs
a separately versioned, audited release, not deletion of existing holdouts.

Verification checked raw file hashes, source-record hashes, strict stage schemas,
original targets, holdout exclusion, shared source/split consistency and actual
tool outputs. Nineteen corpus tests passed. The UI fixture tests passed saving,
idempotency, stale-source rejection, refresh recovery and desktop/mobile layout.
The real release also passed read-only desktop/mobile checks for all four review
categories. No real human reviews were created by tests.

Thirteen small reports, cards, manifests and configuration files were stored in
the existing private Modal volume `ghana-health-understand-train` at
`/corpus-release-reports/twi-stage-v1-20260911/ddc88cb47302`. Remote SHA-256
read-back verification passed. The receipt is
`tmp/corpus-releases/twi-stage-v1-20260911.storage.json`. This is a report backup,
not a remote backup of the corpus: raw data, training rows and the ledger were
not uploaded. The frozen 3.4 GB working release remains local.

## Teacher Result

Candidate: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, revision
`e156cb4efae43fbee1a1ab073f946a1377e6b969`. This is a teacher candidate,
not the selected student foundation.

- CPU weight preparation completed: `fc-01M28RWCTZDHXP2MN9HQHSBCCH`.
- First GPU attempt: `fc-01M2ENHS2G0VWN39EJNZBP01FN`. Agent cancelled it
  after shard loading projected beyond the 20-minute bound.
- Eager-loader retry: `fc-01M2EPC4KAPNY08XPCK6J9JAYT`, run
  `corpus_teacher_20260914T005143Z`. Modal timed out at 1,200 seconds.
- Eager loading still took approximately 150 seconds per shard and completed
  only seven of 24 shards before the timeout. No translations were produced.

This is an infrastructure failure, NOT a finding that Qwen cannot translate Twi.
Receipts and the closed gate are in `tmp/corpus-teacher/qwen235-v2-eager`.
Do not repeat the unchanged loader or silently extend its spend. Before the next
GPU attempt, measure local checkpoint staging/loading and choose a bounded runtime
that fits the measured preparation cost. Bulk annotation must reuse a loaded
teacher; a cold 235B load for every 32-row batch is not a viable production path.

The existing Afrique/Gemma comparison remains unqualified. We did not fill the
corpus with their known-bad translations to inflate completion numbers. General
multi-turn Twi response targets and Twi tool trajectories remain gaps.

## Local Review

Open `http://localhost:3100/research/ase?release=twi-stage-v1-20260911`.
The release index is built. The existing production review page is unchanged.
The local endpoint accepts loopback hosts only and is disabled in production.
Source text and answers are read-only. Corrections select a target field and
optional error category; browser drafts survive navigation/refresh. Saved
decisions are source-hash-bound and append-only under `tmp/corpus-review/`.
Postgres receives references/corrections when configured, not corpus blobs.
If a process dies while saving, inspect the local `.review-lock` directory
before removing that stale lock. Never remove a live writer's lock.

Approving a local review does not automatically certify a medical answer or
modify this frozen export. Corrections belong to the next explicit release.
Existing production reviews have not been counted or overwritten by this pass.
The UI is a correction pool, not an accepted-training-row counter. For example,
Meaning and alignment shows sentence/intent records, while dictionary and speech
records are accessible under Source text, with existing English references when
available. Protected and synthetic records are not offered for routine review.

## Reproduce And Verify

Use Python 3.13 with `scripts/requirements-corpus.txt`. On this device the
prepared interpreter is `/Users/welcome/miniconda3/bin/python3`.
All public source revisions and source file checksums are preserved. The source
cache is required; these commands do not fetch unpinned replacement data.

```sh
python scripts/corpus_release.py build --out tmp/corpus-releases/NEW_VERSION
python scripts/corpus_release.py recover --out tmp/corpus-releases/NEW_VERSION
python scripts/corpus_handoff.py finalize --release tmp/corpus-releases/NEW_VERSION --teacher tmp/corpus-teacher/qwen235-v2-eager
python scripts/corpus_release.py verify --out tmp/corpus-releases/NEW_VERSION
python -m unittest discover -s scripts -p 'test_corpus*.py'
node --import tsx scripts/eval-source-corpus-runner.ts
node --import tsx scripts/test-corpus-review-ui.cjs
```

The UI test needs Playwright and Chrome; it uses isolated fixture records and
does not write real reviews or connect to Postgres. The runtime bundle can supply
Playwright through `NODE_PATH`. TypeScript also passes `npx tsc --noEmit`.

## Research Reference

Masakhane reinforces this plan, rather than replacing it. Relevant references:

- [Twi curated-corpus experiments](https://arxiv.org/abs/1912.02481): separate
  corpus quantity, quality, orthography and domain diversity in experiments.
- [Participatory research](https://arxiv.org/abs/2010.02353): evaluate outside
  religious training domains and retain concrete native-speaker post-edits.
- [Low-resource methods](https://github.com/masakhane-io/masakhane-mt/blob/master/MT4LRL.md):
  source-native monolingual text and qualified back-translation are complementary.

No old translation baseline is adopted as our modern LLM base. No community
contribution or dataset upload occurred. Later contributions could include
Twi meaning-preservation tests, reproducible cleaning tools and failure analyses.

## Next Training Decision

Inspect representative accepted language/alignment records and the coverage
report before setting a mixture. A later controlled comparison can test the
same foundation with and without Twi language adaptation, retaining English.
Do not claim direct-response training readiness while general multi-turn Twi
targets and teacher qualification are unresolved. No new training is authorised
by a source-processing completion flag.
