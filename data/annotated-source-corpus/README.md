# Original-source annotation inputs

This directory extends the Twi medical annotation work with original English
AfriHealth QA and the existing WAXAL/GhanaNLP speech-text manifests. It does not
include synthetic symptom descriptions, small medical seeds, local recordings,
or the old model labels. No audio is sent to the annotation provider.

`sources.v1.jsonl` contains 10,446 unique text records after two exact duplicate
exclusions. Original splits are retained; held-out text wins conflicts with
training copies:

| Source | Training | Validation | Test | Task |
| --- | ---: | ---: | ---: | --- |
| Original English AfriHealth | 4,402 | 1,096 | 0 | Understanding and response |
| GhanaNLP speech-text | 2,249 | 129 | 120 | Understanding only |
| WAXAL Akan | 1,800 | 350 | 300 | Understanding only |

Together with the existing 4,407-row Twi medical training pool, this prepares
12,858 source-training records. This is a source count, not an annotation or
training-ready count. Translated views are not added to that count.

## Evidence and permitted use

- AfriHealth retains pinned upstream revision and original CC-BY-SA attribution
  from the existing import.
- [WAXAL's dataset card](https://huggingface.co/datasets/google/WaxalNLP) identifies
  the University of Ghana Akan data as CC-BY-4.0.
- The [GhanaNLP/Ghana Open AI speech card](https://huggingface.co/datasets/ghanaopenai/twi-speech-text-multispeaker-16k)
  specifies CC-BY-NC-4.0. These rows remain noncommercial research-only.
- Original Hub revisions for the speech manifests are not available in those
  manifests. They are honestly identified by a local manifest SHA-256, not by
  an invented Hub revision. Provenance and text hashes remain attached to rows.

## Run and validate

```bash
pnpm corpus:sources:prepare
pnpm eval:corpus:sources
pnpm eval:corpus:resume
pnpm eval:corpus:runner
pnpm corpus:sources:annotate --train-only --limit 0 --chunk-size 4 --dry-run
```

The runner test uses a local mock service and explicitly generates no real
corpus labels. A live model check of this new runner is still pending: the Twi
full run exhausted the connected OpenAI API credit balance first.

After funding is confirmed, use a separate output for a live smoke check before
the full run. Never overwrite an existing plan with a different source selection.
The research VPS launcher accepts `--script scripts/annotate-source-corpus.ts`.
The normal source annotator records two independent proposals, an adjudicator
decision, exact question-entity grounding, ambiguity, source assessment and
concise replies only where reference answers exist. It does not invent replies
for fragments from speech datasets.

The full-run dry-run plan is saved at `annotations.v1.jsonl.plan.json`, with
8,451 training rows pending. There are no completed live annotations in this
directory yet. Per-stage request caches are ignored by Git. Resumes keep paid
batch boundaries stable and preserve already completed annotations.

All generated labels are model-assisted, never human gold. A model-consensus
status is not clinical verification or permission to bypass training/export
quality gates. The whole goal remains incomplete.
