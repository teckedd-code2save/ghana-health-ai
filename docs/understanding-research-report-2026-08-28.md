# Understanding research report — 2026-08-28

## Scope

Today’s goal was to move from discussion to a working research loop without
confusing three different evidence classes:

1. The 50 synthetic benchmark probes.
2. Dataset-derived and local-recording corpus candidates.
3. Human-reviewed records that may later become training or evaluation exports.

## Benchmark

Modal benchmark completed for `ninte/twi-en-nllb-v2`, then the same benchmark
path was extended to test a PEFT/LoRA Twi→English adapter. A structured LLM
draft-understanding benchmark was also added so translation-only models can be
compared against models that extract meaning and entities directly.

| Field | Value |
| --- | --- |
| Candidate | Cases | Meaning score | Exact cases | Runtime | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| `openai:gpt-5.4-mini` | 50 | 94.2% | 42/50 | 62.426s | `tmp/understanding-results/understanding/openai--gpt-5.4-mini/20260829T100632Z.json` |
| `openai:gpt-5.5` | 50 | 93.5% | 44/50 | 178.773s | `tmp/understanding-results/understanding/openai--gpt-5.5/20260829T112746Z.json` |
| `ninte/twi-en-nllb-v2` | 50 | 77.5% | 30/50 | 5.866s | `tmp/understanding-results/understanding/ninte--twi-en-nllb-v2/20260828T143600Z.json` |
| `facebook/nllb-200-distilled-600M` + `mclanorjeff/NLLB-Twi-Human-Aligned` | 50 | 76.1% | 28/50 | 9.479s | `tmp/understanding-results/understanding/mclanorjeff--NLLB-Twi-Human-Aligned/20260829T100413Z.json` |

The benchmark runner had to be fixed because the model repository tokenizer
metadata resolves to `TokenizersBackend`, which failed in the pinned
Transformers environment. The benchmark now loads the NLLB base tokenizer while
using the fine-tuned `ninte/twi-en-nllb-v2` weights.

The runner now also supports PEFT/LoRA adapters through `--adapter-id`; this was
needed to test `mclanorjeff/NLLB-Twi-Human-Aligned`.

## Benchmark finding

NLLB is fast and useful as a baseline translation candidate, but it is not safe
as the sole health meaning annotator. The human-aligned adapter improved some
phrasing, but it did not beat the existing NLLB v2 baseline on this project
rubric and still failed safety-relevant health meanings.

`openai:gpt-5.5` was also benchmarked on the full 50-case probe set. It had
more exact cases than `gpt-5.4-mini`, but a lower total meaning score and a
dangerous blank/missing interpretation for the chest-pain plus breathing case,
so it is not the current best fit for this product path.

Examples from the benchmark:

| Twi input | NLLB output | Issue |
| --- | --- | --- |
| `Me ho mfa me.` | `I don't care.` | Health/unwell meaning lost. |
| `Me nyinsɛn na me ti yɛ me yaw paa, m'anim nso ahonhon.` | `My pregnancy is the most painful part of my head, and my face is heartburn.` | Safety-critical pregnancy symptom mangled. |
| `M'ani kum paa.` | `I was very sad.` | Eye-pain/body-part meaning lost. |
| `Me kɔn yɛ me yaw.` | `My stomach hurts.` | Body-part mismatch. |
| `Mewɔ Ghana cedis ɔha pɛ.` | `I live in Ghana only one hundred cedis.` | Commerce budget meaning lost. |

Decision:

1. Use `openai:gpt-5.4-mini` as the current best draft-understanding candidate
   for corpus population and product meaning extraction.
2. Keep `openai:gpt-5.5` as a rejected comparison candidate for now; its health
   critical failure outweighs the slightly higher exact-case count.
3. Use `ninte/twi-en-nllb-v2` as a fast translation baseline and disagreement
   signal, not as the final health annotator.
4. Do not promote `mclanorjeff/NLLB-Twi-Human-Aligned` for this product yet; it
   remains useful research context, but it did not outperform the baseline here.

## Corpus candidate queue

The first review-ready candidate artifact is now versioned at:

`data/understanding-corpus/candidates.v0.jsonl`

Summary:

| Metric | Count |
| --- | ---: |
| Candidate rows | 80 |
| Rows with local-recording artifact references | 30 |
| Rows with draft model annotations | 80 |
| Health-domain rows | 60 |
| Commerce-domain rows | 20 |

Source mix:

| Source | Rows |
| --- | ---: |
| `local_recording` | 30 |
| `curated_prompt` | 50 |

Every row remains:

- `review_status=needs_review`
- `eligible_for_training=false`
- `eligible_for_final_evaluation=false`

## Review workflow

The internal workspace at `/research/ase` now has:

- source and storage plan;
- benchmark summary;
- corpus candidate summary;
- review queue switch between corpus candidates and benchmark probes;
- edit fields for normalized Twi, faithful English meaning, literal English,
  intent, entities, ambiguity/uncertainty, decision, and notes.

The expected workflow is:

1. Candidate models populate draft fields.
2. A reviewer corrects Twi transcript/normalization and English meaning.
3. The reviewer marks `reviewed`, `needs_second_review`, or `exclude`.
4. Only reviewed, split-safe rows can be exported for training or evaluation.

## Training export gate

The export gate is implemented:

`pnpm corpus:understanding:export`

It writes manifests under `tmp/understanding-corpus/exports/v0/` only for rows
with saved human reviews marked `decision=reviewed` and non-empty normalized
Twi, faithful English meaning, and intent.

Review decisions are stored in Postgres table
`research_understanding_reviews`; JSONL review storage remains only as a local
fallback for offline development.

Current export state:

| Metric | Count |
| --- | ---: |
| Candidate rows | 80 |
| Saved reviews | 0 |
| Training-eligible rows | 0 |
| Dev rows | 0 |
| Test rows | 0 |

This means the corpus pipeline is ready for review and export, but the corpus is
not yet ready for training. The blocker is not missing code; it is missing human
accepted review decisions.

## Next model step

Do not train from the 50 benchmark rows.

Next useful model work:

1. Review the 20-row assisted training pack from `/research/ase`, correcting
   draft fields and changing only accepted rows to `reviewed`; then continue
   through the remaining 80 corpus candidates after the first strict export
   passes.
2. Export reviewed rows with `pnpm corpus:understanding:export:strict`.
3. If strict export produces enough rows, train the first understanding/adaptation
   artifact from reviewed data only.
4. Add a heavier open model candidate, likely Qwen/Gemma-class, only after this
   review gate proves the corpus labels are useful enough to justify the run.

To benchmark another OpenAI-compatible response model on the synthetic probes
without changing production config:

```bash
pnpm eval:understanding:llm -- --model gpt-5.5
pnpm eval:understanding:score
```
