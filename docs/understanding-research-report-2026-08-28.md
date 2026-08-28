# Understanding research report — 2026-08-28

## Scope

Today’s goal was to move from discussion to a working research loop without
confusing three different evidence classes:

1. The 50 synthetic benchmark probes.
2. Dataset-derived and local-recording corpus candidates.
3. Human-reviewed records that may later become training or evaluation exports.

## Benchmark

Modal benchmark completed for `ninte/twi-en-nllb-v2`.

| Field | Value |
| --- | --- |
| Model | `ninte/twi-en-nllb-v2` |
| Tokenizer | `facebook/nllb-200-distilled-600M` |
| Resolved revision | `79522137844c0aa6dbba3d4da4d1926a9bf945f4` |
| Cases | 50 |
| Runtime | 5.866 seconds on CUDA |
| Artifact | `tmp/understanding-results/understanding/ninte--twi-en-nllb-v2/20260828T143600Z.json` |

The benchmark runner had to be fixed because the model repository tokenizer
metadata resolves to `TokenizersBackend`, which failed in the pinned
Transformers environment. The benchmark now loads the NLLB base tokenizer while
using the fine-tuned `ninte/twi-en-nllb-v2` weights.

## Benchmark finding

NLLB is fast and useful as a baseline translation candidate, but it is not safe
as the sole health meaning annotator.

Examples from the benchmark:

| Twi input | NLLB output | Issue |
| --- | --- | --- |
| `Me ho mfa me.` | `I don't care.` | Health/unwell meaning lost. |
| `Me nyinsɛn na me ti yɛ me yaw paa, m'anim nso ahonhon.` | `My pregnancy is the most painful part of my head, and my face is heartburn.` | Safety-critical pregnancy symptom mangled. |
| `M'ani kum paa.` | `I was very sad.` | Eye-pain/body-part meaning lost. |
| `Me kɔn yɛ me yaw.` | `My stomach hurts.` | Body-part mismatch. |
| `Mewɔ Ghana cedis ɔha pɛ.` | `I live in Ghana only one hundred cedis.` | Commerce budget meaning lost. |

Decision: use NLLB as a fast translation baseline and disagreement signal, not
as the final draft annotator for safety-sensitive health records.

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

## Next model step

Do not train from the 50 benchmark rows.

Next useful model work:

1. Add one stronger structured-understanding candidate beside NLLB, likely a
   Qwen/Gemma-class instruction model through Modal.
2. Use NLLB plus the stronger model for disagreement scoring.
3. Prioritize review rows where the models disagree or where health/commerce
   entities are safety-critical.
4. Train only after enough reviewed rows exist.
