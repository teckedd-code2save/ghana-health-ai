# Model Credit Plan

Date: 2026-08-12

## Decision

Do not spend the remaining credits on another blind full fine-tune. The current evidence shows a real plateau around the 30% WER range on Waxal-style Twi when we only change training recipe. The next spend must either create stronger product data or test a meaningfully different hypothesis.

Current serving stance:

- Twi: keep the best validated Whisper-family route while collecting better product data.
- English: route to the separate `openai/whisper-small` Modal endpoint; `MODAL_ASR_EN_URL` can override the documented default, but English must not be forced through Twi-specialized v6.
- DONDO: keep as a research branch. `teckedd/gha-dondo-w2v-bert-twi-v1` improved from zero-shot to **35.77% WER**, but missed the promotion bar.

## Promotion Gates

No ASR checkpoint should be promoted unless all required gates are present and passing.

| Gate | Required evidence | Pass bar | Why it matters |
| --- | --- | ---: | --- |
| Twi held-out Waxal | Full or representative Waxal test, same decode settings | WER <= 28% | Must beat the current 30% plateau, not just match it |
| English retention | Common Voice English or better English speech eval | WER <= 17% and within +5 pp of `openai/whisper-small` | Prevents the v6 English regression from returning |
| Health-domain Twi | Consented or curated symptom, medicine, maternal, malaria, emergency utterances | WER <= 25% or clear improvement over current route | Product quality depends on health terms, not generic speech only |
| Code-switch | Twi-English mixed health/commerce turns | Intent-critical terms preserved | Ghanaian user speech will code-switch |
| Phone/noise | Mobile mic/noisy room samples | No large degradation versus clean samples | The app is voice-first on real phones |
| Latency | End-to-end ASR timing on deployment hardware | Fast enough for live turn-taking | A better WER model that feels slow is not product-ready |
| HF card | Model card in pushed repo | Valid metadata + metrics + limitations | Avoids undocumented model uploads |

## Spend Order

1. Build the product eval set before more large runs.
   - 100 Twi health utterances.
   - 50 Twi commerce utterances.
   - 50 Twi-English code-switch utterances.
   - 50 English health utterances.
   - 50 noisy or phone-recorded utterances.
   - Manifest format: [`data/asr-product-eval/README.md`](../data/asr-product-eval/README.md).
   - Validate with `pnpm eval:asr-manifest`; use `pnpm eval:asr-manifest:strict` before serious training spend.

2. Run the current models on that set.
   - `teckedd/gha-whisper-small-twi-v6`
   - `openai/whisper-small` for English
   - `teckedd/gha-dondo-w2v-bert-twi-v1`
   - current v7 balanced checkpoints as negative controls, not promotion candidates

3. Only then spend on training.
   - If errors are mostly health vocabulary: targeted ASR data collection + continuation fine-tune.
   - If errors are mostly language routing/code-switch: routing and decoder prompting before model training.
   - If DONDO helps on noisy/phone audio despite worse Waxal WER: run a targeted DONDO phone/health trial.
   - If all current architectures remain above 30% WER on product speech: spend credits on transcription/data generation, not another model recipe.

4. Push every trained model with a real card.
   - Base model.
   - Dataset list.
   - Eval split and WER/CER.
   - Product decision: promote, hold, or do not promote.
   - Medical-device limitation.
   - Verify pushed cards with `pnpm eval:hf-model-cards` for public repos tracked by this project.

## Current Evidence Snapshot

| Model | Twi WER | English WER | Decision |
| --- | ---: | ---: | --- |
| `openai/whisper-small` | 114.54% on Waxal sample | 11.82% | English route only |
| `teckedd/gha-whisper-small-twi-v6` | 30.44% full beam5 / 32.16% sample beam5 | 42.34% | Twi hold-and-validate; not English |
| `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` | 40.86% sample beam5 | 15.10% | English retained; Twi failed |
| `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` | 53.83% train validation | Missing | Frozen recipe failed Twi |
| `teckedd/gha-dondo-w2v-bert-twi-v1` | 35.77% final validation | Missing | Research only; do not promote |

The v7-lite result is useful but negative: adding English retention data preserved English substantially better than v6, but the recipe pulled the Twi model away from the current product baseline. Do not spend more credits on the same recipe unless the training mix or objective changes around product speech.

## Immediate Product Loop

Every real voice turn should become useful training signal when the user consents:

- audio hash and metadata, not raw private audio by default
- ASR transcript
- language route
- confidence/quality metadata
- user correction if given
- intent and final response
- whether the assistant asked for clarification

Current implementation:

- `/api/voice/feedback` stores corrected transcripts and ratings in `asr_feedback`.
- `pnpm eval:asr-feedback:export` writes reviewable JSONL to `tmp/asr-feedback-export.jsonl`.
- `ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-feedback-export.jsonl pnpm eval:asr-manifest` validates the export shape and flags missing audio placeholders.

The next competitive model is more likely to come from this loop than from another Waxal-only fine-tune.
