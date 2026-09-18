# Native Twi Fluency Comparator

## Result

MiniCPM5-1B-Twi completed 27 non-tool development cases in 356.49 seconds on
one T4. Every response reached the 384-token ceiling. More recognizable Twi
wording did not yield reliable question following, English retention, arithmetic,
context preservation or appropriate health responses. It is not enabled in chat
and is not accepted as an unreviewed Twi answer teacher.

The personal-budgeting case becomes a long institutional/elections discussion.
An explicit request for two English sentences produces long Twi text instead.
The quantity correction changes kilograms into cups. The no-tools shopping case
invents price/contact details. Newborn context is lost. These are observed
failures, not fabricated semantic scores or a claim of native/clinical review.

## Reproduction

- Model: `ghananlpcommunity/MiniCPM5-1B-Twi`.
- Revision: `d807ca1a3323972afafabff8f9affe2639e37b5c`.
- Run: `minicpm_twi_20260910T151419Z`.
- Call: `fc-01M25Y4P40NXK4KZMRH22GKNRZ`.
- App: `ghana-native-fluency-comparison`.
- Local result: `tmp/native-fluency-comparison/results.attempt-2.json`.
- SHA256: `a5170598c470a167461ff084bc9cae7f7dd9da971f6a01833af6cfcc68ce556b`.
- Final-answer-only export: `results.attempt-2.diagnostic.json` in that directory.
- Decision: `data/response-adaptation/minicpm-twi-decision.json`.
- Remote volume: `ghana-health-understand-train`,
  `/stronger-response/minicpm_twi_20260910T151419Z/`.

Actual extended vocabulary is 136,560; complete checkpoint loading and a finite
CPU forward pass were checked before retry. Native FP16 weights, SDPA,
Transformers 5.17.0, published chat template with thinking disabled, published
EOS IDs `[1, 130073]`. Sampling uses temperature .7, top-p .9, top-k 50,
repetition penalty 1.3, no-repeat trigrams and seed 42 per case. Prompt inputs
are the same saved product cases used for other comparisons, excluding the three
tool cases. No hidden response model or tool execution.

First attempt `minicpm_twi_20260910T151049Z`, call
`fc-01M25XY9Z9SC64RBXDVD1E44FZ`, failed before generation because `device_map`
required a missing Accelerate dependency. The loader was changed to load on CPU
then move to CUDA, and an actual CPU forward preflight was added. Both receipts
remain. Do not repeat either completed comparison to obtain a preferred result.

These are reused development examples, a single decoding profile, and no
project training. End-of-answer and English failures warrant rejection for this
product; they do not invalidate every possible language-model use.

Source: [publisher model card](https://huggingface.co/ghananlpcommunity/MiniCPM5-1B-Twi).
The card itself identifies weak reasoning and factual accuracy; its fluency
claims are not our independently validated results.
