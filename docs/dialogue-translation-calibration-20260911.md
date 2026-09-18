# Dialogue Translation Calibration - 2026-09-11

## Decision

The bounded comparison completed: 96 translations of 32 turns from eight
existing English training conversations. Do not resubmit. No translation
method is approved for unattended corpus expansion from this result.
No new response model was trained, and no model-quality pass is claimed.

This tests corpus-construction methods, not final response-model quality.
Hand-selected development examples are not an unseen benchmark. The input
dialogues include source caveats and English-dependent reasoning puzzles;
accurate translation alone would not certify them as good training examples.

## Reproduction And Artifacts

- Selection: `data/response-adaptation/dialogue-translation-calibration.v1.json`.
- Runner: `modal/train/translate_conversation_sources.py`.
- Client: `scripts/calibrate_dialogue_translation.py`.
- Results: `tmp/dialogue-translation-calibration/v1/` contains immutable input,
  source records, demonstrations, receipts, per-model results, `review.json`,
  `summary.json`, and code snapshots.
- Private remote artifacts: `ghana-health-understand-train`, under
  `/dialogue-translation/<run_id>/<variant>/`.
- Source: `OpenAssistant/oasst1`, revision
  `fdf72ae0827c1cda404aff25b6603abec9e3399b`, Apache-2.0. Original English
  messages, source groups and hashes preserved. No private conversation/audio
  or production data was used.
- Inputs SHA256:
  `db23854277a92a7731ae96d72a8ec3e96bcaff4d8fdef748ecf800d9c02ac950`.
- Review SHA256:
  `218be2aa7f4607e692f5eb80d9a2120de04b4e9710f4340ec89e6e8a61a98971`.

## Methods And Completed Calls

| Method | Outputs | Run / call |
| --- | ---: | --- |
| Afrique 9B, isolated turns, three fixed translation demonstrations | 32 | `dialogue_20260911T153704Z` / `fc-01M28HV1H7K0H30101WEZDN6FR` |
| Gemma 31B, isolated turns and original-conversation context | 64 | `dialogue_20260911T154334Z` / `fc-01M28J6XYJ9WSV7KZS6HGFMM20` |

Pinned models: `McGill-NLP/AfriqueQwen3.5-9B-50Langs` at
`4358dcbc062421751174279da1efe9f88f85e1d5`; `google/gemma-4-31B-it` at
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Greedy generation, 768-token limit,
no thinking; Afrique continuation stops at the next English demonstration label,
not at a translated paragraph boundary. Runtime and prompt hashes are recorded.
Only the Gemma isolated/context pair is a within-model context ablation.

First Gemma call `fc-01M28HV1JA8BZBBVAJ22J5WZ27` failed after preemption because
our original output-directory creation did not tolerate restart. The failed
receipt remains preserved. Attempt 2 added validated checkpoint recovery and
completed. This was an infrastructure bug, not evidence against model quality.
After completion, local recovery accounting was additionally hardened to save
the restart count before loading weights. That final local hardening was not
redeployed and did not alter these completed results.

## Agent Inspection, Not Native Gold Review

- Afrique turns the book's prince into a king. In the equal-weight example it
  drops the one-pound constraint and later omits an explanatory paragraph.
  A Pluto explanation loops to the token limit.
- Gemma context sometimes helps: the Pluto answer preserves the first TWO
  criteria rather than the isolated version's THREE, and the short question
  about limits of knowledge becomes more intelligible.
- It is not a general repair. Both Gemma methods translate days as weeks in
  the million-years calculation. Numerals survive while their units change.
  Equal-weight examples introduce money and inconsistent material names;
  other outputs have malformed vocabulary or literal escaped formatting.
- The automated flag totals (Afrique 1/32; Gemma isolated 2/32; context 2/32)
  are formatting/termination checks, NOT semantic error rates. Unflagged
  translations can be seriously wrong. Do not report 95%+ translation quality.

All rows remain `training_eligible=false`, `human_verified=false`. Preserve
alternatives; do not silently replace original transcripts or mark generated
text as human reviewed. Context alone does not justify translating the whole
corpus with this teacher. Next selection needs targeted lexical/semantic checks
and a genuinely better translator or audited corrections, not another unchanged
fine-tune on these outputs.

The user's latest request redirected to the shared ASR research. See
`docs/asr-research-shortlist-review-20260911.md`. Keep that review distinct from
this failed corpus-expansion quality gate. Public chat, private model selection,
Voice A, user history and production were not changed by this comparison.
