# R5 IFEval Proposal

Status: source metadata verified; adapter, generation protocol, official scoring
execution, and model results are **not implemented or run**. This is a separate
follow-up, not a blocker for current R5 training. No GPU/model calls were made.

## Verified Source

Use the official Google Research implementation, not replacement regex checks:
[README](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/instruction_following_eval/README.md),
[scorer](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/instruction_following_eval/evaluation_lib.py),
[data](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/instruction_following_eval/data/input_data.jsonl).

Read-only inspection of revision `0eebfd403ba438bfd14cb70b9b24a7745c997083`
verified 541 unique prompts/keys, 834 instruction instances, and 25 instruction
types. Data SHA-256:
`67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`.
Official `evaluation_lib.py` SHA-256:
`35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c`.
These counts are source inventory, not measured model capabilities.

## Minimal Integration

1. Build a new private versioned evaluation package from all 541 original
   prompts, keys, instruction IDs, and kwargs at that revision. Seal every
   scorer/module/dependency/resource checksum, licenses, source provenance,
   source-disjointness audit against training, and expected denominators.
   Reject omissions, duplicates, unknown instruction IDs, and changed bytes.
2. Resolve and lock the official scorer dependencies separately from inference.
   Its upstream requirements are not version locked, and sentence checks use
   NLTK's English Punkt resource. Prepackage/hash needed resources, seed relevant
   nondeterminism, and test with network disabled. Do not alter official scoring
   behavior or silently fetch data during scoring.
3. Create a separate generation protocol/runtime version. The current R5
   development runtime's hard 256-token output cap is **not suitable** and must
   not be inherited. Audit every official word-count constraint and native prompt
   length before selecting an adequate output allowance and supported native
   context limit. Freeze decoding, stop rules, per-request time budget, and model
   identities equally across base/R4/R5. Preserve the exact user prompt and native
   chat serialization; add no Twi/system instruction. No prompt truncation,
   shortened tasks, context substitution, or silent per-case cap reductions.
   Context-infeasible cases must fail preparation visibly, not be dropped.
4. Generate independent fixed-context responses for all 541 prompts/model using
   resumable checksum/call-ID journals and zero automatic ambiguous retries.
   Bind every output to exact prompt IDs, checkpoint, protocol, and source.
   Preserve EOS, output-cap/time-limit stops, partial text, generation errors,
   missing/pending slots, and indeterminate requests. Main authorizes GPU work
   separately after training; no current runner/trainer constants are mutated.
5. Run the unchanged official strict and loose scorer offline on saved outputs.
   Report its four metrics: prompt-level strict/loose and instruction-level
   strict/loose, plus per-instruction-type/category counts. Expected denominators
   are 541 prompts and 834 instruction instances per model, never just completed
   replies. Include per-case generation and scoring errors; do not silently
   remove failures or publish an incomplete subset as a full benchmark. Score
   actual partial outputs with official checks while separately identifying cap
   hits; completion itself is not a correctness condition or accuracy score.
6. Keep official **raw** output evaluation and product-guarded evaluation in
   separate packages, journals, and reports. Official scorer strict/loose modes
   are not the same dimension as raw/guarded generation. Use synthetic known-pass,
   known-fail, empty, incomplete, missing, duplicate, and tampered cases for local
   wrapper tests, plus the upstream scorer's tests. No model calls in those tests.

IFEval measures verifiable instruction following on an English-prompt public
benchmark. It is not evidence of Twi meaning ability, general factual accuracy,
medical safety, or native-human certification. Keep it alongside, not pooled
with, Twi/English retention NLL, reviewed Twi capability tasks, and product
rollouts. Public-benchmark pretraining contamination is not ruled out by a clean
local training-source audit. Do not claim public-leaderboard comparability unless
the complete published generation/scoring protocol is also matched and disclosed.
