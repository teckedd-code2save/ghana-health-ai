# Stronger Open-Weight Comparisons

These are external foundation models evaluated privately on Modal, NOT new
project fine-tunes, production services, or automatic annotation teachers.
The goal is to identify usable native-language supervision and an adaptable
foundation after V5's small auxiliary gains failed to improve direct replies.

## Bounded Runs

| Candidate | Pinned revision | Compute ceiling |
| --- | --- | --- |
| Qwen/Qwen3.5-122B-A10B-FP8 | a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 | Two H100s, 1,800 seconds |
| Qwen/Qwen3.8-27B | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | One H100, 1,800 seconds |

Both are ungated Apache-2.0 publisher weights. No hosted Qwen or proprietary
API calls, HF upload, private voice recording, or production data are involved.
Models are cached by separate CPU preparation calls before GPU inference.

The 122B run is `qwen122_20260910T125452Z`, call
`fc-01M25P5DDJB2WYV2HBMCS5J0J0`, app `ghana-stronger-response-comparison`.
Its original public-weight preparation completed in 673 seconds. The GPU call
hit its 1,800-second ceiling after expensive loading/kernel warmup and generation.
This was a recorded FunctionTimeoutError, NOT a user cancellation. All 30
non-thinking outputs and the first eight thinking outputs were preserved in
`tmp/stronger-response-comparison/partial-final.json` and the private volume's
`/stronger-response/<run_id>/partial.json`. There is no completed 60-output
paired result. SHA-256 of the local partial file:
`f86218c7a52424efc59db0be3fad0d3a379b6c23816dcb8a8b525687653f9b78`.

The newer 27B model was discovered during this continuation. Publisher claims
about broad agentic improvements do not establish Twi competence. Its BF16
weights fit the single-H100 comparison without the 122B FP8 MoE warmup.
CPU preparation call `fc-01M25Q11CR4W7YJCKFJ7RBWD7A` completed in 220 seconds.
Run `qwen38_20260910T131425Z`, call `fc-01M25Q96QG8X3F656CJX9SAD4A`, completed
all 60 outputs in 413.38 seconds (164.75 loading). Full local result:
`tmp/current-response-comparison/results.json`, SHA-256
`03a1138afd79b022017f96087666026d7003ac8445cba7be48e24c23a786f7dd`.

## Actual Decision: Neither Accepted

The 122B candidate preserved the tomato negation and quantity correction but
interpreted Twi breathing difficulty as a cold home. Budgeting and health replies
contained wrong words and unsupported content. One saved thinking output spent
all 4,096 tokens without a final answer. Incomplete thinking coverage remains
explicit; there is no aggregate conversational score.

Qwen 3.8 reversed the tomato negation in BOTH modes. Non-thinking also changed
tomorrow to Thursday. Its child-fever response claimed the assistant's own child
had a fever, and pregnancy/breathing/context cases were not dependable.
Both modes produced valid product-search and web-search requests, but no tools
were executed in this comparison. This does not offset language failures.

Neither is an accepted chat replacement or an unchecked annotation teacher.
Mechanical completion is not semantic correctness: 3.8 had zero empty answers
and no repeated-eight-word flags, despite these clear meaning failures.

## Methods

Identical saved 30 product/development inputs are run in non-thinking and
thinking modes. No answers or rubrics are sent to the models. Three cases show
read-only tool schemas, but no tools actually run. Multi-turn histories are
preserved. A generic Twi/English language hint is used for both models.

Non-thinking: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5,
640 maximum output tokens. Thinking: temperature 1.0, top-p 0.95, top-k 20,
4,096 maximum output tokens. The 3.5 publisher thinking profile uses presence
penalty 1.5; 3.8 uses 0.0 and explicit medium reasoning effort. Thus this is a
practical candidate comparison, NOT a controlled parameter-count or thinking-only
ablation. Batched timing is not private-chat latency.

vLLM 0.28.0 and Transformers 5.17.0, native tokenizer templates and response
parsers, no remote model code. Hidden reasoning stays out of review exports.
Missing answers, length limits, repeated eight-word spans, and invalid tool
handoffs are retained as failures, not silently trimmed or repaired.
No lexical heuristic is called clinical or native-language accuracy.

## Reproduce Summaries

```sh
python3 scripts/summarize_stronger_response.py tmp/current-response-comparison/results.json
```

Use the existing receipts; do not duplicate submissions. Output folders are
`tmp/stronger-response-comparison/` and `tmp/current-response-comparison/`.
No private playground selection has been changed. Owner voice A and existing
histories are untouched. The strict paired summarizer intentionally cannot
summarize the incomplete 122B run as though all outputs exist.

After a completed result, run `scripts/summarize_stronger_response.py` on its
`results.json`. It requires every case in both modes and produces a separate
review-safe diagnostic with final text, tool requests, and mechanical flags.
The semantic failures above were identified from actual final answers, not
inferred from model size, training loss, or mechanical flags.

## Next: Native-Language Foundation Control

`modal/train/evaluate_afrique_native.py` compares two BASE models with plain
few-shot continuation, not a chat template or an untrained assistant claim:

- Qwen/Qwen3.5-9B-Base, revision `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- McGill-NLP/AfriqueQwen3.5-9B-50Langs, revision
  `4358dcbc062421751174279da1efe9f88f85e1d5`.

The publisher explicitly includes Akan/Twi in its extended 50-language recipe.
Its reported cross-language averages do not establish Twi-specific quality.
Unlike the older 4B Afrique experiment, this evaluates the newer 9B foundation
before training it on our response data.

Each variant receives 450 Twi and 450 English AfriXNLI development cases,
restricted one-token A/B/C output, counterbalanced choices, and three English
demonstrations. This differs from the earlier unrestricted instruction-model
score and must not be presented as a direct leaderboard comparison.

Translation uses 32 validation-source pairs in both directions, balanced across
eight source-connected components. Three train-source demonstrations per
direction use unchanged records 69432, 68787, and 64065. They were agent-inspected
before inference, NOT human verified. No train/validation component overlaps;
reference answers remain local. Two source meaning problems were separately
flagged in `data/response-adaptation/source-review-findings.jsonl`; no original
translations were rewritten. chrF++ will measure reference similarity only.

Initial preparation `fc-01M25RD3JA40KW9SY1B535NPHP` failed because Xet kept a
log file open during the second volume reload. No GPU ran in that attempt.
Temporary Xet files now live in `/tmp/hf-xet`; preparation reloads the shared
volume only once. Retry `fc-01M25RP0PCDEVRDXG03Y1NBJ6X` completed, retaining
the original failed receipt.

Run `afrique9_20260910T134056Z` is COMPLETE. Base call:
`fc-01M25RSRJSG4WD1DQYH65ZT49Q`; Afrique:
`fc-01M25RSV09TXJSMRHBGKMKJ1RF`. Each received all 964 inputs on one
A100-40GB, serial, single-use containers, zero retries. Base took 325.52 seconds
(257.11 loading), Afrique 438.48 (354.47 loading). These include JIT warmup and
are not per-request serving latency.

| Measurement | Original base | Afrique 50-language CPT |
| --- | --- | --- |
| Twi AfriXNLI development | 211/450, 46.89% | 214/450, 47.56% |
| English AfriXNLI development | 392/450, 87.11% | 361/450, 80.22% |
| English to Twi chrF++ (32) | 9.11 | 41.47 |
| Twi to English chrF++ (32) | 22.92 | 48.31 |
| English to Twi output limits | 18/32 | 1/32 |

The translation improvement is substantial on this small source sample, but
the meaning gain is only three additional correct cases, with 31 fewer correct
English cases. This is NOT an accepted understanding-model replacement.
Translation outputs still contain meaning errors, especially English to Twi;
do not use it as an unchecked corpus translator. Reference translations also
have errors, so chrF++ is not semantic accuracy.

Local complete artifacts: `tmp/afrique-native-comparison/cachefix/`.
Hashes: base `1e1e51dbeea7b049cde42f0635d35332d7307bcbff61f8ddddc490d919e6ecd9`,
Afrique `318c2b9dee7882a7f9f04f752abe465e3d91c1793d1f6f13a06add7eeaf498fd`,
summary `343dba24527ba041eb0457e601f271cce90b67349f133ce4eeb0f82eeeb8715c`.
Remote copies: `/native-foundation/<run_id>/<variant>/` on the private training
volume. No training, public deployment, private recordings, or proprietary
responses occurred. Do not resubmit completed calls.

For the next data experiment, see `docs/native-answer-synthesis-20260910.md`.

Sources: [Qwen 3.5 FP8 card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8),
[Qwen 3.8 card and generation guidance](https://huggingface.co/Qwen/Qwen3.8-27B),
[vLLM warmup implementation](https://docs.vllm.ai/en/stable/api/vllm/model_executor/warmup/deep_gemm_warmup.html).
