---
base_model: McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1
library_name: peft
inference: false
model-index:
  - name: Ghana Health AI Twi R5
    results:
      - task:
          type: text-classification
          name: Two-choice meaning recognition
        dataset:
          type: gha-twi-reference-choice-v1
          name: Published Twi-English references, disjoint same-domain pairs
        metrics:
          - type: accuracy
            name: Correct in both option orders (900/1092)
            value: 0.824176
        source:
          name: Ghana Health AI matched checkpoint evaluation
          url: https://github.com/teckedd-code2save/ghana-health-ai/blob/main/docs/TWI_UNDERSTANDING_SCALE_EVALUATION.md
      - task:
          type: text-generation
          name: Own-history dialogue state recall
        dataset:
          type: gha-context-rollout-v1
          name: Controlled bilingual corrections (4 templates, 24 scenarios per language)
        metrics:
          - type: twi_both_slots_accuracy
            name: Both values recalled, Twi (24/24)
            value: 1.000000
          - type: english_both_slots_accuracy
            name: Both values recalled, English (18/24)
            value: 0.75
        source:
          name: Ghana Health AI matched checkpoint evaluation
          url: https://github.com/teckedd-code2save/ghana-health-ai/blob/main/docs/TWI_UNDERSTANDING_SCALE_EVALUATION.md
pipeline_tag: text-generation
language:
  - tw
  - en
tags:
  - lora
  - research
---

# Ghana Health AI: Twi R5

**Availability:** This Hugging Face release is a model card only. It contains no
downloadable adapter, base weights, training examples or inference endpoint.

In the matched 11 November comparison, R5 picked the published English meaning
in both answer orders for 900 of 1,092 Twi sentences (base: 636). In independent
conversation histories it recalled both the changed and unchanged fact in
24/24 Twi scenarios and 18/24 English scenarios. The former measures agreement
with fallible references; the latter uses four authored templates and does not
establish general conversational usefulness. See the [side-by-side results and
protocol](https://github.com/teckedd-code2save/ghana-health-ai/blob/main/docs/TWI_UNDERSTANDING_SCALE_EVALUATION.md).

R5 continues the final R4 LoRA adapter to study useful Twi conversation,
meaning preservation and general instruction following with English retention.
It is a text model adapter, not a standalone base, ASR/TTS model or clinically
validated assistant. This pass does not promote it to the public application.

This companion card supplements the unchanged, hash-sealed PEFT README.
Private training sources, model bytes and evaluation artifacts were not uploaded
publicly. Upstream model and individual source terms remain applicable; the
private research selection includes restricted sources and grants no new rights.

## Identity

| Item | Value |
| --- | --- |
| Base | `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` |
| Base revision | `693f12e444a30971ca53889128d7bade9491ecb3` |
| Initial R4 adapter SHA-256 | `1c18338e732f7461901403f5a50c876d64712c21285cf24b2f78a86530d87aca` |
| Final R5 adapter SHA-256 | `2085ee881927ee559d4bd978ae2372e12dea963d588fd136b2130bc551835d25` |
| Training run | `understanding_20261110T103734Z` |
| Final checkpoint | Update 228, complete schedule |
| Dataset version | `r5-reviewed-conversations-v1-20261109` |
| Dataset manifest SHA-256 | `29e23dfa6d363a30b9d3f457963513d3f6777d3c0e88b8d62ed24d30d7340147` |

The final 173,181,072-byte adapter was read back from private Modal storage and
its full checksum verified. The comparison disables adapters for the base and
verifies the selected R4/R5 weights. No other model rewrites its responses.

## Actually Consumed Data

| Dataset | Registered records | Distinct views | Presentations | Supervised tokens |
| --- | ---: | ---: | ---: | ---: |
| Responses | 657 | 908 | 1,099 | 192,703 |
| English retention | 217 | 314 | 314 | 46,920 |
| Twi language replay | 1,187 | 1,187 | 1,187 | 31,856 |
| Meaning replay | 1,047 | 1,047 | 1,047 | 32,115 |
| Total | 3,108 | 3,456 | 3,647 | 303,594 |

The consumed-data receipt matches every scheduled view to its versioned dataset
record and checksum. Complete histories can yield several final-assistant
prefix views; a repeated presentation is not an independent new conversation.
There are 2,849 source IDs across the selected classes, with related bilingual
and derived records retaining their common ancestry.

The response selection includes 106 newly agent-translated/corrected histories
from existing English source conversations, with 191 assistant targets. All 240
candidate histories received dispositions; 134 remain excluded with reasons.
A separate 24 project-authored repair histories supply 72 targets. Existing
R4 records, the complete 333-record instruction export and explicitly marked
spelling derivatives supply the remaining response data. The instruction export
contains prose, language exercises, code and literal-response tasks, not 333
natural conversations. English text/code requested by a task remains explicit.

New source-backed conversations receive two exposures and contribute 52.66% of
response-target tokens. New project-authored repairs contribute 1.63% and are
not multiplied. New conversation derivatives were separately agent-reviewed,
not certified by native-human or clinical reviewers. Original English sources,
explicit corrections, Twi derivatives and review decisions remain preserved.

Language/meaning replay uses GhanaNLP parallel and speech-transcript records,
INJONGO and community parallel material. English retention uses source histories.
The four pack-local dataset snapshots have separate records, manifests and cards.

## Training Method

- Frozen base: 8,953,803,264 parameters; trainable rank-16 LoRA: 43,278,336.
- Exact final R4 initialization; fresh optimizer/scheduler, learning rate `1e-5`,
  accumulation 16, warmup fraction 0.05 and seed 42.
- One complete response pass, then a new-source-conversation consolidation pass,
  with finite, distinct replay. The first-pass snapshot is retained at update 171;
  the final update-228 checkpoint is served, not selected using fresh-test outcomes.
- Native tokenization and unchanged vocabulary/template; complete context,
  final-assistant supervision and EOS checks. Causal supervision for language rows.
  No packing or truncation; 620,048 input tokens consumed.
- Actual target-token mix: 63.47% responses, 15.45% English retention,
  10.49% Twi language and 10.58% meaning.
- Real-weight backward and bitwise interruption/recovery checks passed before
  training. The completed worker reports 2,191.47 seconds, about 36.5 minutes.
- Private Modal H100; bf16 base and fp32 adapters. Runtime: torch 2.13.0,
  transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, numpy 2.4.2,
  triton 3.7.1, flash-linear-attention 0.5.2. The same reference causal-convolution
  implementation remains in use; no unmeasured kernel speedup is claimed.

## Evaluation Status

Training, full adapter readback and all four chat evaluation runs are complete.
Retention completed with 1,160 measurements and one explicitly indeterminate
measurement preserved after interruption; there are no pending slots. Training
loss is not a quality score. All 1,146 chat slots have rating coverage: 573 raw agent judgments,
559 exact raw-to-guarded transfers and 14 separately reviewed changed outputs.

| Raw generation measurement | Base | R4 | R5 |
| --- | ---: | ---: | ---: |
| Development EOS replies / 75 | 70 | 71 | 70 |
| Development strict JSON matches / 8 | 6 | 8 | 6 |
| Development replies with repeated 4-grams / 75 | 9 | 9 | 9 |
| Fresh EOS replies / 116 | 109 | 112 | 109 |
| Fresh replies with repeated 4-grams / 116 | 29 | 26 | 31 |

Fresh guarded EOS counts are 109/116, 112/116 and 110/116 respectively; repeated
4-gram counts remain 29/26/31. No guard-triggered pause occurred in this suite.
Seven changed outputs/histories received separate review; 341 exact counterparts
link to their original raw agent ratings, with explicit transfer provenance.
Small differences without guard triggers are not treated as guard improvements.
Development guarded EOS counts are 70/75, 72/75 and 70/75; guards pause one,
two and one replies respectively. Shortening a loop does not restore meaning.

The two R5 JSON failures contain the correct arithmetic answer inside Markdown
fences, violating JSON-only output. Repeated 4-grams can include legitimate shared
phrases; they are a diagnostic, not a semantic-failure count. Completion is not
accuracy. All 348 fresh raw outputs now have separate agent reviews.

All 225 development outputs have separate agent meaning/usefulness reviews.
R5 versus R4 has four meaning gains, four regressions, 64 ties and three uncertain
pairs across 75 output slots. These ordinal agent judgments do not establish
broad improvement over R4, native-human accuracy or clinical safety. Original
outputs and pending human-review fields are preserved. The local comparison
serves the trained R5 checkpoint for inspection, not as a promoted default.

| Fresh agent meaning judgment | Base | R4 | R5 |
| --- | ---: | ---: | ---: |
| Pass / 116 slots | 49 | 66 | 72 |
| Partial | 42 | 36 | 27 |
| Fail | 24 | 10 | 12 |
| Uncertain | 1 | 4 | 5 |
| Pass among 60 requested-Twi slots | 10 | 24 | 29 |
| Pass among 56 requested-English slots | 39 | 42 | 43 |

R5 versus R4 has 11 meaning gains, eight regressions, 90 ties and seven uncertain
pairs. Usefulness has ten gains, nine regressions and 97 ties. Requested-Twi
failure counts are base 20, R4 nine and R5 ten; more full passes coexist with
some regressions. These are ordinal, non-blinded agent assessments, not gold
accuracy. The 116 slots are related outputs from 40 source groups, not 116
independent samples. Fresh proposal wording and native/domain validation remain
limitations. Gains include some uncertainty/permission distinctions and context
comparisons; clarification, negation, quantity arithmetic, context updates and
unsupported action promises remain uneven. No decisive R5-over-R4 improvement
or production readiness is established.

The protocol compares base, final R4 and final R5 with independent rollout
histories and matched decoding. Raw and guarded generation remain separate.
The reused development suite has 46 fixed cases and 29 rollout turns per model.
Retention planned 343 Twi rows and 44 English source histories per model.
The fresh suite has 40 source groups, 84 cases and 116 scoring slots per model;
related language/spelling variants are not independent samples.

Final source audits found no selected-pack exact, normalized, near or recorded
source-group overlap with prior/fresh protection. Earlier ancestor overlap with
reused development cases remains disclosed. No independence from unknown base
pretraining is established. Native-human and domain judgments remain pending.

### Paired Retention Results

Only the identical measured case intersection is compared. One interrupted
R5 Twi measurement is excluded for all models; it was not automatically retried.

| Cohort | Cases / planned | Tokens/model | Base perplexity | R4 perplexity | R5 perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Twi native causal prediction | 342 / 343 | 9,300 | 15.411176 | 5.197391 | 5.128916 |
| English final-assistant prediction | 44 / 44 | 8,534 | 3.135279 | 3.100235 | 3.101749 |

These are exponentiated token-weighted NLL values with native EOS and identical
token masks within each cohort. Twi and English use different objectives, so
their values are not pooled or compared as language-quality rankings. R5's Twi
loss is slightly lower than R4's; English loss is essentially unchanged on this
cohort. This is language prediction evidence, not conversational accuracy,
clinical safety or proof of a statistically significant improvement. The
base-to-adapter gain includes adaptation inherited from earlier stages.

One user-confirmed phrase correction is attached separately to an R5 arithmetic
reply: `yɛnnya` should be `yɛnya`. It resolves the earlier agent's wording
uncertainty, not approval of the complete answer or dataset. Saved outputs and
aggregate agent ratings remain unchanged; the evaluation example is not training
data. Full source-bound evidence is in the private evaluation viewer.

Official IFEval support is separate: 541 original English-prompt tasks, 834
instructions, the pinned official scorer and an 8,192-token output allowance.
Its dedicated inherited/new-source overlap audit passed. No IFEval score is
claimed before full generation and scoring; it does not measure Twi understanding.

## Use and Limitations

Use the private local base/R4/R5 comparison for research feedback. Repetition,
wrong meaning, invented facts, weak reasoning and language drift remain possible.
The shared guard limits loops without repairing meaning. Incomplete responses
and errors remain visible, and context exclusions are explicit and recorded.
No current web facts or real shopping actions are connected to this comparison.
Do not rely on R5 alone for consequential medical, legal or purchasing decisions.

See the [execution record](https://github.com/teckedd-code2save/ghana-health-ai/blob/main/docs/TWI_UNDERSTANDING_R5_20261109.md) for dataset admission,
private storage, checksums, actual run receipts and the evaluation handoff.
