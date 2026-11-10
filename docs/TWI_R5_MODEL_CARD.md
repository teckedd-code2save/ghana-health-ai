---
base_model: McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1
library_name: peft
pipeline_tag: text-generation
language:
  - tw
  - en
tags:
  - lora
  - research
---

# Ghana Health AI: Twi R5

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

Training and full adapter readback are complete. R5 comparison results are
pending actual evaluation and review; training loss is not a quality score.

The protocol compares base, final R4 and final R5 with independent rollout
histories and matched decoding. Raw and guarded generation remain separate.
The reused development suite has 46 fixed cases and 29 rollout turns per model.
Retention measurements cover 343 Twi rows and 44 English source histories.
The fresh suite has 40 source groups, 84 cases and 116 scoring slots per model;
related language/spelling variants are not independent samples.

Final source audits found no selected-pack exact, normalized, near or recorded
source-group overlap with prior/fresh protection. Earlier ancestor overlap with
reused development cases remains disclosed. No independence from unknown base
pretraining is established. Native-human and domain judgments remain pending.

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

See the [execution record](TWI_UNDERSTANDING_R5_20261109.md) for dataset admission,
private storage, checksums, actual run receipts and the evaluation handoff.
