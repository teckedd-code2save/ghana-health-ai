# Twi Research Overview

This page separates research tasks, measured outcomes, provenance, and reproduction requirements. It accompanies the [README](../../README.md), not a dataset release or clinical readiness certificate.

The presentation takes a cue from [BLUFF's task/metric table and experiment documentation](https://github.com/jsl5710/BLUFF#-benchmark-tasks). Our tasks and datasets differ; this does not imply comparable scores, evaluation on BLUFF, or an affiliation.

## Research Scope

The programme investigates Twi understanding and conversational generation while retaining general capabilities. Speech recognition and synthesis are evaluated separately. Health and commerce provide application settings, not a restriction of the research question to medical classification.

Existing documentation: [research programme](../GHANA_LANGUAGE_UNDERSTANDING_RESEARCH.md), [research stack](../research-stack.md), [ASR decision record](../asr-model-decision.md), and [R3 application access](../R3_GUEST_CHAT.md). These include historical decisions; their dates and limitations matter.

## Completed Development Evaluation

Recorded 6 November 2026, run `r3_eval_20261106T072728Z`. All three variants use `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1`, revision `693f12e444a30971ca53889128d7bade9491ecb3`. R2 and R3 are adaptations of that foundation, not models trained from scratch.

| Measure | Base | R2 meaning | R3 response |
| --- | ---: | ---: | ---: |
| Token-weighted mean NLL | 2.737638 | 1.654256 | 1.641245 |
| Perplexity | 15.450454 | 5.229186 | 5.161592 |
| Strict literal JSON matches with EOS | 6/8 | 8/8 | 8/8 |
| EOS completion / 75 slots | 70 | 71 | 67 |
| Token-limit incomplete replies | 5 | 4 | 6 |
| Error / blocked slots | 0 / 0 | 0 / 0 | 1 / 1 |
| Repeated four-word sequence present | 9/75 generated | 9/75 generated | 13/73 generated |

Protocol and denominators:

- 46 fixed histories: 36 earlier development cases and 10 observed cases. Raw observed conversations are not reproduced here.
- 10 scripted rollouts, 29 turns per checkpoint. Fixed tests use identical frozen messages and native prompt IDs across variants; rollouts retain each model's own replies.
- 343 language-loss rows and 9,315 predicted tokens per variant. Total: 138 fixed generations, 87 rollout slots, and 1,029 loss slots, or 1,254 finalized slots.
- Frozen greedy, non-thinking decoding; maximum 256 new tokens and 4,096-token context. These saved results used the raw protocol, not later interactive repetition containment.
- Eight literal JSON checks are not broad meaning accuracy. Repeated four-word sequences can be harmless; EOS does not imply task success.

R3 perplexity is approximately 1.29% below R2, but this does not establish reliable overall conversational improvement. Regressions and incomplete follow-up behavior remain. Native-human and clinical certification are pending; this previously exposed development set is not a blind final benchmark.

Interactive comparisons must state their history policy. The earlier shared-history comparison supplied candidate replies to both sides: a conditional comparison, not independent conversations. A later six-generation, three-turn-per-model probe verified independent histories with equal within-response containment; it is a protocol check, not a replacement benchmark. Containment does not repair meaning or prevent all cross-turn repetition.

## Data and Training Accountability

R2 completed three stages with 1,999 updates. Counts below describe actual
consumption, not the stored pool or the complete corpus archive.

| R2 class | Distinct views consumed | Source and form |
| --- | ---: | --- |
| Twi language | 16,742 | 5,385 GhanaNLP parallel sentences, 11,350 GhanaNLP speech transcripts, 7 community utterances |
| Bilingual meaning and intent | 13,439 | 10,752 bilingual views and 2,687 intent views |
| Response trial | 306 | 86 Twi prose records and 220 language exercises; each presented twice |
| English retention | 992 | Sampled English conversation histories from a 2,131-view pool |
| Total | 31,479 | 20,731 distinct source IDs across the classes |

R2 consumed 1,193,890 supervised target tokens across all three stages.
These totals include its separate response trial. R3 inherits the earlier
language-and-meaning checkpoint, not that response trial. Source IDs shared
between classes or runs must not be added as new independent sources.

R3 initialized from verified R2 **meaning**, not the R2 final-response stage, with a fresh optimizer and scheduler. Main training completed 105 updates on dataset version `20261106-r3-v1`.

| Class | Unique packed views consumed | Presentations | Effective target tokens |
| --- | ---: | ---: | ---: |
| Responses | 229 | 458 | 89,922 |
| English retention | 147 | 147 | 29,973 |
| Meaning | 491 | 491 | 14,985 |
| Twi language | 582 | 582 | 14,859 |
| Total | 1,449 | 1,678 | 149,739 |

The receipt records 1,411 unique dataset records and 281,243 input tokens. Packed views, records, source inventory, repeated presentations, and target tokens are different units.

The records reconcile as 582 language + 491 meaning + 191 responses + 147
English = 1,411. The 191 response records expand into 229 assistant-target
views; the other classes retain one view per record, totaling 1,449 views.
The full mix traces to 1,388 distinct source IDs after cross-class deduplication.

The response snapshot contains 191 records: 86 inherited R2-source records, 80 source-grounded AI-translated and agent-corrected records, and 25 project-authored AI-translated records. Private research admission is not native-human translation certification, clinical approval, or permission to redistribute underlying sources.

Training accountability binds rows to a dataset/version/record/source, preserves earlier turns as context, and separates selected inventory from checkpoint-backed consumption. Raw private datasets and owner histories are not distributed by this page.

## Provenance and Availability

These aggregates were checked against the locally held `source-evidence.json` for the 6 November final research brief and completed evaluation evidence. Those private files and local research worktrees are **not public download links**. This overview publishes aggregate results and methods; it does not release restricted evidence or weights.

Selected SHA-256 identities for matching authorized receipts:

| Artifact | SHA-256 |
| --- | --- |
| Evaluation summary | `614a6c17c2007a464cb6e42d5a33eddde613ccea6753aea23bc45332f20a4363` |
| Evaluation execution | `62b79d50ab3c3465f9502ad6fd9ee3df5dd357d614912a4178fdb4c14c0697f6` |
| Inference manifest | `62fc8df677147e175fb4f22c63d8a78281cebef9f1f0fc57991bdbf726091af4` |
| R3 response adapter | `5cc0537b21674231b640ae9109a08ffa6fab426c80ac1969705dfac9c9e6b874` |

A checksum identifies bytes; it does not grant access or certify quality. No public R3 weight or private dataset download is promised.

## Reproduction Boundaries

Published code entry points include [R2 training](../../modal/train/train_understanding_r2.py), [R3 training](../../modal/train/train_understanding_r3.py), [R2 evaluation](../../modal/train/evaluate_understanding_r2.py), [R3 evaluation](../../modal/train/evaluate_understanding_r3.py), and the [command inventory](../../package.json). The [ASR benchmark ladder](../../modal/train/benchmark_asr_ladder.py) evaluates a separate speech task.

Exact R3 reproduction requires matching sealed execution, admitted snapshots, pinned base/tokenizer and adapter artifacts, evaluation protocol, and authorized access. A clean public clone alone is insufficient. Do not substitute mutable data or advertise local-only worktree scripts as published assets.

A future public experiment release should include license-reviewed data documentation, task definitions and splits, immutable code/configuration and model revisions, executable evaluation instructions, raw-versus-derived provenance, and explicit failure/approval criteria. Until then, these aggregates document a private development run, not a fully downloadable public benchmark.
