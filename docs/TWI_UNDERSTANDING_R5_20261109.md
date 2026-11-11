# R5 Meaning and Conversation Repair

Issue: #59. Continues the R4 research work in PR #58. R4 and its evidence remain
immutable. The target is useful Twi conversation, with faithful meaning across
turns, not completion scores alone.

## Execution Map

| Node | Work | Status | Evidence or exit condition |
| --- | --- | --- | --- |
| A | Preserve and review lab evidence | Complete, agent review | R4 raw/guarded exports plus local snapshot of 5 saved sessions, 13 rounds and 39 reply slots; no saved human assessments |
| B | Recover the blocked comparison UI | Complete | 46 focused tests passed; actual failed session recovered in the browser, new follow-up completed on all three models, input enabled afterward |
| C | Audit and repair the broader training pool | Complete for this version, agent-reviewed | 240 source histories reviewed; 106 admitted with 191 assistant targets; 134 excluded with reasons. Separately, 24 project-authored histories provide 72 targets |
| D | Freeze fresh held-out scenarios | Final source audit passed | 40 source groups, 84 cases, 116 scoring slots/model; complete prior/fresh protection with zero selected-pack matches; native/domain judgments remain pending |
| E | Seal R5 training data and run | Complete, full readback verified | 228 updates; all 3,647 scheduled exposures and 303,594 target tokens consumed from registered data |
| F | Evaluate and compare locally | Complete, with one preserved missing measurement | All 1,146 chat outputs review-covered; 1,160 retention measurements; matched comparison on 342 Twi and 44 English cases; base/R4/R5 at localhost:7869 |
| G | Back up and publish research handoff | Complete for this research pass | Final 4,791-file evidence backup read back and verified; dataset/model cards and local comparison ready; stacked PR #60 stays draft until normal CI is available |

## Confirmed Starting Findings

- R4 added 33 original agent-translated histories. Their original and spelling-
  augmented targets contributed about 7.4% of response-target tokens. The next
  run must broaden actual supervision, not merely multiply these examples.
- R4 finishes 73/75 raw chat slots versus R3's 67/75. All six additional finishes
  occur in rollouts; two correspond to R3 execution/blocked slots. Fixed-case
  completion is unchanged at 45/46. Finishing is not evidence of correct meaning.
- Meaning failures include invented medicine categories, uncertain plans stated
  as certain, asking for already-declared unknown information, wrong topic
  persistence, and loss of remembered facts. Retain each observation separately
  from any hypothesis about its training cause.
- The supplied lab session shows a truncated R3 response ending in a replacement
  character. The next turn fails validation, then the comparison disallows all
  follow-ups. This is a recoverability bug, separate from model understanding.
- Fresh evaluation is necessary: the existing suite is reused development data.
  English language-loss measurement remains an unexecuted item from R4's plan.

## Data and Evaluation Rules

1. No changes to frozen R2/R3/R4 data, reviews, model bytes or results.
2. Every new training view resolves to a versioned dataset record and source.
   Corrections preserve the original and state their reason and reviewer type.
3. Project-authored, agent-translated, source-preserved and human-reviewed data
   are distinct. Automated passing checks do not grant semantic certification.
4. Full related conversations and bilingual versions share a split group.
   New held-out scenarios stay out of generation prompts, selection by outcome
   and training. Existing failure cases are development diagnostics.
5. Model output is never rewritten by another model in the comparison UI.
   Context recovery must be explicit and recorded, with exact submitted input.
6. No production promotion, public dataset upload, ASR or TTS changes in this pass.

## Benchmark Contract for Every Training Run

Every completed training run is followed by a checkpoint-matched comparison
against its initialization checkpoint and the adapter-disabled base. Data and
evaluation versions, decoding settings, exact inputs, outputs, failures and
checkpoint identities are retained. A checkpoint is not described as improved
merely because training finished, loss fell or more replies reached EOS.

| Capability | Evidence required | How it guides the next data work |
| --- | --- | --- |
| Meaning | Negation, experiencer, uncertainty, time, quantities and entity corrections | Identify which semantic distinctions are lost |
| Conversation | Independent multi-turn histories, follow-ups, topic changes, remembered facts and useful next steps | Repair conversational supervision, not isolated phrases only |
| Language | Twi naturalness, requested language, code-switching and informal spelling; English retention separately | Detect language drift and regression |
| Reasoning and actions | Verifiable arithmetic, constraints, catalogue/tool outcomes and honest capability limits | Distinguish executable success from plausible prose |
| Factual uncertainty | Unsupported claims, time-sensitive questions and appropriate clarification | Target invented facts without rewarding blanket refusal |
| Safety | Harmful or unsupported advice and unnecessary refusals, with domain-review status explicit | Separate language fidelity from safe health-education behavior |
| Reliability | Repetition, truncation, errors, first-text latency and total latency | Separate serving problems from model-quality failures |

The reused development suite measures regression on known cases. Fresh held-out
cases measure transfer only after an audit against inherited as well as new
training sources. Neither suite establishes independence from unknown base-model
pretraining. Raw and guarded results are separate: stopping a loop does not turn
its answer into a correct one.

Each capability reports its own case and source-group denominators, correct,
failed and unreviewed counts, and paired improvements/regressions with examples.
Semantic judgments identify their reviewer type. Missing review remains missing,
not a pass. Spelling and bilingual variants are related cases, not independent
sample-size inflation. Perplexity is reported by language and token protocol;
it measures prediction loss, not factual accuracy or clinical safety.

The handoff must end with an evidence-based list of strengths, unresolved errors,
regressions and the next data/training changes. No single aggregate fluency score
or selected successful demo substitutes for this breakdown.

### External Benchmark Anchors

The benchmark program has three layers: established public tasks, Twi-specific
meaning/conversation evaluation, and end-to-end product behavior. The current
runner implements the latter research comparisons and language-loss cohorts;
standard public task execution is not yet complete. Public benchmark scores must
not be claimed before their official task/scorer has actually run.

- EleutherAI's [evaluation methodology](https://arxiv.org/abs/2405.14782)
  informs pinned task versions, prompts, model configuration and reproducible
  scoring. Retain raw model outputs and enough configuration to rerun the score.
- [HELM](https://crfm.stanford.edu/2022/11/17/helm.html) motivates separate
  capability and reliability dimensions, not one overall fluency number. This
  is methodological guidance, not a claim that this project has run HELM.
- Google's [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
  is the first proposed standard instruction-following anchor, using its own
  pinned data and scorer. English results measure English instruction-following.
  A Twi derivative needs language-aware checking and its own version/validation;
  it is not the original benchmark or directly leaderboard-comparable.

Frozen tests are kept separate from development diagnostics. Repeatedly inspected
release cases become regression evidence; they are not indefinitely described as
unseen. Confidence intervals and paired changes should respect source groups,
not treat translated/spelling variants as independent observations. Native
meaning/naturalness judgments and clinical judgments remain different reviews.

## Current Data Work

### Dataset Registration Before Training

Every selected example must be a record in a versioned dataset, not an untracked
prompt passed directly to the trainer. The R5 builder writes separate `language`,
`meaning`, `responses`, and `english_retention` snapshots, each with records,
a manifest and a dataset card. Every training view carries the dataset ID,
version, record ID, source ID and record checksum. The runner rejects missing
records, changed contents, unadmitted records and history-prefix mismatches.
Spelling variants retain their parent admission and transformation reference.

The consumed-data receipt will distinguish dataset records, derived training
views, repeated presentations and supervised tokens. Until a completed worker
receipt exists, counts describe preparation only. All selected research sources
and review decisions are privately backed up; they are not public datasets.

The current continuation passed 201 local preparation, training-contract,
inference and workbench tests. These verify code behavior, not Twi quality.
The independent conversation reviews are complete and the final pack is sealed.
Repairable source errors are corrected in explicit new
English/Twi derivatives, with original source bytes and earlier decisions kept.
Examples include invented performance-review evidence, inconsistent recipe
ingredients and invalid CSS instructions. Separate reviewers check those
derivatives; a source correction does not admit itself.

The source work plan selects **240 existing English conversation histories from
240 source trees**, with **209 multi-turn histories and 449 assistant occurrences**.
These are translation/review candidates, not newly admitted training records.
The larger source pool contains 2,131 histories, including 879 multi-turn histories.
Four disjoint source shards preserve full histories and source hashes. Selection
protects existing evaluation sources; admission must also check the fresh suite.
Translation and review are separate steps. Source defects can cause exclusion or
an explicitly documented derivative, never a silent edit of the original.

The audit also identifies the full 333-row previously cleared instruction export:
86 Twi prose, 220 language exercises, 17 code exercises, 9 literal labels and one
literal structure example. These are distinct task types, not 333 natural chats.
Their suitability and mix are evaluated separately from the conversation expansion.

Three R4 English replay histories trigger an unanchored-current-information
screen. A broader R2 replay example asserts an old date as today's date without
clock context. Such material needs an explicit decision before reuse. No exact
weather refusal or invented animal-medicine answer was located in R4 response
targets; these observed failures are not evidence that the same text was trained.

The recovered lab session retains every original output. Only R3 turns 5 and 6
were explicitly excluded from its future context, because one is malformed and
the next failed validation. Base and R4 retain their own six turns. The recovery
test was appended as an agent UI check, not a user quality assessment. The raw
mode remains raw; recovery is not a claim that repetitions or meaning are fixed.

There are also 24 project-authored bilingual repair histories, containing 72
assistant turns. They target semantic contrasts and recovery behavior and passed
a separate agent review. They are not counted as naturally collected conversations.
Repairable source defects are corrected in explicit English/Twi derivatives with
reasons; unsafe or unresolved material is excluded from this run, not deleted.

### Previously Consumed Data

The R2 checkpoint-backed consumption audit resolves archive size versus actual
training. Across its stages, it consumed 31,479 distinct training views from
20,731 source IDs, with 31,963 presentations and 1,193,890 supervised target
tokens. This is not a count of natural conversations or the full raw archive.

| Class | Distinct views consumed by R2 | What those views contain |
| --- | ---: | --- |
| Twi language | 16,742 | Source sentences and speech transcripts |
| Meaning | 13,439 | Bilingual and intent examples, from 8,063 source IDs |
| Responses | 306 | 86 prose instructions and 220 language exercises |
| English retention | 992 | Complete English source histories |

Those totals describe the whole R2 experiment, not all inherited R4 training.
R3 initialized from R2's **meaning** checkpoint, before R2's separate response
stage. The ancestral R2 language/meaning stages contain 31,115 distinct views
and 1,084,400 effective target tokens; 364 additional distinct views occur only
in the non-ancestral response stage. R3 and R4 then have their own recorded
response training. The R5 work addresses that conversation supervision, while
preserving the language/meaning adaptation. The private audit verifies the
checkpoint links and separates actual ancestor consumption from sibling stages.

The broad audit's 14 decoded-string mismatch flags (seven instruction rows in
each of R3 and R4) are not evidence of 14 corrupt targets. Rechecking all 84
R4 packed instruction views using the native decoding contract gives 84 exact
matches. The broad string check used default whitespace cleanup and an untrimmed
source; a separate note records this distinction without changing old evidence.

## Spending and Execution

No OpenAI requests were made in this pass. The existing lifetime US$10 ledger
remains authoritative; completed reservations have not been reset. Three private
Modal inference requests verified lab recovery. The real-weight GPU check
`understanding_20261110T102958Z` completed in 235 seconds with exact initial
adapter loading, backward training and bitwise uninterrupted/resumed equivalence.
The full R5 run `understanding_20261110T103734Z` completed all 228 updates in
2,191.47 seconds. Its receipt is `train-v1/launch.json`, with private Modal
app `ap-hLgXKc5nolUhHUmPRLZYTv`. It uses one H100, fresh optimizer/scheduler,
learning rate `1e-5` and a 10,500-second soft budget. The completed consumption
receipt verifies all 3,108 dataset records, 3,456 distinct views, 3,647 exposures
and 303,594 supervised tokens. No additional data was supplied outside the pack.

## Prepared R5 Dataset

The complete reviewed pack is `tmp/understanding-r5-20261109/pack-v1`, version
`r5-reviewed-conversations-v1-20261109`. Its manifest SHA-256 is
`29e23dfa6d363a30b9d3f457963513d3f6777d3c0e88b8d62ed24d30d7340147`.
The completed consumption receipt now matches the preparation counts below.

| Registered dataset | Records | Distinct training views | Scheduled target tokens | Token share |
| --- | ---: | ---: | ---: | ---: |
| Responses | 657 | 908 | 192,703 | 63.47% |
| English retention | 217 | 314 | 46,920 | 15.45% |
| Twi language replay | 1,187 | 1,187 | 31,856 | 10.49% |
| Meaning replay | 1,047 | 1,047 | 32,115 | 10.58% |
| Total | 3,108 | 3,456 | 303,594 | 100% |

A complete history can produce multiple final-assistant training views. The
schedule presents 191 new source-backed targets twice, giving 3,647 exposures
and 228 optimizer updates; other targets are presented once. These are not
3,647 independent conversations. Related English/Twi versions and spelling
derivatives retain common source identity and split protection.

All 130 newly admitted histories survive pack selection: 106 source-backed,
agent-translated/corrected histories plus 24 project-authored repair histories.
The source-backed additions contribute 52.66% of response-target tokens;
project-authored repairs contribute 1.63%. The remaining response supervision
comes from inherited R4 records, the complete 333-record instruction export,
and explicitly marked spelling derivatives. No new record is described as
human-reviewed or clinically certified.

The complete candidate ledger retains 28 duplicate views, four further
screening exclusions and 186 protected-content exclusions. The protection
extension accounts for derived rollout contexts missed by the earlier scan,
and conservatively excludes flagged source groups. The separate broad final
audit cleared this exact pack: all 248 input hashes were rechecked; all 3,456
rows and witnesses resolve to registered records; there were zero selected-pack
exact, normalized, near or source-group matches against prior/fresh protection.
The clearance is `evaluation-design/final-pack-audit-v1/clearance.json`, SHA-256
`d34fd7befb1ecb4ed51db8001fe74d3f5e0c594f5ba810942e714e102d4438fa`.
Earlier development overlap in ancestor training remains disclosed; the audit
does not establish independence from unknown base-model pretraining. Original
sources and earlier decisions remain unchanged.

For scale comparison, the completed R4 run consumed 1,544 registered records,
1,642 distinct views and 161,773 target tokens. This R5 continuation schedules
about 1.88 times those target tokens and about 2.10 times the distinct views.
This measures supervision volume, not an improvement in model quality.

The 2,234 selected language/meaning records comprise 1,272 GhanaNLP parallel,
786 GhanaNLP speech-transcript, 174 INJONGO and two community-parallel records.
These are class-specific registered records, not a count of unique new source
documents. The response origins and complete histories remain inspectable in
`pack-v1/datasets/responses/records.jsonl`; all four snapshots have dataset cards.

The private offline explorer is `dataset-view-v1/index.html` under the R5 artifact
root. It presents exactly the consumed records, with separate class counts,
search, complete registered messages, available English references, review
metadata and source provenance. Its counts are reconstructed from the completed
consumption receipt and checked against the sealed pack, not a separate registry.
It makes no network requests and adds no quality certification. Regeneration uses
`scripts/export_understanding_r5_dataset_view.py --pack <pack-v1> --consumption
<train-v1/completed/consumption.json> --out <new-private-directory>`.
The explorer and training-contract tests passed 43 checks, including nine
explorer tests. Browser interaction was checked by the implementation agent;
the main agent's in-app browser refused the local-file URL. The local artifact
remains directly openable by the user without a server.

The official IFEval adapter and scorer are implemented separately. Its 541
English-prompt tasks are not Twi-understanding results, and no R5 benchmark
score exists before training and actual evaluation.
The dedicated IFEval source audit also passed all 541 original prompts against
the final pack and inherited actual consumption/full source histories, with zero
exact, normalized, near or recorded-lineage matches and no benchmark cases removed.
Its clearance is `evaluation-design/ifeval-source-audit-v1/clearance.json`.

## Local Comparison and Evaluation

The completed adapter is served through the private Modal actor
`ghana-understanding-r5-demo-private`. The loopback UI is
<http://127.0.0.1:7869/?__theme=light>. It compares adapter-disabled base,
final R4 and final R5 with independent histories and matched raw/guarded modes.
Its private bundle SHA-256 is
`6e232b799c622fa55885baa1f76d5d86b1388974144ae16ddf5f5b9ef02076a6`.
The [R5 model card](TWI_R5_MODEL_CARD.md) records architecture, sources,
actual consumption and evaluation limitations.

An agent UI check completed two turns on all three models, verified the actual
inputs/weights, and left the question box usable for another turn. In the
follow-up asking for the speaker's name, R5 answered `Wo din de Kofi.` (your
name is Kofi), while base echoed the earlier request and R4 said `Me din de
Kofi.` (my name is Kofi). This is a small integration/qualitative observation,
not a held-out accuracy score or a saved human assessment. The local session is
`04a769d76a004b8b9f4f278a60587da9`; its first two turns preserve all six outputs.
Cold-start and warm timings are not a fair latency comparison.

Development raw evaluation `r5_eval_20261110T112211Z` has 1,386 planned slots:
225 chat outputs plus 1,161 language-loss measurements across base/R4/R5.
It saved 443 result records before Modal preempted its GPU. Modal's restart on
the same input was rejected by the duplicate-attempt guard. The original call
is terminal with `ValueError: Attempt already claimed`; its app stopped at
11:55 UTC. The result journal and original execution package remain unchanged.
The client also exposed a recovery bug: terminal polling handled wrapped Modal
exceptions but missed a deserialized remote `ValueError`. The repaired predicate
requires an actual result or SDK-proven remote exception; unknown local/transport
errors remain nonterminal. A separately sealed recovery controller changes only
that callback in the unchanged evaluator, records the controller and terminal
proof before resumption, and verifies every saved journal/provenance hash.
It retains the atomic attempt claim and never retries completed or indeterminate
requests. The controller is `recovery-controller-v2`, SHA-256
`e6c67e1507cd79efcfcfaf1e4c0b8c4381b0209df83bea8d1ddf84946bd396da`;
52 recovery and evaluation tests passed. Its attempted deployment at 12:39 UTC
failed before any function was spawned: the wrapper added an image build setting
after file mounts, which Modal rejects. The incomplete launch receipt is retained,
not reused; no recovery GPU or generation ran. A corrected image-construction
order is fixed in `recovery-controller-v3`, SHA-256
`db470bdb00024fabbb5d97df8071f97850ec877b08165131c9fe6e0da1161525`.
The offline SDK-loader test reproduces the original rejection and verifies the
new wrapper's mount order. Recovery, evaluation and both viewer tests pass 80
checks. That controller reached a worker but failed before a claim or model load:
the worker-mounted Modal SDK has no installed distribution metadata. Version
checking now reads the mounted SDK's exact `__version__`, still requiring 1.4.3.
The v3 terminal receipt is retained. Offline regressions cover missing distribution
metadata and reject missing/wrong SDK identity. The complete R5 evaluation test
selection passes 103 checks. Controller v4, SHA-256
`3e4f7235578146cfaff09aacae93ff521d4cb47e4dffe82b642883abe1806daa`,
was submitted as call `fc-01M72TWV82PVEP4JT8216Y0NXF`. It claimed attempt 2 and
loaded the pinned model. It resumes only unstarted retention slots; the original
443 durable results and one indeterminate request remain preserved. Training and
local inference are unaffected. Failed controllers and receipts are preserved.

Attempt 2 subsequently reached its 7,200-second outer timeout with 1,332 saved
records, including 1,106 measured losses and the original indeterminate loss.
There was no newly started request without a saved result. The complete second
snapshot verifies all 899 protected files from the first snapshot unchanged.
Recovery verification occurs before the inner evaluation timer starts, so startup
overhead can exhaust the outer timeout before the inner soft pause. Controller v5
uses eight bounded file readers for every protected checksum instead of serial
reads; it still rejects additions, removals, changed bytes and links. No weights,
data, scoring, prompts, decoding or original journal checks change. Its checksum
is `02375c79f7cf58d8a645a9dafb36a886535512949c482b4426a0265159caaa30`.
Attempt 3 (`fc-01M732DBPVW13TVHZ4ZX55BTYP`) resumes the 54 unstarted slots.
The updated R5 tests complete 357 checks: 352 pass and five optional-dependency
tests skip. The timeout and all earlier records remain part of the final evidence.

Attempt 3 completed the remaining 54 loss measurements in 207 seconds of inner
worker time. Final collection `evaluation-development-raw-v1/results-v1` accounts
for all 1,386 slots: 225 chat outputs, 1,160 measured losses and one indeterminate
loss from the original interruption. There are no pending slots or newly lost
measurements. All 901 protected files in the first snapshot and 2,688 in the second
are byte-identical in the final collection. The 225-chat review checksum is also
unchanged. `continuity-v1.json` records these comparisons; the smaller earlier
899-file check omitted the two protected root files, now included.

The collected interrupted snapshot contains all 225 development chat outputs,
218 loss measurements and one started request without a durable result. A
separate partial report supports review without changing original evidence.
Raw EOS counts are base 70/75, R4 71/75 and R5 70/75. Strict JSON matches are
6/8, 8/8 and 6/8 respectively. Both R5 failures contain the correct arithmetic
value, 28, inside Markdown fences rather than JSON alone: a formatting failure,
not a wrong computed quantity. Repeated 4-gram diagnostics flag 9/75 replies
for each model; that diagnostic is not automatically a harmful loop or a
semantic judgment. All 225 chat slots have now received separate agent review;
retention is incomplete. Review files are under `evaluation-review-v1`, with
hash-bound source references and original human-review fields still pending.

| Development agent meaning judgment | Base | R4 | R5 |
| --- | ---: | ---: | ---: |
| Pass | 26 | 44 | 41 |
| Partial | 12 | 21 | 23 |
| Fail | 35 | 9 | 8 |
| Uncertain | 2 | 1 | 3 |

Across the same 75 output slots per model, R5 versus R4 has four ordinal meaning
gains, four regressions, 64 ties and three uncertain pairs. Versus base it has
31 gains, eight regressions, 31 ties and five uncertain pairs. Usefulness versus
R4 has four gains, five regressions, 65 ties and one uncertain pair. These are
agent judgments on reused development data, not independent native-human
accuracy, clinical certification or 75 independent source groups. Broad R5
improvement over R4 is not established by these results.

Fresh raw run `r5_eval_20261110T115927Z` completed all 348 slots in 2,353 seconds;
its results were collected and checksummed under `evaluation-fresh-raw-v1/results-v1`.
Raw EOS counts per 116 slots are base 109, R4 112 and R5 109. Repeated 4-gram
diagnostics flag 29, 26 and 31 replies respectively. These are mechanical
measurements, not meaning scores. Separate agent semantic review covers all 348
slots. Per 116 outputs, meaning pass/partial/fail/uncertain counts are base
49/42/24/1, R4 66/36/10/4 and R5 72/27/12/5. R5 versus R4 has 11 gains,
eight regressions, 90 ties and seven uncertain pairs; usefulness has ten gains,
nine regressions and 97 ties. On the 60 requested-Twi slots, full-meaning passes
are base 10, R4 24 and R5 29, while failures are 20, nine and ten. This is mixed
progress, not a decisive improvement. Source-group dependence, ambiguous Twi,
non-blinded agent judgments and pending native/domain validation remain explicit.
The reproducible per-language summary is `evaluation-review-v1/fresh-summary-v2.json`.
Fresh guarded run `r5_eval_20261110T124059Z` completed all 348 slots in 2,272
seconds. EOS counts are base 109/116, R4 112/116 and R5 110/116; repeated-4-gram
reply counts remain 29/26/31. No guard-triggered pauses occurred. Seven outputs
or their histories differ from raw and received separate agent review;
341 exactly matching cases reuse their existing judgments with explicit,
hash-bound transfer provenance, not a claim of independent new reviews.
Small differences without guard triggers are not credited to the guard.
Development guarded run `r5_eval_20261110T132122Z` completed all 225 slots in
922 seconds. EOS counts are base 70/75, R4 72/75 and R5 70/75; guards pause
one, two and one replies respectively. The 218 exact matches retain their
original judgments; seven changed outputs received separate reviews. Shorter
loops did not restore the requested answers. Guarded meaning counts are base
26/12/35/2, R4 44/21/9/1 and R5 41/22/9/3 (pass/partial/fail/uncertain).
The guarded fresh R5 count is 73/26/12/5, versus raw 72/27/12/5; no guard
trigger caused this difference, and one downstream history changed as well.
Completion, meaning, language retention, repetition and guard behavior are
reported separately.

### Matched Language Retention

The final retention comparison uses only the same measured cases and token masks
for all three models. One R5 measurement was interrupted before a durable result;
that source is excluded from every model's paired total, without rerunning it.
All other planned measurements are present. Twi causal text and English
final-assistant prediction are separate objectives and are never pooled.

| Cohort | Matched cases | Predicted tokens/model | Base NLL / perplexity | R4 NLL / perplexity | R5 NLL / perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Twi causal text | 342 of 343 | 9,300 | 2.735093 / 15.411176 | 1.648157 / 5.197391 | 1.634894 / 5.128916 |
| English final-assistant targets | 44 of 44 | 8,534 | 1.142718 / 3.135279 | 1.131478 / 3.100235 | 1.131966 / 3.101749 |

NLL is summed token prediction loss divided by the matched token count;
perplexity is its exponential. Lower is better on this prediction task, not a
measure of meaning accuracy or safety. R5 slightly lowers Twi loss relative to
R4 while English loss is essentially unchanged on this small cohort. The large
base-to-adapter difference reflects the cumulative adaptation, not R5 alone.
No statistical significance or broad conversational improvement is claimed.

The private `paired-retention-v1/paired-retention.json` under the development
raw run retains every included/excluded source proof and exact sums. Its
derivation receipt checksum is
`17c2f8dba2dd3ad1322c2666f6ac9af9866f8cc56a32f607077a6f4ec09021b8`.
Reproduce with `PYTHONPATH=<worktree>/scripts python3 -B
<R5-root>/evaluation-design/summarize_paired_retention.py --results
<R5-root>/evaluation-development-raw-v1/results-v1 --out <new-directory>`.
All seven derivation tests pass. Available-case totals in the raw report retain
their original denominators; the table above uses the matched intersection.

The private `evaluation-review-v1/capability-benchmark-v1` report preserves all
authored capability/task labels and exact paired examples. Its 21 tests pass.
Development source-group metadata is missing; case IDs are not substituted for
independent source groups. Fresh task labels are unclassified while its capability
labels and all 40 source groups are available. Categories overlap and cannot be
summed into a larger sample count. Selected raw full-meaning-pass counts are:

| Fresh capability | Source groups | Outputs/model | Base | R4 | R5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncertainty | 6 | 20 | 6 | 11 | 15 |
| Negation | 7 | 20 | 10 | 8 | 11 |
| Multi-turn | 16 | 64 | 22 | 38 | 41 |
| Clarification | 4 | 12 | 2 | 5 | 4 |
| Quantities/math | 9 | 27 | 11 | 16 | 14 |

These are non-blinded agent judgments on this authored suite, not general
capability accuracy. Full partial/fail/uncertain counts and paired regressions
are retained. Independent rollouts intentionally have different prior model
replies; fixed-history tests separately hold the full input constant.

The private offline benchmark viewer is `evaluation-view-v1/index.html` under
the R5 artifact root. It initially contains the development partial report and
all 225 agent ratings, with explicit partial retention coverage. Its generator,
`scripts/export_understanding_r5_evaluation_view.py`, accepts repeated
`--results LABEL=DIR` and `--agent-review FILE` inputs and a fresh `--out`
directory. It validates raw records, identities, source hashes and denominators;
it does not generate answers, add ratings or rewrite the source reports. New
exports include subsequent completed suites without replacing old snapshots.
`evaluation-view-v2/index.html` includes development raw and complete fresh raw,
with all 573 raw-chat agent reviews and exact per-model histories. The local
live comparison remains available at port 7869 throughout evaluation.
`evaluation-view-v3/index.html` includes all four chat executions and their
1,146 review-covered outputs, with transferred judgments visibly distinguished.
Its development retention panel still reflects the preserved partial snapshot;
it will receive a new export after recovery, never an in-place rewrite.
`evaluation-view-v4/index.html` additionally attaches the user's confirmed
wording correction to the exact saved arithmetic reply: `yɛnnya` should be
`yɛnya`. The earlier agent review had marked that phrase uncertain. This is
one phrase-level confirmation, linked to identical raw/guarded occurrences,
not two independent reviews or approval of the complete answer. Original outputs
and aggregate agent ratings remain unchanged. The private sidecar is
`evaluation-review-v1/user-phrase-check-20261110.json`; it stays evaluation
evidence, not training data. The exporter accepts `--phrase-review FILE` and
checks the source hash, exact quote, occurrence identity and limited review scope.
All 355 R5 tests completed: 350 passed, five optional-dependency tests skipped.
The 20 viewer tests include correction binding, unchanged original ratings and
safe rendering. Actual GPU training/resume checks are recorded separately above.
The final `evaluation-view-v5/index.html` includes the completed retention journal
and all four chat executions, agent ratings and the scoped user correction. It
replaces no earlier snapshot. All final reports verify their source manifests,
runtime identities and raw record hashes before export. Per-model available-case
retention totals are labelled separately from the matched retention report.

### What the Evidence Supports Next

The private `evaluation-review-v1/research-findings-v1.md` connects observed
errors to verified data composition without claiming unproven causation. The
106 new source-backed histories contain 21 single-assistant-turn and 85
two-assistant-turn histories; none has a third assistant turn. All 24 repair
histories have three assistant turns but contribute only 1.63% of response
tokens. Longer contexts do exist in inherited records, so this is a depth/mix
imbalance in the new cohort, not total absence of multi-turn training.

The next preparation should measure response acts, actor/negation scope,
known-versus-unknown state, selective updates, clarification resolution and
honest action boundaries across the registered pool. Prioritize varied,
source-separated correction chains and useful alternatives over exact test-prompt
repairs. Preserve these evaluation prompts and outcomes outside data authoring.
Validate calculations, units, requested output format and language separately.
Compare a predeclared capability-balanced mixture before changing several
training factors together. No new run or test-derived training data is launched
by this recommendation. Native ambiguity adjudication and independent domain
assessment remain separate from the completed agent reviews.

## Durable Locations

- Code and progress: branch `codex/59-r5-meaning`, GitHub issue #59.
- Private R5 artifacts: original project `tmp/understanding-r5-20261109/`.
- R4 evidence: original project `tmp/understanding-r4-20261107/`.
- Lab database: `understanding-r4-20261107/workbench/workbench.sqlite3`.
- Existing private remote research volume: `ghana-health-understand-train`.

The initial source plan, repair sources, held-out admission package and progress
snapshot were backed up and read back from the private research volume at
15:02 UTC on November 9. The manifest is
`e4a2c7d682870adb7a941f0f66a84be338ddf2e660e172ce16e1e57d48b1092c`,
under `/research-training-r5/` with that manifest hash as its directory. The
local receipt is `backup-preparation-v1/storage.json`. User lab conversations
were not included.

The complete pack, all conversation proposals/reviews, five admission packages,
protection extension and progress snapshot were backed up with full remote
checksum readback at 10:30 UTC on November 10. The 90-file snapshot is under
`/research-training-r5/d503e9ded94f51987deca6318ba1733bcf094cc26d5fdc28a38ff7c95f9df217`
on the same private Modal volume; `backup-pack-v1/storage.json` is the receipt.
Final audit evidence, training receipts and evaluations receive later snapshots.

The 1,962-file completed-chat snapshot was remotely read back and verified at
14:12 UTC. It contains the fresh raw/guarded and development guarded journals,
reviews, capability report, recovery controls and viewer/progress sources under
`/research-training-r5/601f043242472feb46c0d07908b3604b645148692fa1cf0bfe2ccbae246b1f8b`.
Receipt: `backup-chat-evaluations-v1/storage.json`. User chat stores, secrets and
weights are excluded; weights and the earlier interrupted journal have their
separate verified storage/readback records. Final retention and handoff receive
an additional snapshot after completion.

The final 4,791-file evaluation snapshot was uploaded and fully read back at
16:10 UTC on November 10. It contains all four completed result journals,
hash-bound agent reviews, the scoped user phrase check, matched retention
derivation, interruption continuity proofs, recovery controls and reproducible
viewer sources. Its private Modal location is
`/research-training-r5/c03fa1827e363d212f93986298c0cdc7a2f03eb0bd5471c7c793f427e3c99313`;
receipt: `backup-final-evaluation-v1/storage.json`. The archive checksum is
`a1d46d23f0e726f2e4c643e878ab65be5be3f242142e3a8ba0f4f5119fbeee8a`.
HTML is reproducible from backed-up inputs and generators; user conversation
stores and credentials are excluded. Training weights and the full registered
pack retain their separate verified backups. Final public documentation is
versioned on the issue branch and PR #60, not silently pushed to main.

The final evidence viewer checksum is
`b59e5926274bfc0fa6666bcdf630daeed0afab58ac2a43cf392a50df16ccf079`.
The live local UI remains at <http://127.0.0.1:7869/?__theme=light>. Public
application behavior and its default model have not changed in this pass.
