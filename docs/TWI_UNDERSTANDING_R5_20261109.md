# R5 Meaning and Conversation Repair

Issue: #59. Continues the R4 research work in PR #58. R4 and its evidence remain
immutable. The target is useful Twi conversation, with faithful meaning across
turns, not completion scores alone.

## Execution Map

| Node | Work | Status | Evidence or exit condition |
| --- | --- | --- | --- |
| A | Preserve and review lab evidence | Complete, agent review | R4 raw/guarded exports plus local snapshot of 5 saved sessions, 13 rounds and 39 reply slots; no saved human assessments |
| B | Recover the blocked comparison UI | Complete | 46 focused tests passed; actual failed session recovered in the browser, new follow-up completed on all three models, input enabled afterward |
| C | Audit and repair the broader training pool | In progress | Count usable original histories, targets and tokens; retain source lineage and reasons for exclusions; do not inflate counts with spelling variants |
| D | Freeze fresh held-out scenarios | Frozen; final source audit pending | 40 source groups, 84 cases, 116 scoring slots/model; 232 protected text units; native/domain judgments are still pending |
| E | Seal R5 training data and run | Pending C/D | Actual admitted counts, source snapshots, target-token mixture, masking/EOS/native encoding, initialization and recovery proof |
| F | Evaluate and compare locally | Pending E | Base/R4/R5, independent histories, matched decoding and guards, raw and guarded evidence, failures retained |
| G | Back up and publish research handoff | Pending F | Private data and weights backed up; public progress and dataset/model cards contain measured results only |

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

There are also 24 project-authored bilingual repair proposals, containing 72
assistant turns. They target semantic contrasts and recovery behavior and require
a separate reviewer. They are not counted as naturally collected conversations.
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
Modal inference requests verified lab recovery. R5 training has not started.

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
were not included. Later reviews and the final pack still require their own
verified backup. R5 training has not started.
