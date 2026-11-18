# R7 Execution Record

Updated 18 November 2026. Issue #61. Branch `codex/61-r7-data-training`, stacked
on corrected R6 commit `587e375`. No R7 model has been trained yet.

## Goal

General-purpose Twi understanding and useful multi-turn replies, with executed
tool use, grounded health education and commerce. Keep the corpus and evaluation
contracts portable to future smaller-model distillation. R7 keeps the same
pinned AfriqueQwen 9B base and continues R6; more data does not add parameters.

## Execution Map

| Node | State | Exit evidence |
| --- | --- | --- |
| A. Source inventory and provenance | Full selected-source scan produced; review in progress | Original OASST, AfriHealth, parallel references and R5/R6 audit with immutable source hashes |
| B. Broader evaluation freeze | Behavior pack frozen; all 117 R6 outputs collected and agent-assessed | Eight bilingual source scenarios, 112 derived views, five separate observed regressions; public-task and tool comparisons being prepared |
| C. Data expansion and correction | Selected admissions sealed; deferred sources remain explicit | 12,067 original bilingual pairs; 129 executed English tool histories; 7,921 replay records; all selected response findings reviewed |
| D. Encoding and leakage checks | Complete for pack-v1 | 38,607 views native-reencoded; inherited, behavior, tool-dev and 2,004 exact public-benchmark protections; 855 exact duplicate views removed |
| E. Training and recovery smoke | Credits clarified; prepared smoke launch in progress | Pinned R6 initialization, real backward/resume proof, then detached training |
| F. Matched R6/R7 evaluations | Pending E | Raw outputs, per-category gains/losses, uncertainty, failure examples |
| G. Local comparison and handoff | Pending F | Inspectable calls/results, checkpoint identity, durable Modal backup |

## Data Policy

- Preserve R5 and R6 originals and review decisions. Register all R7 supervision.
- Include earlier useful response examples explicitly; R6's builder replaced
  rather than combined the old and new response pools.
- Separate source-preserved English, AI-translated/adapted Twi, source-grounded
  medical targets and explicitly simulated commerce/tool trajectories.
- No invented web results, purchases or payments. Store actual sandbox outputs
  and pinned public-source snapshots; live search is a separate integration.
- Native-language and clinical review are separate from automated screening.
  Admission to private research does not certify medical safety.
- Preserve existing holdouts and freeze new groups before generation/training.
  Never use benchmark answers as repair training targets.
- Do not silently count related views as independent examples or diagnoses.

## API and Compute

The user explicitly approved reuse of the existing project API credential.
It remains in protected VPS configuration; no key is copied locally or committed.
OpenAI supports data preparation/validation, not replacement of model replies.
All new requests have durable receipts and token-based cost estimates. Unknown
billing states are not retried blindly. Initial R7 spending ceiling is $50 of
authorized credits; this is not a claim that the account still has $50 available.
No automatic top-up or account billing changes are authorized.

**Observed access blocker:** the first qualification request to `gpt-6.2-sol`
returned HTTP 429, `insufficient_quota` / `credit_balance_exhausted`. No generated
data was returned. Receipt is retained under `tmp/r7-20261117/api/`; no automatic
retry. User chose **Modal for now**; no further OpenAI requests are authorized by
this execution path. Corpus inspection and source-preserving exports continue.

Current teacher qualification: pinned Gemma-4-31B-it and Qwen3.5-35B-A3B on Modal,
22 development controls covering meaning, Twi production, source-grounded
correction/follow-ups and truthful tool results. This is teacher selection, not
a student benchmark or selection of a replacement training base. No control is
training eligible. The old Qwen-235B qualification failed five critical meaning
checks; it is not being rerun unchanged at scale.

Gemma attempt v1 failed before generation because the offline loader requested
uncached repository documentation. Attempt v2 restricted snapshot selection to
actual model/tokenizer files. Both receipts are preserved. Qwen35 preparation
downloaded its pinned public checkpoint on CPU before requesting a GPU.

Both completed all 22 controls on private Modal H200 workers. Neither qualified
for bulk Twi generation/translation. Three inspected critical findings per model
are bound to their actual saved output in
`tmp/r7-20261117/teacher-reviews-v1/reviews.json`:

| Source distinction | Gemma-4-31B-it observation | Qwen3.5-35B-A3B observation |
| --- | --- | --- |
| Child has become hot | Changed hot to cold | Changed the event to giving a gift |
| Book is half the size of mine, owner-confirmed | Changed to a book the speaker took | Changed to a book the speaker used |
| Two kilograms / eye pain | Changed two kilograms to five | Changed eye pain to heart/joy |

These are attributed agent findings, not native-certified total scores. The
remaining controls are not implicitly passed. One fruit-distinctness reference
needs lexical adjudication and is not used to score three distinct types.
The two failed methods are not scaled unchanged. Qwen35 passed all 32 controlled
English source-review checks, but a real-source batch exposed a calibration error:
it treated an explicitly requested fictional superhero role-play as a claim of
real assistant capabilities. The pass was stopped after 68 checkpointed results
out of 4,087 planned histories. No training admission or removal follows those
ratings. Its Modal call is verified TERMINATED; the saved progress file's last
checkpoint still says `running`, so the separate collection receipt is authoritative
for lifecycle. The rubric and representative natural-source calibration are being
revised before any new bulk pass. The 32-control success is not a quality claim
about the full corpus, Twi translation or clinical assessment.

The expanded English-review v2 qualification is complete: **42/48 controls
passed**, including eight natural-source controls in the denominator. Failures
include malformed evidence, missing required context anchors, and unsupported
acceptance of externally checkable claims. It is **not qualified for bulk
admission**. Results are preserved in `english-review-qwen35-v2/qualification.json`.
No new OASST admission follows these ratings. The source-preserved and inherited
admissions continue without waiting for another teacher experiment.

Modal workspace `createdliving1000`, environment `main`, training volume
`ghana-health-understand-train`. New private artifacts will live under
`understanding-adaptation-r7/` and a versioned research-handoff prefix. Existing
R6 evidence, app and adapters remain unchanged. No production promotion implied.

## Evaluation Boundaries

Retain corrected R6 public-task measurements for comparability, but add direct
tests of meaning, useful follow-ups, correction handling, language switching,
tool argument/execution success, grounded final answers, unsupported claims,
repetition, justified abstention and wrongful refusals. Report health safety
separately; automated judgments do not establish clinical readiness.

Current R6 evidence: [corrected results](R6-STATUS.md). R6 improved topic
classification and the combined Twi knowledge subset, but not its math subset;
five English task scores were slightly lower than R5. These differences motivate
R7's coverage and replay design without proving one data class caused them.

## Current Root-Cause Evidence

Two saved R6 conversations were read from the local lab and audited against their
recorded request hashes. The correction was present; histories were independent.
The fresh bird-example question also failed, returning three copies of "Bird".
With prior errors in context, R6 instead repeated the wrong animal list. This
establishes a fresh-task failure plus failed correction uptake, not a diagnosis
of context overflow or a proven causal defect in a particular training record.

Private archive: `tmp/r7-20261117/observed-regressions-v1/evidence.json`, six saved
turns across both sessions at export time, 12 model-side protected units.
Evidence digest: `cf021af14b92f99699322ebb1b334b769064b73ba99364d68d156e032dc6d196`.
These are reused development regressions, not fresh holdout cases. The broader
suite adds fresh chats, fixed-history corrections and independent follow-ups,
with category membership, distinctness and changed-constraint checks reported
separately. New phrasing alone does not create an independent source group.

Infrastructure verification: **380 tests passed** in the latest integrated R7
suite. Earlier interim UI/path test failures were corrected; old failed runs are
not counted as passes. An independent preflight found no blocking assembler/tool
mask defect: 5,160 native views and 129 reconstructed tool histories passed.
All 7,921 replay records were separately native-reencoded. The assembled final
pack still requires complete verification before launching training.
This does not claim that teacher output or a new model passed semantic evaluation.

## Selected-Source Audit

Snapshot: `tmp/r7-20261117/source-inventory/v4/summary.json`. These rows belong to
different source classes; do not sum them into a count of usable conversations.
All candidate sets still require scoped admission and actual training-pack checks.

| Source class | Original scope inspected | Resulting candidate queue | Remaining work |
| --- | --- | --- | --- |
| OASST English dialogue | 84,437 training messages; 4,401 validation messages preserved | 4,087 complete ranked histories across 2,938 trees; 2,424 histories from families absent in R6 | Target review, privacy and factual-risk triage; no Twi translation claim |
| Existing bilingual references | 130,535 original rows from two sources | 28,025 pairs: 22,668 lexicon, 5,352 sentence, five utterance | Keep record types separate; resolve conflicts, duplicate mappings and routing flags |
| AfriHealth | 29,538 train, 6,673 validation, 2,612 test rows | 10,683 English source candidates, still pending | Cross-language grouping, source/clinical checks; no new health admission |
| Prior R5 material absent as exact R6 views | 3,108 registered R5 records inspected | 1,848 replay candidates, including 645 response records | Preserve prior review; recheck protection and target quality |
| R6 Twi response targets | All 997 histories, 2,038 assistant occurrences | Full source-linked audit | Distinguish genuine duplication and unsupported claims from legitimate code/list repetition |

The R6 pattern scan found 77 repeated-sentence/line flags and seven repeated-list
flags across assistant occurrences. These overlapping lexical flags are review
selectors, not 84 proven errors or a causal explanation of every bad reply.
Source-preserved English and Twi references remain available even while a
reliable bulk Twi authoring method is unresolved.

## Current Admission Outputs

These are separate source classes, not additive counts of natural conversations.

| Collection | Available after scoped checks | What the count means |
| --- | --- | --- |
| Original sentence alignment | 4,782 pairs / 9,564 directions | Unchanged English and Twi references |
| Original lexical alignment | 7,280 pairs / 14,560 directions | Dictionary/lexicon entries, not conversations |
| Original utterance alignment | 5 pairs / 10 directions | Short bilingual utterances |
| Executed English tool trajectories | 129 histories / 32 source groups / 378 supervised prefixes | Templated synthetic sandbox tasks, with actual tool results |

Alignment totals: **12,067 pairs, 24,134 views, 369,545 target tokens** for a
single pass. This is source-preserved, automatically screened private research
admission, not native-certified semantic gold. Two additional pairs were excluded
because the native template would trim their original target. Mapping/polysemy
flags remain in a separate review queue; they are not automatically wrong.

Tool release: `tmp/r7-20261117/tool-admission-v2/`. All 129 candidate train
histories passed the declared replay, holdout and native-encoding checks. Eighteen
deliberately invalid probe calls remain context only; they are never supervised
assistant targets. The 378 supervised prefixes contain 15,168 target tokens.
This covers English sandbox behavior, not demonstrated Twi tool competence,
real web search, clinical reasoning or actual purchasing. The catalogue contains
only four fictional products. Source-grouped dev trajectories are preserved.

Conversation audit: all 997 histories were source-linked and scanned, covering
2,038 assistant occurrences. All 191 selected findings across 122 histories now
have attributed agent reviews: 27 prior decisions plus 164 newly inspected
findings. The new set contains 159 issue-specific clearances and five holds
across four histories. Two deletion-only repair candidates remove exact repeated
material; they do not authorize new Twi wording or override other record concerns.
Final replay admission integrates these decisions separately. No whole-translation,
native or clinical certification is claimed. Useful prior R5 responses retain
their original reviews and provenance.

Replay v2 is now sealed at `replay-admission-v2/`, manifest
`c923b3c777819a9691298ccedf52a5da7941a85d5db7ed3291b450395bbd6f0b`.
It admits 7,921 full records: 1,501 response histories (633 R5 + 868 R6), 3,758
meaning records, 1,475 English-retention records and 1,187 language texts.
Of the 9,398 prior records examined, the remainder comprises 183 protected,
1,215 exact duplicates, six genuine-issue holds, 61 record-concern holds and 12
native-boundary failures. The two exact-deletion histories are included with
their inherited research tier. These counts precede combined prefix expansion,
cross-source deduplication and the additional public-benchmark protections.

Controlled follow-up augmentation is now sealed at
`tmp/r7-20261117/alignment-followups-v2/`: **4,782 exercises across 2,391 paired
source groups**, with 1,594 examples of each of three forms: a new request,
replacement of an earlier request, and wrong-language correction. The same 4,782
original sentence pairs supply these targets; they are not new independent text
sources. The exercises add 197,002 target-token presentations for one pass.
Instruction scaffolding is English and synthetic; no new Twi wording was generated
and these must not be counted as naturally collected Twi dialogues. Only the final
source reference is supervised. Transfer to free-form Twi correction handling
will be measured against R6, not assumed.
V2 additionally binds transformation-code and pinned tokenizer hashes; v1 remains
unchanged for audit.

## Sealed Training Package

Path: `tmp/r7-20261117/pack-v1/`.
Manifest: `999d53cdbc79f4d0268edf4e2f9ec270756155f3341c81069deec82ad8243d53`.
36,966 admitted input records produce 39,462 candidate assistant-prefix/text
views. After 855 exact within-collection tokenized duplicates are removed,
38,607 remain. No additional source/content holdout matches were found by the
declared checks; this does not claim unknown foundation training was clean.
17,856 distinct `source_id` values are present, not 38,607 independent sources
or conversations. The one-pass schedule has 2,413 optimizer updates, 6,178,465
input tokens and 1,849,881 effective target tokens, with no truncation.

| Training collection | Views | Effective target tokens | Token share |
| --- | ---: | ---: | ---: |
| Meaning and original alignment | 31,902 | 603,289 | 32.61% |
| Direct responses | 2,644 | 735,911 | 39.78% |
| English retention | 2,554 | 466,685 | 25.23% |
| Language replay | 1,187 | 31,856 | 1.72% |
| Executed English tools | 320 | 12,140 | 0.66% |

The source-backed expansion is largest in alignment. Controlled follow-ups are
translation exercises, not natural dialogue. New medical-source expansion and
bulk generated Twi conversations remain deferred; R7 does not silently include
failed teacher output. Full native verification succeeded before the package
was atomically published. GPU training has not started at this checkpoint.

## Frozen Behavior Evaluation

Pack: `tmp/r7-20261117/behavior-eval-v1/`.
Manifest SHA256: `8029612cea3110d31d9b8a1f717e0f43a38148c613c739341ae46f50d3a68e13`.

- Eight semantic roots cover category selection, reference resolution, entity
  binding, time, quantity changes, negation, spatial relations and state updates.
- 64 controlled fixed-history requests plus 48 independent-rollout turns yield
  112 derived views, balanced 56 Twi/56 English. These are not 112 independent tasks.
- Five saved R6 regressions are labelled reused development evidence separately.
- No matches were found under the declared exact/group/shingle checks against
  the scanned R6 pack. Unknown foundation-model exposure remains unknown.
- Expected answers and rubrics stay local; the worker receives only model inputs.
- Raw outputs, failed slots, actual histories, checkpoint identity and decoding
  are retained. R6 runs separately from the live workbench, without changing it.
- This is a controlled behavior diagnostic, not a complete natural-conversation,
  real-world-tool or medical-safety benchmark. Those gaps remain explicit.

Current execution receipt: `tmp/r7-20261117/r6-behavior-v4.receipt.json`.
Detached Modal call: `fc-01M7NY8CADGXV8HNVESN4AXJ6M`.
The earlier v1-v3 setup failures are preserved and did not produce model outputs.
V4 uses a local Python 3.11 launcher matching the remote image; model, inputs and
decoding remain pinned. All **117 outputs are now complete**, with zero execution
errors. This is separate from correctness:

| Fresh controlled views | R6 English exact answers | R6 Twi exact answers |
| --- | --- | --- |
| Fixed-history requests | 31/32 | 27/32 |
| Independent three-turn rollouts | 21/24 | 19/24 |

The broader agent judgment over 112 fresh views is 52 pass, 14 fail and 46
unknown; exact-answer success does not certify Twi naturalness or the entire reply.
Eight semantic roots underlie these derived views. Observed errors include
returning operands instead of totals, entity/attribute confusion and failure to
apply corrections across turns. Raw outputs: `r6-behavior-v4-collected/latest.json`;
hash-bound assessments: `r6-behavior-v4-score-reviewed-v1/agent-review.json`.

## Durable Handoff

Primary working tree: `/Users/welcome/.codex/worktrees/r4-train/ghana-health-ai`.
Do not overwrite the original dirty project worktree. Private outputs are under
`tmp/r7-20261117/`; source files are read-only. Run receipts include detached
Modal call IDs, model revisions and exact input hashes.

Private backup volume: `ghana-health-understand-train`.
Backup prefix: `/research-handoffs/twi-understanding/2026-11-17-r7/`.
Each archive is named by its SHA256 and verified by reading it back from Modal.
Local receipts are in `tmp/r7-20261117/backups/`. The backup contains work in
progress, not an admitted training release. Existing R6 evidence is untouched.

Latest completed backup: `1ba6aab9f10d275e6587804b6305fd27f7c6e7567af5ad05b62a34b1f1953544.tar.gz`,
378,014,184 bytes / 372 files, remote read-back verified. It predates final pack
assembly; a later snapshot must include that release and its training receipts.
Data preparation code is committed as `e1d7c0f`; the training/tool contracts are
in `82e2513`. UI/evaluation extensions remain in progress on the same branch.

Resume from the active complete-pack assembly and verification,
then run the real-weight backward/resume smoke and detached training. The R6
baseline is complete. R7/R6 UI and matched public/tool comparisons are being
prepared separately without altering the current R6 workbench. No R7 training
is running yet.

## Credit Checkpoint

The user clarified that **Codex weekly usage**, not Modal credits, is running low
and explicitly instructed continuation on Modal. The temporary GPU hold is lifted.
The prepared smoke launch is in progress; its receipt determines submission state.
The complete CPU-verified smoke package is `tmp/r7-20261117/execution-smoke-v1/`,
execution SHA256 `d5bdc11126ffab4261e89cee4283f95b5c0e7810a858949c050fd35d80fd920d`.
Do not duplicate the active launch or rebuild the data. Its receipt is
`tmp/r7-20261117/smoke-v1.receipt.json`. The next stages are real backward/resume
smoke, full training, matched R6 comparisons and the local R7/R6 UI.
The supplementary public source-group audit checked all 38,607 witnesses and
found zero identity/hash intersections, addressing the original protection's
group-field naming mismatch without altering its sealed artifacts. Supplement:
`public-benchmark-protection-supplement-v1`, manifest
`ee762fa34ad52c1e6b0d35d21c02e59badaa341187c786c8328d3369c202111a`.
The detailed dataset profile is `pack-profile-v1/REPORT.md`; it reports 15,365
alias-connected groups (not proven independent tasks), 390 target-only groups
shared across collections, and 2,766 views without explicit language labels.
