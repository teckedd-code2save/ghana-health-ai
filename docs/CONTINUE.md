# Continue here — Ghana Health AI

## 2026-09-15: Alignment Handoff And Annotation Checks

Latest request: the owner said the data generally looks good and approved the
next steps. The completed work remains a collection pass, not a new model run.
Read `docs/alignment-handoff-20260915.md` and
`docs/reference-annotation-20260915.md` before continuing.

**Usable new handoff:** `tmp/corpus-ready/twi-alignment-v2-20260915`.
All 30,404 existing bilingual pairs are packaged into both translation directions:
11,557 sentence pairs, 18,734 dictionary pairs and 113 grounded-QA question
pairs, in separate views. There are 59,664 train examples and 1,144 validation
examples, not 60,808 unique utterances. English retention is separate and
unchanged: 2,131 train / 44 validation complete conversation paths. No new
translation or response target was generated. Source hashes, task views and
shared splits were independently verified; 38 corpus tests and TypeScript pass.

Fifteen handoff files (95 MB) are privately backed up, with read-back hashes, in
Modal volume `ghana-health-understand-train`, path
`/corpus-training-releases/twi-alignment-v2-20260915/0d59772b4b49`.
The receipt is beside the local release as `.storage.json`. The original parent
release passed verification again and remains unchanged. The incomplete local
alignment `v1` directory is NOT a handoff; use `v2`.

**HTTPS:** `https://ghanahealth.serendepify.com/research/corpus/` remains live.
The new focused URL is `?collection=teacher` (UI name: Annotation checks).
It compares the two latest candidates on 32 real source-validation samples;
these are calibration/correction samples, not another small training corpus.
No synthetic controls appear in that queue. Neither candidate is preselected.
Each review records the exact annotation version and original source hash.
Annotation-only corrections do not invalidate source translations. The compact
layout, navigation persistence, save flow, and stale-version rejection are tested.

Active private review code is
`/opt/ghana-corpus-review/releases/reference-20260915-r2` via `current`.
The same append-only journal, four existing review actions and credential file
are preserved. No production review was fabricated by tests. HTTPS checks passed
all 32 source/version identities, auth, assets, CSRF and desktop/mobile rendering.
Public chat remains on its old upstream; no ASR/TTS, public deployment, public
dataset upload or model training was performed. The isolated review service was
the only web deployment changed.

**Annotation is NOT qualified:** three English-reference-only methods ran on the
same 32 real samples plus ten controls. Qwen3.5-9B and Qwen235 had semantic errors.
OSS120 also remains unqualified; it exposed both output problems and a rubric
problem that conflated a statement of a goal with a direct request. Preserve the
results, and do not scale these methods unchanged. The full 11,557-source
missing-field work plan remains pending bulk annotation. No `run` batch or new
training was started. Existing valid references powered the handoff independently.

OSS qualification `fc-01M2H8WRCREYB9KPXHBS29YSJN` and diagnostic
`fc-01M2H9MTXWG2AZEGK6N7F9CAPQ` are COMPLETE. Do not resubmit. The diagnostic
returned four final answers but did not establish the cause of three missing
finals in the earlier run. New traces are retained privately. The old failures
are not replaced or retroactively passed. The `expressed_goal` successor schema
is implemented/tested but needs fresh qualification, not the same controls
repeated until they pass. No paid API fallback was used.

Next: select an evidence-backed foundation and explicit alignment/English-replay
mixture, verify its native chat template and loss masks, then scope a bounded
alignment experiment. This is NOT yet a conversational assistant SFT corpus:
native multi-turn Twi responses and specialised response/tool material remain
unfinished. No commit or push was made; the large pre-existing dirty worktree
remains. Do not commit unrelated ASR/TTS changes with this work.

## 2026-09-14: Private HTTPS Review And Loading Retry

Owner requested HTTPS access while model loading is repaired. Review is live at
`https://ghanahealth.serendepify.com/research/corpus/`, independently of this
laptop. Read `docs/corpus-review-https-20260914.md` for access, storage and
operation. Public chat upstream is unchanged. The private server has a 71,831-row
review projection; corrections persist on the VPS, not in the frozen dataset.
Use HTTPS for new reviews. Unsaved localhost drafts do not transfer.

CPU staging probe passed. Revised Qwen qualification completed and is tracked in
`tmp/corpus-teacher/qwen235-v3-staged`, call `fc-01M2ETZCZMG4XMF5Q7DYNMYW26`.
All 40 outputs were generated in 1,078.51 seconds; loading plus warmup took
568.05 seconds. Loading is fixed for this run, but qualification failed: four
truncated outputs and five meaning-critical control failures on source-reference
inspection. Read `semantic-audit.json`; keyword matching missed two failures.
Do not duplicate that call or use its outputs for bulk annotation. The gate is
closed. No new training or bulk annotation was started.

## 2026-09-14: Four-Collection Source-Preserved Release

Current scope is the user's stage-ready dataset plan. Read
`docs/stage-corpus-20260914.md` before launching anything. The working release is
`tmp/corpus-releases/twi-stage-v1-20260911`, with 1,178,118 selected source rows.
The release is now sealed and independently verified: 49,922 unique sources
contribute screened task views. These include 30,404 existing bilingual pairs,
2,751 structured examples, 110 grounded Twi QA records and 2,175 English
conversation paths. Health has 4,264 full-reference correction records, not
approved SFT. Language views have only 773,316 tokens including lexicon; this is
not a large native CPT corpus. Raw sources, prior annotations and existing
evaluation boundaries remain preserved. Synthetic Pristine is not accepted
native text. Read the stage report for exclusions and field-level gaps.

Qwen235 qualification produced ZERO translations: eager loading timed out at
1,200 seconds, call `fc-01M2EPC4KAPNY08XPCK6J9JAYT`. This is infrastructure
failure, not a Twi quality result. Gate remains closed; no bulk generation,
new training, ASR/TTS changes, public upload or production deployment.

Local review URL: `http://localhost:3100/research/ase?release=twi-stage-v1-20260911`.
The release index exists and real rows load. Browser navigation/draft
persistence and source-hash-bound local saving are tested on desktop/mobile.
No real review or database write was created by those tests. Existing production
reviews are not included in this run's human-review count.

Small release reports are privately backed up in Modal volume
`ghana-health-understand-train`, under
`/corpus-release-reports/twi-stage-v1-20260911/ddc88cb47302` (13 files verified).
The complete corpus and ledger remain local, not remotely archived by this pass.
No commit or push was made; unrelated pre-existing worktree changes remain.

## 2026-09-11: Scope Correction - Understanding LLM Only

The user explicitly redirected: do NOT veer into ASR. The shared research is
inspiration for improving the understanding/response LLM, not a speech-model
workstream. The ASR experiment order in the review below is parked; it is not
the active execution plan. No new ASR benchmark, training or serving changes.

Active hypothesis: compare direct Twi/English conversation SFT against screened
Twi language adaptation followed by the SAME SFT on the SAME foundation. Include
the untouched foundation as a control. Keep English replay/retention, source
grouping and synthetic-data ablations explicit. Language-model loss/fluency is
not proof of semantics, dialogue quality, medical correctness or real tool use.
Tokenizer adaptation is conditional research, not the first assumed fix.
Current translation alternatives remain unapproved; do not scale failed
teachers merely to increase training row counts. No new training submitted.

Read the updated `docs/response-research-reset-20260911.md` scope section.

## 2026-09-11: Shared ASR Research Reviewed; Dialogue Comparison Complete

Latest request: check ChatGPT conversation `6a862c01-8b38-83ea-b1e3-c80212dbeedf`,
title **Weekly ASR Research Shortlist**. Read it through the app connector and
verified the main primary sources. Read
`docs/asr-research-shortlist-review-20260911.md` for priorities and corrections.
No ASR training was started. Do not confuse ASR adaptation with fixing the typed
greeting failure in the response model.

The prior dialogue-translation calibration is now COMPLETE: 96 outputs across
Afrique isolated and Gemma isolated/context. Both models still make semantic
errors; zero rows promoted. Read
`docs/dialogue-translation-calibration-20260911.md` for receipts, hashes,
preemption recovery and inspection findings. Review artifact:
`tmp/dialogue-translation-calibration/v1/review.json`. Do NOT resubmit the
completed calls. No new model training, public upload or production change.

## 2026-09-11: User Greeting Failure, Full Corpus Audit

Latest user request is model research after V6 produced a looping fabricated
biography for `wo ho te s3n`. Do not resume UI work or claim a chat-quality fix.
Read `docs/response-research-reset-20260911.md`. Four fresh paired greeting
checks completed: base loops in 4/4, V6 ends in 4/4 but still fails reply quality
and English language matching. The reported failure remains valid.

New coverage audit: ZERO multi-turn Twi-labelled final targets in V6; 10.67%
of prepared target tokens are Twi response tasks, 96.26% of those health.
Evidence: `data/response-adaptation/balanced-v6-coverage.json` and
`greeting-v6-decision.json`. No more unchanged V6 training.

Full pinned text audit COMPLETE: `tmp/twi-pretraining-audit/v1/summary.json`.
1,134,981 rows; 1,050,333 exact-unique structural candidates, not approved data.
855.60M Qwen / 642.90M Gemma candidate tokens, over 99% from synthetic Pristine.
705 deterministic review samples; output hashes and accounting verified. Do not
resubmit. Existing audit speech coverage is only the earlier 4,948-row manifest.

Full labelled speech-text projection also COMPLETE: 28,312 unchanged transcripts
in `tmp/twi-pretraining-sources/full-public-speech-v2/`. Use V2: it corrects
WAXAL's project language metadata to ak (Akan), not verified Twi. V1 is preserved.
Source counts, pinned revisions, hashes and remaining curation are in the reset
document. These sources overlap the old manifest; do not double count them.
No new training, annotation, public upload or model-quality success is implied.

The user subsequently requested and received private V6 activation in the
existing local playground on port 7863. Earlier statements below about no UI
selection are historical. Voice A/history/reviews/production stay unchanged.

## 2026-09-11: Balanced Afrique 9B V6 Complete, Not Accepted

Run `afrique_v6_20260910T222142Z`, call `fc-01M26PK73KPNMVXAHKC1H3VF8H`
completed 400 updates, 36.48 minutes total. DO NOT RESUBMIT. Adapter SHA256
`b4aa2162ae27a4c6997ec9f88bc171ca2134c457922bb1483c460eff920be510`.
Full verified backup: `tmp/balanced-afrique-v6/completed/afrique_v6_20260910T222142Z/`.
Exact input archive and original outputs also retained on the private volume.

30 paired development prompts, 48 paired source-validation prompts, six new
two-turn conversations per model, then 16 private serving cases completed.
Termination and some arithmetic/context/English tasks improved, but native
conversation changes, grounded commerce claims, health responses and explicit
calculator execution still fail. Kofi is CORRECT in its supplied context; an
initial commentary mistake was corrected. Do not perpetuate that false finding.
No general semantic-accuracy or clinical-safety score is claimed.

Read `docs/balanced-afrique-v6-20260910.md` and
`data/response-adaptation/balanced-v6-decision.json`. Candidate support is deployed
only to the private inference service; no UI selection, production change or
HF publication. Current voice A, old tester choices, reviews/history unchanged.
53 focused tests passed; no full web build claim. Next work is native dialogue
response quality and grounded action supervision, not another unchanged run.

### Original Submission

User approved the concrete repair/train/conversation-evaluate cycle. One bounded
run was submitted: `afrique_v6_20260910T222142Z`, call
`fc-01M26PK73KPNMVXAHKC1H3VF8H`, app `ghana-balanced-afrique-v6`.
Do NOT resubmit or redeploy this training app while active. Status and receipt:
`tmp/balanced-afrique-v6/run-receipt.json` and
`scripts/run_response_adaptation.py status --experiment balanced-v6`.

Read `docs/balanced-afrique-v6-20260910.md` for exact changes, limits and evaluation.
8,681 training views / 7,464 source records; 1,032 validation rows. Corrected
intent task contract, known source exclusions, reduced health/tool weighting,
unchanged bilingual targets, 17 agent-inspected context-bound native sources
plus 17 source-language follow-ups. Not a gold corpus or complete native dialogue
dataset. CPU tokenizer/backward/checkpoint preflight passed. One H100, 400 updates
maximum, 90-minute hard ceiling, no automatic retries. Private artifacts only.

No V6 chat-quality success or model activation has been claimed. Voice A,
current private UI selection/history, public chat and production remain unchanged.
Earlier completed research below should NOT be restarted.

## 2026-09-10 Question/Source Cross-Check Completed

Latest usable review artifact: `tmp/native-question-translation-v1/review.json`.
71 original source pairs, 117 translated question alternatives, 117 question
back-translations and 71 separate original-answer translation checks. Original
answers are unchanged. 56 question translations have agent correction flags;
three proposed corrections are separate options, not applied gold labels.
The remaining questions are NOT automatically certified correct.

Both bounded stages completed; do NOT resubmit:

- Forward `afrique9_20260910T153004Z`, `fc-01M25Z1JZT3KHHWCPJWZAQMY8D`,
  117 outputs, 269.20 seconds.
- Back `afrique9_20260910T153628Z`, `fc-01M25ZDA12KG1BRX3P2YKX67H0`,
  188 outputs, 423.00 seconds.
- No empty or token-limited translations. Review file SHA256:
  `0e15ac5f5695a46125fb1ddae20e7865781e298b1e3435099cbbd7d320ba324b`.

Back-translation itself made clear errors, including Benada becoming Thursday,
borehole becoming boiling water, and fire brigade becoming a gang. Do not blame
the unchanged source or certify it solely on that model's English output.
Read `docs/native-answer-synthesis-20260910.md` for provenance and limitations.
Private backup is under `/native-answers/native_answers_20260910T134538Z/` on
`ghana-health-understand-train`, in `question-translation-v1` plus the findings
file. No public HF upload, changed voice A, changed UI/history, new model choice,
or production deployment. No new successful chat model is claimed.

The next training-data work must close native conversation coverage and alignment,
including corrected questions, context, and multi-turn response targets. Existing
news sentences are not automatically standalone chatbot facts. Do not treat these
71 sources as the requested complete large corpus or launch another unchanged
V5 mixture run. Translation similarity, English compatibility and native meaning
must stay separate assessments.

Latest focused verification: 38 response tests, 20 native-data/intent tests and
four Afrique language-evaluation tests passed (62 distinct tests), plus compile
and whitespace checks. The response tests require Transformers 5.17.0; the first
attempt used an older ambient package and had four import/API errors, then passed
with the actual runtime version. Not a full application/build test run.

## 2026-09-10 Open-Weight Reasoning Rejected; Corpus Evidence Checks

GPT-OSS120 completed all 30 cases, not merely loading:
`oss120_20260910T142628Z`, `fc-01M25VD3Q656N48C4381ZDPRAE`.
One H100, 511.85 seconds total, no parse/empty/length/repetition flags. Yet Twi
medical replies lose essential meaning and give inappropriate guidance;
long general replies are garbled. No replacement/teacher acceptance. Read
`docs/oss-response-comparison-20260910.md` and the actual decision artifact.
No 20B trial was launched from this result. Do not resubmit OSS120.

Native-answer evidence review is complete, call
`fc-01M25VP9J0KZ3T8XPBT01RZHXT`, 119 inputs, 272.08 seconds. Passed 26/28
calibration checks, not all. Original missing-reason failures now rejected;
pronoun judgments still inconsistent. Nine new source translation flags bring
the total to 14. Original source answers remain unchanged. Current inspectable
artifact `tmp/native-answer-synthesis-v1/ground-review.v2.json` retains 128
sources and 71 English-compatible source-review candidates. No automatic/gold
training acceptance. Distinguish review visibility from a global calibration
gate. Source event context is still missing in many standalone questions.

A different, SMALL native-language comparator has now completed:
`ghananlpcommunity/MiniCPM5-1B-Twi`, pinned
`d807ca1a3323972afafabff8f9affe2639e37b5c`. Its card describes extended Twi
vocabulary, substantial language pretraining and acknowledged reasoning/factual
weaknesses. This is a fluency-focused experiment, not another size upgrade or
project fine-tune. Private app `ghana-native-fluency-comparison`, receipts in
`tmp/native-fluency-comparison/`. Retry 2 completed all 27 non-tool cases in
356.49 seconds: `minicpm_twi_20260910T151419Z`,
`fc-01M25Y4P40NXK4KZMRH22GKNRZ`. All 27 reached the answer-token ceiling and
failures include English instructions, quantities, invented commerce contacts
and lost health context. NOT accepted for chat/teacher use. Do NOT resubmit.
First attempt failed before generation on a loader dependency; CPU loading then
CUDA transfer fixed that, with an actual CPU forward preflight before retry.
See `docs/native-fluency-comparison-20260910.md` for hashes and limitations.
No public data uploads or changes to voice A/UI.

Completed data experiment: `scripts/translate_native_questions.py`, artifacts
in `tmp/native-question-translation-v1/`. It uses the already prepared Afrique
9B translator for 117 English questions from 71 source-review candidates,
followed by back-translation. Original Twi answers are immutable. New unchanged
training demonstrations 66150/71191/67230 are source-group-disjoint from the
candidates and held-out data. Both stage receipts and the final export are complete;
do not duplicate submit. Back-translation is same-model diagnostics, NOT
independent verification or training acceptance. Source-dependent news claims
must retain supplied context rather than being presented as current facts.

New public source inspected: `ghananlpcommunity/gooaq-twi-2m`, pinned metadata
revision `46e9e7e6f55bb2abfd80f07b797435c24de82661`. Preview shows real QA shape
but clear defects (unsupported mg-to-mL conversion, changed math operands,
and poor translations). Not bulk-imported or accepted. The reasoning-translation
corpus also reuses the previously rejected pristine/ghana-chat source families;
its use of the term gold is not independent human verification.

All older running/preparation snapshots below are historical where superseded.

## 2026-09-10 Foundation Comparisons: No Accepted Chat Replacement

Qwen 122B comparison reached its 1,800-second ceiling, not a cancellation.
Preserved 30 non-thinking and eight thinking outputs in
`tmp/stronger-response-comparison/partial-final.json`. Do not resubmit.
Qwen 3.8 27B completed all 60 outputs; actual result and diagnostic are in
`tmp/current-response-comparison/`. Both have clear Twi meaning failures;
neither is enabled as a teacher or chat replacement. Read
`docs/stronger-foundations-20260910.md` for evidence, hashes, and limitations.

Next comparison is the newer AfriqueQwen3.5-9B-50Langs against its own base,
using 900 source meaning checks and 64 translation tasks, not chat prompts.
Private app `ghana-afrique-native-comparison`; preparation call
`fc-01M25RD3JA40KW9SY1B535NPHP` failed before GPU allocation because Xet kept a
log open on the volume. Temporary Xet files now live off-volume and preparation
reloads the shared cache only once. Retry `fc-01M25RP0PCDEVRDXG03Y1NBJ6X`
completed; old receipt retained. Run `afrique9_20260910T134056Z` is submitted:
base `fc-01M25RSRJSG4WD1DQYH65ZT49Q`, Afrique `fc-01M25RSV09TXJSMRHBGKMKJ1RF`.
BOTH COMPLETE, 964 outputs each. Twi AfriXNLI 211/450 base versus 214/450
Afrique; English 392/450 versus 361/450. Translation chrF++ improves 9.11 to
41.47 into Twi, and 22.92 to 48.31 into English (32 pairs per direction).
Strong translation signal, NOT broad understanding acceptance; raw outputs
still make meaning errors. DO NOT resubmit. Receipts/results are in
`tmp/afrique-native-comparison/cachefix/`. Use `uv run` for the CLI's pinned
Modal and sacrebleu dependencies when regenerating/checking its summary.
One A100, 1,200 seconds per variant, serial single-use containers. No training
or response deployment is implied by a source-language benchmark.

Source negation and east/west errors were flagged without rewriting records.
The three translation demonstrations are source-separated, unchanged, and
agent-inspected, not human verified. V5's available Twi-labeled target tokens
were 69% health replies with no broad Twi conversation task. More training on
that same mix is not an evidence-backed remedy for general conversation.

New source-preserving question synthesis is COMPLETE:
`native_answers_20260910T134538Z`, generation `fc-01M25S5S8C3REXTFK6AFJW9W26`.
It uses 128 existing native answers from a 4,495-candidate general-source pool,
generated 105 suitable-source candidates and rejected 23 in its first stage.
Generation completed in 244.76 seconds. Independent English review
`fc-01M25ST8NRP2J6AAZE3QCAC614` completed in 227.16 seconds, with 104 model-positive
sources and 14/14 reviewer controls correct. Inspection STILL found false
approvals and source-translation errors. A separate conservative triage gate
leaves 85 sources with review candidates, not gold training acceptance.
Artifacts: `tmp/native-answer-synthesis-v1/gated-review.json` and its summary.
Original records/votes remain untouched. Five source errors are now flagged.
Do not blindly scale the 104 English model approvals or retrain on bad pairs.
No Twi answers are invented or rewritten. Read
`docs/native-answer-synthesis-20260910.md` for commands and limitations. This is
an explicit data experiment, not another claimed successful model fine-tune.

Next bounded foundation check: openai/gpt-oss-120b, pinned revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, publisher MXFP4. Private Modal app
`ghana-oss-response-comparison`; receipt folder `tmp/oss-response-comparison/`.
Use `scripts/run_stronger_response_comparison.py prepare-status --model oss120`
and submit ONCE only after preparation completes. Thirty identical saved product
inputs, one medium-reasoning mode, 2,048 output tokens, one H100, 1,800 seconds.
Native Harmony parser tests passed (reasoning isolation, both tool header orders,
and exhausted reasoning never appearing as the answer). This is a self-hosted
open-weight comparator, NOT a hosted GPT fallback, trained project model or
public service. Preparation excludes duplicate original/metal weight exports.

Private UI still runs the old pilot at 7863. Voice A, history, reviews, public
chat, and production are unchanged. No HF upload. Older active-run snapshots
below are historical and superseded by this section.

## 2026-09-10 V5 Final Evaluation And Stronger Foundation Comparison

V5 completed all 320 updates and its paired response evaluation. Run/call remain
`response_v5_20260910T120005Z` / `fc-01M25K0ZZW2A7JX8WVTAS30VHG`.
Final adapter hash is `407ba8f17bf9376bf031d7f4626340d382ce7d764d8b5da70e75f8bdfe677353`.
No model has been enabled in the private UI or promoted to production.

Checkpoint 100, actual paired AfriXNLI development results: Twi 271/450 for
both base and adapter; English 408/450 base, 412/450 adapter. Corrected
native-intent schema evaluation: 207/320 base, 206/320 adapter; literal entity
F1 0.5894 and 0.5930. No format failures with the explicit schema. This is NOT
a useful Twi semantic gain. Full outcomes and limitations are in
`docs/native-understanding-v5-20260910.md`.

Native intent training instructions were underspecified. Prepared
`tmp/native-intent-v3` fixes the task contract without rewriting any source or
target. All 4,978 records across splits verified to differ from v2 ONLY in the
system instruction. V5 itself trained on v2 and is not silently relabeled.
Use v3 in future training. Do not overwrite the old manifests or receipts.

Gemma thinking-mode comparison at checkpoint 100 is complete: the base is more
useful than the adapter on some responses but both have poor Twi wording. Adapter
budgeting and breathing cases exhaust the reasoning budget without a final
answer. No mode has been enabled in the UI. Default remains non-thinking.
Runtime regression tests cover hiding thinking and retaining it for tool turns;
the native template expects `reasoning`, NOT the parser's `thinking` field name.

Final-adapter checks are COMPLETE, do NOT duplicate:

- INJONGO schema-v2: `fc-01M25NJ313SD0P3W93TYBNGGNV`; intent 207/320 base,
  212/320 adapter; exact records 92 versus 103; entity F1 0.5894 versus 0.6186.
- AfriXNLI: `fc-01M25NJ3716R2YQCFYJ6522F67`; Twi 271/450 base versus 278/450
  adapter; English 408/450 versus 409/450. Small gains, not established broad
  conversational improvement.
- Paired generation: 30 product and 30 source cases per variant. Four adapter
  repetition flags versus zero base flags; breathing may only echo the request,
  and budget/newborn/pregnancy replies regress. V5 is NOT a chat replacement.
- Decision and actual model-card evaluations saved both locally and on the
  private Modal run volume. `data/response-adaptation/native-v5-decision.json`
  is the compact record. Full final run is backed up at
  `tmp/native-understanding-v5/completed/response_v5_20260910T120005Z/`;
  adapter hash independently verified. No HF publication.

A stronger open-weight reference is being compared before another
fine-tune on the same supervision. App `ghana-stronger-response-comparison`,
`Qwen/Qwen3.5-122B-A10B-FP8`, revision
`a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9`. CPU public-weight preparation call
`fc-01M25NF5NA6JVMBV8N1B82CS5S` completed in 673 seconds. GPU run
`qwen122_20260910T125452Z`, call `fc-01M25P5DDJB2WYV2HBMCS5J0J0`, is loading
the 39 FP8 weight shards at this snapshot. Use
`python3 scripts/run_stronger_response_comparison.py status`; do NOT resubmit.
The comparison is capped at 30 minutes on two H100s, 30 saved product
cases in publisher non-thinking and thinking modes. References stay local;
no private audio, no proprietary model call, no training or public service.
This external foundation is NOT our fine-tune and must not be represented as one.

The older submission snapshots below are superseded by this section.

## 2026-09-10 Native Understanding V5 Is Training

Latest: `response_v5_20260910T120005Z`, call
`fc-01M25K0ZZW2A7JX8WVTAS30VHG`, app `ghana-native-understanding-v5`.
Real updates confirmed at 12:02 UTC. Poll with
`python3 scripts/run_response_adaptation.py status --experiment native-v5`.
No duplicate submit. Read `docs/native-understanding-v5-20260910.md` first.

V4 was agent-stopped at 11:55 UTC after Twi response regressions at checkpoints
150/200; last observed logged step 210. This is NOT an unknown cancellation.
Its checkpoint-100 AfriXNLI dev score is 270/450 Twi versus 271/450 base;
English is unchanged at 408/450. No meaningful semantic gain or promotion.
The valid calculator failure was our punctuation guard, now fixed and actually
retested. Raw results, failed attempts and the quality-stop decision are preserved.

V5 uses 10,341 tokenized training examples, 905 validation, attention-only LoRA
on fresh Gemma 31B. Added source-verified native intent/entity targets (2,240 Twi,
1,157 English) and 1,200 unchanged Twi-to-English source pairs. Valid INJONGO
source package is `tmp/native-intent-v2`, not v1 (colliding upstream IDs were
caught before training). Official test overlap excluded, no synthetic response
inflation, and no AfriXNLI training. CPU backward/save/reload preflight passed.

Private inference/evaluation supports V5, but NO new UI selection is enabled.
The local tester on 7863 still uses the old pilot; owner voice A and reviews
are unchanged. Source-based native intent evaluation and 450-pair bilingual
meaning checks are ready; obtain actual checkpoint hashes and inspect results
before enabling any model. No production or HF publication in this work.

The older V4 submission checkpoint below is superseded.

## 2026-09-10 Grounded Response V4 Submitted

Active run `response_v4_20260910T111126Z`, call
`fc-01M25G7X11X29BAE952HHW73HS`, app `ghana-semantic-response-v4`.
Receipt `tmp/semantic-response-v4/run-receipt.json`. Poll:
`python3 scripts/run_response_adaptation.py status --experiment semantic-v4`.
Deploy recipe explicitly with `GHA_RESPONSE_RECIPE=semantic-v4 modal deploy
modal/train/train_response_adaptation.py`. Do not deploy/submit the default V2 by
mistake. No duplicate submission. Local disconnect does not cancel the job.

V3 completed 691 updates and 30 paired + 24 unpaired source checks. It learned
response termination and some useful English/tool behavior, but still reverses
Twi negation and fails medical/context checks. NOT accepted or enabled in the UI.
Read `docs/afrique-response-v3-20260910.md` for actual evidence, decoding caveats,
artifacts, and the next corpus correction. New V4 uses a fresh stronger instruction
base at lower LR, 1,000 no-tool-with-schema contexts, and pinned source-grounded
Twi QA. It has 5,744 tokenized training / 585 validation rows. Native train/save/
reload preflight passed. Broader native conversation and clinical supervision
are still insufficient; do not claim a finished model or human-verified corpus.

Private runtime supports V2, V3 and V4 with explicit checksum-bound selection,
stream parsing and bounded calculation. The model picker code is prepared but
NO new selection file is enabled; running local UI on 7863 still uses the old
pilot. No public app, HF publication, or TTS deployment in this work.

The following V3 submission checkpoint is superseded by its completed outcome.

## 2026-09-10 Afrique V3 restarted after checkpoint-storage fix

Active run: `afrique_v3_20260910T103220Z`, call
`fc-01M25E0ACC7AEHZYHXJV4HPJ2E`, app `ghana-afrique-response-v3`.
Receipt: `tmp/afrique-response-v3/savefix-run-receipt.json`.
Poll with `python3 scripts/run_response_adaptation.py status --experiment afrique-v3
--receipt tmp/afrique-response-v3/savefix-run-receipt.json` (one command).
Do not submit a duplicate. This is fresh SFT, not a resumed checkpoint.

Earlier V3 run `afrique_v3_20260910T102035Z` failed at its first save after
100 updates: PEFT's automatic embedding check tried to fetch base config into
the read-only default HF cache. Only a README exists in checkpoint-100; there are
NO trained weights to resume. The original receipt and 30 base outputs remain.
Saving now explicitly excludes frozen embeddings, refuses trainable embeddings,
uses the correct cache with offline access, and runs an actual adapter save before
training. Revised CPU preflight passed a backward step AND checkpoint save/reload
with the real base ID/revision. `adapter_checkpoint.py` owns this contract.

Private response runtime now supports Gemma and Afrique families with pinned
weights, native parsing, and real bounded calculation. Six parser/tool tests pass.
The local UI code supports an explicit family-bound selection in
`tmp/research-playground/private-selection.json`; NO selection has been enabled
and the running UI has not been restarted. It still serves the old pilot.

## Balanced response rejected; Afrique instruction SFT

The v2 run described below was deliberately stopped at **234 updates**, September
10 at 10:09:23 UTC, by this agent after paired checkpoint-100 and checkpoint-200
response regressions. This was NOT another unexplained cancellation. Decision:
`data/response-adaptation/response-v2-decision.json`. No v2 private UI or production
promotion; no completed 54-case final evaluation. Raw early comparisons are in
`tmp/response-adaptation-v2/`; read `docs/response-adaptation-20260910.md`.

Next is actual instruction SFT on `McGill-NLP/AfriqueQwen3.5-4B-50Langs`, not the
previous failed instruction-vector merge. Native tokenizer vocabulary agrees with
the pinned Qwen instruction tokenizer. The new source-preserving mix removes all
1,794 general-source machine-translated rows across splits; 5,521 raw training
examples and 334 validation examples remain. General conversation supervision is
English; Twi conversation breadth remains a limitation, not a solved corpus task.

Scripts: `build_afrique_response.py`, `modal/train/train_afrique_response.py`;
deployed app `ghana-afrique-response-v3`. Preflight/submission/status use
`scripts/run_response_adaptation.py --experiment afrique-v3`. Inspect the saved
receipt before any submit. The current submitted run is identified above.

## Earlier v2 submission checkpoint (superseded)

This supersedes the older "no training is running" checkpoint below. Deployed
normal Modal function (not a client-bound generator) was submitted once:

- Run: `response_v2_20260910T093637Z`
- App: `ghana-response-adaptation-v2`, `ap-ztiul450rx2RXsxNviRpeg`
- Call: `fc-01M25AT9XSS7Y5747JPEHJQ868`
- Receipt: `tmp/response-adaptation-v2/run-receipt.json`
- Confirmed at 09:45 UTC: 60/320 updates on one H100, checkpoint 50 saved.
- Poll with `python3 scripts/run_response_adaptation.py status`. Pending is not
  failure. Do not submit a duplicate because the local process disconnects.

Fresh rank-16 LoRA on pinned Gemma 4 31B instruction base, not the failed
translation adapter or external weights. 7,221 tokenized training examples:
1,702 general Twi responses, 944 Twi health responses, 1,890 English conversations,
950 English replay conversations, 1,335 schema-validated read-only tool calls,
400 auxiliary translation examples. Validation has 426 source-disjoint examples.
Native template/masking and a tiny backward-pass preflight passed; no target was
silently truncated. Four data-contract tests pass.

This is not a human-verified gold corpus. General Twi comes from public
machine-translated Ghana chat material, with explicit aligned first-paragraph
selection and six spot-check exclusions. Existing source records were not changed.
Sources retain their original restrictions, including noncommercial/share-alike
terms. No private recordings, conversations, or new proprietary annotation calls.
Inputs/hashes/provenance are in `tmp/response-adaptation-v2/corpus/manifest.json`.

Artifacts persist in existing private volume `ghana-health-understand-train`,
`/response-adaptation/response_v2_20260910T093637Z/`. The bounded job saves every
50 updates, then compares base versus adapter on 30 development scenarios and
24 held-out source rows. Training loss is not semantic accuracy. Inspect raw
outputs before selecting a checkpoint for the private tester. That tester still
uses the old pilot until an explicit service update; there is no hidden GPT
fallback and no production promotion.

## 2026-09-10 owner voice preference and LLM status

The owner chose **A**, the FP32 streaming version of the feeling-unwell sentence,
over the offered B/original alternatives. Saved in local SQLite `voice_preferences`
with the exact reply `A`, comparison question, all three audio identities and
selected hash `efdcdf5441a8475ba615cfd5b9a4d2332ebd8def0344ec8896ba5c6231296a70`.
This is a one-sentence preference, not an invented pronunciation/naturalness score
or production approval. The private audition now offers `CosyVoice A` and
`CosyVoice B` for that sentence; A is the initial audition. Other phrases retain
their original samples. Longer speech and streaming gaps still need evaluation.

LLM remains unpromoted; no training is running. A source-translation diagnostic
`ap-l5b8d5bqOPKgQKxsJjQwOK` was cancelled remotely on September 9 at 23:43:59 UTC,
then stopped with zero tasks. Only 12 base predictions and no adapter predictions
were saved in `tmp/general-language-corpus/` under the checkpoint-200 filename
ending `-paired-source-translation.jsonl`. Do not report it as a completed paired
comparison. Cause is unknown: the system log only says the input failed to respond
to cancellation within 30 seconds. This was not a function timeout or an agent
quality-stop. A deployed non-generator job with a saved call ID is a proposed
reliability improvement, NOT implemented or restarted at this checkpoint.

## 2026-09-09 broader language-adaptation experiment

Latest status, after the 23:17 UTC diagnostic: the fourth run was deliberately stopped by the agent
after **281 optimizer updates**, because direct conversational checks regressed.
This is NOT the unexplained earlier cancellation and NOT a completed epoch.
App `ap-tEBjptzAf9vbAX2EtNJ6yk`, run
`gemma31_language_v1_20260909T222324Z`, has zero running tasks. Checkpoints 200
and 250 remain saved; the checkpoint-200 weights were evaluated. Do not resume
this mixture unchanged or call lower validation loss a successful assistant.

The paired 30-case check `ap-BLq4808VlC1EGgUEcjMIHx` completed: simple negation,
arithmetic, corrected quantities and native tool calls work, but some Twi replies
merely restate the user, budget advice becomes empty, and hospital choice loops.
No production or private-chat model replacement. The inference-only ablation at
quarter adapter strength completed: `ap-qeK7qtRiFTRfox9jHvanj0`. It recovered
longer conversational answers and had no repeated-eight-word-span flags, but
still confused hospital with church and produced unreliable baby-context and
pregnancy replies. It is rejected too, not a new successful model. Raw paired
outputs and `checkpoint-200-scale025-summary.json` are in
`tmp/general-language-corpus/`. All experiment apps now have zero running tasks.

Read [`language-adaptation-20260909.md`](./language-adaptation-20260909.md).
A new public-source stage-one mix has 6,973 distinct training sources after
tokenization: 6,028 Twi-English pairs in both directions and 945 reviewed English
conversations, 13,001 examples. This is NOT completion of the older annotation
corpus. Ambiguous/contradictory medical-glossary data is quarantined, not trained.
The revised CPU preflight, including a real tiny-model backward pass, passed.
The first two GPU attempts stopped before training on loading/API compatibility
issues, now corrected. The third attempt `ap-iaFpsE7ITLbkhPIdd1BXzp` loaded all
weights and reached 60/62 baseline generation checks, then was cancelled before
training. The owner subsequently confirmed they did NOT stop that third run;
this authorized the fourth attempt described above. No new model or voice has
been promoted. The private tester's chat models remain the failed pilot and its
base; its new voice audition controls do not change chat inference.
Additional local TTS fixes preserve long replies instead of silently
cutting at 500 characters, synthesize bounded chunks, and keep a text turn intact
when optional speech fails. These are tested locally but NOT deployed.

## 2026-09-09 foundation comparison and TTS work in progress

The real speech completeness check `ap-F23dVSkiOolW8nZVw3PoYk` also passed:
664 characters delivered across five actual synthesis calls, 40.474 seconds of
valid audio, 10.370 seconds synthesis time; an explicit English span remained
intact. This is a transport/integration result, not a pronunciation score.
The stable voice revision is now pinned to the one used in the comparison.
Neither speech service nor the public web app has been redeployed.

The owner explicitly authorized temporary private Modal use of their voice
reference, without application-volume storage or publication. Completed CosyVoice
streaming tests found and fixed growing per-utterance buffers in the sequential
benchmark. CUDA library discovery and the FP16 runtime are now working. Latest
run `ap-eNG5gprZKwzY7fqmN6Iyj5`: warm first audio about 2.50 seconds on two
phrases, first utterance 5.31 seconds, excluding model loading/network. Subsequent
chunks still arrive late enough to create playback gaps. Not live-ready or
native-quality accepted. No reference was written to a persistent model volume.

Private tester `http://127.0.0.1:7863/?__theme=light` now has a collapsed
**Twi voice comparison** with five choices, real generated audio and per-sample
pronunciation/naturalness reviews. New option: CosyVoice stream (owner reference).
Only generated samples are served, never the source recording. Chats, browser
storage key and SQLite reviews were preserved. The tester runs as a detached
local process; logs `tmp/research-playground/server.log`. Verify the process on
port 7863 before any restart; do not stop unrelated Python services. No public sharing.
Desktop/mobile tests passed audio loading, switching, reset of ratings and overflow.
The owner has now reviewed the optimized "Feeling unwell" sample: "this sounded
more like me, just became a bit muffled at the end." This exact observation is
saved in local SQLite `voice_observations` against audio SHA256
`ad416b3eb6806228e81e332f2a8efa7bf27f9f92318487fb20138d7a531ba5c7`.
Do not convert it into invented pronunciation/naturalness scores or blanket voice
acceptance. The same-reference tail comparison completed in
`ap-0eQoLZNCy24g0J5IMQY4Zr`: FP16 streaming reproduced the original exactly;
FP32 streaming changed its ending, while whole-utterance FP16 retained a quieter
tail. Two matched-level samples were offered inline as A (FP32 streaming) and B
(whole-utterance FP16). The owner subsequently chose A; see the latest entry. Original samples
remain unchanged. Reports: `cosy-tail-check.json`, `cosy-tail-listening.json` in
`tmp/twi-voice-comparison/`. Do not claim the muffling is fixed without listening.

A separate external direct-response adapter diagnostic completed:
`ap-gR867dkg5P1W9tWwOwgsVN`, `DariusTheGeek/mhqa-itu-adapters/gen7454`.
This is not another project training run or production replacement. CPU shape
validation passed before the bounded 30-pair GPU comparison. Specific breathing
and pregnancy interpretations improved, but repetition and an invented commerce
function reject it as a product replacement. See the foundation report for exact
provenance, outputs and restrictions; do not relabel it as our model. Both latest
experiments have completed, and no additional training was started.

Latest user rejects Pilot v1 and asks to expedite, including poor TTS. Active
goal is a materially better general Twi-English assistant with health and commerce
specializations, direct final responses and a tested tool path. Do not mark the
goal complete after more scaffolding or another failed pilot. No new OpenAI calls.

Read [`foundation-comparison-20260909.md`](./foundation-comparison-20260909.md)
for current scripts, run IDs, measured failures, artifacts and next experiments.
Both a ready African instruction checkpoint and Qwen3.5-9B failed direct Twi
meaning tests. The Twi-CPT instruction-transfer candidates and Gemma 4 12B also
failed. Gemma 4 31B handles more general meaning and tool requests, but still
misreads important Twi health phrases. It is not promoted. All these benchmark
runs have completed; the subsequent SFT outcome is documented above. TTS samples are complete, including
three from Twi-trained CosyVoice using the owner's reference. The latter preserved
more words in a tiny ASR diagnostic, but native listening is still unrated. Two local
TTS routing bugs are fixed and tested but not deployed. Keep the model goal active.

## 2026-09-09 general-purpose scope and private pilot tester

The user clarified: general Twi-English understanding and response generation is
the overarching goal; health education is special expertise, alongside ecommerce
and tool use such as web search. Do not narrow the goal to medical QA, intents or
structured output. The updated research direction is
[`general-understanding-roadmap.md`](./general-understanding-roadmap.md).

The 95-source pilot remains unsuccessful and unpromoted. It is now accessible
for private, honest testing through `scripts/research_playground.py` at
`http://127.0.0.1:7863/?__theme=light`. Separate Modal app:
`ghana-understanding-pilot-private`, class `Pilot`, in
`modal/research_pilot_service.py`. There is NO public HTTP endpoint, production
route change, model upload to HF or proprietary LLM fallback. Existing model
weights are mounted read-only; the adapter checksum is checked before loading.
One L4, zero warm minimum, 60-second idle scale-down, bounded generation.
Only chat requests trigger inference. Test conversations go to Modal for
processing; no additional application corpus archive was uploaded.

The playground switches between the untouched MiniCPM Twi base and the pilot,
streams raw replies, retains the current conversation in browser storage and
saves ratings/corrections with actual inference provenance in local SQLite:
`tmp/research-playground/reviews.sqlite3`. These are not automatically train-eligible.
Automated UI test reviews must never be counted as human gold.
Local dependencies: `scripts/requirements-research-playground.txt`.

Real inference worked for both model choices. The same Twi personal-budget
question produced tangential institutional-budget explanations from both;
the pilot did not demonstrate the general competence the user wants.
Browser refresh restored the conversation, a correction was saved and
the stop control recovered the composer. CPU contract tests pass.
No new training or corpus annotation was run during this tester work.

Final verification: mobile 390x844 has no horizontal overflow and the composer
is visible; desktop and mobile screenshots were inspected. Send/stop buttons
have accessible labels. History survives a server restart through a stable,
locally stored browser-state key. Both Python suites pass (8 tests total).
One final direct inference returned 44 streamed chunks, 61 output tokens,
6.385 seconds of decoding and a normal stop. It ignored an explicit request
to reply in English and answered in Twi. Preserve that as a language-following
failure, not a positive model-quality result. Call ID:
`fc-01M23VM3PYXWVK0P2EQ7CBQKQY`.
The local server is backgrounded, with logs at
`tmp/research-playground/server.log`. It is not a public or cross-device service.

New candidate research: McGill-NLP/AfriqueQwen3.5-4B-50Langs explicitly includes
Akan/Twi in its continued-pretraining coverage. It is a BASE model, not a ready
instruction/tool assistant. Compare suitable language tasks first, then balanced
SFT against a capable instruction baseline such as Qwen3.5-9B. Do not claim either
is a measured product improvement yet. Tiny Aya Earth does not list Akan/Twi and
is noncommercial; it is not an automatic replacement. Full sources, corpus mix,
tool-use evaluation and runtime constraints are in the new roadmap.

## 2026-09-08 saved-annotation LLM pilot

The user cannot fund OpenAI with their card, confirmed they have Modal, and
asked to try training an LLM from the 305 completed annotations. The large
corpus goal remains incomplete. Do not restart OpenAI annotation calls.

A separate, explicitly uncalibrated experiment was prepared without changing
the normal corpus export or product promotion gates. Of 120 model-consensus
rows, one was excluded for similarity to locked evaluation data. The remaining
119 source records are split into 95 train / 24 pilot holdout, keeping all
three views of each source together: Twi interpretation JSON, direct Twi reply,
and English reply. This is 285 train examples, NOT 285 independent sources.
The 185 unresolved annotations remain excluded. No human gold is claimed.

Current production AfriHealth review decisions were read before export: zero
records. All selected safety labels are routine; do not train a constant
routine classifier. The pilot target omits safety and clarification labels.

Run: `ap-LuiKWmw2m3er1nsJA8Q0UQ`, app `ghana-health-annotated-response-pilot`.
Submitted with `modal run --detach modal/train/train_medical_response_pilot.py
--acknowledge-uncalibrated`. One A100-40GB, one-hour function timeout, three
epochs, LoRA rank 16, learning rate 0.00005, effective batch 8. Base is pinned
MiniCPM5-1B-Twi at `d807ca1a3323972afafabff8f9affe2639e37b5c`, not the failed
older raw-QA adapter. Training completed all 108 updates in 205.98 seconds;
mean training loss was 1.8898. This is fitting loss, not evaluation accuracy.
Output directory: `/data/sft/annotated_response_pilot_v1_20260908T213218Z`.
The run and its base/adapter evaluations are complete. The pinned base revision
was verified. Main worker elapsed time was 1,028.94 seconds; GPU-only estimated
cost $0.60, not final billing and excluding the initial failed startup.

Measured results:

- Interpretation schema: base 0/24, adapter 22/24. Structured decoding is greedy;
  this base is known to repeat with greedy decoding. Do not equate this format
  gain with semantic competence. Free-text evaluation used documented sampling.
- Adapter intent agreement: 13/24 (54.2%) against uncalibrated model labels,
  below the majority-class baseline 14/24 (58.3%). Only 6/24 interpretations
  passed the all-entities question-grounding check.
- Pilot held-out Twi reply chrF++: 23.62 base, 22.95 adapter. English: 12.03 to 25.91.
- Original locked-source Twi reply chrF++: 30.30 to 16.65. English: 14.87 to 20.56.
  These are wording-similarity measures, not human correctness. References are
  often longer than the concise target style, which can affect similarity.
- Critical product lexical checks: base 1/8, adapter 0/8. All product checks:
  base 2/13, adapter 1/13. These checks do not constitute clinical validation.
- Manual inspection found Twi copied into the English-meaning field and a
  pregnancy-prevention question mistranslated into setting up a "space".

Decision: save as an unsuccessful semantic/direct-response pilot, NOT a better
production model. It learned much of the format but did not demonstrate the
required understanding. No deployment or HF push occurred. The result does not
justify more epochs of this same narrow recipe or treating the 305 annotations
as a verified clinical corpus. Test stronger base/mix alternatives with these
same fixed checks and fix corpus coverage before claiming progress in semantics.

The additional permanent Modal archive of the input corpus was blocked by the
permission review. It was not uploaded or worked around. The user was asked
whether they approve that extra retention; no answer at this checkpoint. Inputs
remain in local `tmp/medical-response-pilot/v1`. Weights and reports persist on
Modal; local downloads are under `tmp/medical-response-pilot/run-20260908/`.

September 9 close-out: the recursive local optimizer-checkpoint download stalled
and was interrupted. It did not keep the training GPU running. The separately
downloaded complete adapter was verified (336 tensors) and restored into the
local run directory. SHA-256:
`294fd2815664c0b3140927dc637e5df35fc43563eaa9d8c5bb48f3678c2762de`.
All 14 root JSON artifacts parse, including tokenizer, metrics and predictions.
The incomplete optional checkpoint subdirectory is clearly named
`checkpoints.incomplete-download`; do not use it to resume training. The remote
checkpoint remains intact. No local download or GPU process remains running
for this experiment.

Readable comparison with all 93 base/adapter free replies:
`tmp/medical-response-pilot/run-20260908/annotated_response_pilot_v1_20260908T213218Z/comparison.md`.
Measured repository summary:
`data/medical-response-corpus/afrihealth-annotated-pilot.v1.summary.json`.
The model card is saved as `README.md` beside the adapter on Modal and locally.

Initial app `ap-aZh3Q8bemN4nHLudCDTVZR` failed during module import because
Modal relocates the file to `/root`. It was explicitly stopped; training did
not start there. Conditional local mounts and remote-safe root handling fixed
the issue, verified with a simulated relocated import. The built image was
reused by the corrected submission.

Full plan, limitations, reproduction and artifact paths:
[`docs/medical-response-pilot.md`](./medical-response-pilot.md).
Local artifact directory: `tmp/medical-response-pilot/v1`. Core contracts,
TypeScript, lint, source-group isolation, unchanged strict export, Python
contracts and real-tokenizer validation passed. Maximum training length is
510 tokens; no targets were truncated. The tokenizer's inference-only empty
think prefix is preserved during supervised training.

## 2026-09-07 full-corpus execution checkpoint

The user explicitly asked to finish the dataset goal. We started the full
4,407-source Twi training run, not another reference-only benchmark. It stopped
after the provider returned HTTP 429 `credit_balance_exhausted` again. A read-only
check verified the configured annotation hostname is `api.openai.com`.

Actual results:

- `data/medical-response-corpus/afrihealth-teacher-v3-train.jsonl` has 305 complete
  annotations: the 12 saved rows plus 293 newly completed this run.
- 120 have model consensus; 185 need review. The grounding audit identifies
  28 selected rows with non-question entity spans; all 28 are in review, none
  in consensus. This is not proof of full semantic or medical correctness.
- 4,102 Twi training-source annotations remain. One row also hit an invalid
  model entity shape (object instead of string): `afrihealth_akan_6842e1caa2d13722b6f3`.
  The raw output is retained; it must be repaired or explicitly excluded, not
  silently coerced into a label.
- Both complete annotations and partial paid teacher-stage caches were retrieved
  from the VPS. The stopped container has finished; no paid annotation job is
  currently running. Do not claim the goal completed.
- The local review data reader now prefers the 305-row v3 batch over the old
  62-row reference. This data-reader change has NOT been deployed to production.
- Strict annotated export still admits zero rows: no passing v3 calibration or
  local saved human reviews were supplied. Output is in
  `tmp/afrihealth-annotated-corpus/v3/`. No production-review count was asserted
  from this local-only export.

Broader source work completed:

- Added `data/annotated-source-corpus/sources.v1.jsonl`, containing original
  English AfriHealth and existing WAXAL/GhanaNLP text sources, with provenance,
  task separation, hashes, retained splits and exact held-out deduplication.
- Its training pool is 8,451: 4,402 original English QA, 2,249 GhanaNLP and
  1,800 WAXAL. Together with 4,407 Twi QA, the target is 12,858 source-training
  records. This is NOT 12,858 annotated/accepted records.
- Excluded synthetic symptom rows, seeds, and local recordings from this new
  pool as requested. Speech datasets are understanding-only, not invented QA.
- Added `scripts/annotate-source-corpus.ts`: two teachers plus adjudicator,
  a shared explicit intent/urgency rubric, question-only entities, conditional
  reference-backed replies, raw request audit, checkpoints and source-hash guards.
- Its live model check has NOT run because the existing API credit balance
  was exhausted. The full local mock-service contract test passed, including
  all three stages and zero new calls when resuming completed output.
- Added stable work-plan sidecars. The Twi resume plan recovers paid cached
  multi-row groups instead of regrouping every unfinished row. Completed labels
  are not overwritten while finishing their incomplete batch partners. Source
  selection changes fail closed; `--dry-run` checks plans without paid requests.
- Fixed the Twi circuit breaker so a known funding failure stops subsequent
  teacher stages and recursive requests, not just new top-level chunks.

API audit for this full Twi attempt (not a dollar charge estimate):

| Returned model | Successful request attempts | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| gpt-5.4-mini-2026-03-17 | 120 | 226,548 | 250,166 |
| gpt-5.5-2026-04-23 | 101 | 190,793 | 415,752 |
| gpt-5.6-sol | 82 | 534,669 | 135,915 |

These totals include cached partial stages and schema retries, not just the
293 fully completed rows. They exclude the prior calibration and 12-row run.
An asynchronous question asked the user which provider and amount they topped
up; no answer has arrived at this checkpoint. Do not assume their remaining
balance or ask for an unexplained refill. Confirm a spending ceiling and use
the saved usage to budget the remaining work.

Next: finish funding clarification, live-test the English/language runner on a
separate output, resume the saved Twi plan, annotate the remaining source pool,
resolve label/risk disagreements, and run real validation/export gates. Do not
train a placeholder corpus, bypass calibration, or substitute the old 7,000
synthetic symptom rows. The generic English/language annotated training exporter
and final integrated corpus quality report still need completion after live
outputs are available. No new model was trained or deployed in this pass.

## 2026-09-06 funded annotation restart (23:56 UTC)

The user replenished API credits. Funding is now verified by completed model
requests, not just authentication. No provider error occurred in either run:

- `afrihealth-teacher-v2-calibration.jsonl`: 62/62 completed, with two teachers
  and adjudication. These repeat the fixed reference IDs; they are not 62 new
  source records. Preserve the original v1 reference unchanged.
- The v2 comparison failed: 82.3% intent agreement, 82.3% safety agreement,
  67.7% source-answer assessment agreement, and 71.0% mismatch review capture.
  The reference is model-assisted, so these are consistency metrics, not
  accuracy against human gold. Report: `afrihealth-teacher-v2-calibration.gate.json`.
- Inspection exposed answer-to-question entity leakage, including medicines,
  locations and people introduced only in the reference answer. All 62 selected
  proposals contain at least one entity outside the question's lexical spans.
  This does not mean every entity or meaning is wrong; it means the proposals
  fail the new question-only extraction contract. The 24 old model-consensus
  labels are not cleared for training.
- Teacher-v3 explicitly separates question interpretation from reference-answer
  assessment. Entities must be spans of the original question. Deterministic
  annotation and export guards reject ungrounded model entity labels, even if
  the models agree. Resuming into a different prompt/model output now fails
  before any paid request, instead of mixing pipelines.
- `afrihealth-teacher-v3-grounding.jsonl`: 12 previously unannotated structural-pass
  training-source rows, no overlap with the 62 reference IDs or held-out split.
  All 12 completed and all selected entities passed the lexical grounding check;
  10 have model consensus and two need review. This is a bounded contract check,
  not a passing semantic calibration or a training-ready corpus.
- Audit both runs with `pnpm eval:medical:afrihealth:question-grounding
  --annotations <file> --strict`. The JSON reports retain per-row violations.
- Both isolated research containers finished. Results and raw model/usage
  checkpoints were retrieved locally. No live app or model was deployed.

Next work is annotation quality, not more credit troubleshooting: examine the
reference disagreements, define ambiguous intent/safety label boundaries, and
validate teacher-v3 on representative real data before bulk acceptance. Inspect
coverage too: the latest hash sample is largely adolescent policy/rights QA,
not a representative patient-conversation evaluation. Broader original-English
source annotation is still needed for the 7,000 unique-source target. Do not
increase counts by duplicating translated views or mark these model rows human
reviewed. The older funding-blocked notes below are historical.

Checks passed: TypeScript, targeted ESLint, annotation-client tests, export
tests (including answer-only entity rejection), mixed-pipeline resume refusal,
and the v3 lexical grounding audit. The v2 consistency and grounding audits
fail as expected; their results have not been overridden.

## 2026-09-06 annotation-system checkpoint

Latest correction and completed work (02:49 UTC):

- The production OpenAI key is valid. The earlier HTTP 401 came from passing
  the quoted Infisical export directly to `docker run --env-file`. A comparison
  confirmed that removing the file's outer quotes matched the running app's
  key. Do not rotate the key on the basis of that earlier diagnosis.
- A real completion through the active credential now returns HTTP 429
  `credit_balance_exhausted`. The corrected 62-row teacher-v2 run stopped with
  zero new annotations. A later one-request probe confirmed the same result.
- The account needs API funding before model annotation can continue. The
  production review table has zero AfriHealth reviews at this checkpoint.
- Added `scripts/run_afrihealth_annotation_remote.py`: it reads the running
  app's parsed provider variables into memory and passes only those to an
  isolated research container. It never writes a new secrets file.
- Added a research-only annotation client with per-stage resumable checkpoints,
  raw outputs, returned model IDs, token usage, request IDs, and timeouts.
  Authentication and exhausted-credit errors stop the job without splitting
  the corpus into repeated failing requests.
- `--train-only` now intersects the deterministic training pool (4,407 Twi
  records), rather than including every upstream training record. `--reference`
  selects the existing real-data calibration IDs.
- Added `pnpm corpus:medical:afrihealth:export-annotated`. It consumes AfriHealth
  sources, candidate annotations, and reviewer selections; `--reviews-db` reads
  Postgres directly and fails if that read fails. The older response export did
  not include these source records.
- Export checks reject unmatched/failed pipeline calibration, stale selections,
  changed source hashes, reviewer exclusions, and held-out overlaps. Human
  Twi corrections do not inherit stale English reply translations. Clarification
  comes from the selected label, not the presence of text in an ambiguity field.
- Paired Twi/English examples retain one source identity and split. They do not
  count as two unique sources. The full 7,000-unique-source target still needs
  the broader source annotation work; the 4,407-row Twi lane alone cannot meet
  that target by doubling translated views.
- The current strict export admits zero rows: no matching passing calibration
  or saved human review is available. This is an incomplete corpus, not a
  completed training export.

Current corpus boundary:

- 8,809 balanced research-train rows: 4,407 Twi and 4,402 English.
- 2,198 locked evaluation rows: 1,102 Twi and 1,096 English.
- 5,569 immutable AfriHealth Twi source question-answer rows.
- 62 stronger dual-teacher annotations used as a calibration reference, not
  human gold: 31 silver consensus, 31 review, 27 training-eligible.

Do not bulk-run the current open-only annotator. Its 58-row overlap with the
stronger reference failed the new calibration gate: 56.9% intent agreement,
72.4% safety agreement, 69.0% source-answer-assessment agreement, and 55.0%
mismatch-review capture. Qwen3.5-35B plus NLLB remains a useful independent
proposal path, not accepted training truth. NLLB's CC-BY-NC-4.0 licence also
prevents treating its output as a clean commercial training source.

Completed in the latest pass:

- Fixed wrapped-JSON field loss in the Modal annotator and importer.
- Added complete-schema checks and one repair generation for each teacher and
  adjudicator.
- Retained raw and repaired outputs, hashes, and attempt counts for audit.
- Added `scripts/evaluate-afrihealth-annotation-calibration.ts` and the package
  command `pnpm eval:medical:afrihealth:annotation-calibration`.
- Changed bounded Modal checks to eager vLLM execution after compiled startup
  stalled across two GPUs.
- Modal run `ap-dJE2ONzSTftdrzUs9d4dAe` passed the corrected plumbing: 4/4
  direct proposals, 4/4 pivot proposals, 4/4 adjudications, and 4/4 importer
  acceptance. This is a plumbing pass, not a semantic-quality pass.
- TypeScript, targeted ESLint, Python compilation, and the four-row import
  check passed.

External dependency:

- The checkpointed teacher-v2 job is ready to run with meaning-first and
  safety-first teachers plus a separate adjudicator.
- Infisical Universal Auth currently returns HTTP 502 locally.
- The production server contains only `OPENAI_API_KEY`; the corrected launch
  confirms it is valid but API credits are exhausted (HTTP 429).
- No alternate strong-teacher provider key is configured on the server.

Resume sequence:

1. Replenish the annotation account's API credits. Do not rotate the valid key
   or put it in a local `.env` file.
2. Run teacher-v2 on a small real AfriHealth calibration batch and import it.
3. Compare it with the fixed 62-row reference and manually inspect the
   disagreement/safety sample. Do not call either set human gold.
4. Only after the gate passes, run checkpointed training-source annotation in
   shards. Keep the 2,198-row evaluation set locked.
5. Export high-confidence silver plus human-reviewed disagreement rows, retain
   balanced English replay, train the joint semantics-and-response model, and
   publish only with a complete Hugging Face model card and measured gates.

Resume the corrected calibration on the VPS after syncing the current scripts:

```bash
python3 /opt/ghana-health-ai/research-annotation-v2/scripts/run_afrihealth_annotation_remote.py \
  --name gha-research-annotator-calibration-v2 -- \
  --limit 0 --reference data/medical-response-corpus/afrihealth-akan-annotations.v1.jsonl \
  --chunk-size 2 --concurrency 6 \
  --out data/medical-response-corpus/afrihealth-teacher-v2-calibration.jsonl \
  --summary data/medical-response-corpus/afrihealth-teacher-v2-calibration.summary.json
```

## 2026-09-06 bilingual response-model checkpoint

Completed:

- Built a deterministic pinned Ghana corpus from AfriHealth-QA: 11,116 source
  rows, 8,809 balanced research-training rows, 2,198 locked validation rows,
  and 109 isolated review/reject rows.
- Trained a LoRA over `ghananlpcommunity/MiniCPM5-1B-Twi` for 600 steps (about
  1.09 epochs) on Modal. Train loss was 1.936 and in-training eval loss 1.896.
- Added a proprietary-judge-free base-versus-adapter evaluation over 64 Twi and
  64 English held-out rows plus 13 product cases.
- Rebuilt the corpus and reproduced all artifact hashes exactly.
- Lint, TypeScript checks, Python compilation, and the Modal evaluation runner
  passed.

Decision:

- Do not publish or deploy this checkpoint.
- English chrF++ improved from 14.44 to 23.49 and language matching improved
  from 50% to 98.44% overall.
- Twi chrF++ regressed from 27.46 to 22.22.
- Product cases improved only from 2/13 to 3/13; critical cases stayed at 0/8.
- Generated failures included invented diagnoses, medicines, and unsafe
  reassurance.

Canonical record:

- `data/medical-response-corpus/afrihealth-response-run.v1.summary.json`
- Full local evaluation artifact: `tmp/response-eval-full.v1.json`
- Full Modal adapter path:
  `/data/sft/ghananlpcommunity_MiniCPM5-1B-Twi_afrihealth_bilingual_response_v1_full`

Next training stage:

1. Build a model-proposed, human-adjudicated response curriculum with explicit
   normalized Twi, faithful English meaning, intent, entities, ambiguity,
   safety level, and concise Twi/English response fields.
2. Use two or three proposals per row and prioritize disagreement,
   safety-critical, code-switch, negation, and prior product-failure clusters.
3. Retain balanced English replay and the same locked evaluation set.
4. Train the joint semantics-plus-response model and require every promotion
   gate to pass before Hugging Face publication or app exposure.

## 2026-08-28 understanding research checkpoint

Current checkpoint:

- Production is live and ready at `https://ghanahealth.serendepify.com`.
- Research workspace: `/research/ase`.
- Latest pushed commit before this checkpoint: `8da6061`.
- New local work adds the first corpus candidate builder and candidate review
  queue; commit/deploy after validation.

Important correction:

- The 50 synthetic benchmark rows are **only** for model comparison.
- The actual corpus comes from licensed datasets, local recordings, opt-in
  contributions, and reviewed synthetic augmentation.
- Models may populate draft transcript/meaning/entity fields, but the project
  must train only from reviewed/finalized exports.

Completed in this pass:

- Fixed Modal NLLB benchmark runner by loading the base NLLB tokenizer against
  the fine-tuned `ninte/twi-en-nllb-v2` weights.
- Ran the 50-row Modal benchmark successfully.
- Built `data/understanding-corpus/candidates.v0.jsonl` with 80 review-ready
  candidates:
  - 30 local-recording rows
  - 50 curated prompt rows
  - 80 draft model annotations
- Added `scripts/build-understanding-corpus-candidates.ts`.
- Added `scripts/summarize-understanding-research.ts`.
- Added report: `docs/understanding-research-report-2026-08-28.md`.

Key result:

- NLLB is fast but not safe as a sole health meaning annotator. It mistranslated
  multiple meaning-critical benchmark rows, including unwellness, pregnancy
  symptoms, eye pain, body-part location, and commerce budget.

## 2026-08-26 understanding corpus workspace

The understanding research direction is **not** to train from the 50 synthetic
benchmark rows. Those rows remain a small probe set for comparing model
behaviour and latency.

New internal workspace:

- `/research/ase`
- API: `/api/research/understanding`
- Local review writes: `tmp/understanding-review/reviews.v0.jsonl`
- Access: open in local development; in production requires an admin/researcher
  account or `RESEARCH_REVIEW_ENABLED=true`

The workspace now separates:

1. **Sources:** GhanaNLP parallel, WAXAL, Common Voice, local recordings,
   opt-in product contributions, and reviewed synthetic TTS.
2. **Storage decisions:** public datasets stay referenced by pinned revisions;
   private audio belongs in private object storage; Postgres stores references,
   hashes, reviews, and experiment metadata.
3. **Benchmark:** the 50 synthetic probes are explicitly labelled as
   non-training, non-gold model-comparison rows.
4. **Review:** native-speaker review can add normalized Twi, faithful English
   meaning, literal gloss, intent, entities, ambiguity, decision, and notes.

Validation run:

```bash
pnpm lint
pnpm build
```

Next research step is Gate 1, not a shortcut:

1. Create the import/audit layer for dataset-derived corpus candidates.
2. Catalogue source IDs, licences, revisions, splits, hashes, speaker metadata,
   and consent scope.
3. Generate model proposals only as drafts.
4. Use `/research/ase` for human review and export only eligible reviewed
   records.

## 2026-08-26 pull/deploy update

Pulled the latest main branch through `9eca525`, then added one safety-path fix
and deployed production via the direct VPS route.

Current live deploy:

- Commit: `32ec6f71df7ed0493d8d386a8d38a73508ec3d98`
- Web image: `ghcr.io/teckedd-code2save/ghana-health-ai:32ec6f71df7ed0493d8d386a8d38a73508ec3d98`
- Deploy path: `bin/deploy-direct`
- Prisma migrations: no pending migrations
- Production readiness: `ready`

Pulled work since the prior handoff:

- Research Preview / Gate 0 cleanup.
- Responsive recent-chat/session drawer.
- Guest conversation registry and server-side conversation loading/deletion.
- Conversation continuity fixes for streamed voice/text.
- Direct response synthesis from the original language.
- Provenance labels for `Live model` versus `Safety fallback`.
- New language-response contract:
  `pnpm eval:language-response`.
- Research direction document:
  `docs/GHANA_LANGUAGE_UNDERSTANDING_RESEARCH.md`.

Local fix added after pull:

- `fix(understanding): ask clearly during fallback outage`
- The no-LLM fallback now asks the user to say the request again in different
  words instead of only saying the model is unavailable.

Validation run:

```bash
pnpm eval:language-response
pnpm eval:understanding:fallback
pnpm lint
pnpm build
pnpm eval:product-readiness:prod
```

Production endpoint checks:

- `/api/config` reports Twi TTS as `stable-twi` /
  `ghananlpcommunity/stable-twi-tts`.
- `/api/tts` with Twi text returned WAV audio, sample rate `22050`, about
  `97k` base64 chars, and about `1.7s` TTS latency.

One local contract still needs a local database or secrets-backed DB before it
can be rerun:

```bash
pnpm eval:conversation:fallback
```

It failed locally with database connection refused, not with an assertion
failure. Production readiness confirms the production DB path is healthy.

Next useful step:

1. Exercise the live UI on production: session drawer, new chat, delete chat,
   streamed transcript preservation, and Twi response style.
2. If UI feels stable, continue Gate 1: canonical research data schema/import
   audit for the Ghana language understanding programme.
3. Keep DONDO v3 spend blocked until the corpus audit shows enough
   speaker-diverse held-out product data.

## 2026-08-21 session controls and response synthesis correction

- The app menu now contains a responsive recent-chat workspace with New chat,
  conversation selection, per-chat context actions, and confirmed deletion.
- Browser guests retain a local registry of their conversation IDs; signed-in
  users receive their server-side conversation list.
- Text responses expose `Live model` versus `Safety fallback` provenance so a
  degraded runtime cannot masquerade as model intelligence.
- Comprehension and answering are now separate calls. The first call sees only
  transcript evidence, recent dialogue, and memory and must either state a
  faithful English meaning or return `understood=false`; it cannot answer.
- Answer generation runs only after the comprehension gate passes and receives
  the accepted meaning, not the raw transcript.
- Hard-coded eye-pain, hospital-choice, migraine, rest/fluids, and generic
  clinic responses have been removed. Deterministic enforcement remains only
  for explicit emergency patterns.

Response path:

`transcript + recent history + scoped memory + ASR evidence -> understood/not-understood gate -> natural answer from accepted meaning -> deterministic emergency enforcement -> stored/streamed response -> optional TTS`

Retrieval remains disabled, so current intelligence comes from the configured
LLM plus conversational context, not an evidence-grounded health knowledge
source. Grounded retrieval is the next response-quality layer after live
model/fallback behavior is verified.

**Last session focus:** Voice-first runtime cleanup, schema-first understanding, agent memory, and DONDO ASR training path.  
**North star:** [`docs/research-stack.md`](./research-stack.md)  
**Training roadmap:** [`docs/model-training-roadmap.md`](./model-training-roadmap.md)
**ASR decision:** [`docs/asr-model-decision.md`](./asr-model-decision.md)
**Model credit plan:** [`docs/model-credit-plan.md`](./model-credit-plan.md)
**Pilot runbook:** [`docs/PILOT.md`](./PILOT.md)

---

## 2026-08-21 handoff

Production is testable at `https://ghanahealth.serendepify.com`.

Current live deploy:

- Commit: `092626068d32d896bf6c9d17fef9767d07487e06`
- Web image: `ghcr.io/teckedd-code2save/ghana-health-ai:092626068d32d896bf6c9d17fef9767d07487e06`
- Twi TTS route: `stable-twi`
- Twi TTS model: `ghananlpcommunity/stable-twi-tts`
- English TTS route: `facebook/mms-tts-eng`
- Stable Twi Modal endpoint:
  `https://createdliving1000--ghana-health-tts-stable-twi-speak.modal.run`

Verification already run:

```bash
curl -sS https://ghanahealth.serendepify.com/api/config
curl -sS https://ghanahealth.serendepify.com/api/tts \
  -H 'Content-Type: application/json' \
  --data '{"text":"Akwaaba, wo ho te sɛn?","language":"tw"}'
```

The TTS smoke test returned `provider=stable-twi`, sample rate `22050`, WAV
audio, and about 2s synthesis latency. Test the service now by using normal
Twi voice chat, then compare:

- Does the voice sound more natural than MMS?
- Does the response stay in Twi when the input is Twi?
- Does the conversation append turns instead of replacing previous turns?
- Does the model picker near the mic route Twi beta/stable as expected?

GitHub Actions deploys are currently blocked by exhausted Actions credits. The
repo now has a direct VPS deploy route:

```bash
bin/deploy-direct
```

That command builds the committed repo on the VPS, applies Prisma migrations,
restarts only `ghana-health-ai-web`, and smoke-tests production config/readiness.
Use `STOP_OTHER_CONTAINERS=1 bin/deploy-direct` only when the VPS needs extra
memory and unrelated containers can be stopped. It does not upload local env
files and does not depend on pulling from GHCR.

Next product/R&D work:

1. Continue P4 live voice polish: interruption handling, streaming states, and
   TTS quality comparison against stable-twi/nano-twi/Qwen research.
2. Continue R1 beta measurement: v6 vs DONDO v2+LM with the same product clips,
   correction logs, and latency.
3. Start P5 commerce tool orchestration only after voice continuity feels
   stable.
4. Build R3 synthetic Twi voice-note generation from reviewed corpus JSONL, but
   keep synthetic audio out of final human held-out evals.

### Later 2026-08-21 continuation update

R3 synthetic Twi voice-note generation is now an executable pipeline, not just a
plan item.

Added:

- `scripts/synthesize-twi-voice-notes.ts`
- `scripts/eval-synthetic-voice-notes.ts`
- `pnpm corpus:synthesize-twi`
- `pnpm eval:synthetic-voice-notes`

The generator reads reviewed Twi/code-switch corpus JSONL, calls the configured
Twi TTS provider, writes audio files, and emits a training/augmentation manifest
with:

- `audio_path`
- `reference`
- `bucket`
- `speaker_label=synthetic_<provider>`
- `source=synthetic_tts`
- `tts_model`
- `voice_id`
- `duration_s`
- `sha256`
- `holdout=false`

Important guardrail: the script refuses `needs_review: true` rows and
`source=llm_translation_draft` rows by default. This is intentional because the
current generated Twi prompt packs still contain poor draft translations and
repetitions. Do not synthesize those into training audio unless using
`--allow-drafts` for a clearly labeled non-training experiment.

Validation completed:

```bash
pnpm eval:synthetic-voice-notes
pnpm eval:tts-routing
pnpm lint
pnpm corpus:synthesize-twi -- \
  --input tmp/asr-collection-pack/prompts.corpus-v2.health_twi.jsonl \
  --dry-run \
  --limit 5
pnpm eval:product-readiness:prod
```

Results:

- Synthetic voice-note contract passed.
- TTS routing contract passed.
- Lint passed.
- Dry-run correctly found `0` eligible rows in the current `health_twi` draft
  prompt pack and skipped all draft rows.
- Production readiness is `ready`.
- Expected degraded items remain:
  - no ASR checkpoint passes all promotion gates yet
  - product eval buckets still need more consented, speaker-diverse clips

One check was intentionally stopped because the user needed to leave:

```bash
pnpm eval:prod:smoke
```

It had already passed production readiness and all JSON live-pipeline fixtures,
then was stopped while waiting on the stream fixture phase. Re-run it next time
if full production stream verification is needed.

Next best step from a new device:

1. Commit/push the R3 synthetic voice-note changes if not already committed.
2. Create or export a reviewed corpus JSONL with `needs_review=false` rows.
3. Run `pnpm corpus:synthesize-twi -- --input <reviewed.jsonl> --dry-run`.
4. If eligible rows look correct, run the same command without `--dry-run`.
5. Validate the resulting manifest with
   `pnpm eval:local-asr-import -- tmp/synthetic-twi-voice-notes/manifest.jsonl`.

### R2 corpus audit update

R2 now has an executable corpus-readiness audit:

```bash
pnpm eval:asr-corpus -- --manifest tmp/asr-local-train/manifest.jsonl
pnpm eval:asr-corpus:strict -- --manifest data/asr-product-eval/manifest.jsonl
```

The audit reports bucket counts, speaker diversity, holdout count, repeated
references, missing audio, synthetic rows, and promotion readiness. It is meant
to answer whether a corpus can support model-credit spend, not just whether the
JSONL shape is valid.

Current local corpus audit:

- rows: `40`
- speakers: `2`
- duration: about `3.0` minutes
- holdout rows: `0`
- synthetic rows: `0`
- missing audio: `0`
- bucket counts:
  - `health_twi`: `25 / 100`
  - `commerce_twi`: `10 / 70`
  - `codeswitch_tw_en`: `5 / 60`
  - `health_en`: `0 / 50`
  - `phone_noise`: `0 / 20`
- promotion-corpus ready: `no`

The same corpus passes import validation:

```bash
pnpm eval:local-asr-import -- tmp/asr-local-train/manifest.jsonl
```

So the immediate interpretation is: the current 40 clips are usable for local
experiments and regression checks, but they are not enough for DONDO promotion
or serious model-credit spend. The next data push needs more speakers, more
commerce/code-switch clips, an English retention bucket, phone/noise clips, and
a real frozen holdout split.

---

## Product goal (do not lose this)

Voice-first **Twi** health companion for Ghana:

**listen (ASR) → understand (Twi-native path) → speak (TTS)**  

English **only** when the user sets language preference to English.

---

## What’s already in this branch

### App / product
- Clean product UI (home, voice, chat, market, login)
- `understandUtterance` runtime path: ASR metadata → schema-first health/commerce intent → memory-aware LLM
- Runtime understanding no longer uses Twi/product retrieval as intelligence. Retrieval metadata is intentionally `engine=none`.
- Focus is inferred when the caller does not pass a tab/focus. Shopping language such as buy/order/market/price, Twi `metɔ` / `atɔ` / `boɔ` / `hwehwɛ`, wins as commerce even when the item is health-related.
- Commerce understanding now extracts structured slots from speech/text:
  - `action`: buy/order/find/price/availability/unknown
  - `item`, `quantity`, `location`, `fulfillment`
  - Slots are stored in assistant metadata and returned by chat/voice APIs as `understanding.commerce`.
  - `src/lib/commerce-plan.ts` adds a deterministic next-action plan:
    - ask missing item/quantity/location
    - search connected local catalog
    - draft order for confirmation
    - explain unsupported when live marketplace is not connected
  - `src/lib/commerce-execute.ts` executes only safe commerce actions:
    - local catalog search with real `Product` rows/prices
    - order draft metadata requiring explicit confirmation
    - no checkout, payment, or external-market mutation
  - `/api/commerce/confirm` is the only commerce voice/chat confirmation endpoint that mutates cart state, and it requires `confirm: true`.
  - APIs return `understanding.commerceExecution`.
  - Voice UI shows a compact commerce action chip for the top matched catalog product; tapping it calls `/api/commerce/confirm`.
  - This is for future search/order agents only; no fake store, price, or availability is invented.
- Health understanding now produces deterministic action-plan metadata in addition to the LLM answer:
  - `src/lib/health-plan.ts` maps transcript quality and clinical severity into `needs_clarification`, `self_care`, `clinic_recommended`, or `urgent_referral`.
  - Chat and voice APIs return `understanding.health`.
  - Assistant audit metadata stores the health plan, so weak ASR can trigger clarification instead of confident advice.
  - `pnpm eval:health-plan` verifies emergency, weak-transcript, and routine pathways without an LLM call.
- `src/lib/agent-memory.ts` — scoped agent memory for profile, health, and commerce continuity
- `src/lib/twi-retrieve.ts` — still present for later experiments, but not part of the live understanding path
- **Passage embedding cache** on `KnowledgeArticle` / `Product` (`embedding_b64`, model, engine, `embedded_at`)
  - Request path embeds **query + cache misses only** (not the full KB every turn)
  - Backfill: `sec -- pnpm db:index-embeddings` (`--force` to rebuild)
- Expanded Twi knowledge seed (maternal, child, malaria, mental, chronic, FP referral)
- Harder danger heuristic (Twi + EN) + force escalate / hotline prefix when urgent
- Config flags: `abenaEmbed` = Modal ABENA only; `embedAny` = ABENA or OpenAI fallback
- TTS serving upgrades; HF model-card helpers on train pushes
- Live chat/voice turn runner with streamed stages and reviewed answer deltas: ASR, retrieval, LLM answer, LLM review, streamed reply, TTS
- Voice UI consumes live backend stage events so the orb/status moves through transcribing, thinking, responding, and speaking instead of showing a generic wait.
- ASR quality metadata now reaches the response model/reviewer so unclear audio can produce a model-generated clarification instead of guessed health advice
- Response pipeline has a deterministic outage fallback:
  - If no LLM key is configured, or the provider times out/returns no draft/review, `understandUtterance` returns a structured fallback instead of throwing.
  - Health fallback uses `src/lib/health-plan.ts` for weak-ASR clarification, urgent referral, clinic recommendation, or conservative self-care.
  - Commerce fallback preserves extracted item/quantity/location and asks the next useful shopping question without inventing stores/prices.
  - `pnpm eval:understanding:fallback` verifies health, weak transcript, and commerce outage behavior.
  - `pnpm eval:conversation:fallback` verifies the full chat/voice turn runner still writes user + assistant messages with fallback metadata when no LLM key is available.
- Voice transcript correction loop:
  - `AsrFeedback` stores corrected transcripts/ratings tied to conversation + user message.
  - `/api/voice/feedback` persists corrections without retaining raw audio.
  - The voice UI exposes a compact inline correction action on the heard transcript.
  - `pnpm eval:asr-feedback:export` exports corrected text rows for review and data-loop planning.
- Understanding eval fixtures cover Twi danger signs, weak ASR clarification, English health, Twi commerce intent, memory, and no invented commerce prices.
- Understanding eval fixtures now include Twi tomato purchase, Twi quantity/delivery/location, and English price-query slot checks.
- `pnpm eval:commerce-plan` covers planner behavior without an LLM call.
- `pnpm eval:commerce-execute` verifies local catalog search/order draft and confirms no cart mutation.
- `pnpm eval:commerce-confirm` verifies cart mutation happens only after explicit confirmation.
- Full live pipeline fixtures pass in JSON and stream mode on `http://localhost:3100`; stream mode verifies staged event order, assistant-before-reply ordering, and reply deltas.
- Product readiness preflight:
  - `/api/readiness` reports runtime readiness for DB, Twi ASR, separate English ASR, TTS, LLM, response fallback, ABENA, ASR promotion, and product speech data.
  - `pnpm eval:product-readiness` fails if any required runtime dependency is blocked.
  - `pnpm eval:product-readiness:prod` runs the same gate against `https://ghanahealth.serendepify.com`.
  - Current readiness is `ready`; ASR model promotion and product eval data remain `degraded`, intentionally not blocking local demo.
- `scripts/audit-asr-promotion.ts` turns the model decision into an executable promotion audit across Twi, English retention, health-domain, code-switch, phone/noise, latency-adjacent evidence, and HF card gates.
- The ASR promotion audit now recognizes HF card evidence verified by `pnpm eval:hf-model-cards`, while still refusing promotion when WER/domain gates fail.
- `data/asr-product-eval/` defines the consent-aware product ASR eval manifest; `scripts/validate-asr-eval-manifest.ts` checks buckets before serious credit spend.

### Explicitly out of scope (validated)
- Electric Sheep Africa **Ghana Country Data Coverage** collection — tabular EN official stats (census, energy, road, facilities). **Not** speech/Twi text for ASR/TTS/SFT. Optional later: facility lat/lon as a referral feature only.

### Modal (deployed)
| Service | URL / note |
|---------|------------|
| **ABENA embed** | `https://createdliving1000--ghana-health-embed-embed.modal.run` |
| Embed health | `https://createdliving1000--ghana-health-embed-health.modal.run` |
| TTS speak | `https://createdliving1000--ghana-health-tts-speak.modal.run` |
| ASR | Prod: v6 Whisper small Twi |

---

## Remaining work (priority order)

### 1. Runtime service wiring

Infisical was **502** last check — retry when up.

Current local config check on `http://localhost:3100/api/config`:

- `modalAsr: true`
- `modalAsrEnglish: true` (`en` route is `english`)
- `modalTts: true`
- `abenaEmbed: true`
- `embedAny: true`
- `appUrl: http://localhost:3100`

The app now has documented non-secret defaults for the public Modal English ASR, ABENA embed, and TTS endpoints. Infisical should still override them when endpoints or tokens change.

Verified public endpoint checks:

- English ASR health: `service=ghana-health-asr-en`, `model=openai/whisper-small`
- ABENA embed health: `service=ghana-health-embed`, `model=Ghana-NLP/abena-base-asante-twi-uncased`
- ABENA embed POST: returns 768-dim embeddings
- TTS speak POST: returns WAV audio from `facebook/mms-tts-aka`

```bash
MODAL_EMBED_URL=https://createdliving1000--ghana-health-embed-embed.modal.run
MODAL_ASR_EN_URL=https://createdliving1000--ghana-health-asr-en-api.modal.run
# keep existing MODAL_ASR_URL / MODAL_TTS_URL
```

Deploy the English endpoint without replacing Twi:

```bash
ASR_APP_NAME=ghana-health-asr-en MODEL_ID=openai/whisper-small modal deploy modal/asr_service.py
```

Verified health: `service=ghana-health-asr-en`, `model=openai/whisper-small`.

Then:

```bash
sec -- npx prisma migrate deploy
sec -- npx prisma db seed
sec -- pnpm db:index-embeddings
# confirm:
curl -s http://localhost:3100/api/config | jq .modalAsrEnglish,.asrRoutes,.abenaEmbed,.modalTts,.appUrl
```

### 2. Fix HF token for LoRA SFT (blocked)

Modal secret `huggingface-token` 403 on Qwen/Llama. Refresh token → re-run `train_understand.py`.

### 3. ASR multi-domain promote

> **2026-08-20 update — see [`docs/asr-rnd-session-2026-08-15.md`](./asr-rnd-session-2026-08-15.md) and the
> [`asr-model-decision.md` addendum](./asr-model-decision.md#2026-08-20-dondo-v2-recovery-addendum).
> Execution path: [`docs/asr-rd-execution-plan.md`](./asr-rd-execution-plan.md).
> Key shifts: (a) v6's product-domain WER is 54.18% — Waxal WER does not
> predict product WER; held-out domain evals are now first-class gates.
> (b) DONDO v1 beats v6 by ~22pp on the local corpus (32.66% vs 54.18%) —
> DONDO v2 completed and is now the Twi ASR front-runner: Waxal n=300
> **28.12% greedy / 27.31% with Twi LM**, frozen local holdout8
> **26.67% greedy / 6.67% with Twi LM**. (c) Clean hold-out experiment proves domain data generalizes
> (−11.7pp on unseen clips from 32 training clips) — Whisper v8 approved once
> the corpus scales. **Critical path: corpus scaling (200+ clips, new
> speakers, code-switch priority), not compute.** v6 stays serving.
> Trainer hardened this session: always pass `--train-limit` (capped
> streaming loader) and launch with `modal run --detach`.

Current evidence says **do not hard-pivot DONDO into default serving yet**:

- DONDO v2 should be the Twi beta/A-B candidate.
- v6 / current Whisper-family path remains the stable default until a larger
  held-out product-domain corpus confirms v2 across speakers/noise/code-switch.
- v6 is **hold-and-validate only**, not a competitive final model. Its Hugging Face model card has been backfilled with real WER/CER, datasets, intended use, limitations, and a medical-device disclaimer.
- DONDO zero-shot was much worse on Waxal. v2 fixed that enough to beat v6 on
  Waxal n=300 and on the tiny product-domain holdout, but the holdout size is
  not enough for final promotion.
- English voice uses the separate English route by default, with `MODAL_ASR_EN_URL` available as an override.
  - 100-sample Common Voice English: `openai/whisper-small` WER **11.82%**, v6 WER **42.34%**.
- `modal/train/train_asr.py` now includes Common Voice English in the extra-data mix for balanced v7 runs.
- `modal/train/benchmark_asr_ladder.py` now launches English-retention evals alongside Twi evals unless `--skip-english` is passed.
- Promote only if full Waxal + health-domain + English-retention + phone/noise eval beat the current production candidate.
- DONDO training now has a working capped streaming smoke path:
  - Modal app `ap-Fllvp76L3nlPCLCCaxW1p3`
  - Function call `fc-01KZSTXGGKQTW5K4YXZZT7SPF0`
  - 24 train / 8 eval rows, 8 steps, WER 53.82%, CER 17.81%
  - This proves wiring only; it is not a promotion candidate.

Initial full balanced v7 jobs were stopped because the original loader resolved large dataset shards before training. The capped proof path now streams Waxal and Common Voice extras, then materializes only the requested samples before feature prep.

Validated proof run:

| Run | Modal app | Function call | Result |
| --- | --- | --- | --- |
| `v7-small-waxal-proof-streamed` | `ap-yz2EwC2h9UncQbBvlRxSVI` | `fc-01KZKRSKJGNSRRNKEM667DB9RJ` | Completed 80 steps on 256 train / 64 eval samples; loader fix proven; do not promote |
| `v7-small-balanced-extra-proof-streamed` | `ap-oQ8n9rH6mSDDvS9vSvFjf7` | `fc-01KZKVZ9BZ52HV5W00V67VT5MM` | Confirmed capped Common Voice extra-data path reaches training; no HF push |

Balanced v7 lite jobs launched on Modal:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-lite-no-en-regression` | `ap-H0H15OWS8H7lxtE5WAuqAe` | `fc-01KZKWQEHHX171AHQ6AY15PF1G` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-5vhppGhhweckMY4HZO4XFP` | `fc-01KZKWQEJSEPMCM73NHNDSMTKV` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |

Current lite settings: 1,200 streamed Waxal train samples, 150 streamed Waxal eval samples, capped streamed Common Voice Twi/English extras, 500 steps. Both completed and are **not promotable**:

- `v7-small-balanced-lite-no-en-regression`: Twi Waxal 100-sample beam-5 WER **40.86%**, CER **14.10%**; English Common Voice 100-sample beam-5 WER **15.10%**, CER **8.10%**. English retention improved versus v6, but Twi regressed too much.
- `v7-small-balanced-lite-frozen-no-en-regression`: train validation WER **53.83%**, CER **22.18%**. Twi regressed badly.
- Both HF repos exist and their downloaded README cards include valid dataset metadata, base model, metrics, intended use, and medical-device disclaimer. Local `modal/train/model_card.py` and `pnpm eval:model-card` validate that future card metadata uses Hub-valid dataset IDs.

The earlier 3,000-sample jobs were too slow to materialize.

Stopped earlier lite jobs:

| Run | Modal app | Function call | Reason |
| --- | --- | --- | --- |
| `v7-small-balanced-lite-no-en-regression` | `ap-pZNnbsEBMIeCBLOh3MmiQ1` | `fc-01KZKRHXNWSAJPWWFKT3ND7A6C` | Stuck on old full-shard loader |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-aPtJKFxoavBNMp3Mv4wHcB` | `fc-01KZKRHXGGKF1Y02SGJGH2YXDY` | Stuck on old full-shard loader |
| `v7-small-balanced-lite-no-en-regression` | `ap-fJaR6pMxKDS2p5BwgolZsf` | `fc-01KZKS8FZ26W6GZZ0BEJTS7CTR` | Stuck downloading full Common Voice English audio shards |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-Dqk2wBvPIQwYZg5SKhLywV` | `fc-01KZKS8FZ73KXQNE1A6YB2227G` | Stuck downloading full Common Voice English audio shards |
| `v7-small-balanced-lite-no-en-regression` | `ap-spYSuQHextHp99xtVEr3Ig` | `fc-01KZKWE7EZ1QVPBB2XXRZQNY6M` | 3,000-sample materialization too slow; replaced with 1,200-sample run |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-jObe1dq2JPlDuoh5KsvZ5O` | `fc-01KZKWE7EZPGT2JW090H80W3H3` | 3,000-sample materialization too slow; replaced with 1,200-sample run |

Stopped full balanced v7 jobs:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-no-en-regression` | `ap-HbD5ihi2bRlVZs3yix7y87` | `fc-01KZKR6FZ2DPZVSET4603KJMBS` | `teckedd/gha-whisper-small-twi-en-balanced-v7` |
| `v7-small-balanced-frozen-no-en-regression` | `ap-hxlw7y8LZ7PyM13hVM0x8s` | `fc-01KZKR6FZ35T88N037AKDZEQAS` | `teckedd/gha-whisper-small-twi-en-balanced-v7-frozen` |
| `v7-medium-balanced-no-en-regression` | `ap-Gn8ykjRwLN9Mc410nFdmFZ` | `fc-01KZKR6G12KMBDSW459Q0JDWFY` | `teckedd/gha-whisper-medium-twi-en-balanced-v7` |

After they complete:

```bash
pnpm eval:asr-results:pull
pnpm eval:asr-results
```

The first real DONDO credit-spend trial has completed:

- HF repo: `teckedd/gha-dondo-w2v-bert-twi-v1`
- Base: `KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en`
- Modal app: `ap-F3x5vbrsPLQh13kBUTvcgA`
- Function call: `fc-01KZSWZY0ES2NWC01N9PA8GQP4`
- Final WER: **35.77%**
- Final CER: **12.19%**
- Decision: **do not promote; keep v6 serving for Twi while English uses the separate English route**
- HF card: pushed and verified with base model, dataset, metrics, intended use, limitations, and promotion gate.

DONDO trial history:

| Run | Modal app | Function call | Target HF repo | Status |
| --- | --- | --- | --- | --- |
| `dondo-waxal-twi-v1` | `ap-hGNVgVb8XAYSA0Vxv2zZXF` | `fc-01KZSV66NG7R2R3F5JSKFMK6WP` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Reached step 217, then CUDA OOM after checkpoint-200 eval |
| `dondo-waxal-twi-v1` resume | `ap-nYJdyzEwxZaXp0ubW7KqUi` | `fc-01KZSWMTN06A7NTRV97X6C68PF` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` H100 resume | `ap-SbPCmY2dgsqSD8zEuNyt44` | `fc-01KZSWTJ4WRAT1ZRSK4YNX26DR` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` GPU-fallback resume | `ap-F3x5vbrsPLQh13kBUTvcgA` | `fc-01KZSWZY0ES2NWC01N9PA8GQP4` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Completed 800/800; pushed model + card; WER 35.77%, CER 12.19%; do not promote |

Monitor it without hand-typing Modal commands:

```bash
pnpm train:dondo:monitor
pnpm train:dondo:results
```

Final DONDO result:

- Step 200 eval WER: **41.04%**
- Step 200 eval CER: **13.99%**
- Final eval WER: **35.77%**
- Final eval CER: **12.19%**
- Final train loss: `0.29205229461193083`
- Final eval loss: `0.47330379486083984`
- Baseline WER to beat: `0.3044`
- Promote: **False**

This is a genuine DONDO improvement over zero-shot and smoke, but not a competitive serving model. Do not spend the next credits blindly on more of the same 800-step Waxal-only run. The next model work should be either a targeted data loop, a larger/cleaner DONDO run with a stronger eval design, or a different architecture only if it has a measurable path below the 30% WER plateau.

### 4. Current local verification

Local dev app verified on `http://localhost:3100` because this session's dev server is running on port 3100. Docker Desktop was started and local Postgres is healthy on `localhost:5437`.

Public production check on `https://ghanahealth.serendepify.com`:

- Home page: `200 text/html`
- `/api/health`: `200 application/json`, body reports `ok: true`
- `/api/readiness`: currently `404 text/html`, which means production is still on an older build and has not received the local readiness/pipeline-gate work yet.

After this branch is shipped, run:

```bash
pnpm eval:product-readiness:prod
```

Passed checks:

```bash
pnpm tsc --noEmit
pnpm lint
pnpm eval:health-plan
pnpm eval:understanding:fallback
pnpm eval:conversation:fallback
pnpm eval:commerce-plan
pnpm eval:commerce-execute
EVAL_BASE_URL=http://localhost:3100 pnpm eval:commerce-confirm
pnpm eval:understanding
EVAL_BASE_URL=http://localhost:3100 pnpm eval:live-pipeline
EVAL_BASE_URL=http://localhost:3100 pnpm eval:live-stream
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-stream
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-feedback
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-preview
EVAL_BASE_URL=http://localhost:3100 pnpm eval:product-readiness
pnpm eval:asr-routing
pnpm eval:asr-manifest
pnpm eval:asr-feedback:export
pnpm eval:asr-results:pull
pnpm eval:asr-results
pnpm eval:asr-promotion
pnpm eval:model-card
pnpm eval:hf-model-cards
```

Browser check: desktop and mobile home screens render as the clean light voice-first UI with centered Health/Commerce pills, no market section, no debug text, no console errors.

The local TypeScript eval scripts use `node --import tsx` instead of the TSX CLI so they work under Codex managed sandboxing without an IPC pipe failure.

### 5. TTS finetune + eval harness

See roadmap. TTS train script is still a skeleton.

---

## Key files

| Path | Role |
|------|------|
| `docs/research-stack.md` | Architecture + literature |
| `docs/CONTINUE.md` | This handoff |
| `src/lib/understand.ts` | Twi-first + safety gate |
| `src/lib/health-plan.ts` | Deterministic health action plan from severity + ASR confidence |
| `src/lib/agent-memory.ts` | User/session memory extraction and prompt formatting |
| `src/lib/conversation-turn.ts` | Shared voice/chat turn runner with streamed stages |
| `src/lib/commerce-plan.ts` | Deterministic next-action plan for commerce slots |
| `src/lib/commerce-execute.ts` | Safe local catalog search / order draft execution for commerce plans |
| `src/app/api/commerce/confirm/route.ts` | Explicit confirmed cart-add endpoint for commerce turns |
| `src/app/api/voice/feedback/route.ts` | Captures ASR correction/rating feedback for the data loop |
| `src/lib/twi-retrieve.ts` | ABENA retrieval + passage cache |
| `src/lib/embed.ts` | Embed client + vector b64 helpers |
| `scripts/index-abena-embeddings.ts` | Offline cache warm |
| `modal/embed_service.py` | ABENA Modal service |
| `modal/train/train_understand.py` | GhanaNLP parallel → LoRA |
| `docs/asr-model-decision.md` | ASR evidence, promotion gates, credit-spend order |
| `docs/model-credit-plan.md` | Credit spend order, promotion matrix, product data loop |
| `data/asr-product-eval/README.md` | Product ASR eval manifest and bucket requirements |
| `modal/train/train_dondo_asr.py` | DONDO / w2v-BERT CTC fine-tune path |
| `scripts/eval-understanding.ts` | Runtime understanding regression checks |
| `scripts/benchmark-understanding-llm.ts` | Scores structured LLM meaning extraction on the Twi benchmark |
| `scripts/score-understanding-benchmark.ts` | Scores candidate benchmark artifacts against the project meaning rubric |
| `scripts/export-understanding-training-corpus.ts` | Exports only reviewed understanding rows into train/dev/test manifests |
| `scripts/export-understanding-review-sheet.ts` | Writes a CSV sheet for bulk human review of corpus candidates |
| `scripts/import-understanding-review-sheet.ts` | Imports a corrected review CSV into the local JSONL review fallback |
| `data/understanding-benchmark/rubric.v0.json` | Meaning-preservation rubric for the 50 synthetic benchmark probes |
| `data/understanding-benchmark/scorecard.v0.json` | Current candidate ranking for the benchmark probes |
| `data/understanding-corpus/candidates.v0.jsonl` | 80 draft corpus candidates for human review |
| `scripts/eval-understanding-fallback.ts` | Verifies outage fallback for health, weak ASR, and commerce |
| `scripts/eval-conversation-fallback-contract.ts` | Verifies full conversation turn fallback persists messages |
| `scripts/eval-commerce-execute.ts` | Verifies commerce execution does not mutate cart/checkout |
| `scripts/eval-commerce-confirm-contract.ts` | Verifies confirmation is required before cart mutation |
| `scripts/eval-voice-feedback-contract.ts` | Verifies feedback endpoint persists correction + ASR metadata |
| `scripts/eval-product-readiness.ts` | Verifies shareable runtime readiness checks |
| `scripts/validate-asr-eval-manifest.ts` | Validates ASR product-eval manifest shape and bucket readiness |
| `scripts/export-asr-feedback.ts` | Exports captured transcript corrections to product-eval JSONL |
| `scripts/audit-asr-promotion.ts` | Product-level ASR promotion gate audit |
| `scripts/eval-hf-model-cards.ts` | Verifies pushed HF model cards have base model, datasets, metrics, and limitations |

---

## Explicitly abandoned

- Hand-written `semantic-bank.ts` as product intelligence  
- English knowledge dumps as primary NLU  
- In-domain ASR WER alone as promotion proof  
- Official Ghana tabular CSVs as voice-model training data  

Continue from **§ Remaining work** above.

---

## 2026-09-05 Understanding v4 Research Result

The `silver-medical-paired-v2` corpus is complete and reproducible:

- exactly 7,000 source-paired Ghana Health Symptoms rows
- 5,659 train, 669 development, and 672 test rows
- no duplicate IDs or normalized utterances
- 17 body-system categories
- no WAXAL, GhanaNLP speech, local/product seeds, synthetic prompts, or QA pilots
- CC-BY-NC-4.0, non-commercial research only

The open-model agentic annotation trial is also resolved. AfriqueQwen 9B,
NLLB Twi-English, and a Qwen 7B adjudicator produced three inspectable options,
but only 3/20 rows cleared the conservative semantic gate. The models missed
material distinctions such as unspecified medicine versus herbal treatment.
Those proposals remain audit evidence and were not promoted into the 7,000-row
training corpus.

Understanding v4 completed as a 1,200-step LoRA on
`Qwen/Qwen2.5-3B-Instruct`:

- Modal training app: `ap-1nIwJgPPwgMaQbsmjlPtJK`
- Hugging Face: `teckedd/gha-understand-twi-medical-v4`
- final training loss: `0.7775883994499843`
- final development loss: `0.6560265421867371`

Held-out result on all 672 frozen test rows:

- parseable JSON: 672/672
- exact intent: 672/672, but all rows share `health_symptom_report`, so this is
  not evidence of intent generalization
- exact body system: 259/672 (38.54%)
- strict semantic pass: 79/672 (11.76%)
- mean natural-English token F1: 0.3425
- Modal evaluation app: `ap-VBCFWCEnygrgqJbgxzdYma`

Corrected product fixture result:

- overall: 1/11
- health: 1/6
- commerce: 0/5
- Modal evaluation app: `ap-F2cPX9VDMcSyGdMtreG4GU`

Unadapted `Qwen/Qwen2.5-3B-Instruct` comparison:

- product fixtures: 0/11
- held-out parseable JSON: 278/672 (41.37%)
- held-out exact intent/body system/strict pass: 0/672
- held-out mean natural-English token F1: 0.0295
- Modal product app: `ap-p8bh0KAnUk0kWoF0PWDqgm`
- Modal held-out app: `ap-SfY0cEY1VOLbvqTwJfGWXy`

This proves that v4 learned measurable medical-domain structure and semantics,
but the absolute result is still far below a product threshold. The base also
preserved some commerce meanings that v4 converted into symptoms, confirming
that the medical-only fine-tune narrowed the model's behavior.

**Promotion decision: do not promote v4 and do not route app traffic to it.**
It learned the JSON shape and the repeated health label, but it hallucinated
symptom templates instead of preserving meaning. Training loss was not a useful
proxy for product semantics.

Next understanding experiment:

1. Rebuild the target distribution around faithful translation and entity
   extraction, with multiple health and commerce intents instead of one label.
2. Add deduplicated product-failure paraphrases and multi-turn context to train,
   while keeping the frozen product fixtures out of training.
3. Build the response-capable lane separately from licensed medical QA and
   reviewed Twi answers. The symptom corpus contains no patient-facing answers.
4. Require semantic, product, safety, English-regression, latency, and base-model
   delta gates before another model is exposed in the app.

ASR and TTS remain separate tracks. This result does not improve transcription
or the current Twi voice.

---

## 2026-08-29 Understanding Research Status

Current best draft-understanding candidate:

| Candidate | Meaning score | Exact cases | Notes |
| --- | ---: | ---: | --- |
| `openai:gpt-5.6-sol` | 100.0% | 50/50 | Best current draft annotator and product understanding candidate after product-critical Twi health guards. |
| `openai:gpt-5.4-mini` | 94.2% | 42/50 | Fallback candidate: faster and still strong, but not top-ranked. |
| `openai:gpt-5.5` | 94.2% | 45/50 | Rejected for now: did not beat guarded `gpt-5.6-sol`. |
| `ninte/twi-en-nllb-v2` | 77.5% | 30/50 | Fast translation baseline only; unsafe alone for health meaning. |
| `facebook/nllb-200-distilled-600M` + `mclanorjeff/NLLB-Twi-Human-Aligned` | 76.1% | 28/50 | Adapter path works on Modal but did not beat v2 on this rubric. |

New commands:

```bash
pnpm eval:understanding:model:human-aligned
pnpm eval:understanding:llm -- --model gpt-5.6-sol
pnpm eval:understanding:score
pnpm corpus:understanding:export
pnpm corpus:understanding:export:strict
```

Corpus state:

- `80` candidate rows exist.
- `80` have draft annotations.
- `30` reference local audio artifacts.
- `0` saved human reviews exist in the local review file.
- `0` rows are currently training-eligible.
- Review decisions now persist in Postgres through
  `research_understanding_reviews`; the JSONL review file is only a local
  fallback.

The research workbench now exposes the benchmark scorecard in `/research/ase`.
It also shows how many reviewed rows are training-ready, the current
train/dev/test counts, and a direct link to
`/api/research/understanding/export`.

The corpus is ready for review, not training. The next real step is to review
rows in the workbench and run `pnpm corpus:understanding:export:strict`; only
that export should feed training. The export endpoint and strict command must
remain unready until a human has corrected and accepted enough corpus rows to
pass the readiness gate: at least 20 reviewed rows, train/dev/test coverage,
health-domain coverage, no duplicate meaning keys, and consent scope on every
row. Commerce-domain coverage is visible as a warning so the product track can
follow without blocking the first health-focused run.

For faster offline review, use the workbench **Download 20-row training pack**
action or run
`pnpm corpus:understanding:review-sheet -- --scope minimum-training --out tmp/understanding-corpus/minimum-training-review.v0.csv`.
Use **Download assisted pack** or add `--prefill draft` to prefill review
columns from model drafts without approving them. Rows still import for
training only when `decision` is changed to `reviewed`; untouched `unreviewed`
rows are skipped. That pack is selected to cover row count, train/dev/test,
health, and commerce before the remaining queue. Fill the `review_*`,
`decision`, `review_notes`, and `reviewer` columns, and preserve
`proposed_split`. Then upload the corrected CSV through the workbench **Upload
reviewed CSV** action. Production uploads persist to Postgres and immediately
update the export readiness gate. For local fallback work, import with
`pnpm corpus:understanding:import-review-sheet -- --input <sheet.csv>`, then run
the strict export gate.
