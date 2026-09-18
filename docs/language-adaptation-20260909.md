# General Language Adaptation, 2026-09-09

Stage: bounded experiment, not a trained-model success or production release.
The objective is a general Twi-English assistant, not a translation-only product.
This first stage addresses the measured language gap in the stronger foundation;
it does not replace the later response, health, commerce and tool curriculum.

## Why This Run

The 95-source MiniCPM pilot did not improve Twi semantics. More epochs on that
small medical set are not justified. Gemma 4 31B handles substantially more
general questions, corrections and tool requests, but still misreads Twi health
phrases. This experiment tests whether broader parallel supervision can improve
that foundation while preserving English instruction following. The comparison
must use the actual adapter and raw outputs, not a hidden proprietary responder.

Base: `google/gemma-4-31B-it`, pinned revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`.

## Public Data

`scripts/audit_parallel_sources.py` downloads pinned public artifacts locally,
keeps their cards and hashes, and records deterministic samples. No owner audio,
private reviews, production conversations or application secrets are inputs.

- Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT, revision
  `5f59d16167c9432a6fa0dac5e7e6a7e48e161bdb`: actual CSV has 6,090 rows,
  not the 14,875 advertised in parts of its card. Broad sentence pairs; the card
  describes professional translation. This is not project-native-verified gold.
- ghananlpcommunity/twi-english-parallel-text, revision
  `b9ae4bc01ff2324ca6d73d1f86b38bb4f7f224cf`: 124,445 rows, mostly dictionary
  entries and Bible verses, not 124k everyday conversations. Only language-guide,
  sample-phrase and census-glossary sources are eligible in this first stage.
- OpenAssistant/oasst1, revision
  `fdf72ae0827c1cda404aff25b6603abec9e3399b`: reviewed, rank-zero, nonsynthetic
  English assistant paths. Keep official conversation-tree splits; reject flagged
  privacy, spam, inappropriate-content and language-mismatch rows, low-quality
  targets, high toxicity scores and oversized conversations. One path per tree.

The community medical glossary is explicitly quarantined. Inspection found
symptom-to-disease conflation and contradictory translation variants. A published
dataset label is not evidence that an entry is safe supervision. Old dictionaries,
Bible material and synthetic dictionary-generated mega-corpora are not added to
inflate the count. No source transcript is rewritten or assigned a made-up reply.

Source terms permit private noncommercial research with attribution; the first
GhanaNLP card has inconsistent license metadata. Preserve its stated research,
noncommercial and share-alike restrictions. No commercial or public distribution
clearance is asserted. OpenAssistant is Apache-2.0. Full source cards are retained.

## Filtering And Splits

`scripts/build_language_adaptation.py` preserves original pairs in a separate
artifact. It quarantines empty/identical pairs, encoding/markup problems, extreme
length ratios, dictionary notation and numeric mismatches. Digit-to-word pairs
may be legitimate but stay quarantined rather than being automatically repaired.

Shared English, shared Twi and adjacent GhanaNLP source-ID blocks form connected
groups before splitting. This keeps duplicate translation variants together.
True document IDs are unavailable, so complete document independence is not
claimed. Existing locked evaluation texts, product checks and foundation checks
are excluded by normalized exact matching and eight-word overlap. Common task
instructions are not included in overlap matching. Near-duplicate detection is
not a claim of perfect semantic deduplication.

Final pre-tokenization export:

| Split | Distinct sources | Examples | Composition |
| --- | ---: | ---: | --- |
| Train | 6,980 | 13,008 | 6,028 pairs in both directions + 952 English conversations |
| Validation | 704 | 1,281 | 577 pairs in both directions + 127 English conversations |

Two translation directions are two views of one source, not extra independent
records. Seven training conversations exceed the 896-token budget and are excluded
whole, without truncating their targets. GPU input is therefore **13,001 examples
from 6,973 sources**. This is separate from the existing 12,858-row annotation
program; it does not mean that WAXAL/GhanaNLP/AfriHealth annotation is complete.

Supervised tokens: 186,387 Twi targets, 84,702 English translation targets and
194,349 English conversational targets. About 40% Twi and 60% English. A first
preflight revealed that longer English responses dominated the larger replay
sample; replay was reduced by source selection, not by cutting answers off.

Artifacts in `tmp/general-language-corpus/adaptation-v1/`:

- `train.jsonl`: `d8ae3b9e93af15a7988b9a1cfe240d8ff16c94f1077f67051df5bfbdce24029c`
- `validation.jsonl`: `93b45221c94e1a88b57f289bab959b4967bed0ba77144471d443edc92a0c4c6f`
- `source_pairs.jsonl`, `excluded.jsonl`, `manifest.json`: original records,
  quarantine reasons, source revisions, counts and limitations.
- Final accepted training-ID digest after tokenization:
  `cda2326a4e0560cfe416dd7a652ec46d9cb7ae08a1e46d74f7921a8e1d5403bd`.

## Training And Evaluation

Runner: `modal/train/train_language_adaptation.py`.
Core: `modal/train/language_adaptation_core.py`.
One H100, two-hour function ceiling, one-pass maximum, rank-16 LoRA,
alpha 32, learning rate 2e-5, effective batch 16, BF16 and gradient checkpointing.
Only the final assistant target and its correct Gemma turn-ending token receive
loss. No fabricated thought traces. Whole overlength examples are excluded.
Periodic adapter checkpoints and actual metrics remain private on the existing
`ghana-health-understand-train` volume. Public base cache is reused separately.
No raw private corpus archive, HF upload or production replacement is performed.

Planned before and after: held-out losses for both translation directions and English
conversations, 32 held-out translation generations, and the same 30 development
product/general checks. Correct API calls and final responses are inspected as
raw output; tool execution is not implied by a valid call string. Similarity
scores and loss are diagnostics, not native understanding or clinical accuracy.

Initial CPU checks passed tokenization. First GPU attempt
`ap-A2UtN1aMnXANREXNNGn3Kq` failed before training: the full multimodal checkpoint
requires explicit text-weight name mapping; a loading-report serialization issue
also occurred. Second attempt `ap-OErBQabb84j0Ud9W1lo1a7` loaded the text weights
after that fix but stopped on the newer Trainer API's removal of `warmup_ratio`.
No optimizer steps or trained adapter resulted from either attempt. The requested
stop of the second attempt found it already stopped. It was not left running.

The runner now uses `warmup_steps` and performs a tiny real CPU optimizer/backward
test in preflight. Final CPU check `ap-dV98jIg7pe6yQ5TxkfW2W6` passed with the
revised corpus. That random tiny-model check verifies runtime compatibility only;
its loss is NOT a result for Gemma 31B or Twi.

Do not call this experiment successful until its real before/after results are
read. Do not promote it merely because it trained or because a few outputs look
better than the rejected pilot. Preserve all private playground conversations.

Third attempt `ap-iaFpsE7ITLbkhPIdd1BXzp`, run folder
`gemma31_language_v1_20260909T215322Z`, loaded all text weights correctly and
measured baseline losses: Twi translation 4.10005, English translation 4.99947,
English replay 2.76115 (32 held-out examples each, not accuracy scores). It reached
60/62 baseline generation checks, then received a cancellation signal at
22:00:39 UTC, before any optimizer step. The cause is not yet known; do not
silently restart an intentionally cancelled job. No trained adapter exists from
this run. The system-log query is being inspected, and the user was asked whether
they stopped it. All three attempts are now stopped, not spending GPU time.

The runner is improved for resumption: adapter-only checkpoints every 50 steps,
partial generation results every eight examples, and generation comparisons after
the first completed training checkpoint. Frozen-base generation with LoRA disabled
is an unchanged baseline; teacher-forced baseline losses still run before SFT.
At this checkpoint the revised order had not yet run on the GPU. The subsequent
owner-approved restart and its unsuccessful outcome are recorded below.

Read-only Modal app listing confirmed zero tasks for this run. The system-log
query returned no additional events explaining the cancellation. Browser console
access requires login; no new login or permission grant was performed. The user
confirmation question was then unanswered; the owner later confirmed they did not
cancel it. Do not conflate this unknown cause with
the earlier explicit stop request for the second, already-failed attempt.

Local verification: ten data/training contract tests, six TTS text/routing tests,
and ten existing benchmark/playground/pilot tests pass (26 total). TypeScript
type checking and TTS routing checks pass. Existing SQLite ResourceWarnings are
still present. No production deployment, HF publication, or new UI/model choice.

## Restart After Owner Confirmation

The owner explicitly answered that they did not stop the cancelled job. The
corrected runner was restarted as `ap-tEBjptzAf9vbAX2EtNJ6yk`, run folder
`gemma31_language_v1_20260909T222324Z`. All text weights loaded successfully,
the baseline losses reproduced, and actual training reached 281 optimizer updates.
The agent deliberately stopped it at 23:03 UTC after the paired conversational
diagnostic showed regressions. This is not a completed 813-step pass. Checkpoints
200 and 250 are preserved on the training volume; earlier checkpoints were removed
by the two-checkpoint retention policy. There is no final completed-run adapter,
full-generation evaluation or success card from this attempt.

## Early Results And Stop Decision

`modal/train/evaluate_language_checkpoint.py` reads checkpoints without changing
training. It checks corpus provenance and adapter hashes, computes losses on the
same 32 examples per task, and saves every raw generation immediately on this
device. `scripts/summarize_language_checkpoint.py` verifies paired IDs/prompts and
reports repetition and incomplete output separately, without semantic scores.

| Validation task | Frozen base | Step 100 | Step 200 |
| --- | ---: | ---: | ---: |
| English to Twi | 4.10005 | 1.93214 | 1.63792 |
| Twi to English | 4.99947 | 1.86454 | 1.66136 |
| English conversation replay | 2.76115 | 1.39213 | 1.31811 |

These losses do NOT establish response quality. Step 100 produced five obvious
repetition loops. Step 200 reduced that to one, but several direct Twi responses
became short repetitions/normalizations of the user instead of answers. The
budgeting response lost actionable steps. Breathing, pregnancy and baby-context
answers remained semantically unreliable. The unchanged base also has serious
Twi health failures; it is not a clinically accepted alternative.

The step-200 comparison used identical prompts, BF16 Transformers runtime,
greedy decoding, batch size two and a 384-token limit for base and adapter.
The base's long budget answer reached the length limit without a repetition
loop; the adapter's hospital-choice answer looped to that limit. Do not equate
these failure types. An isolated eye/sleepiness phrase has an ambiguous rubric;
do not claim a confident eye-pain gold label without contextual/native review.
Negation, arithmetic, Kofi's future trip, corrected one-kilogram quantity and
the two read-only tool calls were retained. Valid call strings are not execution.

Runs and local artifacts:

- Step 100: `ap-T7a9yNZrjH1aed0q2jfQAx`,
  `tmp/general-language-corpus/gemma31_language_v1_20260909T222324Z-checkpoint-100-diagnostic.jsonl`.
- Paired step 200: `ap-BLq4808VlC1EGgUEcjMIHx`, same folder,
  `gemma31_language_v1_20260909T222324Z-checkpoint-200-paired.jsonl` and
  `checkpoint-200-paired-summary.json`.
- Step-200 adapter SHA256:
  `b17e6bf8d2b2d28c6f2959636462909c480e16e10a221df7658928fb618524aa`.

The bounded quarter-strength ablation `ap-qeK7qtRiFTRfox9jHvanj0` completed at
23:17 UTC, with 30 base and 30 adapter replies. It used the same checkpoint
tensors with LoRA alpha 8 instead of 32. This was inference-only scaling, not
retraining. Validation losses were 2.89741 for English-to-Twi, 3.25900 for
Twi-to-English and 2.05965 for English conversation replay.

Neither variant triggered the repeated-eight-word-span flag; both budget answers
reached the 384-token limit without a loop. Lower adapter strength recovered
conversational structure, but did not resolve Twi semantics. The hospital-choice
reply directs the user to "asɔre" (church); the pregnancy reply repeats the
same wrong place with "(hospital)" beside it. Baby-context replies remain
unreliable and the code-switched child-fever case switches to English. Preserved
arithmetic, quantities and tool-call syntax do not offset these failures.
Decision: reject this scaled variant for product promotion too. These development
checks cannot become a held-out success claim after tuning on them.

Artifacts in `tmp/general-language-corpus/`:

- `gemma31_language_v1_20260909T222324Z-checkpoint-200-paired-scale025.jsonl`
  (SHA256 `dd78d84821ad04397c5d56b321b3e35516bcc2e36205fd752bb85b68d57043b1`).
- `checkpoint-200-scale025-summary.json`: paired raw answers and mechanical flags,
  not a fabricated semantic accuracy score.
- `early-stop-decision.json`: training stop decision, also preserved beside the
  private checkpoints as `operator-stop.json` on the existing training volume.

All training and diagnostic apps have zero running tasks. No more training was
started after this comparison, and no checkpoint or voice was promoted.

The observed failure is consistent with translation supervision displacing
conversational behavior; this is a hypothesis, not a proven causal attribution.
Next training requires audited direct Twi conversational targets plus English
replay and a smaller auxiliary translation share. Do not simply run more epochs
or fill targets with unverified fluent-looking medical text.
