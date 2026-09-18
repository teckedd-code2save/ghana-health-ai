# Response Research: Greeting Failure And Corpus Audit

## Active Scope After Shared Research Review

The user explicitly wants understanding/response LLM research only. Do not start
ASR work from the shared weekly shortlist. Borrow experimental principles,
not speech-specific architecture or claims of transferable WER gains.

The language-model work cited by BuzzASR is directly relevant:
[monolingual language modeling](https://aclanthology.org/2026.lrec-1.300/)
finds that language-specific models can improve grammatical modeling relative
to larger multilingual models. Its
[preprint](https://arxiv.org/abs/2408.10441) also reports weaker reasoning than
larger multilingual models. This supports separating language exposure from
reasoning/assistant evaluation; it does not justify replacing our capable
foundation with a tiny monolingual model or promise a Twi result.

Our application of that principle is a controlled comparison: untouched
instruction foundation; conversation SFT only; screened Twi continued
pretraining plus the identical conversation SFT. Protect English and preserve
reasoning capability. Evaluate both language-adaptation and final checkpoints.
Measure Twi fluency, meaning/negation/entities, multi-turn corrections, English,
code-switching and executed tool behavior separately. If gains cost English
retention or instruction following, do not promote the checkpoint.

Raw text needs no invented intent/answer labels. Direct assistant supervision
does need correct answers and real turn structure. Keep synthetic Pristine
separate from source-backed text in ablations, and do not use the failed
dialogue translators to mass-produce the missing supervision. The absence of
multi-turn Twi final targets remains a measured gap, not proof that fixing
that count alone will fix model quality. A tokenizer change is a later
controlled option only if measured fragmentation and a small adaptation test
justify the compatibility and retention costs.

## What Failed

The user tested `wo ho te s3n` on the privately enabled V6. It invented a
male human identity and age 25, then repeated until 512 tokens. That is a
basic assistant failure, not an ASR/TTS issue. The local turn is
`84cbf7d5-46d9-4d31-904f-d103f701aaa5`; the real inference call is
`fc-01M275V96KTEWRPCT9HP21MA48`. Earlier Pilot v1 replies were present in
the retained conversation. Do not blame the user or dismiss the failure as
history contamination. No original private conversation was resubmitted.

Four separate development cases compared the same frozen Afrique 9B with
and without V6, using identical inputs, system/tool contract and publisher
sampling, seed 42. Both informal and standard Twi, English, and a short
synthetic Hello/Hello history were tested. The base hits the output limit
on all four; V6 ends normally on all four. However, V6 answers English in
Twi and produces awkward Twi greeting replies. Normal termination is not
an acceptance result. These are not unseen benchmark scores.

Evidence: `data/response-adaptation/greeting-v6-decision.json` and the
raw result paths/checksums it records. No hidden fallback, hardcoded
greeting, changed decoding policy, voice change or production promotion.
Fixtures are development-only, not future training examples.

## Actual Supervision

The reproducible coverage audit validates the training/manifest checksums
against the original tokenization evidence before counting. It found:

- 8,681 prepared rows, 509,384 assistant-target tokens.
- Only 422 Twi-labelled response-task rows, 54,361 target tokens (10.67%).
- Health contributes 52,327 of those tokens (96.26%). The remaining 2,034
  are short AfriQA answers and 17 source-context answers.
- Zero multi-turn rows with a Twi-labelled final assistant target.
- The 2,332 multi-turn views end in English; earlier Twi context does not
  turn an English follow-up target into Twi response supervision.
- Translation views and intent/entity JSON are separate auxiliary tasks,
  not fluent Twi conversation. Labels are not language verification.

This is a concrete coverage gap, not proof of one exclusive causal mechanism.
Counts describe the prepared pool; the 400-update run used less than one
epoch. Do not present prepared totals as exact tokens seen by that run.

```sh
python3 scripts/audit_response_coverage.py tmp/balanced-afrique-v6/corpus --preflight tmp/balanced-afrique-v6/preflight.json --output NEW_OUTPUT.json
```

Saved result: `data/response-adaptation/balanced-v6-coverage.json`.

## Full Text Audit

The CPU audit measures original Twi independently from English alignment.
It reads all cached GhanaNLP/community parallel rows, the existing public
speech-text manifest, and all four pinned original Pristine Twi shards.
It does not count Pristine's English derivative as another independent corpus.
No recordings, real chats, proprietary annotation or GPU training are inputs.

Runner: `scripts/audit_twi_pretraining.py`; source registry:
`data/response-adaptation/twi-text-audit-sources.v1.json`.
Completed output: `tmp/twi-pretraining-audit/v1/summary.json`. Never overwrite
this directory to retry. Counts and output/sample checksums were verified:

| Source | Audited rows | Exact-unique structural candidates |
| --- | ---: | ---: |
| GhanaNLP original parallel | 6,090 | 5,524 |
| Community compiled parallel | 124,445 | 43,514 |
| Existing GhanaNLP speech manifest | 2,498 | 1,189 |
| Existing WAXAL speech manifest | 2,450 | 1,793 |
| Original synthetic Pristine Twi | 999,498 | 998,313 |
| Total | 1,134,981 | 1,050,333 |

Structural-candidate token totals are 855,600,539 with the pinned Qwen tokenizer
and 642,900,829 with the pinned Gemma tokenizer, without chat wrappers/EOS.
Pristine contributes 853,085,092 of the Qwen tokens, more than 99% of the pool.
Shorter tokenization is not evidence of a better language model. The 705
deterministic review samples are inspection aids, not gold labels.

Audit code snapshot: `tmp/twi-pretraining-audit/v1/audit-code.tar.gz`, SHA256
`4697c578d1a7b6cc353e29e440213885cb3a99ca1bbe839726fbe5d6911de916`.

Outputs preserve original source hashes and row positions, normalized exact
duplicate links, structural flags, protected-evaluation matches, source/style
counts, native Qwen/Gemma token counts, and deterministic review samples.
All rows retain `training_eligible=false`: this is a measurement artifact,
not a silently accepted training export. Bad English alignment does not
automatically discard the independently useful Twi text. Short fragments
remain in the review ledger; they may matter for dialogue even when unsuitable
as long-form pretraining documents.

Pristine's own card says **the Twi is Gemini-generated**, across four styles
based on news. The linked generator currently validates character length and
removes repetition, not independent semantic correctness. Its model/version
may have changed since the pinned dataset; do not infer exact generation
provenance from today's script. The exported schema lacks original article
IDs and speaker roles. Do not automatically parse its Dialogue label into
user/assistant gold or split four style variants as independent topics.
An initial eight-row preview also mixes unrelated news topics; it is not
a corpus-wide defect-rate estimate.

Sources: [dataset card](https://huggingface.co/datasets/ghananlpcommunity/pristine-twi),
[linked generator](https://github.com/GhanaNLP/NLP-scripts/blob/main/text/generate_twi-gemini.py).
Synthetic origin does not automatically make every row unusable, but its
contribution must remain separate in quality checks and training ablations.

Important limitations: exact deduplication is not near-duplicate/topic grouping;
speech coverage is the existing manifest, not the whole upstream audio corpus.
The newly added greeting fixtures postdate this audit's protected-set snapshot.
The shared language-corpus exclusion code now protects their final queries on
future runs; this finished audit was not rewritten to claim it already did so.
No full linguistic or clinical validation is implied.

### Full Published Speech Text

Separately completed all labelled WAXAL Akan ASR shards and all three GhanaNLP
16k speech-text shards, using Parquet column projection rather than full audio
downloads. Corrected metadata projection:
`tmp/twi-pretraining-sources/full-public-speech-v2/sources.jsonl` and `summary.json`.
28,312 unchanged transcripts: WAXAL train 10,107, validation 1,123, test 1,522;
GhanaNLP train 15,560. WAXAL revision
`5f4d8ca24f2b9d168b2ee545f1febaaff4b40580`; GhanaNLP revision
`410ef3e8ed3d16829247339e7f4fc75b519271a8`.

The initial v1 projection assigned the project's tw label too broadly. V2
corrects only the project language metadata on 12,752 WAXAL rows to ak (Akan);
it does not rewrite their source text, source language, split or speaker IDs.
V1 and its code archive remain for provenance. The extractor is corrected for
future runs. Akan scope is not independent Twi dialect verification; inspection
also found letter-like symbols in some original transcriptions. Do not silently
normalize or certify these. All source text hashes were preserved/verified.

These 28,312 rows overlap the older 4,948-row manifest in the main audit. They
must replace/union it with deduplication, not be added as independent sources.
The full projection still needs text quality, dialect, source/speaker grouping,
and all local plus upstream holdout exclusions before training selection. It is
not automatically QA, newly annotated data or a training-ready dialogue corpus.

## Corrective Experiments

Do not rerun the unchanged V6 mixture or treat additional random epochs as
the solution. Repair the missing training objective and measure the change:

1. Keep raw language exposure, parallel alignment, assistant dialogue and
   executable tool trajectories as separate corpora with separate counts.
   Deduplicate/group by source before splitting and budget by tokens.
2. Build broader Twi response supervision: ordinary conversation, questions,
   corrections, uncertainty, language switches, short answers and multi-turn
   continuations. Preserve original source passages/answers and attribution.
   Use grounded open-model synthesis plus calibrated filtering and targeted
   native review; unverified model agreement is not correctness.
3. Compare SFT-only with additional Twi-language pretraining followed by the
   **same** SFT, from the same capable instruction foundation. Include the
   untouched instruction model and V6 as controls. English replay and English
   retention checks are required. Base, training exposure and mixture must
   not all change at once as they did between V5 and V6.
4. Within the language-data comparison, separate the compiled/speech text
   contribution from synthetic Pristine. More synthetic tokens might worsen
   biography drift, verbosity or coherence; this is an empirical question.
5. Evaluate actual model-generated multi-turn histories, common conversational
   acts, Twi/English/code-switching, entities/negation, refusal/uncertainty and
   grounded tool execution. Health correctness is a separate acceptance gate.
   Seen regressions remain development tests, not new unseen accuracy claims.

No new training run is submitted in this audit. Decide token budgets and
compute from the completed inventory and measured throughput, not row count.
There is no requirement to label every raw Twi pretraining paragraph with
intent, meaning and an answer; such labels belong to supervised task data.
OpenAI funding is not a prerequisite for the proposed Modal workflow.

## Private Audition State

At the user's request V6 was enabled in the existing local playground at
`http://127.0.0.1:7863/?__theme=light`. The explicit private-selection file
pins run `afrique_v6_20260910T222142Z`, checkpoint `adapter`, SHA
`b4aa2162ae27a4c6997ec9f88bc171ca2134c457922bb1483c460eff920be510`.
Inference runs on private Modal compute, not the Mac. Voice A, saved history,
reviews and older model choices were preserved. This is a user-requested
audition of a rejected candidate, not a release or a repaired model.

Verification: 36 focused Python tests passed across runtime (11), playground
(5), greeting diagnostics (2), response coverage (3), text audit (5), public
speech projection (3) and shared language exclusions (7). Runtime tests use the
actual Transformers 5.17.0 dependency set. Whitespace checks passed. This is not
a full web build, a successful new training run or human language certification.
