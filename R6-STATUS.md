# R6: Training and Corrected Benchmark Status

Updated 17 November 2026. R6 is a completed private research checkpoint, not a
production promotion. The corrected base/R5/R6 benchmark is **complete**: 2,004
questions per checkpoint and 6,012 saved predictions independently rescored
locally. Historical results remain unchanged. [Try R6 against R5](http://127.0.0.1:7870/?__theme=light).

## What Was Trained

R6 continued the final R5 adapter on the pinned
`McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` foundation, revision
`693f12e444a30971ca53889128d7bade9491ecb3`.

| Training quantity | Verified count | Meaning |
| --- | ---: | --- |
| Registered collection records | 6,290 | 997 response, 1,395 English-retention, 2,711 meaning, 1,187 language records; collection membership is not a unique-source count |
| Source identities | 5,293 | Distinct source IDs in the prepared pack |
| Prepared training views | 8,373 | 2,038 response, 2,437 English-retention, 2,711 meaning, 1,187 language views |
| Distinct consumed views | 7,902 | Views actually visited in the completed run |
| Presentations | 9,940 | Counts repeat exposure, not new data |
| Optimizer updates | 622 | Completed updates |
| Supervised target tokens | 1,845,073 | Tokens contributing to the training loss |

Evidence: `understanding_20261113T152209Z`, its training terminal receipt and
`pack-v1/manifest.json`. The data version is
`r6-relaxed-review-v1-20261113`; its admission flag is not independent native
review or clinical certification. No new training is part of this correction.

## Why The Benchmark Was Corrected

1. AfriMMLU stores `choices` as a serialized list. The old runner iterated its
   characters, presenting malformed choices rather than the actual answers.
2. The old scorer compared A-E for every question, including three- and
   four-choice questions. The old prompt builder also omitted a fifth choice.
3. ARC's numeric source labels needed mapping to the displayed letters.
4. SIB-200 accepted any output containing the gold label, even when the output
   listed multiple labels. The corrected metric requires one explicit category.
5. The first 300 AfriMMLU rows contain 100 elementary mathematics, 100 geography
   and 100 international-law questions. Their combined score is not a maths score.
6. Non-thinking mode must be selected in the chat template. Setting an arbitrary
   generation-config attribute, as the old SIB path did, did not reliably select
   that template mode. Every corrected task now uses an explicit matched prefix.

Consequently, the earlier conclusion that Twi maths was at chance is withdrawn:
the corrected mathematics counts are 28/100 for base, 37/100 for R5 and 36/100
for R6. The historical SIB gain of 4.9 percentage points is superseded by the
corrected matched gain of 2.94 points. The claim of no English regression is also
withdrawn: all five English subset scores are slightly lower for R6 than R5.
Old-to-new changes are measurement changes on the same checkpoints, not additional
learning. Only comparisons within the corrected run estimate checkpoint differences.

## Corrected Protocol

- All three models receive the same frozen source records, in the same order.
- Dataset revisions, source hashes, prompts, answer labels, code and adapter
  checksums are recorded before inference.
- MCQ scores use the summed conditional log-probability of each **presented**
  answer letter, including its complete token sequence. No answer is extracted
  from generated prose. Invalid source answers fail preparation rather than
  disappearing from the denominator.
- SIB uses greedy generation and strict single-category parsing. Raw output and
  the legacy substring score are retained separately for diagnosis.
- Thinking is explicitly disabled in the chat template for every checkpoint.
- Per-question evidence is checkpointed on Modal. Restarts reuse saved results
  only under the same frozen manifest.
- The sample matches the earlier scope: first 300 source rows per task and all
  204 SIB test rows. This is a custom zero-shot subset evaluation, not an official
  leaderboard result. HellaSwag uses a choice-letter prompt, not its standard
  continuation-likelihood protocol. PubMedQA uses its labeled training partition
  as an evaluation source; training-set overlap has not been comprehensively
  certified. Scores do not establish clinical safety or conversational competence.

## Historical Results Preserved

These are the original aggregate files, with the known scoring limitations.

| Task | Base | R5 | R6 |
| --- | ---: | ---: | ---: |
| Twi AfriMMLU, malformed choices / 300 | 39 | 75 | 74 |
| Twi SIB-200, substring scoring / 204 | 123 | 156 | 166 |
| MMLU / 300 | 148 | 191 | 196 |
| HellaSwag / 300 | 139 | 152 | 160 |
| ARC-Challenge / 300 | 228 | 266 | 266 |
| MedQA / 300 | 88 | 168 | 170 |
| PubMedQA / 300 | 164 | 167 | 165 |

The old files remain at `understanding-adaptation-r6/benchmark-results/` on
Modal volume `ghana-health-understand-train`. A byte-identical copy and checksums
are also included under the corrected run's `legacy/` directory.

## Corrected Results

Counts are correct answers / questions, followed by accuracy. Differences are
**percentage points**, comparing R6 with R5, not relative percentage increases.
Machine-readable [verified summary](docs/research/r6-benchmark-v2-summary.json)
includes dataset revisions, adapter identities, subject breakdowns and paired
gains/losses. The original protocol did not pin dataset revisions, so the new run
matches its task/sample scope without claiming byte-identical old source inputs.

| Task | Base | R5 | R6 | R6 - R5 |
| --- | ---: | ---: | ---: | ---: |
| AfriMMLU Twi, three subjects | 79/300 (26.33%) | 112/300 (37.33%) | 120/300 (40.00%) | +2.67 |
| SIB-200 Twi, topic classification | 154/204 (75.49%) | 160/204 (78.43%) | 166/204 (81.37%) | +2.94 |
| MMLU English, three subjects | 164/300 (54.67%) | 201/300 (67.00%) | 197/300 (65.67%) | -1.33 |
| HellaSwag English, choice-letter protocol | 179/300 (59.67%) | 200/300 (66.67%) | 199/300 (66.33%) | -0.33 |
| ARC-Challenge English | 265/300 (88.33%) | 280/300 (93.33%) | 279/300 (93.00%) | -0.33 |
| MedQA English | 118/300 (39.33%) | 198/300 (66.00%) | 193/300 (64.33%) | -1.67 |
| PubMedQA English | 187/300 (62.33%) | 175/300 (58.33%) | 174/300 (58.00%) | -0.33 |

### What Improved, and What Did Not

- **Observed Twi gains:** R6 answers eight more AfriMMLU questions and six more
  SIB questions correctly than R5. Relative to base, the gains are 41/300 and
  12/204 respectively. These are task-specific measurements, not proof of broad
  conversational understanding.
- **Meaning of the eight-question AfriMMLU gain:** mathematics changes from
  37 to 36, geography from 41 to 42, and international law from 34 to 42, each
  out of 100. The net gain is driven by the law subset, not arithmetic.
- **English tradeoff:** R6 loses between one and five correct answers per task
  against R5. Four English scores remain above base; PubMedQA is below base for
  both adapters. We cannot claim that English retention is unchanged.
- **Uncertainty:** paired R6/R5 results are 24 gains / 16 losses on AfriMMLU and
  7 gains / 1 loss on SIB. Exact paired two-sided p-values are 0.268 and 0.070,
  unadjusted for multiple comparisons. Neither reaches 0.05. Source-order
  samples and unverified training overlap further limit generalisation.
- **Scope:** MMLU's 300 rows cover 100 abstract-algebra, 135 anatomy and 65
  astronomy questions; they do not represent all MMLU subjects. SIB measures
  topic selection, not generation quality. No new evidence of reduced
  repetition, safer medical advice or improved multi-turn usefulness is claimed.

| Paired R6 vs R5 | R5 wrong, R6 correct | R5 correct, R6 wrong | Same correctness |
| --- | ---: | ---: | ---: |
| AfriMMLU Twi | 24 | 16 | 260 |
| SIB-200 Twi | 7 | 1 | 196 |
| MMLU | 2 | 6 | 292 |
| HellaSwag | 7 | 8 | 285 |
| ARC-Challenge | 0 | 1 | 299 |
| MedQA | 1 | 6 | 293 |
| PubMedQA | 5 | 6 | 289 |

All new SIB outputs were valid single categories; strict and legacy substring
counts coincide in this run. The old-to-new shift therefore cannot be attributed
to strict parsing alone: the chat-template correction and newly pinned inputs
also change the protocol. A controlled ablation would be needed to isolate them.

**Next research step:** review saved R5/R6 disagreements and independent
conversation rollouts, keep these benchmark questions out of repair training,
and add fresh source-separated evaluation cases for arithmetic, meaning retention
and useful next steps. A broad R6 promotion is not justified by these subset gains.

## Local Comparison

Open [R6 and R5](http://127.0.0.1:7870/?__theme=light). R6 is the left/first model;
R5 follows. Requests run serially on one private H100 service: R6 starts and
streams before R5 is submitted. Each model receives the same new question and
only its own previous answers. Both use the same decoding and repetition guard;
raw mode is available in a new separate session. The responses are generated by
the selected checkpoint, with no external LLM replacement.

Saved conversations and assessments remain in a separate local SQLite database.
Earlier R5 sessions are not migrated, rewritten or deleted. Cold-start loading
affects the first reply; the serial timings must not be compared as a latency
benchmark. The model service scales to zero when idle.

The collapsed **Benchmark results** section presents R6 and R5 counts, percentages
and differences; full base results and exact evidence remain available separately.
The verification label checks saved experimental identity and scoring, not whether
a live conversational reply is factually or medically correct.

Restart from the research worktree, only when port 7870 is free:

```sh
python scripts/understanding_r6_workbench.py chat \
  --bundle tmp/r6-demo/bundle-v1 \
  --expected-sha 529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a \
  --port 7870 --benchmark-report tmp/r6-benchmark-v2-20261117/results
```

## Storage and Reproduction

- Worktree: `~/.codex/worktrees/r4-train/ghana-health-ai`.
- Original training artifacts: project `tmp/understanding-r6-20261113/`.
- New local launch receipt: worktree `tmp/r6-benchmark-v2-20261117/launch.json`.
- Collected private evidence: `tmp/r6-benchmark-v2-20261117/results/`, including
  the transport-checksummed archive, original result copies, 6,012 predictions,
  frozen prompts and locally verified summary.
- Private Modal volume: `ghana-health-understand-train`, profile
  `createdliving1000`, environment `main`.
- New run prefix: `understanding-adaptation-r6/benchmark-corrections/r6-matched-v2-20261117/`.
- Each `predictions/<variant>/<task>/<source_index>.json` contains the prompt,
  prediction, gold answer, scores or raw output, and input/checkpoint identity.
- Private UI bundle backup:
  `research-handoffs/twi-understanding/2026-11-17-r6-benchmark-v2/ui-bundle/` on
  the same Modal volume. Its manifest was read back and matches the SHA above.
- That handoff directory also holds `benchmark-evidence.tar.gz` and
  `verified-summary.json`, both read back in full and checksum-verified. The
  evidence archive SHA256 is
  `2940c123fd4237beb5d02ec623ef843c6a57a064c73f2714675dfd077cc48d1f`.
  The summary SHA256 is
  `417ba9cc4ea2d558f38a1aa49fff8547190278eca3d88885dc4574bfde5acc6b`.

From the worktree, with an unused run ID and receipt path:

```sh
python scripts/benchmark_launch.py --run-id YOUR_UNIQUE_RUN_ID --limit 300 \
  --receipt tmp/YOUR_UNIQUE_RUN_ID/launch.json
python -B -m unittest discover -s scripts -p test_r6_benchmark_protocol.py -v
```

After all three calls complete, collect and independently rescore the evidence:

```sh
python scripts/collect_r6_benchmark.py --run-id r6-matched-v2-20261117 \
  --out tmp/r6-benchmark-v2-20261117/results \
  --summary-out docs/research/r6-benchmark-v2-summary.json
```

Use `--verify-only` to recheck an already collected directory without starting
inference or replacing the original archive. The public summary contains only
aggregates, dataset references and identities, not prompts or private chat history.
The CPU archive job also detaches and saves `results-collection-call.json` beside
the collection folder; repeating the command after a connection loss retrieves
the same call instead of starting model inference again.

Verification checks every row against frozen inputs, recomputes predictions from
saved scores/raw labels, matches model/tokenizer/template/runtime identities, and
checks frozen-input/code/legacy hashes and recorded adapter identities. Counts and accuracy must match exactly;
only the derived chance baseline allows 1e-12 rounding for Python-version float
summation differences. This validates the evidence, not independent clinical or
native-language judgments.

### Code Checks

101 automated tests passed: 56 for the corrected benchmark, collection, R6
identity, resumable launches and comparison UI; 45 existing R5 comparison tests
also passed. Coverage includes malformed choices, numeric answer labels,
multi-token scoring, tampered evidence, independent histories, R6-first streaming,
stop/recovery and saved-session refresh. Gradio test teardown emits event-loop
resource warnings; no tests failed. Production application code is unchanged.
Live R6/R5 streaming was exercised; desktop and 390px mobile checks confirmed
the seven real result rows, saved-chat persistence and no horizontal overflow.
This smoke check establishes a working interface, not response quality.

```sh
python -B -m unittest discover -s scripts -p 'test_*r6*.py' -q
python -B -m unittest discover -s scripts -p test_understanding_r5_workbench.py -q
git diff --check
```

The launcher detaches the Modal jobs. Resume only after confirming the prior
calls are terminal; use the same receipt with `--resume` and the exact unchanged
code. Existing v1 results are never overwritten. Do not upload private evaluation
records or adapters with a public documentation update.
