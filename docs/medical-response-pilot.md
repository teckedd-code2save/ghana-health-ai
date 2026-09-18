# Saved-annotation response pilot

## Scope: September 8, 2026

The user confirmed available Modal compute and requested a first LLM experiment
from the saved annotations while OpenAI API funding is unavailable. This is an
explicitly uncalibrated, isolated learning experiment. It does not complete the
large-corpus goal or relax the accepted corpus and product promotion gates.

No OpenAI requests, source-label regeneration, production deployment, or public
Hugging Face upload are part of this run.

## Data

- Input: 305 completed teacher-v3 annotations, 120 marked model consensus.
- 185 unresolved rows are excluded. One further consensus row is a near duplicate
  of locked evaluation data and is also excluded.
- Selected: 119 unique source records, not 305 approved training records.
- Deterministic source-group split: 95 train, 24 pilot holdout.
- Three views per source: Twi interpretation JSON, direct Twi reply, direct
  English reply. This gives 285 train and 72 holdout examples, not new sources.
- Exact question entities are checked against source text. English views teach
  replies only: Twi entity spans are not incorrectly copied into English labels.
- All selected safety labels are routine. Safety and clarification classifiers
  are deliberately absent from the supervised target; a constant routine label
  must not masquerade as learned triage.
- Labels are model-assisted, not human gold. Most are adolescent/reproductive
  health education, not acute clinical conversations or commerce.
- Production review decisions were read from Postgres before export: zero
  AfriHealth review records on September 8. The empty decision snapshot is
  `tmp/medical-response-pilot/reviews.v1.jsonl`. Future runs must obtain a current
  snapshot and respect all human decisions, rather than assuming this stays empty.

The pilot artifacts set `experimental_only: true` and retain
`eligible_for_research_training: false`, `eligible_for_final_evaluation: false`,
and `eligible_for_production_training: false`. Only the separate explicitly
acknowledged pilot runner accepts them. Normal corpus export remains closed.

## Recipe

- Base: `ghananlpcommunity/MiniCPM5-1B-Twi` at
  `d807ca1a3323972afafabff8f9affe2639e37b5c`, not the failed previous adapter.
- LoRA rank 16, alpha 32, dropout 0.05; attention and MLP projections.
- Three epochs, learning rate 0.00005, effective batch 8, seed 42.
- Train assistant tokens only, including EOS, against the exact inference
  prefix. This tokenizer inserts an empty think block in inference prompts but
  not completed training conversations; using those templates interchangeably
  would misalign the training prompt.
- All 285 examples were tokenized locally: maximum 510 tokens; no truncation.
- Single A100 40GB, 2 CPU cores requested, 16GiB memory, 3,600-second function
  timeout, no automatic function retries. The one-hour GPU-only list-price
  ceiling is approximately $2.10, not a billing statement; startup, CPU, memory
  and storage are additional. No existing service is stopped or changed.
- Results persist on `ghana-health-understand-train` under a unique
  `/data/sft/annotated_response_pilot_v1_<UTC timestamp>` directory.

## Evaluation

The untouched base and adapter use identical prompts and decoding within each
task. The 24 pilot holdout sources and all their translated views are excluded
from training. These are fixed evaluations, not checkpoint-selection data.
Additional diagnostics use 16 original locked source-validation questions per
language and all 13 pre-existing product fixtures, including multi-turn cases.

Structured generation uses greedy decoding with a small repetition penalty and
no n-gram blocking: blocking repeated JSON delimiters can corrupt the schema.
Free-text replies use the base author's documented sampling settings:
temperature 0.7, top-p 0.9, repetition penalty 1.3, and no-repeat trigram blocking.
The earlier response experiment used greedy free-text decoding, against the
base card's recommendation. This is a fresh comparison, not a claim that the
older run's failures were only decoding problems.

Report schema validity, question-grounded entities, intent agreement, meaning
and reply chrF++, free-text repetition, and lexical product checks. Agreement
with uncalibrated model labels and lexical checks are not medical correctness.
No automatic production promotion is allowed, even if these proxies improve.

## Reproduce

```bash
node --import tsx scripts/build-medical-response-pilot.ts \
  --acknowledge-uncalibrated-experiment \
  --reviews tmp/medical-response-pilot/reviews.v1.jsonl
node --import tsx scripts/eval-medical-response-pilot.ts
python3 scripts/test_medical_pilot_core.py
modal run --detach modal/train/train_medical_response_pilot.py \
  --acknowledge-uncalibrated
```

The runner writes checkpoints, raw base/adapter predictions, training metrics,
corpus and configuration manifests, comparison JSON, and a model card. It does
not publish weights. Any later Hub upload must include the completed card,
measured results, source attribution, and the unvalidated research-only status.

## Result

Modal app `ap-LuiKWmw2m3er1nsJA8Q0UQ` completed. The saved adapter is
`/data/sft/annotated_response_pilot_v1_20260908T213218Z` on
`ghana-health-understand-train`. Training took 205.98 seconds (108 updates,
mean loss 1.8898). Training plus evaluation worker time was 1,028.94 seconds;
the GPU-only list-price estimate is $0.60, not the final bill.

| Check | Base | Adapter |
| --- | ---: | ---: |
| Valid structured interpretation | 0/24 | 22/24 |
| Intent agreement with model labels | 0/24 | 13/24 |
| All extracted entities grounded in question | 0/24 | 6/24 |
| Pilot holdout Twi reply chrF++ | 23.62 | 22.95 |
| Pilot holdout English reply chrF++ | 12.03 | 25.91 |
| Locked source Twi reply chrF++ | 30.30 | 16.65 |
| Locked source English reply chrF++ | 14.87 | 20.56 |
| Critical lexical product checks | 1/8 | 0/8 |

The majority-class intent baseline scores 14/24 (58.3%), exceeding the adapter's
13/24 (54.2%). The experiment learned format-following, but it did not demonstrate
semantic competence. Inspected mistakes include copying Twi into `natural_english`
and mistranslating pregnancy prevention as setting up a space. Word-similarity
scores can also penalize the concise trained style against long original source
answers. They must not be reported as accuracy or as proof of clinical safety.

The checkpoint is not promoted, deployed, or published. Keep it as an evaluated
negative result. The larger annotation goal is still outstanding.

The first submission, `ap-aZh3Q8bemN4nHLudCDTVZR`, failed before training because
of a relocated-module path bug and was explicitly stopped. Its startup costs
are not included in the successful worker's estimate. A test now covers the
remote module import.

An additional permanent cloud archive of the input corpus was blocked by the
permission check. It was not uploaded. The user was asked for approval; the
local input files remain intact. The completed model and its reports are already
stored on Modal as part of the authorized training run.

Local close-out on September 9: the optional recursive checkpoint download
stalled and was stopped. The complete adapter was independently downloaded,
verified as 336 tensors, and restored into the local run directory. Its SHA-256
is `294fd2815664c0b3140927dc637e5df35fc43563eaa9d8c5bb48f3678c2762de`.
All 14 root JSON artifacts parse. Optional partial resume files are isolated in
`checkpoints.incomplete-download`; use the intact remote checkpoint if needed.
The training job and local download are no longer running.

## Sources

- [Base model card](https://huggingface.co/ghananlpcommunity/MiniCPM5-1B-Twi)
- [Modal resource pricing](https://modal.com/pricing)
- [AfriHealth QA source](https://huggingface.co/datasets/ImhotepSystems/AfriHealth-QA)
- [Challenge license clarification](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge/discussions/33230)
