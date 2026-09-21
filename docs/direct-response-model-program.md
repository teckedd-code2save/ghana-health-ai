# Twi-English General Understanding Model Program

> **Research scope note (2026-09-21):** This document describes the
> conversational-response stage and its corpus/safety requirements. It is one
> stage of the broader Twi adaptation programme, not the overarching research
> question. The authoritative programme framing is
> [general-understanding-roadmap.md](./general-understanding-roadmap.md): compare
> the untouched foundation, translation-mediated baseline, bilingual alignment,
> native conversational SFT, and alignment followed by conversational SFT.
> The target contribution is an independently serving model that understands
> and responds naturally in Twi while retaining useful foundation capabilities.


## Scope correction: 2026-09-09

The user clarified that the overarching goal is a general-purpose understanding
and response model, with health education as a special expertise, alongside
ecommerce and tool use such as web search. A medical classifier or medical-QA
adapter is not the whole product. The 95-source pilot tests a narrow recipe;
it is not evidence of achieving this broader goal.

The current research and training direction is specified in
[general-understanding-roadmap.md](./general-understanding-roadmap.md).
The medical corpus contract and safety gates below remain applicable to the
health specialization, not to every general-conversation training example.
General replies must not be forced into a HEALTH intent or medical-advice
template. Structured meaning extraction is a separately evaluated capability,
not a requirement to print JSON before every conversational response.

## Earlier health specialization plan

## Decision

The next research checkpoint is a direct-response LLM, not another hidden hint
to a hosted model. For each turn it must produce both a machine-checkable
semantic record and its own short Twi reply.

```text
Twi speech/text + short history
  -> normalized Twi, faithful English meaning, intent, entities, uncertainty
  -> safety level, clarification decision, Twi reply
```

The response model must not depend on `gpt-5.6-sol` at inference time. A
deterministic safety gate remains around the model for known red flags and for
cases where the model declares uncertainty.

## Corpus contract

Research response SFT can use either human-reviewed rows or high-confidence
multi-model silver rows that pass every deterministic and calibration gate.
Each accepted row must contain:

- normalized Twi;
- faithful English meaning and intent;
- a source-grounded Twi response and its exact English meaning;
- entities, ambiguity, and safety level;
- source, consent scope, and stable split.

Silver rows are never labelled human gold. Two materially different proposal
paths and a separate adjudicator must agree at high confidence. Disagreement,
negation, vulnerable populations, medicine, urgency, questionable source
answers, code-switching, and prior product failures route to focused human
review. A native-speaker sample audit remains required before model promotion.

`pnpm corpus:response:export` writes the candidate JSONL files to
`tmp/response-capable-corpus/v1/`. `pnpm corpus:response:export:strict` blocks
unless there are at least 100 reviewed response rows distributed across train,
dev, and test.

The existing large semantic corpus is still valuable: it teaches meaning
recovery. It is not silently converted into response examples, because that
would teach generic advice without reviewed target replies.

## Review loop

The chat review control opens a one-at-a-time sample card. It starts with
medical rows that already contain source-linked Twi response drafts. A reviewer
can correct the meaning, intent, Twi reply, and safety level, then mark the row
reviewed. A consented recording is attached to the exact row with its speaker
ID, allowing several speakers to record the same phrase.

Recordings are private review data stored in Postgres for this first research
lane. They are not public assets and are not included in a model or dataset
push without a separate export, consent, and licence review.

## Promotion gate

A direct-response checkpoint can be exposed as a research choice only after:

1. It emits parseable structured output and a Twi reply on every held-out case.
2. It preserves held-out meanings, including negation, person, symptom, time,
   and code-switching.
3. It makes no critical emergency-routing errors on the locked safety set.
4. Native review finds the Twi reply useful and faithful rather than templated.
5. Its Hugging Face card includes source licences, corpus counts, the split
   policy, evaluation results, limitations, and its non-production status.

Until then, `Research v1` remains an interpreter experiment. It should not be
described as a better assistant simply because the product can obtain a good
final reply elsewhere.

## 2026-09-06 response pre-adaptation result

The first balanced direct-response experiment used 8,809 deterministic-pass
AfriHealth Ghana rows: 4,407 Twi and 4,402 Ghanaian English. It was deliberately
limited to about one epoch and was evaluated against the untouched base on 128
locked source-validation rows and 13 product cases.

The experiment answered English prompts in English and improved English
reference similarity, but Twi reference similarity fell and all eight critical
product cases failed. Several responses invented diagnoses or medicine advice.
The checkpoint is not published and is not wired into the app.

This result narrows the next training stage: raw bilingual question-answer SFT
is useful pre-adaptation, but it does not satisfy the corpus contract above. The
next response corpus must explicitly supervise faithful meaning, intent,
entities, uncertainty, safety level, and a concise language-matched reply. Human
review should target model disagreement and safety-critical clusters rather
than require exhaustive review of all source rows.

## 2026-09-06 annotation calibration checkpoint

The response pool and locked evaluation set are stable:

- 8,809 balanced source-train rows: 4,407 Twi and 4,402 English;
- 2,198 locked source-validation rows: 1,102 Twi and 1,096 English;
- 5,569 immutable Twi question-answer source records;
- source revision `61befdaa19e12afdbf9032a602067f0daa3e6c68`.

The stronger dual-teacher pilot contains 62 real AfriHealth Twi rows. It is a
calibration reference, not human gold: 31 are silver consensus, 31 need review,
and 27 currently clear the training-eligibility rules.

The cheaper open pipeline uses Qwen3.5-35B-A3B directly on Twi, plus an
independent NLLB Twi-to-English pivot, followed by Qwen adjudication. On 58
importable rows shared with the stronger pilot it achieved only 56.9% intent
agreement, 72.4% safety agreement, 69.0% source-answer-assessment agreement,
and a 55.0% mismatch-review capture rate. Its automatic promotion gate fails.
It must remain a proposal/disagreement signal and must not be scaled into
training labels in its current form.

The annotation runner now retains every raw attempt, validates complete
schemas, retries malformed output once, and uses eager vLLM execution for
bounded correctness checks. Modal run `ap-dJE2ONzSTftdrzUs9d4dAe` produced
4/4 complete direct proposals, 4/4 pivot proposals, and 4/4 adjudications with
zero importer rejection.

The user replenished API credits on September 6. Teacher-v2 completed the 62
reference rows, but failed the consistency gate: intent and safety agreement
were each 82.3%, and source-answer assessment agreement was 67.7%. These are
agreement with model-assisted references, not human-verified accuracy.

Inspection found another concrete issue: reference-answer entities leaked into
question interpretation. Teacher-v3 now requires question-only intent/entities
and keeps source-answer assessment separate. Deterministic guards prevent
answer-only entity values from entering model-silver exports. A 12-row check on
new source-train records completed with zero selected lexical span violations,
10 model consensus and two review rows. This bounded check does not establish
semantic or medical correctness, and full calibration remains outstanding.

The dedicated annotated export is
`pnpm corpus:medical:afrihealth:export-annotated`; add `--reviews-db` to include
reviews saved in Postgres. Model silver requires a passing calibration report
matching the prompt and teacher/adjudicator identities. Human-selected
proposals and corrections carry reviewer provenance and take precedence over
model recommendations. Sources and held-out questions are checked before
export, and a changed Twi reply cannot silently reuse the old English meaning.

Each Twi source can also supply an English training view through the selected
translation. Both views keep the same source identity and split. Report these
as two examples of one source, never as two independent corpus records. The
larger unique-source target still requires annotating the broader source pool.
The current export is not ready: its calibration and review requirements have
not been met.
