# MORENA Baseline Evaluation — 2026-09-21

## Purpose

This is a **baseline evaluation only**. It is not a MORENA research programme,
training run, foundation bake-off, or model-selection reset.

The broader research goal remains to determine how to adapt a capable foundation
so that it genuinely understands and responds in Twi while retaining useful
general capabilities. MORENA is one candidate used to obtain evidence toward
that goal.

## Immediate question

> How much Twi linguistic capability does the untouched
> `vamboai/morena-1.5b-base` already demonstrate on the project's newest
> source-backed corpus?

This run must not be used to claim that MORENA is or is not a complete Twi
conversational assistant. The current corpus primarily measures bilingual
language alignment.

## Data

Use only the verified September 15 release:

`twi-alignment-v2-20260915`

Private Modal copy:

`/corpus-training-releases/twi-alignment-v2-20260915/0d59772b4b49`

Use the existing validation split. Do not rebuild, reshuffle, augment, translate
or substitute older evaluation fixtures.

Report the views separately:

| Validation view | Examples | Purpose |
| --- | ---: | --- |
| Sentence alignment | 454 | Primary language-alignment evidence |
| Dictionary alignment | 686 | Lexical evidence only |
| Grounded-QA question alignment | 4 | Descriptive examples only; too small for a broad claim |
| Total | 1,144 | Never collapse the three views into one headline score |

Each alignment source is evaluated in its existing direction(s). Preserve source
IDs, groups, references and provenance exactly as stored in the release.

## Execution

1. Resolve and record the exact immutable MORENA model revision before GPU work.
2. Load the existing verified validation artifacts from the private Modal volume.
3. Run **inference only** with the untouched model. No LoRA, SFT, continued
   pretraining, prompt-generated demonstrations, retrieval, proprietary fallback
   model or reference-answer exposure.
4. Use one explicit, reproducible generation configuration appropriate to the
   model. Record tokenizer revision, prompt format, decoding parameters, GPU,
   package versions, run/app/function IDs and wall time.
5. Save one raw prediction record per example with source ID, evaluation view,
   direction, input, reference, raw model output, stop reason and token counts.
6. Score sentence and dictionary views separately with reproducible automatic
   reference-similarity metrics. These metrics are proxies, not proof of semantic
   understanding or native quality.
7. Inspect a deterministic sample of successes and failures, especially gross
   meaning changes, non-Twi output, copying, looping, empty output and truncation.
8. Produce a signed/hashed evaluation receipt and a short result document. Preserve
   the raw outputs even if the result is poor.

## Hard boundaries

- **No training in this run.**
- Do not run the old 74-case foundation suite as the primary evaluation.
- Do not rerun Qwen, AfriqueQwen, MiniCPM, Gemma, previous Ghana Health AI
  adapters or translation baselines.
- Do not create a new benchmark to make MORENA look better or worse.
- Do not spend additional credits after a failed run until the failure is
  classified as infrastructure versus model behaviour.
- Infrastructure retries must preserve the same model, corpus and evaluation
  contract and be recorded rather than hidden.
- Do not publish a model or alter Ghana Health AI production routing.
- Do not infer conversational competence from alignment scores.

## Deliverable

The output of this step is one answer:

> **What does untouched MORENA do on our newest held-out Twi alignment data?**

The report must include:

- immutable model revision;
- exact corpus/release checksum or receipt identity;
- counts actually attempted/completed by view and direction;
- automatic metrics by view/direction;
- generation failures and stop reasons;
- representative raw examples;
- compute/runtime receipt;
- limitations and the next decision.

## Decision after results

Do not pre-authorize MORENA training.

After the baseline exists, combine it with the research evidence already saved
for other foundations. Decide the next adaptation experiment from the broader
research question, not from enthusiasm for MORENA.

If MORENA warrants adaptation, write a **separate experiment plan first** stating
the hypothesis, training objective, exact training split, held-out evaluation,
changed variable, success threshold and maximum compute budget. Only then may a
training run begin.

If it does not warrant adaptation, retain the baseline as a negative result and
move to the next research action without rerunning the evaluation.
