# Balanced Response V2

Status: V2 rejected and deliberately stopped after 234 updates. No private UI or
production promotion. The current experiment is Afrique V3, described below.

## Experiment

Fresh rank-16, alpha-32 LoRA on `google/gemma-4-31B-it`, revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Attention and MLP projections,
bf16, learning rate `1e-5`, effective batch 16, seed 42, 320-update ceiling.
One H100; training stops after 3,900 elapsed seconds if the step limit is not
reached. Total function ceiling is 5,400 seconds, including paired generation.
No failed translation adapter or external adapter was used for initialization.

Run `response_v2_20260910T093637Z`, call `fc-01M25AT9XSS7Y5747JPEHJQ868`,
deployed app `ghana-response-adaptation-v2`. Submission is durable and has a saved
receipt. A pending poll must not be mistaken for a cancellation or failure.
Original inputs, hashes and preflight report are under
`tmp/response-adaptation-v2/`. Artifacts persist in the existing private Modal
volume `ghana-health-understand-train`, beneath `/response-adaptation/<run>/`.

## Actual Mixture

| Task | Accepted training examples | Supervised tokens |
| --- | ---: | ---: |
| General-source Twi answers | 1,702 | 237,288 |
| Twi health answers | 944 | 219,775 |
| English everyday conversation | 1,890 | 63,493 |
| English OpenAssistant replay | 950 | 196,990 |
| English native tool calls | 1,335 | 55,223 |
| Auxiliary Twi translation | 400 | 12,811 |
| Total | 7,221 | 785,580 |

There are 426 source-disjoint validation examples. Two additional training rows
were excluded at native tokenization for length, not truncated. The table is the
measured mixture, not the earlier aspirational 25% tool-token allocation.

Source revisions, file hashes, original row IDs, terms and attribution are
preserved by `prepare_response_sources.py` and `build_response_mixture.py`:
AfriHealth-QA, michsethowusu's Ghana chat corpus, Hugging Face everyday
conversations, OpenAssistant, Argilla APIGen, and existing public parallel text.
This run uses no private voice recording, real conversation archive or new
proprietary annotation call. Original transcripts remain unchanged.

Important limits:

- Ghana chat Twi is machine translated, including an Amharic pivot upstream.
  The source joins assistant turns. Only a first aligned complete paragraph is
  selected, with the English counterpart preserved for inspection. This does not
  establish that the answer is native, complete, or correct.
- Six observed bad source records are excluded explicitly. An agent spot check
  is not blanket human validation of the remainder.
- AfriHealth quality is an upstream claim, not project clinical certification.
- APIGen trains function selection and arguments, not executed result-to-answer
  trajectories. Called function names are split across train/validation; this
  does not imply every distractor schema is unseen.
- Noncommercial and share-alike source restrictions remain. No public
  redistribution or commercial-use clearance is asserted.

## Evaluation

The trainer was designed to compare base and adapter on 30 existing development scenarios and
24 held-out source rows. Raw outputs, truncated-turn indicators, input messages,
references and adapter identity are saved. `summarize_response_adaptation.py`
checks paired completeness and reports inspectable repetition/empty/limit flags.
It does not invent a semantic accuracy or clinical score. This final evaluation
did not run because V2 was deliberately stopped. Do not claim 54 paired results.

An early checkpoint-100 comparison also exercises the private runtime:

- Hash `3e35a6f81e0c609b4015955b32259ab03a726ad504c8a70a6b2a5ad68f020f60`.
- Greedy decoding: budget answer loops; English language switch works; no child
  age is invented in the English context check; real decimal tool execution
  returns 15.00 GHS and the model states that result in Twi.
- The unchanged base also performs the tool calculation and language switch.
  Those are integration successes, not evidence of an adapter improvement.
- Both variants are additionally compared with the publisher's sampling
  settings. Do not discard the greedy regression or select only attractive
  outputs from different seeds.

The first sampling adapter diagnostic hit an old method signature during the
deployment transition (`fc-01M25C426WNY8P6HE26W2VCN1P`), before inference. Its
failed receipt is preserved. An explicit second attempt was submitted; the main
training run was not cancelled by that error. It was later stopped deliberately
at update 234 after checkpoint-200 health responses also regressed. That sampling
comparison produced incoherent Twi, language mismatches, and unsupported symptom
questions. Decision: `data/response-adaptation/response-v2-decision.json`.

## Private Runtime

`modal/research_response_service.py` is a separate private Modal class with no
public HTTP endpoint. Existing base and training volumes are mounted read-only.
Every adapter load is bound to an experiment, checkpoint and SHA256. Zero warm
minimum, one H100, 60-second idle scale-down, bounded context/output/time.

Ordinary replies stream through the native Transformers response parser, with
the actual prompt prefix supplied. Thought and raw tool syntax do not appear in
chat bubbles. The only connected tool is `calculate_total`: validated, bounded
decimal multiplication using supplied price/quantity. No live inventory, search,
ordering or payment service is implied. Partial calls and unknown functions
cannot execute. The model consumes the actual tool result and generates its own
final answer. Tool traces and model provenance are retained with local reviews.

Local private selection is explicit in
`tmp/research-playground/private-selection.json`. Without an enabled valid
selection the UI retains its old pilot choices. Do not label the old pilot as v2,
silently substitute the foundation for the adapter, or route through GPT.

Primary implementation references:
[Gemma sampling guidance](https://huggingface.co/google/gemma-4-31B-it#best-practices),
[native response parsing](https://huggingface.co/docs/transformers/main/chat_response_parsing),
[function calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).

## Afrique V3: Actual Instruction SFT

Pinned foundation: `McGill-NLP/AfriqueQwen3.5-4B-50Langs`, revision
`ea443ca5e6674e17c271fb66e54e3282fe78d21a`. This is continued pretraining, NOT an
instruction assistant. The experiment performs instruction SFT, not another
task-vector merge. Its Twi-inclusive pretraining does not establish product quality.

The source-preserving corpus excludes all 1,794 general-source translated rows
across splits. Remaining: 5,521 training / 334 validation, all accepted without
truncation by native Qwen tokenization. Training includes 1,890 English everyday,
952 English replay, 1,335 tool, 944 Twi health, and 400 auxiliary translation rows.
English dominates general dialogue supervision; broad Twi conversation remains a
data gap. Sources are not automatically human-verified gold.

Rank-16 all-linear LoRA, alpha32, LR5e-5, batch8, 691-update ceiling, one H100.
The native template's empty thought prefix is masked, not duplicated in targets.
30 raw base outputs, four probes per checkpoint, 30 final adapter outputs and 24
source-heldout outputs are saved. The latter are NOT a paired base comparison.
The base model's observed repetitive control strings are a baseline, not usable chat.

First V3 attempt failed at save100 due to a read-only cache lookup, not bad loss
or cancellation. No checkpoint weights were written. `adapter_checkpoint.py`
now saves LoRA with frozen embeddings explicitly excluded and validates an actual
save/reload before training. A new run was submitted at 10:32:20 UTC:
`afrique_v3_20260910T103220Z`, `fc-01M25E0ACC7AEHZYHXJV4HPJ2E`.
Receipt: `tmp/afrique-response-v3/savefix-run-receipt.json`.
Artifacts: existing private training volume, `/afrique-response/<run>/`.

The private runtime has a separate Qwen native XML response parser and schema
casting; character-stream tests verify tools/thought are hidden, real results
feed back to the model, and partial calls do not execute. No model is yet selected
in the private UI. No proprietary fallback, public HF upload or production change.

## Continue

1. Poll the save-fixed V3 call above, not the failed first attempt or a new submission.
2. Inspect both early sampling outputs and completed paired outputs.
3. Compare meaning, language, corrections, health context and real tool behavior;
   lower loss or valid syntax alone is not improvement.
4. Select a retained, verified checkpoint only when the evidence justifies private
   testing, preserve its shortcomings, then restart the local tester with its
   stable browser-state key. Keep the public product unchanged.
5. If quality fails, change the implicated data, training or decoding variable
   and continue bounded experiments. Another failure report is not the goal.
