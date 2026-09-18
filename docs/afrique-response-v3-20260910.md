# Afrique Response V3 Outcome

The corrected run completed all 691 updates on 2026-09-10. Run
`afrique_v3_20260910T103220Z`, call `fc-01M25E0ACC7AEHZYHXJV4HPJ2E`.
One H100; total observed duration 1,841.8 seconds including evaluation. Adapter
SHA256 `6901a6464f1c86ae7d20e08458ed56d879bdec72c9b4a1c07b5aa088c88b6db2`.
5,521 training examples; 334 source-disjoint validation examples. Original model
and source limitations are in `response-adaptation-20260910.md`.

## What Improved

All 30 final development responses terminate without the 384-token limit;
the unchanged CPT base hits that limit on all 30. The eight-word repetition
heuristic flags 24 base responses and zero final adapter responses. These are
format/termination diagnostics, NOT understanding accuracy. The base was not an
instruction assistant, so this comparison does not establish superiority to a
capable instruction model or to every previous pilot response.

English shopping-list explanation, budgeting, and language-switch follow-up are
responsive in the final batch. The corrected Twi tomato quantity is retained.
Offline tool calls preserve tomato/2kg/Adenta arguments and request a web search.
Those are generated calls, not actual searches or inventory results.

Checkpoint 200 was separately tested through the private serving runtime. It
requested `calculate_total(7.5, 2)`, consumed the actual result and replied with
15 cedis in Twi. The calculation-plus-final-answer took 4.109 seconds excluding
loading/network. No order, search or payment occurred. This is one narrow
integration success, not a general tool-use accuracy score.

## Why It Was Not Enabled

Final responses reverse the meaning of a negated tomato purchase, choose the
wrong person for a future trip, and give contradictory arithmetic amounts.
Health responses lose important context and are not acceptable patient guidance.
The separate 24 source-heldout generations include four length-limit stops;
they have NO paired base evaluation and cannot support a comparative claim.

At checkpoint 400, tools became overused: the model invented price/quantity
arguments for budgeting and a child-health question. Both temperature/top-p
sampling and a separate presence-penalty test showed this. Raw pre-guard traces
remain, rather than being relabeled as successful tool use.

The name `publisher` in saved Qwen runtime results refers to temperature .7,
top-p .8 and top-k20, NOT all publisher settings: presence penalty was zero.
`qwen-presence` is the explicit 1.5 additive generated-token presence experiment.
It uses the same checkpoint and seed. It does not fix meaning preservation.

The runtime now rejects calculation values not present numerically in user
messages. That is a minimum guard, not proof of correct units, relevance, or
correction handling. Errors remain visible as model failures; no other LLM
substitutes a reply.

## Artifacts

- Raw paired and source results: `tmp/afrique-response-v3/completed/`.
- Per-checkpoint probes and serving traces: `tmp/afrique-response-v3/`.
- Verified checkpoint 200 backup: `tmp/afrique-response-v3/archive-200/checkpoint-200/`;
  hash `ca3270412a9cc1bc22707ef7417663bd5c49aeb48ec78bf8f6c89eaa845cb6c6`.
- Private volume: `ghana-health-understand-train`, `/afrique-response/<run>/`.
- Decision: `data/response-adaptation/afrique-v3-decision.json`.

No public HF publication, production promotion, or private tester selection.
The owner voice A preference and existing local chat history remain unchanged.

## Next Controlled Work

`prepare_tool_context_repair.py` attaches an available calculator to 1,000
existing no-numeric-input training conversations, retaining all original
questions, answers, source IDs and splits. It adds zero source rows. Another 80
validation contexts have the same treatment. This addresses tool-availability
bias, not broad Twi semantic or clinical deficiencies.

`prepare_grounded_twi_response.py` imports pinned AfriQA human-translation-labelled
questions and gold passages. Raw records are retained, exact English answer
offsets are checked, and official held-out questions/article titles are excluded
from training. Added before tokenization: 225 training and 251 validation sources.
It excludes 132 reserved-source overlaps, 83 span mismatches, 36 missing-field
records and eight duplicate questions. It does not repair or silently rewrite
those source records. AfriQA adds short grounded answers, not native full dialogues.

Grounded Response V4 uses the stronger measured Gemma 31B instruction foundation,
fresh LoRA at LR3e-6, rather than continuing the failed CPT adapter. Native
preflight accepts 5,744 training and 585 validation examples; two overlength
English replay rows are excluded without truncation. Six per-save development
probes and final paired checks are planned. Multiple variables change, so no
single-variable causal effect should be claimed.

Sources: [AfriQA](https://huggingface.co/datasets/masakhane/afriqa),
[pinned source repository](https://github.com/masakhane-io/afriqa/tree/5c29933753692c28773c34ac1f68482007f1d2bb),
[Qwen decoding guidance](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).
