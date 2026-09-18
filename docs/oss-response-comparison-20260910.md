# Open-Weight Reasoning Comparison

Completed `oss120_20260910T142628Z`, call
`fc-01M25VD3Q656N48C4381ZDPRAE`, private Modal app
`ghana-oss-response-comparison`. Do not resubmit.

Model: `openai/gpt-oss-120b`, revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. CPU preparation downloaded
65,248,893,184 bytes in 147.32 seconds. The GPU run used one H100, publisher
MXFP4 weights, medium reasoning, temperature 1, top-p 1, seed 42, and a
2,048-token output cap. Loading took 165.37 seconds; total was 511.85 seconds.
This is self-hosted open-weight inference, not GPT-5.6 or a hosted API call.
It is also NOT a model trained by this project.

## Actual Outcome

All 30 saved development cases completed, with no parse errors, empty answers,
eight-word repetition flags or length stops. Two tool requests were schema-valid:
tomato product search and English web search. Tools were not executed.

Meaning successes include tomato negation/time translation, the corrected 1kg,
Kofi as the future traveler, and arithmetic yielding 20. These are specific
observations, not a general semantic accuracy score.

Long Twi replies remain badly garbled. Breathing/chest pain becomes stomach,
diet and antacid advice; pregnancy context is lost; the newborn follow-up
ignores the supplied age and corrupts time units. No-tool shopping invents
unverified local services. A code-switched child-fever answer adds Chinese text.
Neither smooth termination nor tool syntax implies useful or safe Twi responses.

**Decision:** no chat replacement, no automatic Twi teacher acceptance, no
private UI enablement, no production promotion, no new 20B size trial based
only on this model's limited meaning successes.

## Evidence

- Raw private result: `tmp/oss-response-comparison/results.json`, SHA256
  `0a37e818f1aa6f69ddf8eb10e8d50fcd090e7b151c31a9b6a6669d3a81fb1ff0`.
- Final-answer-only review: `tmp/oss-response-comparison/results.diagnostic.json`.
  Private reasoning is not copied into this review export.
- Compact decision: `data/response-adaptation/oss120-decision.json`.
- Durable remote results: volume `ghana-health-understand-train`,
  `/stronger-response/oss120_20260910T142628Z/`.

The native Harmony parser uses Transformers' response parser, with explicit
analysis/final/tool regions. Tests cover hidden reasoning, history isolation,
both tool header orders, incomplete reasoning, and incomplete/invalid tool
handoffs. Diagnostics reject missing or duplicate result IDs and changed inputs.

## References And Next Research

[Publisher model card](https://huggingface.co/openai/gpt-oss-120b) and
[Transformers MXFP4 documentation](https://huggingface.co/docs/transformers/quantization/mxfp4)
explain the native quantization path. Published multilingual scores do not
establish Twi competence; our observed replies reject that assumption here.

The independent source-answer experiment now requires verbatim supporting
evidence and explicit answerability judgments. Read
`docs/native-answer-synthesis-20260910.md`. Source translation defects and
context-dependent news fragments remain obstacles to scaling direct-response
supervision. No larger unreviewed corpus is declared gold because a model agrees.
