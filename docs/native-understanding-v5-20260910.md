# Native Understanding V5

## Current Run

The run finished all 320 updates and all final evaluations;
no new private UI choice or production model is enabled. The checkpoint-100
weights are backed up locally at `tmp/native-understanding-v5/archive-100/checkpoint-100/`,
SHA256 `ec2c30c3b67eba3c85e3d4db711c2bd413a9eb29b1c09f32cc6d39d18491768b`.
Final adapter SHA256:
`407ba8f17bf9376bf031d7f4626340d382ce7d764d8b5da70e75f8bdfe677353`.

Submitted September 10 at 12:00:05 UTC. Run `response_v5_20260910T120005Z`, call
`fc-01M25K0ZZW2A7JX8WVTAS30VHG`, app `ghana-native-understanding-v5`.
Confirmed real optimizer updates at 12:02 UTC. This is a normal durable Modal
function, not a client-bound generator. Do not submit duplicates.

Recipe: fresh Gemma 4 31B instruction base, pinned revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Rank-16 LoRA, alpha 32, dropout
0.05, attention q/k/v/o projections only, LR 1e-5, effective batch 16,
maximum sequence 1024, at most 320 updates. One H100, training elapsed stop at
3,900 seconds, function ceiling 5,400 seconds including evaluation. The example
count is the available mixture, NOT a claim that all rows have been seen in this
bounded run. This changes data and trainable layers together; it is not a
single-variable causal ablation. The narrower update is a hypothesis to test,
not a demonstrated preservation guarantee.

Preflight manifest SHA:
`9443eebbe2b5d8d3395755299782612201e171c7ac7a24c508b13bc6e4a5aa05`.
Actual tiny-model training and save/reload passed. 10,341 training examples and
905 validation examples pass native tokenization. Two overlength English replay
rows are excluded; no target is truncated.

## What Changed From V4

V4 was stopped by this agent at 11:55 UTC after repeated response regressions.
Last logged update before the stop was 210; exact stop update is not known.
Checkpoints 150 and 200 loop under greedy generation; checkpoint 200 also loses
meaning under publisher sampling. Its breathing reply invents fever/sore throat,
and its newborn-malaria reply loses context. No UI or production promotion.
The final paired response evaluation was not completed. Decision is recorded in
`data/response-adaptation/semantic-v4-decision.json` and on the private run volume.

V4 checkpoint 100 was independently evaluated on all 450 AfriXNLI development
pairs in both Twi and English, with counterbalanced A/B/C choices:

| Model | Twi | English |
| --- | ---: | ---: |
| Frozen base | 271/450 (60.22%) | 408/450 (90.67%) |
| V4 checkpoint 100 | 270/450 (60.00%) | 408/450 (90.67%) |

Four Twi cases improved and five regressed. All labels were valid. No improvement
is claimed; these are strict generated-label results, not chat or clinical
accuracy. English instructions are used for both languages. Foundation pretraining
contamination is unknown. Official AfriXNLI test data remains untouched, and no
AfriXNLI examples are in the training mix.

The first semantic-evaluation submission failed on a Python argument-name
mismatch before model loading. That receipt is preserved. Corrected attempt 2
`fc-01M25HYX6D9VMCR7XVF7ZMRH4Q` completed all 1,800 base/adapter predictions
in about 200 seconds including loading. Outputs and scoring are under
`tmp/semantic-response-v4/afrixnli/`.

The numeric tool grounding guard also incorrectly rejected price/quantity at
sentence-ending full stops. It is fixed and regression-tested. The real
7.50 x 2 calculation now executes and returns 15.00 GHS; the model follows with
`Tomato kilogram mmienu bo yɛ cedi 15.00.` This is an infrastructure fix, not
evidence that the adapter learned arithmetic better than its base.

## V5 Interim Evaluation

Checkpoint 100, AfriXNLI dev, identical counterbalanced prompts and greedy
decoding: Twi base **271/450**, adapter **271/450**; English base **408/450**,
adapter **412/450**. Twi has nine corrected and nine newly wrong cases;
English has four corrected and no newly wrong cases. No invalid labels.
This is not a demonstrated Twi understanding gain. Corrected attempt 2 is
`fc-01M25MJQ6KY69PRZCA4FVA6BTX`; the original attempt failed before loading
because a reused Modal worker retained an old volume snapshot. The evaluator
and private checkpoint inspector now reload before reading new checkpoints.

Checkpoint 50, INJONGO dev: 204/320 base versus 206/320 adapter intent labels
match when scored independently of the entity format. Full-structure scoring
was much worse: 280 base and 265 adapter rows do not conform, and neither
has a correct literal entity match under the required schema. The original
prompt did not specify entity field names or label types, so these are NOT
credible standalone entity-understanding measurements. Raw outputs remain.
The scorer accepts one enclosing JSON fence for semantic scoring while counting
it as a strict formatting failure, and never extracts labels from prose.

A separately named `schema-v2` comparison defines the exact JSON layout and
23 entity types derived only from training data, for BOTH base and adapter.
It does not rewrite the running training corpus or inject dev reference answers.
Checkpoint-100 call `fc-01M25MJQAPBW5068SXPAE1RMXE` completed: base 207/320
intent, 92 exact records, entity F1 0.5894; adapter 206/320 intent, 92 exact
records, entity F1 0.5930. Neither has format failures under this explicit task.

Final adapter call `fc-01M25NJ313SD0P3W93TYBNGGNV` also completed: adapter
**212/320** intent labels, **103/320** exact records, entity F1 **0.6186**.
Base scores remain identical. There are no format failures. This is a small
auxiliary-task gain, not demonstrated general conversation or clinical quality.
Final AfriXNLI call `fc-01M25NJ3716R2YQCFYJ6522F67` completed. Twi base
271/450 (60.22%) versus adapter **278/450 (61.78%)**; English base 408/450
(90.67%) versus adapter **409/450 (90.89%)**. Twi has 27 corrected and 20 newly
wrong cases; English has 10 corrected and nine newly wrong cases. No invalid
labels. These small gains are not a statistically established improvement in
general conversation, clinical judgment, or Twi fluency.

Final paired generation is also complete: 30 product and 30 source cases per
variant. Base has eight length-limit outputs and zero repeated eight-word-span
flags; V5 has four length-limit outputs and four repetition flags. Budgeting,
newborn-fever follow-up, pregnancy warning, and one source response repeat.
The breathing reply can merely echo the user. Correction and negation successes
are shared with the base. Neither length limits nor these mechanical flags are
a complete semantic assessment; actual replies drove the no-promotion decision.

Decision: retain V5 as a research artifact, NOT a chat replacement. Compact
evidence is in `data/response-adaptation/native-v5-decision.json`. Full final
weights, paired outputs and model card are preserved locally under
`tmp/native-understanding-v5/completed/response_v5_20260910T120005Z/` and on the
private Modal volume. The adapter SHA was independently checked after download.
The private model card now includes the actual final independent scores and
limitations. No HF upload or UI enablement occurred.

The instruction defect is fixed in the prepared `tmp/native-intent-v3` package
using `prepare_native_intents.py --output tmp/native-intent-v3 --schema-v2`.
All 4,978 records across train/development/test were compared with v2: ONLY the
system instruction differs. Text, answers, source evidence, IDs and splits are
unchanged. V5 still trained on v2, not retrospectively relabeled v3.

Response checks remain poor. Checkpoint 100 loops on budgeting with publisher
sampling; some health replies change language or invent symptoms. At update 200,
the greedy budget probe loops and breathing/hospital replies merely echo the
request. Lower training loss does not resolve these failures.

An explicitly opt-in `gemma-thinking` diagnostic now compares the same base
and checkpoint 100 with the publisher's sampling parameters and native reasoning
template. Its budget is 1,536 tokens per pass instead of the existing 512, with
the same 150-second overall per-response stop and three-pass tool bound. Thus
comparison against non-thinking also changes token budget; it is not a pure
single-variable latency comparison. The four-call batch ceiling is 720 seconds.
Reasoning is parsed separately, not shown in chat, and retained under the
template's `reasoning` field only for an in-progress tool turn. Defaults and
the existing user-facing tester are unchanged.

Completed context/health calls, respectively:

- V5: `fc-01M25MP7H8HMYMGXH6CHVSEKQ2`, `fc-01M25MP7MDCESA80NQ89NFMYDV`.
- Base: `fc-01M25MP815MJ6F7F8GQEB3H323`, `fc-01M25MP80592TNMNG5GMK5W814`.

Poll existing receipts with `check_response_candidate.py status`, profile
`gemma-thinking`; for base use `--run-receipt tmp/native-understanding-v5/run-receipt.json`.
No proprietary response model participates. The adapter budget and breathing
cases exhaust their reasoning budget without a final answer; other health
wording remains poor. The base gives fuller replies but also contains serious
Twi wording errors. Reasoning mode is not accepted as a solution or enabled in
the user-facing playground. Successful calculator execution is preserved.

Sources: [Gemma's native thinking guidance](https://huggingface.co/google/gemma-4-31B-it#best-practices),
[Modal volume reload semantics](https://modal.com/docs/guide/volumes#volume-commits-and-reloads).

## Source Additions

[INJONGO](https://aclanthology.org/2025.acl-long.464/) contains native-speaker
utterances and intent/entity annotations. Pinned source repository:
`masakhane-io/masakhane-nlu`, revision
`1f8be590da6699aee3dc23de6f63e801e2352eff`, Apache-2.0.

Prepared source addition: 2,240 Twi and 1,157 English training examples;
320 official Twi development examples. The official test texts were read only
to exclude leakage, not used for checkpoint selection. 622 English training
rows also occur in the test set and are excluded. One overlapping entity example
in English test is quarantined. Native targets and source text remain unchanged.

Two source-format issues were handled explicitly:

- `start_byte` and `limit_byte` actually index Unicode characters in all inspected
  splits. Every extracted span must exactly reproduce the published target;
  original offset fields remain in source evidence.
- `example_id` restarts within each intent. The first prepared v1 had colliding
  IDs and was rejected by the mixture check before training. The valid v2 uses
  language + intent + original ID. The flawed v1 is retained for diagnosis, not
  training. Unique-ID assertions now run before writing the prepared artifacts.

`tmp/native-intent-v2/` is the valid prepared source. Do not use v1.
These are auxiliary annotation targets, not invented assistant replies or
authorization to perform the user's requested action. Intent class alone does
not capture negation, missing parameters, permissions or multi-turn corrections.

V5 also adds 1,200 unchanged Twi-to-English professional source pairs from the
existing language corpus, alongside the earlier 400 English-to-Twi examples.
Some source groups appear in both directions, so task views are not independent
conversations or speakers. It retains 944 Twi health replies, 225 short grounded
AfriQA targets, 2,840 English conversation/replay rows and 1,335 tool-call rows.
The earlier no-tool-with-schema contexts remain. Broad natural Twi conversation
and clinical correctness remain gaps. More task rows alone do not solve them.

The saved tokenization audit makes the imbalance clearer than row counts.
Of 318,481 available supervised tokens in Twi-labeled tasks, 219,775 (69.0%)
are narrow health replies, 84,669 are native intent/entity structures, 12,811
are translation targets, and only 1,226 are short grounded QA answers. There is
no general Twi conversational-response task in this mixture. These are available
dataset tokens, not a claim about exact sampled exposure during the 320 updates;
JSON keys in Twi-labeled tasks are also not native Twi words. This is an observed
supervision gap, not a proved single cause of the response regressions.

Next data work must add source-preserving, natural multi-turn Twi responses and
corrections, with independently checked meaning, rather than add copies of the
same health translations or relabel auxiliary annotations as conversation.
Stronger public foundation comparisons are separate from project fine-tuning.
No benchmark case should be recycled into training or accepted as an annotation
target simply because a larger model generated it.

## Reproduction And Evaluation

Run from the repo with the installed Modal Python environment:

```sh
python3 scripts/run_response_adaptation.py status --experiment native-v5
python3 scripts/check_response_candidate.py submit --variant response_v5 --checkpoint checkpoints/checkpoint-100 --profile publisher
python3 scripts/check_response_candidate.py status --variant response_v5 --checkpoint checkpoints/checkpoint-100 --profile publisher
python3 scripts/evaluate_semantic_pairs.py submit --run-receipt tmp/native-understanding-v5/run-receipt.json --checkpoint checkpoints/checkpoint-100
python3 scripts/evaluate_native_intents.py submit --checkpoint checkpoints/checkpoint-100
```

Each submission writes an exclusive receipt; use `status`, not another submit.
Use `--suite health` for the private four-case response check. Use the same
checkpoint and decoding when comparing base/adapter. Native intent scoring checks
strict JSON, intent accuracy and literal entity micro-F1; it does not score free
conversation. Raw source references stay on the client, not in model prompts.

Code: `prepare_native_intents.py`, `build_native_understanding_mix.py`,
`evaluate_semantic_pairs.py`, `evaluate_native_intents.py`. Deploy training only
with `GHA_RESPONSE_RECIPE=native-v5`; default remains V2. The private response
runtime recognizes V5 IDs and pinned checkpoint hashes. Nothing is automatically
enabled in the local model picker. Existing owner voice A, conversations and
reviews are unchanged. No HF publication, proprietary API calls, private audio
upload or production changes in this continuation.

V4 checkpoint 100 is backed up locally at
`tmp/semantic-response-v4/archive-100/checkpoint-100/`, SHA256
`63cbde5b213f756667a36249b9ed978c08c48ac733a4c9cfc41fae05d7971fd1`.
Later training retains only three checkpoints: preserve any selected V5 weights
before retention removes them. Download Modal directories into an existing
local directory, then verify the adapter hash.
