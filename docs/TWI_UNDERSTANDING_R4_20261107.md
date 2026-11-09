# R4: Conversation Continuation

Issue: #57. User approved the four-stage plan on 2026-11-07.

## Execution State

| Stage | State | Evidence / next gate |
| --- | --- | --- |
| 1. Audit R3 supervision and execution | Complete | All 1,449 rows match native encoding; 867 chat rows have masked context and one supervised final EOS. No blanket finding that repeated training text caused the model loops. |
| 2. Correct and broaden conversation data | Final pack v3 prepared | 35 source frames considered, 33 admitted, two held. Three additional kinship findings resolved in part-A v2; no native-human certification claimed. |
| 3. Continue R3 into R4 | Completed | `understanding_20261109T103659Z`, call `fc-01M6ZY42GWDKP5CYK7TCZ8B6VS`: 125/125 updates, 1,997 presentations and 161,773 effective target tokens committed. Final and update-63 adapters saved. |
| 4. Compare base, R3 and R4 | Local UI verified; saved evaluations running | Both adapters passed full readback. Private actor deployed; UI at `127.0.0.1:7868`. Raw and guarded runs are separate. No new final-test claims. |

R3 is the reference, not an editable dataset or replaceable checkpoint. Its final
adapter SHA256 is `5cc0537b21674231b640ae9109a08ffa6fab426c80ac1969705dfac9c9e6b874`.
The base and tokenizer stay unchanged. R4 training and a private local comparison
are authorized; a production rollout or public data upload is not part of this pass.

## Prepared Run: 9 November (Final Pack v3)

This is a focused continuation, not processing the entire corpus again. Every
training row resolves to a versioned source record in `pack-v3/datasets/`.

| Component | Distinct training views | What it contributes |
| --- | ---: | --- |
| R3 response replay | 229 | Preserve previously admitted response supervision |
| New checked response targets | 67 | 33 complete histories: roles, negation, time, correction, quantity, uncertainty and ordinary follow-ups |
| Synthetic user spelling variants | 59 | User-only lowercase epsilon to `3`; original assistant answers unchanged; no new source conversations |
| Twi language replay | 600 | Retain language patterns |
| Meaning replay | 535 | Retain bilingual/structured meaning |
| English replay | 152 | Retain English conversations |

There are 355 response views and 115 follow-up views (47 inherited, 34 new and
34 spelling variants). The 33 source conversations have two or three assistant
turns each; prefix views are not separate independent conversations. Eight unchanged
prefixes deduplicate from the 67 possible augmented views. Original English frames
are project-authored; new Twi derivatives are agent-translated and agent-checked,
not natural recorded conversations or native-human-certified data.

Prepared two-pass schedule: 1,997 presentations, 161,773 effective target tokens,
125 optimizer updates; a pass-one snapshot at update 63. Target-token proportions
are approximately 60% responses, 20% English, 10% meaning and 10% Twi language.
Continuation learning rate is `1e-5`, half the earlier R3 rate. This is a
conservative experimental choice, not an established optimum. Rank-16 LoRA and
the frozen base/tokenizer are unchanged. No new API requests were made in this pass.

The completed checkpoint and `train-v1/completed/consumption.json` now confirm
the prepared schedule was consumed in full: 1,997 presentations, 315,357 input
tokens and 161,773 effective target tokens. These are exposures, not new unique
examples. The saved restore proof verifies equivalent model, optimizer, scheduler,
RNG and position at update 63 before the second pass.

Final R4 weights SHA256:
`1c18338e732f7461901403f5a50c876d64712c21285cf24b2f78a86530d87aca`.
First-pass weights SHA256:
`f7ca7d04f0fe089100394c8f3b41f2b703ce3b39214bd4535d324b4de0cf3631`.
Both adapters passed SHA-256 verification of their complete local readback bytes.
The readback covers 24 files / 347,460,704 bytes and is recorded in
`readback-v1/READBACK.json`. This proves identity, not response quality.

## Testing R4 Locally

Open <http://127.0.0.1:7868/?__theme=light>. The three panes run the unchanged
base, R3 final response adapter, and R4 final response adapter. Each model keeps
its own history. Guarded mode is the default for all three; switching to raw mode
starts a new comparison. A stopped repetition is not counted as an EOS-complete
answer. No external response model rewrites or substitutes the outputs.

Browser verification completed on 9 November: streaming, second-turn history,
refresh recovery and mode isolation work. At desktop width 1280 the three columns
remain side by side; at mobile width 390 they stack without horizontal overflow.
Per-pane clear buttons are hidden so only New chat resets a comparison. The first
observed cold request took 82.76 seconds to first text; subsequent observed starts
were 1.19-3.34 seconds. Execution is serial and this is not a fair latency benchmark.

An initial two-turn, project-authored name-retention probe is mixed: R4 follows
the Twi-language request, but fails the subsequent name question that R3 answers.
The saved request proves R4 received the supplied name and its own prior answer.
This is one agent-observed development failure, not a corpus-wide metric. Its
local session is `f10cd0f8d3704b19a309337d19ce28b9` in the dedicated workbench store.
No human assessment has been recorded on the user's behalf.

Private inference bundle:
`4e44fc697e44a3aaa43cfb89ef2e17ff97f240643e60b869d7975c66cce238c6`.
Private Modal app: `ghana-understanding-r4-demo-private`.
The public service remains on R3; R4 has not been promoted.

## Saved Evaluation Runs

| Policy | Models | Scope | Receipt |
| --- | --- | --- | --- |
| Raw | Base, R3, R4 final, R4 first pass | 75 chat slots and 343 language-loss rows per model; 1,672 slots total | `eval-raw-v1/launch.json`, `r4_eval_20261109T114144Z` |
| Guarded | Base, R3, R4 final | 75 chat slots per model; 225 slots total | `eval-guarded-v1/launch.json`, `r4_eval_20261109T114213Z` |

Each 75-chat set contains 46 fixed-context cases and 29 turns across 10 independent
dialogue rollouts. Rollout histories belong to each model, so later-turn token
inputs intentionally differ. These are reused development cases, not a new blind
test. A durable result is not automatically a successful or semantically correct
answer. Full results and denominators will be reported after collection.

Raw execution SHA256:
`1bbd843e39dab784022401d53fa5e0a40315ddfb592a63496cbe8376b3d3a469`.
Guarded execution SHA256:
`d34e3b18bb1fe1b850cfd60dc7c8f641ec07f8527194693b6e78d4a1f6f48d49`.

The two held histories concern cupboard/storeroom and gate/outer-door meaning.
Source amendments explicitly remove unsupported conclusions about other windows
and umbrella ownership. Two synthetic scenarios explicitly specify maternal
uncle/aunt rather than claiming those narrower meanings came from broad English
kinship terms. The cousin phrase repeats the full relationship to avoid a mother-
versus-cousin ambiguity. Original English, original facts and review notes remain
available. Original and corrected wording protect all related split groups.

New original targets contain 1,879 target tokens per pass; spelling-variant views
add 1,721 target tokens but no new answers. The 229 R3 response views contain
44,961 target tokens per pass. This exposes the limited scale of the addition:
new/augmented material is about 7.4% of response-target tokens. The run tests a
bounded improvement; it cannot establish broad conversational mastery.

Pack v3 manifest SHA256:
`b71682b47512506f9621faba7370f9f2d941437acc44e4dbe6c2716c0bf50cc7`.
Eight identical training views are removed; 12 replay candidates are excluded by
the protected-content check. Remaining replay candidates are accounted for as
available but outside this run's token budget in `selection-ledger.jsonl`.

## Data Contract

Every training view must resolve to a versioned dataset record and source. Complete
histories remain intact. Source documents and existing annotations are immutable;
corrections are derived records. A machine screen is not a semantic pass, and agent
review is not native-human or clinical certification. Existing source holdouts stay
protected. Previously reported failures are development evidence, not final-test
evidence after targeted repair.

The source audit distinguishes full-history occurrences, deduplicated assistant
prefixes and actual consumed training views. These are different denominators.

## Selection and Evaluation

R4 targets useful responses, follow-ups, corrections, topic/language changes and
clarification. Retain Twi language, meaning and English replay. Set the mixture
from measured target tokens, not raw record totals. Inspect intermediate checkpoints
on development data before selecting a candidate.

Compare the base, preserved R3 and R4 under the same decoding contract. Fixed-context
tests and independent dialogue rollouts answer different questions and must remain
separate. Raw model behavior and guarded behavior are reported separately. Measure
meaning and usefulness with explicit judgments; repetition metrics are diagnostic,
not semantic or safety certification. Keep protected final evaluation separate.

## Costs and Durability

The original offline API repair ledger retains its cumulative USD 10 ceiling. At
start: USD 1.242120 recorded usage; USD 7.372715 conservatively reserved. Unused
reservations are not silently released. No additional spending cap was granted.
Infisical returned HTTP 502 at the start of this pass; secrets must remain in
protected stores or process memory, never committed or saved locally.

Local working evidence is under the original checkout's
`tmp/understanding-r4-20261107/`. A local file is not an off-device backup.
Remote checksum/readback receipts must be recorded before claiming durability.

First private backup is verified off-device: 93 artifacts in the Modal volume
`ghana-health-understand-train`, prefix
`/research-training-r4/6dfe6fa0b71165c65991b2a8b0f0212ccad38678ff16bd258e9a36e04963c5e1`.
Its local receipt is `tmp/understanding-r4-20261107/backup-pretrain-v1/storage.json`.
This preserves pack v2, admission evidence, original frames and first smoke
execution. It does not yet contain subsequent kinship corrections or final results.
The backup shares the training provider; it is not an independent-provider backup.

Final pack and corrections are backed up separately (101 artifacts), prefix
`/research-training-r4/ac8d5657d9f79bac7d7e4f12437f5b48166873216d272a2eedcca00daed79fd0`.
Full remote readback was verified at 2026-11-09 10:35 UTC. Local receipt:
`tmp/understanding-r4-20261107/backup-pretrain-v2/storage.json`.

Implementation: branch `codex/57-r4-conversation`, draft PR #58.
Training receipt: `tmp/understanding-r4-20261107/train-v1/launch.json`.
The Modal job is detached from this computer and checkpoints are on its private
volume. Poll the receipt; do not submit a duplicate job if the client disconnects.

Publishing correction: a dataset-specific materialization helper was briefly
included in the public branch. It contains authored examples and agent judgments,
not user recordings or secrets. It was moved into ignored private research storage
and removed from the branch's reachable history. This does not certify deletion
from third-party caches or GitHub's retained unreachable objects. Future public
changes must exclude example-specific review payloads as well as corpus blobs.
