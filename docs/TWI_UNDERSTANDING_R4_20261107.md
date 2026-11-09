# R4: Conversation Continuation

Issue: #57. User approved the four-stage plan on 2026-11-07.

## Execution State

| Stage | State | Evidence / next gate |
| --- | --- | --- |
| 1. Audit R3 supervision and execution | Complete | All 1,449 rows match native encoding; 867 chat rows have masked context and one supervised final EOS. No blanket finding that repeated training text caused the model loops. |
| 2. Correct and broaden conversation data | Final pack v3 prepared | 35 source frames considered, 33 admitted, two held. Three additional kinship findings resolved in part-A v2; no native-human certification claimed. |
| 3. Continue R3 into R4 | First pass committed; second pass running | `understanding_20261109T103659Z`, call `fc-01M6ZY42GWDKP5CYK7TCZ8B6VS`. At 10:52 UTC: 63/125 updates, 1,008 presentations, 80,842 effective target tokens. Final-pack smoke passed in 193 seconds. |
| 4. Compare base, R3 and R4 | Implementation tested; waiting for actual adapter | 139 tests passed. Separate model histories and equal guard. Saved development cases; no new final-test claims. |

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

The counters above are preparation, not proof of training consumption. Actual
consumption must come from the committed training checkpoint and receipt.

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
