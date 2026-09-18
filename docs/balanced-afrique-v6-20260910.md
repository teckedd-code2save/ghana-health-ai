# Balanced Afrique Response V6

## Subsequent Private Audition, 2026-09-11

The user requested private activation after the completed evaluation below.
V6 is now an explicit local playground choice, with its verified adapter SHA;
this does not change the rejection decision or production. Voice A and saved
history/reviews were preserved. The user immediately exposed a fabricated,
looping greeting reply. Clean paired checks and the training-coverage audit
confirm that stopping gains are not reliable conversational quality.
See `docs/response-research-reset-20260911.md` and
`data/response-adaptation/greeting-v6-decision.json`. Statements below about
unchanged UI choices describe the earlier, pre-activation checkpoint.

## Completed Outcome

**Completed, not accepted for chat or health use.** All 400 updates finished in
1,706.17 seconds of training; 2,188.97 seconds total including evaluation. The
verified local adapter hash is
`b4aa2162ae27a4c6997ec9f88bc171ca2134c457922bb1483c460eff920be510`.
Weights, checkpoints, original results and exact run inputs are backed up locally
and on the private Modal volume. No further training job is running.

All 30 development replies and 12 new conversation turns terminate normally,
unlike the untrained CPT base. This is not semantic accuracy. Six of 48
source-validation replies still hit the token ceiling and five trigger the
eight-word repetition heuristic. Shorter repetition exists even when the flag
does not fire. The eight intent examples are all alarm examples, not a balanced
intent benchmark. No full post-training NLI/translation benchmark is claimed.

Useful results: English conversation, arithmetic, quantity correction, the
Ama/Kofi context, and English picnic memory. The Kofi answer is CORRECT for its
actual prompt; an initial commentary misjudgment was explicitly corrected.

Failures: phone-only and no-eggs follow-ups, school-fee meaning lost in the Twi
thank-you, vague sibling-health replies, unsupported illness speculation, and
unconnected delivery/price claims. The private app serving path was tested with
eight cases under each of greedy and the app's default sampling settings. It
corrects the negation that failed offline, but this is context sensitivity, not
stable correctness. Greedy invents a price. Default sampling gives medicine
suggestions without resolving missing age. Neither profile executes the
explicitly requested calculator call (default gets the amount right without it).

Decision and per-case evidence: `data/response-adaptation/balanced-v6-decision.json`.
The next work needs better native conversational response targets and grounded
action/uncertainty examples, not another replay of the same mixture. Existing
private model choices, voice A, saved reviews/history and production are unchanged.

53 focused tests passed (38 response, eight balanced-data/summary, five private
playground, two training-receipt tests), plus compilation and whitespace checks.
This is not a full Next.js build/test claim.

## Run Record

The user approved repair, one bounded training candidate, and real conversation
evaluation. Submitted once on 2026-09-10 at 22:21:42 UTC:

- Run: `afrique_v6_20260910T222142Z`
- Call: `fc-01M26PK73KPNMVXAHKC1H3VF8H`
- App: `ghana-balanced-afrique-v6`
- Receipt: `tmp/balanced-afrique-v6/run-receipt.json`
- Private volume: `ghana-health-understand-train`, `/afrique-response/<run>/`

This is an actual instruction adaptation of Afrique **9B**, not another
unchanged Gemma V5 run or the earlier Afrique **4B** V3. The 9B model showed a
translation gain but was NOT accepted as a capable chat model. No promises that
scaling alone fixes semantics. No proprietary answer fallback.

## Data Repair

8,681 training views from 7,464 distinct source records and 6,000 source groups;
1,032 validation examples. There are 2,332 multi-turn training views, largely
English. These counts are not unique speakers, native Twi conversations, or gold
human-reviewed examples.

| Training task | Rows |
| --- | ---: |
| English everyday conversation and replay | 2,842 |
| Original bilingual translation pairs, both directions | 2,400 |
| INJONGO explicit-schema intent/entities: Twi / English | 1,600 / 800 |
| Original AfriHealth Twi responses | 180 |
| Source-grounded AfriQA Twi questions | 225 |
| Read-only tool-call supervision | 600 |
| Selected native source-context QA and English follow-up | 17 / 17 |

Changes: use corrected INJONGO v3 task contracts; quarantine known source flags
and the rejected multi-hop response family; reduce health and tool-call weights;
retain English conversation and 682 direct-answer examples with tools available.
Parallel translations retain both original texts. One question per selected
source prevents inflating the count with paraphrases. The 17 reviewed native
source answers retain supplied context, rather than becoming unsupported current
facts or invented assistant biography. Original records/reviews are unchanged.

Selection is **agent-inspected research silver**, not human gold. No saved user
chats or private recordings are added. Broad natural multi-turn Twi conversation
data is still insufficient. The 71-source review set is not silently promoted.
Unselected and flagged translations remain accessible in the original review.
Source licenses/attribution stay attached; no public redistribution is implied.

## Training Checks

CPU preflight passed with all 8,681 / 1,032 examples tokenized without truncation.
Vocabulary IDs of the pinned existing Qwen chat template equal the 9B foundation
vocabulary. A real toy backward pass, adapter save and reload passed. Training
uses native assistant-only targets and native tool serialization.

Twi-labelled health target tokens: 52,327 / 165,740 = 31.57%. Note that the
denominator includes JSON intent labels, so this is NOT the fraction of fluent
Twi dialogue. Native target-token breadth still needs work.

One H100, at most 400 updates, effective batch 16, LR 3e-5, LoRA rank 16 / alpha
32 on linear layers, BF16, max 1,536 tokens. At most 6,400 training views in this
run, less than one epoch. Soft training stop at 3,900 seconds from function start;
hard total timeout 5,400 seconds including evaluation. Save every 100 steps;
keep three adapter checkpoints and final adapter. No automatic retry or GPU pool.

Manifest SHA256: `7fccbca50a45bd8bfa4daf3c55d1111e908bc503c9f1533d814451d42897e6ed`.
Full pinned identities and row-level provenance: local corpus manifest.

## Evaluation And Continuity

Compare original base and adapter on the saved 30 development prompts and 48
source-validation prompts. Also run six new two-turn conversations, feeding
each model its actual previous answer. Evaluation targets/rubrics do not enter
training or generation prompts. Greedy decoding, 384-token maximum.

The new sibling-health diagnostic says `Menim` (I know), not `Minnim` (I do not
know). Judge the actual prompt, not its misleading "uncertain" case name; no
specific illness or age is given. This wording note predates result inspection.

The untrained CPT base is not a strong assistant baseline. Improved termination
or lower loss is not a claim of better semantics. Inspect Twi meaning, English
retention, language switching, multi-turn corrections, and inappropriate tool
calls before offering a private candidate. Never relabel a failed answer as a
success because the text is fluent or the function finished.

Commands:

```sh
python3 scripts/run_response_adaptation.py status --experiment balanced-v6
mkdir -p tmp/balanced-afrique-v6/completed
modal volume get ghana-health-understand-train /afrique-response/afrique_v6_20260910T222142Z tmp/balanced-afrique-v6/completed
python3 scripts/summarize_balanced_afrique.py tmp/balanced-afrique-v6/completed/afrique_v6_20260910T222142Z
```

Do NOT resubmit this completed run. The download folder must exist before using
Modal's recursive downloader; otherwise its path handling can fail. The exact
run input archive is 3,563,521 bytes, SHA256
`f5a97304c5f478ef7fdbd23edfebfb08305d60d3351c504d9fc31fcfdfed9fbc`.
V6 private inference support is deployed and checked, but it has NOT been
activated as a UI model choice. No public Hugging Face upload.
