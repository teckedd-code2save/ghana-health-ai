# R5 Evaluation Runner

The R5 runner compares `base`, `r4_response`, and `r5_response` using the sealed
R5 inference bundle. It preserves the original R4 runner and evaluated bundles.
Preparation and verification are local; launch requires separate explicit GPU
authorization. No model execution is part of this implementation handoff.

## Two Separate Suites

The default suite is the unchanged development `evaluation-v2`, manifest
`413c6e3b7e5737ffa3360219202bfb4e8e7a86437eb0d84dc9e2c72385612a06`.
It remains `development_not_blind_final`: 46 fixed cases and 29 rollout turns,
225 chat slots across the default three models. Optional raw language loss now
requires both Twi343 and the original 44 English OASST validation conversations:
1,386 total slots (1,848 with the optional fourth checkpoint). Guarded chat and
raw loss must be separate executions. Earlier R4 actual results are not backfilled.

The fresh adapter consumes the private design owner's
`r5-fresh-evaluation-proposal-v1` schema, its `HELDOUT_SEAL.json`, and `audit.json`.
It snapshots the generic `scripts/test_r5_fresh_evaluation.py` validator and uses
its `validate(corpus)` output without rewriting stimuli, IDs, reviews, or
spelling-expansion rules. Fixed authored assistant context is preserved. Both
languages and synthetic spelling variants retain the original source group.
Fresh counts are derived from the materialized cases, never hardcoded.

The adapter verifies the corpus, validator, audit, seal, and canonical case
inventory together. All artifacts remain private, with a 0700 suite directory
and 0600 files. The adapter does not edit the source design, grant training
admission, authorize execution, certify meaning, or relabel the proposal as a
blind final test. Its scope is `fresh_proposal_not_blind_final`.

The separately produced admission-heldout package is **exclusion-only**, not an
executable suite. Do not pass its manifest hash as the evaluation hash.

## Offline Usage

Run from the R5 worktree. Set the paths and hashes to the supplied private source,
seal, completed inference bundle, training inputs, and consumption receipt.
Use new directories under the original project's ignored
`tmp/understanding-r5-20261109/`; never commit generated suites or results.
Use the existing local tokenizer environment for `prepare` and `preflight`:

```sh
cd /Users/welcome/.codex/worktrees/r4-train/ghana-health-ai
SOURCE_ROOT=/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai
CPU_PY="$SOURCE_ROOT/tmp/notebook-env/bin/python"
R2_PACK="$SOURCE_ROOT/tmp/understanding-r2-20261105/pack"
ENGLISH_VALIDATION="$SOURCE_ROOT/tmp/corpus-ready/twi-alignment-v3-20260929/english-retention/validation.jsonl"
DEVELOPMENT="$SOURCE_ROOT/tmp/understanding-r3-20261106/evaluation-v2"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$CPU_PY" \
  scripts/run_understanding_r5_evaluation.py preflight \
  --tokenizer-bundle "$SOURCE_ROOT/tmp/understanding-r2-20261105/eval-bundle-v2" \
  --evaluation "$DEVELOPMENT" --r2-pack "$R2_PACK" \
  --english-validation "$ENGLISH_VALIDATION"
```

After training, Main must supply the actual `R5_BUNDLE`, `TRAINING_INPUTS`, and
`CONSUMPTION` paths and choose new output directories. Obtain `R5_BUNDLE_SHA`
from `jq -r .manifest_sha256 "$R5_BUNDLE/SEALED.json"`. No completed R5 bundle
or executable fresh-suite hash is assumed while dataset work is ongoing.

```sh
python3 scripts/prepare_understanding_r5_evaluation.py prepare \
  --source "$FRESH_SOURCE" --source-sha "$FRESH_SOURCE_SHA" \
  --heldout-seal "$HELDOUT_SEAL" --heldout-seal-sha "$HELDOUT_SEAL_SHA" \
  --out "$FRESH_SUITE"

python3 scripts/prepare_understanding_r5_evaluation.py verify \
  --evaluation "$FRESH_SUITE" --evaluation-sha "$FRESH_SUITE_SHA"

"$CPU_PY" scripts/run_understanding_r5_evaluation.py prepare \
  --bundle "$R5_BUNDLE" --bundle-sha "$R5_BUNDLE_SHA" \
  --evaluation "$FRESH_SUITE" --evaluation-sha "$FRESH_SUITE_SHA" \
  --training-inputs "$TRAINING_INPUTS" --consumption "$CONSUMPTION" \
  --source-audit "$FULL_SOURCE_AUDIT" --policy raw --out "$FRESH_RAW_EXECUTION"
```

Repeat execution preparation with `--policy guarded` and a different output
directory. For development, pass the original development directory and omit
`--evaluation-sha`. To include both retention cohorts in the raw execution:

```sh
"$CPU_PY" scripts/run_understanding_r5_evaluation.py prepare \
  --bundle "$R5_BUNDLE" --bundle-sha "$R5_BUNDLE_SHA" \
  --evaluation "$DEVELOPMENT" \
  --training-inputs "$TRAINING_INPUTS" --consumption "$CONSUMPTION" \
  --language-loss --r2-pack "$R2_PACK" --english-validation "$ENGLISH_VALIDATION" \
  --policy raw --out "$DEVELOPMENT_RAW_EXECUTION"

"$CPU_PY" scripts/run_understanding_r5_evaluation.py prepare \
  --bundle "$R5_BUNDLE" --bundle-sha "$R5_BUNDLE_SHA" \
  --evaluation "$DEVELOPMENT" \
  --training-inputs "$TRAINING_INPUTS" --consumption "$CONSUMPTION" \
  --policy guarded --out "$DEVELOPMENT_GUARDED_EXECUTION"

for EXECUTION in "$DEVELOPMENT_RAW_EXECUTION" "$DEVELOPMENT_GUARDED_EXECUTION"; do
  "$CPU_PY" scripts/run_understanding_r5_evaluation.py verify \
    --execution "$EXECUTION" \
    --execution-sha "$(jq -r .execution_sha256 "$EXECUTION/SEALED.json")"
done
```

Missing English source arguments fail closed; there is no Twi-only fallback.
Fresh language-loss rows are not bound, so `--language-loss` fails closed there.

`run_understanding_r5_evaluation.py preflight --tokenizer-bundle ...
--evaluation ... --evaluation-sha ...` checks native context locally using the
original pinned R2 tokenizer bundle. It does not require R5 weights. Fixed and
first rollout prompts are checked immediately; later rollout prompts are checked
against actual independent histories before each generation. No truncation or
case replacement is allowed.
For a pre-training check of both retention cohorts, add `--r2-pack "$R2_PACK"`
and `--english-validation "$ENGLISH_VALIDATION"` to development `preflight`.
This uses the installed pinned tokenizer dependencies, no model weights or API.

`verify --execution ... --execution-sha ...` rechecks the sealed execution.
`launch`, `status`, and `collect` are future remote actions, not local checks.
Launch requires `--authorize-gpu`, a new receipt, and the sealed execution hash.
There are zero automatic retries, one H100 worker, a 6,900-second soft budget,
and a 7,200-second hard timeout. Resume uses `--previous-receipt`; completed slots
are replay-verified, not regenerated. Started slots without durable results are
indeterminate and are never silently retried.

Main-only commands **after separate authorization**, one sealed execution at a
time; these commands have not been run by the implementation agent:

```sh
python3 scripts/run_understanding_r5_evaluation.py launch \
  --execution "$EXECUTION" \
  --execution-sha "$(jq -r .execution_sha256 "$EXECUTION/SEALED.json")" \
  --receipt "$NEW_RECEIPT" --profile "$MODAL_PROFILE" --authorize-gpu
python3 scripts/run_understanding_r5_evaluation.py status --receipt "$NEW_RECEIPT"
python3 scripts/run_understanding_r5_evaluation.py collect \
  --receipt "$NEW_RECEIPT" --out "$NEW_RESULTS_DIRECTORY"
```

Every completed training checkpoint needs its own receipt-bound benchmark runs.
Keep development raw+retention, development guarded, fresh raw, and fresh guarded
as four separate executions/results. Fresh requires the executable adapter seal
and Main's actual clearance. This workflow does not automatically launch GPUs
or block current R5 training on future public-benchmark scaffolding.

## Full Source Audit Gate

Every execution preserves the training pack's hash-bound `overlap-audit.json`,
`source-manifests.json`, admitted dataset manifests, and exact committed
consumption receipt. Development must also match the training evaluation binding
(the original wrapper, its entry in an `evaluations` list, or Main's exact
`development` member). In Main's wrapper, `fresh_path` and
`fresh_manifest_sha256` remain exclusion-only metadata: neither substitutes for
the executable fresh suite hash or its separately supplied source clearance.

Fresh execution additionally requires `--source-audit` with this metadata
contract, produced by Main after actual full training/protection checks:

```json
{
  "version": "r5-evaluation-source-audit-v1",
  "training_manifest_sha256": "<completed training input manifest SHA-256>",
  "evaluation_manifest_sha256": "<fresh adapter manifest SHA-256>",
  "evaluation_source_sha256": "<original private proposal file SHA-256>",
  "overlap_free": true,
  "checks": {
    "source_groups_disjoint": true,
    "exact_text_disjoint": true,
    "normalized_text_disjoint": true,
    "full_protection_coverage": true
  },
  "evidence": {
    "evidence/full-admission-report.json": "<report file SHA-256>"
  }
}
```

Evidence paths are relative to the audit file, must remain under `evidence/`,
and must exist with the stated hashes. The runner copies and rechecks them. The
audit is a supplied clearance receipt, not an overlap computation by this runner.
Do not set these fields from a partial prior-data scan: the original design
audit and its unresolved limitations remain preserved separately. No clearance
receipt for actual R5 training has been fabricated by these tools.

## Retention Protocol

The English source chain is pinned R2 pack manifest -> source snapshot manifest
-> batch file manifest -> complete English validation file
`40a6012a7055efdbb79767e4ea27f8abb560057217f964968291caa39dc6f038`.
Only those 44 original conversations and minimal source metadata are copied;
no test rows, unrelated training payload, replacement cases, or fresh-suite
payload enter retention evaluation. The source proof is replayed during verify.

`twi_native_causal` preserves evaluation-v2's original-text encoding, no BOS,
one EOS, no chat template, and causal-shift targets. `english_oasst_final_assistant`
uses the pinned native complete-history encoder and final-assistant-only labels,
including EOS. Earlier assistant/user/system tokens remain context, not targets.
There is no truncation, history concatenation, gradients, or optimizer update.
English and Twi token losses are never pooled.
Local native-tokenizer preflight verified 343/343 Twi rows (9,315 predicted
tokens) and 44/44 English rows (8,534 final-assistant predicted tokens), with
zero encoding errors. These are per-checkpoint denominators, not model NLL or
accuracy measurements; no weights were loaded.

Native preflight reports every case, cohort counts, token/mask hashes, predicted
token denominators, and sanitized encoding errors; any failed context check
blocks sealing. Runtime failures remain slots, not exclusions or silent retries.
`summary.json` reports expected/measured/error/indeterminate/pending row counts,
planned and measured token denominators, summed NLL, token-weighted mean NLL,
and exp(mean NLL), separately per cohort and checkpoint. A partial mean explicitly
describes measured rows only; paired complete coverage is flagged separately.
`language-loss-cases.json` includes all cases and their error/status/proof fields,
but no protected text or token arrays. Completion is never reported as accuracy.

## Histories and Review

Raw/guarded runs and development/fresh suites have different execution hashes,
bindings, journals, and output roots. Each rollout case/model/policy maintains
only its own exact generated replies. Verified partial outputs retain their stop
reason; guard pauses are not EOS. Errors or indeterminate requests block only
their own descendants. Fixed cases use identical authored inputs across models.

Only `messages` or scripted `user_turns` enter model input. Facts, structured
claims, rubrics, source audits, and `output_languages` are review metadata, never
prompt instructions. Review queues preserve source group, case hash, language,
spelling status, and the correct one-based per-turn review. Mechanical counts,
pending/blocked slots, raw output, native token proof, and human-review-pending
states remain separate from any semantic-quality claim.

`benchmark.json` and `summary.json.benchmark` provide side-by-side base/R4/R5
denominators by category, task, input language, requested output language, cohort,
execution type, and category/task/language intersection. Categories come only
from authored `category`, `tags`, or `phenomena`; task labels use `task`,
`task_kind`, or `family`. Missing labels are `unclassified`, not inferred from
case names, prompts, output text, or source-group IDs. Historical metadata values
are preserved as supplied. Output-language labels are requests, not detected
language adherence or accuracy.

Each subgroup contains expected slots, unique cases, known source-group IDs and
counts, unclassified source-group case counts, finalized/verified counts, all
seven states (complete/incomplete/paused/error/blocked/indeterminate/pending),
finish reasons, and strict literal-check denominators. Source groups are counted
within each model, not multiplied by variants. Multiple category membership is
explicitly non-additive. `mechanical_attention_slots` identifies incomplete or
unresolved work, not semantic weaknesses. Strengths and next meaning improvements
remain pending human review; completion never becomes meaning accuracy.

## Local Tests

```sh
PYTHONPATH=scripts python3 -m unittest \
  test_prepare_understanding_r5_evaluation test_run_understanding_r5_evaluation \
  test_r5_retention test_r5_evaluation_reporting
```

Fresh tests use synthetic schema fixtures only, not the private proposal. Runner
integration tests use the existing local pinned helper/development bundles and
synthetic R5 completion receipts. No remote or model calls are made.
The two optional `test_r5_retention.NativeCPUChecks` tests use real CPU loss
arithmetic and the pinned local tokenizer, without loading a model. They can run
in the existing original repository's `tmp/notebook-env/bin/python` environment;
the default `python3` environment skips these when dependencies are absent.
