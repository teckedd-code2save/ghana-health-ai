# R5 Official IFEval

Implemented: pinned source adapter, separate native generation module, private
bounded/resumable worker, offline official scorer, and local tests. **No model
generation, GPU launch, or real model benchmark scores have been produced.**
R5 training and existing inference/development evaluation code are unchanged.
This integration does not block training. Benchmark each completed checkpoint
with its own inference bundle, registered-data receipts, and source audit.

## Frozen Sources

- Official revision: `0eebfd403ba438bfd14cb70b9b24a7745c997083`.
- [Official README](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/instruction_following_eval/README.md).
- [Official scorer](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/instruction_following_eval/evaluation_lib.py).
- Full inventory: 541 unique prompts/keys, 834 instruction instances, 25 types.
- Input JSONL SHA-256: `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`.
- Source package manifest SHA-256: `35400682ad754f137519e3e4bbe1769ccd2b56e7a2a486781dc198338d3003cf`.

The adapter retains exact upstream bytes, Apache-2.0 LICENSE, README, source
headers, upstream tests, and SHA-256 pins for every upstream file. It also pins
NLTK English Punkt tables and their resource README/archive at nltk_data revision
`550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`. No downloaded source is edited or
committed. Source packages, outputs, and scorer environments stay in ignored
private storage. Scorer dependency versions, distribution RECORD hashes,
resource hashes, Python version, and seeds are included in scoring provenance.

## Generation Protocol

`r5-ifeval-generation-v1` is separate from the old 256-token contract:

| Setting | Value |
| --- | --- |
| Models | base, pinned final R4, receipt-bound final R5 |
| Input | One exact original user prompt; no added system/Twi instructions |
| Native context | 16,384 tokens |
| Output allowance | 8,192 tokens for every case/model |
| Decoding | Greedy, one beam, repetition penalty 1.0, thinking disabled |
| Seed | 17, reset per generation |
| Per-case wall limit | 240 seconds |
| Worker | One H100, zero retries, single-use container |
| Invocation budget | 6,900 soft / 7,200 hard seconds |
| Chunk | 24 new finalized slots by default; explicit bound 1-96 |

The actual largest explicit minimum is **1,200 words** (official key 3425);
the 8,192-token allowance provides about 6.82 tokens/word at that minimum, not a
guarantee of completion. The maximum sentence constraint is 100 and paragraph
constraint is 7. Native CPU preflight passed all 541 prompts with no truncation;
the longest prompt is 375 tokens. The pinned model config supports up to 262,144
positions; the new finite 16,384 limit remains within that native bound.

There are 1,623 generation slots in the full comparison. Chunking does not select
a subset or change the full denominator. Each case retains the full per-case
allowance: the runner pauses before starting a case if that allowance cannot fit
in the invocation budget. EOS, cap hits, time-limit stops, empty responses,
product-guard stops, exceptions, indeterminate and pending slots remain visible.
No hidden continuation, prompt rewriting, truncation, or automatic regeneration.

`raw` is the official-output generation condition. `product_guarded` uses the
same pinned repeat-block guard but has its own sealed execution, journal, and
scores. Official strict/loose scoring is a separate dimension, not raw/guarded.

## Local Setup And Preflight

These commands do not launch a GPU. The source fetch downloads public pinned
files only. The source and private scorer environment already exist locally at
the paths below; do not rerun creation over existing immutable directories.

```sh
cd /Users/welcome/.codex/worktrees/r4-train/ghana-health-ai
ROOT=/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai
PRIVATE="$ROOT/tmp/understanding-r5-20261109"
SOURCE="$PRIVATE/ifeval-source-v1"
SOURCE_SHA=35400682ad754f137519e3e4bbe1769ccd2b56e7a2a486781dc198338d3003cf
CPU_PY="$ROOT/tmp/notebook-env/bin/python"
SCORER_PY="$PRIVATE/ifeval-scorer-env-v1/bin/python"

# On a new machine/directory only:
# python3 scripts/r5_ifeval_source.py fetch --out "$SOURCE"
# python3 -m venv "$PRIVATE/ifeval-scorer-env-v1"
# "$SCORER_PY" -m pip install absl-py==2.3.1 langdetect==1.0.9 nltk==3.9.2 \
#   immutabledict==4.2.2 click==8.1.8 joblib==1.5.3 regex==2026.5.9 tqdm==4.67.1 six==1.17.0

python3 scripts/r5_ifeval_source.py verify --source "$SOURCE" --source-sha "$SOURCE_SHA"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$CPU_PY" \
  scripts/run_understanding_r5_ifeval.py preflight \
  --tokenizer-bundle "$ROOT/tmp/understanding-r2-20261105/eval-bundle-v2" \
  --source "$SOURCE" --source-sha "$SOURCE_SHA"
```

## Training And Overlap Gate

Preparation replays the completed checkpoint's inference proof, exact cumulative
consumption receipt, registered admitted dataset manifests, source record hashes,
and training overlap evidence. Archive inventory is not treated as consumption.
No IFEval row is admitted as training data. Main must additionally supply an
actual all-source audit for this full public suite, not reuse a blanket or
exclusion-only fresh-suite seal. Required audit JSON:

```json
{
  "version": "r5-evaluation-source-audit-v1",
  "training_manifest_sha256": "<actual completed R5 training-input manifest hash>",
  "evaluation_manifest_sha256": "35400682ad754f137519e3e4bbe1769ccd2b56e7a2a486781dc198338d3003cf",
  "evaluation_source_sha256": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
  "overlap_free": true,
  "checks": {
    "source_groups_disjoint": true,
    "exact_text_disjoint": true,
    "normalized_text_disjoint": true,
    "full_protection_coverage": true
  },
  "evidence": {"evidence/ifeval-training-audit.json": "<actual report SHA-256>"}
}
```

Evidence paths are relative to the audit, must exist, and are copied/hash-checked.
This tool validates supplied clearance; it does not fabricate an overlap audit
or establish independence from unknown base-model pretraining. If local overlap
is found, do not delete benchmark cases and still call the result full IFEval.
Resolve/disclose contamination before authorizing the run.

Main supplies actual completed-run values for `R5_BUNDLE`, `R5_BUNDLE_SHA`,
`TRAINING_INPUTS`, `CONSUMPTION`, `IFEVAL_AUDIT`, and a new `EXECUTION` directory:

```sh
"$CPU_PY" scripts/run_understanding_r5_ifeval.py prepare \
  --bundle "$R5_BUNDLE" --bundle-sha "$R5_BUNDLE_SHA" \
  --source "$SOURCE" --source-sha "$SOURCE_SHA" \
  --training-inputs "$TRAINING_INPUTS" --consumption "$CONSUMPTION" \
  --source-audit "$IFEVAL_AUDIT" --policy raw --out "$EXECUTION"
EXECUTION_SHA="$(jq -r .execution_sha256 "$EXECUTION/SEALED.json")"
python3 scripts/run_understanding_r5_ifeval.py verify \
  --execution "$EXECUTION" --execution-sha "$EXECUTION_SHA"
```

Prepare `product_guarded` separately with another output path. Never mix policies
or checkpoint generations in a journal or benchmark report.

## Main-Authorized Generation

Not run by this agent. Main authorizes only after R5 training. An append-only
shared Modal ledger prevents overlapping IFEval launches across runs/policies;
each prior call must be terminal before the next lease is reserved. Terminal
acknowledgements survive output expiry. Missing/ambiguous submission evidence
fails closed; it is not automatically cleared or resubmitted. This ledger does
not control unrelated training jobs. There is no automatic chunk-launch loop.

```sh
python3 scripts/run_understanding_r5_ifeval.py launch \
  --execution "$EXECUTION" --execution-sha "$EXECUTION_SHA" \
  --profile "$MODAL_PROFILE" --receipt "$NEW_RECEIPT" \
  --chunk-slots 24 --authorize-gpu
python3 scripts/run_understanding_r5_ifeval.py status --receipt "$NEW_RECEIPT"

# Only after the previous call is terminal; a new receipt for each chunk:
python3 scripts/run_understanding_r5_ifeval.py launch \
  --execution "$EXECUTION" --execution-sha "$EXECUTION_SHA" \
  --profile "$MODAL_PROFILE" --receipt "$NEXT_RECEIPT" \
  --previous-receipt "$NEW_RECEIPT" --chunk-slots 24 --authorize-gpu

# A new result directory for each terminal snapshot:
python3 scripts/run_understanding_r5_ifeval.py collect \
  --receipt "$LATEST_RECEIPT" --out "$RESULTS"
```

Each slot is durably marked before generation and committed afterward. Resume
checks prompt/protocol/checkpoint/native-output proofs and never reruns finalized
slots. A started request without a result becomes `indeterminate`, not an
automatic retry. A failed/indeterminate case remains in the 541/model denominator.

## Offline Official Scoring

The unmodified official `evaluation_lib` performs strict and loose checks on
the exact saved response. No custom replacement regexes, rewriting, or semantic
judge is used. Official loose-mode normalization happens inside upstream code.
Scoring uses the private pinned CPU environment, fixed seed/hashseed, local-only
NLTK resources, and a 10-second wall limit per case/mode. Errors are retained.

```sh
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 "$SCORER_PY" \
  scripts/score_understanding_r5_ifeval.py \
  --execution "$EXECUTION" --execution-sha "$EXECUTION_SHA" \
  --results "$RESULTS" --out "$SCORES" --seconds 1800 --max-records 1623

# Resume scoring the same immutable result snapshot only:
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 "$SCORER_PY" \
  scripts/score_understanding_r5_ifeval.py \
  --execution "$EXECUTION" --execution-sha "$EXECUTION_SHA" \
  --results "$RESULTS" --out "$SCORES" --seconds 1800 --max-records 1623 --resume
```

`scores.json` contains prompt-level and instruction-level strict/loose metrics
for each model, plus official instruction-type/category breakdowns. The full
denominators are 541 prompts and 834 instructions per model. Full accuracy fields
are null when scoring coverage is incomplete; observed fractions and full-
denominator lower bounds are clearly separate. Actual cap/time-limit outputs
are scored with the official functions while their incomplete state is retained.
No-response generation errors and scoring errors are unscored, never silently
dropped or replaced with fabricated responses. Subgroup prompt scores mean all
constraints on prompts containing that category, not category-only correctness.

`score-cases.json`, `generation-cases.json`, immutable journals, runtime receipts,
`scored-outputs-<variant>-<mode>.jsonl`, and file manifests retain case-level
evidence. A later generation snapshot needs a new score directory. Started CPU
scoring without a durable result becomes scoring-indeterminate rather than
silently retried. No-response errors can leave final coverage incomplete.

These are English-prompt verifiable instruction-following scores, not Twi
understanding, broad factual accuracy, native certification, or medical safety.
Do not pool them with retention NLL, Twi capability review, or product rollouts.
No public-leaderboard comparability is claimed for this disclosed protocol.

## Local Verification

```sh
PYTHONPATH=scripts python3 -m unittest test_r5_ifeval
PYTHONPATH=scripts PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 "$SCORER_PY" \
  -m unittest test_r5_ifeval.OfficialTests
PYTHONPATH="$SOURCE/upstream" NLTK_DATA="$SOURCE/nltk_data" \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 "$SCORER_PY" \
  -m unittest instruction_following_eval.instructions_test instruction_following_eval.instructions_util_test
```

Tests use synthetic replies, receipts, and mocked remote services. The full 541
official schemas/arguments are checked with synthetic empty replies, not model
responses. Native preflight loads only the tokenizer, never model weights.
