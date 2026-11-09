# R5 Training and Inference Handoff

Issue #59, branch `codex/59-r5-meaning`. These are contained R5 copies of the
verified R4 infrastructure. Data preparation belongs to the main task; the
existing R4 workbench belongs to the UI delegate. No training, remote API calls,
deployment, inference, or production promotion was performed for this handoff.

## Training Contract

`scripts/run_understanding_r5.py` retains `verify_pack(folder)` and its
`(manifest, rows, stages)` return value. R4 pack formats remain unchanged:

- `manifest.json` and `PREPARED.json` seal every artifact. Set `version` to
  `understanding-adaptation-r5` and `initialization` to the runner constant.
- Preserve `rows.jsonl`, `source-witnesses.jsonl`, `stages.json`,
  `evaluation-manifest.json`, `overlap-audit.json`, `source-manifests.json`,
  admitted dataset manifests/records, and the four pinned tokenizer assets.
- Every packed row, including unscheduled rows, needs a matching witness and
  `dataset_ref` containing `dataset_id`, `dataset_version`, `record_id`,
  `source_id`, and `record_sha256`. References resolve to admitted, versioned,
  pack-local training records. Chat views are exact history prefixes.
- One `responses` stage contains integer row indexes in `schedule`, optional
  `primary_collections`, and optional sorted `snapshot_after_updates` boundaries.
  Preserve primary multi-turn chat plus retention coverage. Stage accounting is
  recomputed; supplied counts must match. No silent truncation or resampling.
- The current bound is **10,000 scheduled exposures**, not 10,000 unique records,
  with at most four exposures per row. Await actual admitted counts before
  increasing it. A larger finite bound needs a code change, fresh package, and
  new matching smoke evidence; checks must not be bypassed.
- Learning rate is pinned to `1e-5`. Other R4 optimization fields and bounds are
  unchanged. H100, rank-16 pinned adapter, bf16 base/fp32 adapters, seed 42,
  no packing, no truncation, complete native histories, final-assistant masking,
  and EOS checks are retained.
- `verify` checks the sealed structural/data contract offline. The worker also
  performs exact native re-encoding of every packed row before training.

Initializer:
`/artifacts/understanding-adaptation-r4/understanding_20261109T103659Z/responses-adapter`

Weights SHA-256:
`1c18338e732f7461901403f5a50c876d64712c21285cf24b2f78a86530d87aca`

All source pins were verified against original-project
`tmp/understanding-r4-20261107/train-v1/completed` and `readback-v1`.
The runner also pins the adapter config, status, execution, exports, and
consumption receipt. New R5 runs start fresh optimizer/scheduler state; only
explicit R5 resume restores full optimizer, scheduler, RNG, cursor, parameters,
and buffers.

## Offline Usage

Run from this worktree. `PACK` is the path chosen by the main data builder; the
example path below is not a claim that the pack exists.

```sh
ORIGINAL=/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai
PRIVATE="$ORIGINAL/tmp/understanding-r5-20261109"
PACK="$PRIVATE/pack-v1"
python3 scripts/run_understanding_r5.py verify --pack "$PACK"
python3 scripts/run_understanding_r5.py prepare --pack "$PACK" \
  --source "$ORIGINAL" --out "$PRIVATE/smoke-v1/execution" --mode smoke
```

Preparation does not submit work. Only after main authorizes GPU work, `launch`
requires `--execution`, a new `--receipt`, `--profile`, and `--authorize-gpu`.
Its submission receipt is written before submission and cannot be reused.
`status --receipt ...` polls the saved call; `collect --receipt ... --out ...`
collects identity-bound metadata into a fresh directory. These are remote actions.

After a completed R5 smoke has been collected, training preparation is local:

```sh
python3 scripts/run_understanding_r5.py prepare --pack "$PACK" \
  --source "$ORIGINAL" --out "$PRIVATE/train-v1/execution" --mode train \
  --smoke-directory "$PRIVATE/smoke-v1/completed"
```

Smoke is bounded to 900 seconds; training has a 10,500-second soft budget and a
10,800-second hard timeout. One H100 worker, zero automatic retries. A graceful
`paused_budget` result is resumable using the identical training execution
package, a fresh receipt, and `--resume-run <previous-run-id>` on a separately
authorized launch. Both launcher and worker check the claimed previous call and
terminal identity. A running, failed, ambiguous, or mismatched call cannot resume.
Consumption receipts count the committed cumulative schedule prefix, not archive
inventory or duplicate resumed work.

## Training Code Readiness

The training `CODE` list contains exactly these files. This is the readiness
snapshot before any GPU smoke, subject to a deliberate bound change after the
main task supplies actual counts:

| File | SHA-256 |
| --- | --- |
| `scripts/run_understanding_r5.py` | `0f40d689c05217649116d74565528896b21718f67945e259ee5d793cb64d4183` |
| `modal/train/train_understanding_r5.py` | `d7b58086a85a8ba2699af97fd9b75b1cbab275bea2b90de9d247b7350c2422e2` |

All frozen/reused source hashes, pack hashes, native encoding, initializer,
real-weight backward, and bitwise full-state recovery proof must match smoke.
Changing training code or data invalidates smoke; do not edit a sealed package.

## Private Inference

New files: `scripts/r5_inference.py`,
`scripts/collect_understanding_r5_inference.py`,
`scripts/prepare_understanding_r5_inference.py`, and
`modal/research_understanding_r5_demo.py`.

The new bundle exposes `base`, `r4_response`, and `r5_response`; no R3 variant.
The base path disables adapters, and R4 is pinned to the final initializer bytes.
An optional `r5_pass1` requires its own committed in-run snapshot proof. The
original R4 bundle is pinned at
`4e44fc697e44a3aaa43cfb89ef2e17ff97f240643e60b869d7975c66cce238c6`.
No existing bundle is modified. Pinned R3-named streaming/guard helpers are reused
unchanged as implementation dependencies, not as a selectable model.

After actual completion and separate authorization for readback, the collector's
CLI accepts `--receipt`, `--terminal`, and a fresh `--out`. It reads only that
completed R5 run, verifies complete adapter bytes and checkpoint/consumption
proofs, and rechecks terminal status. It does not launch, deploy, mutate the
volume, or download training conversations or checkpoint state binaries.

With that readback available, bundle preparation is offline:

```sh
python3 scripts/prepare_understanding_r5_inference.py \
  --r4-bundle "$ORIGINAL/tmp/understanding-r4-20261107/inferencebundle-v1" \
  --receipt "$PRIVATE/readback-v1/receipt.json" \
  --terminal "$PRIVATE/readback-v1/terminal.json" \
  --readback-root "$PRIVATE/readback-v1" \
  --helpers-root /Users/welcome/.codex/worktrees/r2-train/ghana-health-ai \
  --out "$PRIVATE/inferencebundle-v1"
```

The private actor uses `GHA_R5_DEMO_BUNDLE` and `GHA_R5_DEMO_SHA`, class
`UnderstandingR5Demo`, app `ghana-understanding-r5-demo-private`. It has read-only
volumes, one H100 container, zero retries, and no public HTTP endpoint. Imports
do not deploy or start inference. No real R5 bundle or model comparison exists
until training completes; tests use synthetic R5 receipts and bytes.

## UI Integration Plan

The UI delegate can adapt the robust workbench later with a new R5 entry point
and separate private database/output root. Parameterize bundle path/hash,
runtime loader, app/class, display labels, and the base/R4/R5 variant list. Do not
migrate, rewrite, or reuse the current R4 store as the new session store.

Keep independent per-variant histories, exact submitted inputs, explicit context
recovery, request/call/model/checksum identities, matched decoding and guards,
raw and guarded outputs, and failed/blocked slots. Use the fresh held-out suite
for actual base/R4/R5 comparison. Completion is not semantic certification.

## Local Tests

```sh
PYTHONPATH=scripts python3 -m unittest \
  test_run_understanding_r5 test_collect_understanding_r5_inference \
  test_prepare_understanding_r5_inference test_understanding_r5_demo
```

Tests include native-mask/source/consumption invariants, R4-format compatibility,
source readback pins, fake launch/call-ID guards, resume gating, readback
tampering, sealed packaging, true base selection, and raw/guarded streaming.
Integration fixtures need the local original R4 bundle and pinned helper source;
they skip when those private dependencies are absent. Generated packages,
readbacks, weights, stores, and results stay private under `tmp/`, outside Git.
