# R7 Launch and Continuation

Current state and measured results live in `R7-STATUS.md`. This runbook describes
operations; a command listed here is not evidence that it has run.

## Working Locations

- Worktree: `/Users/welcome/.codex/worktrees/r4-train/ghana-health-ai`.
- Branch: `codex/61-r7-data-training`, issue #61.
- Private local artifacts: `tmp/r7-20261117/` in that worktree.
- Modal profile: `createdliving1000`, environment `main`.
- Training volume: `ghana-health-understand-train`.
- R7 runs: `/understanding-adaptation-r7/<run_id>` on that volume, mounted
  as `/artifacts/understanding-adaptation-r7/<run_id>` inside workers.
- Private backups: `/research-handoffs/twi-understanding/2026-11-17-r7/`.

Preserve the original dirty project worktree. Never overwrite a sealed release,
old evaluation, checkpoint or receipt. Use a new directory for each readback of
changing progress. Do not create local secret files or call OpenAI: its current
project has no available credits, and the user selected Modal.

## Training Contract

R7 continues the **final R6 response adapter** on the pinned AfriqueQwen 9B
foundation. The base is frozen; 43,278,336 existing LoRA parameters are trainable.
It does not introduce vocabulary or change the total architecture. Exact model,
revision, adapter/config hashes and R6 completion metadata are enforced in
`scripts/run_understanding_r7.py`.

The planned schedule makes one shuffled pass over every retained registered
view. Full histories are preserved; no packing or truncation is allowed. Only
the selected final assistant target and native end marker receive loss. Language
replay uses causal language modeling. Counterfactual context and invalid tool
probe calls are not targets. The optimizer uses target-token-weighted causal
cross-entropy, accumulation 16, learning rate 0.00001 and seed 42.

## Before GPU Submission

1. Finish scoped replay admission and keep its decisions separate from whole-row
   semantic, native-speaker and clinical certification.
2. Assemble original alignment, source-bound follow-ups, replay and executed
   tools using `scripts/prepare_understanding_r7.py`.
3. Supply all inherited/fresh behavior, observed-regression, tool-development
   and frozen public-benchmark protections to the assembler.
4. Run `run_understanding_r7.py verify --pack <pack>`: it recomputes checksums,
   source identity, full-history masks, overlap, finite exposures and native
   encoding. An inventory count or a hand-written clearance is insufficient.
5. Run the focused/full R7 tests. Freeze the source revision and document exact
   source records, derived views and target-token totals separately.

## Real-Weight Smoke and Training

Use the unchanged proven core at:
`/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai/tmp/understanding-r6-20261113/execution-train/frozen`.

`scripts/run_understanding_r7.py prepare --mode smoke --pack <pack> --source
<frozen-core> --out <new-smoke-package>` builds a hash-bound package. Launch with
`launch --execution <package> --receipt <new-receipt> --authorize-gpu`.
The receipt is persisted before detached submission. Poll with `status --receipt
<receipt>` and collect with `collect --receipt <receipt> --out <new-readback>`.

The smoke loads and verifies real R6 tensors, performs backward passes on the
longest relevant history/tool/text types, and compares uninterrupted training
with a checkpoint-resumed run bit-for-bit. It is not a research model release.

Only after that smoke succeeds, prepare `--mode train` with the exact same pack
and `--smoke-directory <completed-smoke-readback>`. Launch once using a separate
receipt. Do not change the sealed data/code between smoke and training.

Training checkpoints are committed every 25 updates. `consumption.json` records
only committed row visits and tokens, by dataset and collection. A running status
does not prove that all examples have been consumed.

The worker has an eight-hour bound and a safe earlier pause. If it returns
`paused_budget`, first verify the prior call is terminal, then use the same
training package with `launch --resume-run <prior-run-id>` and a **new receipt**.
The launcher rejects a duplicate or nonterminal prior call. It restores model,
optimizer, scheduler, random state and exact schedule position. Never restart
from R6 and call that a continuation.

## Evaluation and Local Test

- Preserve the existing R6 117-output behavior evidence. Use the same frozen
  prompts and decoding for R7, retaining distinct model-generated rollout histories.
- Keep fixed-history comparisons, independent rollouts, reused regressions,
  public-task benchmarks and executed-tool outcomes separate.
- Public multiple-choice scoring uses conditional answer-token log probabilities;
  SIB classification requires a valid label. Report denominators and invalids.
- Exact-answer correctness does not certify natural Twi or medical safety.
- The local comparison places R7 first and R6 second, with separate histories,
  the same response guard/decoding setting and inspectable checkpoint identity.
- Do not substitute another model's answer, silently retry failures or hide
  regressions. Read back and hash the completed adapter before serving it.

Back up progress with `scripts/backup_r7_progress.py --upload`; retain the receipt
whose `remote_verified` field is true. The archive is private and read back from
Modal to verify its checksum. Public dataset release and production promotion
are separate decisions.
