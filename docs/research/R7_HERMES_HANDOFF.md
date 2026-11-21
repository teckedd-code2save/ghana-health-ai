# R7: Hermes Continuation Handoff

## Latest Handoff: 21 November 2026

The no-paid-work restriction still applies. All 49 approved Ghana Health AI and
akan-speech Modal deployments have now been stopped and verified; unrelated
deployments were not changed. Private receipt: `tmp/cost-hold-20261121/app-shutdown.json`.
Do not reactivate old endpoint or demo commands. Storage still needs separate
attention; a new exact cache-only proposal is pending, not deletion permission.

Priority: finish **private R6 and R7 weight backups on HF** using
[R6_R7_HF_BACKUPS.md](R6_R7_HF_BACKUPS.md). Packages are locally verified at
`/Users/welcome/Documents/Research-Exports/ghana-health-ai/hf-backups-20261121-v2/`.
The owner is signing in locally; no HF write credential was available at the last
check. Do not claim uploads succeeded until both exact revisions and readback
receipts exist. Both original adapters and R7 recovery remain in the existing
private archive and on Modal. Do not upload private corpus/billing archives.

The owner confirmed unmetered internet. The exact public base download resumed
using HTTP fallback after Xet stalled. Inspect the active downloader and partial
file before resuming; never duplicate it or substitute a different base. Actual
local model loading and R7/R6 comparison remain pending. No R7 gain is claimed.
The durable local downloader has `download-base-status.json` and
`download-base-20261121.log` beside the private archive. It performs one bounded
public download and verifies the full weight hash; it cannot launch cloud compute.
Inspect the recorded PID/status before starting another download.

## Superseding Cost Restriction: 18 November 2026

Read `docs/research/COST_CONTROL.md` before the earlier continuation instructions.
The user reported a billing problem and authorized a US$0 monthly out-of-pocket
Modal spend limit, which is now saved. Do not launch evaluations, restart jobs,
deploy paid inference or add large caches without renewed budget authorization.
Storage still accrues charges; do not delete anything without approval.

Read-only reconciliation found the existing R7 training call **completed**, with
2,413 updates and all 38,607 scheduled views consumed. The reported final adapter
SHA256 is `f6e6b484ca404146016eefabd170738e66791880775238e435e31577300e57a7`;
this has now been independently confirmed by downloading and hashing the full
adapter locally. It is not evidence of improved quality. Local result:
`tmp/r7-20261117/train-v1.receipt.result.json`. R7 evaluations remain pending.
No active app containers were reported in the workspace check at this audit.
The earlier submitted/running descriptions below are historical checkpoints.

### Local Export and Next Authorized Action

- Full R6/R7 adapters and 16 verified evidence files:
  `tmp/r7-20261118-local-export/` (373,855,185 verified bytes before recovery-state export).
- Offline inference bundle: `tmp/r7-20261118-local-bundle/`, SHA256
  `9162b42765a5d2f93b4d1f642495093e753570bb16b023fac78354a42475c614`.
- Final full-state checkpoint is independently downloaded and verified too:
  520,958,547 bytes, SHA256
  `2fb8e9310d2c889316ae6a066efb9d18b863edd305d13a1115db1cfbb0221a23`.
  See `recovery-readback.json` inside the local export. No resume was launched.
- Verified private local recovery archive, outside the worktree:
  `/Users/welcome/Documents/Research-Exports/ghana-health-ai/r7-offline-recovery-20261118.tar.gz`.
  SHA256 `7fc173660b9b82cd77efa7e8568c212e7bc97580f260b7e040c5ec72b3ef5fd2`.
  Includes frozen training inputs/code and the offline inference bundle, not the
  public base weights or entire research corpus. Read the adjacent README.
  Do not publish this archive; no new Modal backup was uploaded.
- Exporter `scripts/prepare_understanding_r7_inference.py collect` now downloads
  weights through Volume reads and hashes locally; it no longer launches a paid
  CPU sandbox. Check egress allowance before future downloads.
- Approved cleanup completed: exactly five public-cache directories, 533.425 GiB.
  Private evidence: `tmp/modal-storage-audit-20261118/cleanup-receipt.json`.
  All retained cache metadata and the five protected research volumes matched
  their pre-cleanup inventories. No corpus, trained weights or recovery files
  were deleted. No further deletion is authorized.
- Local MLX R7/R6 adapters are converted and checked; the isolated Mac runtime
  is installed. Pinned 17.9 GB base download remains incomplete. Continue from
  [R7_LOCAL_MAC.md](R7_LOCAL_MAC.md), not a new Modal launch. Confirm an unmetered
  connection before resuming a large download. Full local inference is unverified.
  Do not deploy the Modal bundle or resume GPU work under the current restriction.
- Preserve R6/R7 independent histories, identical decoding/guards and raw outputs.
  Quantized-local measurements must be labeled separately from BF16 benchmarks.

Updated 18 November 2026 UTC. Issue #61. Read this before starting another job.
The latest cost restriction supersedes earlier permission to continue on Modal.
Do not call OpenAI; its project was reported out of credits.

## Goal and Current Boundary

Finish the already-sealed R7 continuation from R6, evaluate actual improvements
and regressions, and serve R7 against R6 in a private local comparison UI.
Keep each model's own conversation history and identical decoding/guard settings.
Do not substitute a stronger model's answers or restart corpus exploration.
No public dataset upload, production promotion or ASR/TTS changes in this pass.

**R7 full training has completed.** The real-weight smoke and full training
receipts are preserved. Evaluation remains pending. Do not prepare or launch a
duplicate training run. The submission section below records the original job.

## Historical Training Submission

- Receipt: `tmp/r7-20261117/train-v1.receipt.json`.
- Run: `understanding_20261118T010701Z`.
- Call: `fc-01M7P3443CKHMZGBK47M0V2CCG`.
- App: `ap-8NwjfRswPmhUIOSMIPlkzu`.
- Package: `tmp/r7-20261117/execution-train-v1/`.
- Execution SHA: `cd4f6824907a179552b26a575fa97a1807a4b3540cb6ede8513b3f27df5e1aa2`.
- Remote path: `/understanding-adaptation-r7/understanding_20261118T010701Z`.

```bash
/Users/welcome/miniconda3/bin/python scripts/run_understanding_r7.py status --receipt tmp/r7-20261117/train-v1.receipt.json --wait 60
/Users/welcome/miniconda3/bin/python scripts/run_understanding_r7.py collect --receipt tmp/r7-20261117/train-v1.receipt.json --out tmp/r7-20261117/train-readback-hermes-01
```

The launcher exited successfully with a real call ID. Its final app-log timeout
was not a submission failure. Verify remote startup, committed consumption and
completion; submission alone does not prove optimizer progress. Training is
detached from this Codex conversation and local launcher.

## Exact Locations

| Item | Location |
| --- | --- |
| Active worktree | `/Users/welcome/.codex/worktrees/r4-train/ghana-health-ai` |
| Branch | `codex/61-r7-data-training` on `https://github.com/teckedd-code2save/ghana-health-ai` |
| Committed implementation | `de2ecde`; submission documentation `a3af959` |
| Data and receipts | `tmp/r7-20261117/` inside the active worktree |
| Current execution map | `R7-STATUS.md` |
| Dataset explanation | `docs/research/R7_DATASET_CARD.md`, `R7_SOURCE_COVERAGE.md` |
| Operational instructions | `docs/research/R7_RUNBOOK.md`, `R7_EVALUATION_CONTRACT.md` |
| Python for preparation | `/Users/welcome/miniconda3/bin/python` |
| Python for Modal launches | `tmp/r7-20261117/baseline-launch-py311/bin/python` (Python 3.11) |
| Modal | profile `createdliving1000`, environment `main` |
| Artifact volume | `ghana-health-understand-train` |
| Model cache volume | `ghana-health-understanding-hf-cache` |
| Existing R6 UI | `http://127.0.0.1:7870/?__theme=light` |

The original `/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai`
worktree is dirty and must not be overwritten. The older Hermes tree at
`/Users/welcome/ghana-health-twi-r6` is not the active R7 worktree. No credential
belongs in this document, a command argument, Git or a local environment file.

## Verified Data, Not an Inventory Estimate

Sealed pack: `tmp/r7-20261117/pack-v1/`.
Manifest SHA256: `999d53cdbc79f4d0268edf4e2f9ec270756155f3341c81069deec82ad8243d53`.

| Collection | Retained training views | Effective target tokens |
| --- | ---: | ---: |
| Original bilingual alignment and meaning | 31,902 | 603,289 |
| Direct responses | 2,644 | 735,911 |
| English retention | 2,554 | 466,685 |
| Language replay | 1,187 | 31,856 |
| Executed English tools | 320 | 12,140 |
| Total | **38,607** | **1,849,881** |

36,966 admitted source records produced 39,462 candidate views; 855 exact
within-class duplicates were removed. There are 17,856 distinct source IDs and
15,365 alias-connected groups, not 38,607 independent conversations. Total input
tokens: 6,178,465. Every view passed native re-encoding, source/hash, mask and
declared holdout checks. The public protection's group-field mismatch has a
separate full-witness supplementary audit with zero intersections; it did not
change the sealed pack. See `pack-profile-v1/REPORT.md` for coverage limitations.

Inputs: `alignment-admission-v1`, `alignment-followups-v2`,
`replay-admission-v2`, `tool-admission-v2`. Preserve these immutable releases.
Follow-ups reuse original Twi targets with English synthetic instruction
scaffolding; do not call them natural Twi conversations. Tool examples use real
CPU sandbox results and a fictional four-product catalogue, not real shopping.
New medical expansion and bulk generated Twi dialogue remain deferred: the
Modal teachers failed meaning controls. Their rejected outputs were not trained.
Inherited health examples are research data, not clinically certified.

**466 integrated tests passed** before submission. This proves infrastructure
checks, not semantic quality or model improvement. No R7 evaluation result exists yet.

## Supporting Jobs and Evidence

### R7 Smoke

**Completed and passed** in 627 seconds. Final readback:
`tmp/r7-20261117/smoke-final-readback/`. All 38,607 rows were native-reencoded;
real backward passes and bitwise-equivalent resumed state passed.

- Receipt: `tmp/r7-20261117/smoke-v1.receipt.json`.
- Run: `understanding_20261118T004044Z`.
- Call: `fc-01M7P23T89EHSKVG0VMC6HXMKR`.
- App: `ap-4Lh0d9CcZ6vgkbCfNrSYNl`.
- Execution: `tmp/r7-20261117/execution-smoke-v1/`.
- Execution SHA: `d5bdc11126ffab4261e89cee4283f95b5c0e7810a858949c050fd35d80fd920d`.
- Remote path: `/understanding-adaptation-r7/understanding_20261118T004044Z`.
- Initial package upload took several minutes; the launch exited with a real
  call ID. The final app-log timeout did not mean submission failed.

From the active worktree:

```bash
/Users/welcome/miniconda3/bin/python scripts/run_understanding_r7.py status --receipt tmp/r7-20261117/smoke-v1.receipt.json --wait 60
/Users/welcome/miniconda3/bin/python scripts/run_understanding_r7.py collect --receipt tmp/r7-20261117/smoke-v1.receipt.json --out tmp/r7-20261117/smoke-final-readback
```

Use a **new readback directory** each time progress changes. A missing artifact
can mean startup is still pending; inspect the call before resubmitting. The smoke
must prove actual backward passes and bitwise-equivalent full-state recovery.

### R6 Tool Baseline

- Receipt: `tmp/r7-20261117/tool-r6-v1.receipt.json`.
- Call: `fc-01M7P1XC5TYKMX9FCCW4248TKQ`.
- App: `ap-XBGD6tfe7WZ8MSFsePToQh`.
- Run: `r7-tool-r6-20261118-v1`.
- Package: `tmp/r7-20261117/tool-r6-execution-v1/`.
- Execution SHA: `7804c2c42047ba5e52187e240634586e1606d21e3a953f2ba58889bbed2acada`.

Completed readback: `tmp/r7-20261117/tool-r6-final-readback/`. All **72/72** cases
were scored; **0 valid native tool calls**, hence 0 executed successes. Final
answer grounding is not evaluable when no tool executed. This measures this
controlled English tool protocol, not every possible tool prompt. Preserve the
outputs and compare R7 on the identical protocol after training.

## Training Launch Recipe (Already Executed; Do Not Repeat)

Preparation uses Python 3.13 with local tokenizer dependencies. Launches use
the prepared Python 3.11 environment to match Modal serialization.

```bash
/Users/welcome/miniconda3/bin/python scripts/run_understanding_r7.py prepare --pack tmp/r7-20261117/pack-v1 --out tmp/r7-20261117/execution-train-v1 --mode train --source /Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai/tmp/understanding-r6-20261113/execution-train/frozen --smoke-directory tmp/r7-20261117/smoke-final-readback
tmp/r7-20261117/baseline-launch-py311/bin/python scripts/run_understanding_r7.py launch --execution tmp/r7-20261117/execution-train-v1 --receipt tmp/r7-20261117/train-v1.receipt.json --authorize-gpu
```

Before using either command, check whether its output directory/receipt already
exists. Do not overwrite or duplicate it. Save the actual detached call ID.

Training continues final R6 weights `6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c`
on pinned `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1`, revision
`693f12e444a30971ca53889128d7bade9491ecb3`. Frozen base, 43,278,336 trainable LoRA
parameters, BF16 base/FP32 adapters, LR 1e-5, accumulation 16, seed 42.
One full pass: **2,413 optimizer updates**. No packing or truncation.
Checkpoint/consumption committed every 25 updates. The worker copies the full
sealed package to Modal before training. Check actual `consumption.json`, not
just the existence of a checkpoint. An eight-hour bound pauses safely; for
`paused_budget`, verify the old call is terminal, then use the same execution
package with `--resume-run <old-run>` and a new receipt. This restores optimizer,
scheduler, RNG and exact data cursor; never silently restart from R6.

## After Training: Evidence and UI

1. Verify `training_completed`, full consumption and adapter bytes. Use
   `scripts/prepare_understanding_r7_inference.py collect`, then `prepare` with
   the existing R6 bundle `tmp/r6-demo/bundle-v1`. Both require the actual
   training-receipt SHA. Read `--help`; do not invent a checkpoint or bundle SHA.
2. Run R7 on `behavior-eval-v1` with `scripts/run_r7_behavior_candidate.py`.
   Frozen manifest: `8029612cea3110d31d9b8a1f717e0f43a38148c613c739341ae46f50d3a68e13`.
   R6's 117 outputs are already complete in `r6-behavior-v4-collected/latest.json`.
   Eight semantic roots yield 112 fresh derived views; five old regressions are
   separate. Preserve independent model histories. Agent reviews are not human review.
3. Run `scripts/r7_public_benchmark_launch.py` on
   `public-benchmark-frozen-v1` and the completed R7 bundle. It runs R6 and R7
   on the same 2,004 original public-task inputs with the verified native loader.
   Earlier R6 public scores lacked this exact loader proof: preserve them but
   do not call a comparison against those old numbers matched. Conditional
   answer-letter log probabilities and strict SIB labels are implemented.
4. Stage/run R7 with `scripts/r7_tool_eval_run.py` on `tool-eval-v1`, then compare
   with the completed R6 tool baseline. Report valid calls, arguments, execution
   and grounded final answers separately. There are 72 derived cases, 15
   scenarios and eight source groups, English only; no live-web capability claim.
5. Serve `modal/research_understanding_r7_demo.py` privately. Start
   `scripts/understanding_r7_workbench.py chat` with the verified R7 bundle and
   exact SHA on port 7871 if free. Leave R6 on 7870 untouched. R7 must appear
   first, R6 second. Verify fresh chat, correction, follow-up, streamed identity
   and repetition guards in both columns before handing the URL to the user.
6. Report actual gains, regressions, failed/unknown judgments and limitations in
   `R7-STATUS.md`; keep the execution map current. No claim of a trained tool
   agent or safe medical service from loss/completion alone.

## Backup and Resume Discipline

Private backups live in volume `ghana-health-understand-train`, prefix
`/research-handoffs/twi-understanding/2026-11-17-r7/`. Local verified receipts:
`tmp/r7-20261117/backups/*.receipt.json`. Select the latest receipt with
`remote_verified: true`; verify its SHA when restoring. Earlier confirmed
archive `673ac4802412935319fe6871f8dd06967511c5a79f747a8fa5f78b46f663f0a5.tar.gz`
contains the final sealed data and smoke package: **607,656,162 bytes, 6,534
files, remote readback verified**. It predates the completed smoke/training
receipt and this final handoff; use the accompanying Hermes supplement for
those. Earlier archives are preserved, not the latest state.

Run `MODAL_PROFILE=createdliving1000 /Users/welcome/miniconda3/bin/python
scripts/backup_r7_progress.py --upload` after updating this handoff and receipts.
It includes this document and private JSON evidence, excludes disposable Python
environments, and verifies remote bytes. Full executable training packages also
live with their run on Modal once worker startup reaches artifact persistence.
Push code/docs to the research branch, never directly to main. Do not touch the
original dirty tree, frozen datasets or old model/evaluation evidence.

## Prompt for Hermes

Continue R7 from docs/research/R7_HERMES_HANDOFF.md in
/Users/welcome/.codex/worktrees/r4-train/ghana-health-ai on
codex/61-r7-data-training. First reconcile the existing smoke, tool-evaluation
and training receipts against Modal; never duplicate jobs. The data pack is
sealed and fully checked, smoke passed, and full training is already submitted
as fc-01M7P3443CKHMZGBK47M0V2CCG. Monitor it, verify consumption, run matched R6/R7 behavior,
public and executed-tool evaluations, and serve R7 first against R6 in the local
UI. Preserve independent histories and equal guard settings. Keep R6's existing
UI and production unchanged. Continue on Modal, no OpenAI calls. Record honest
results and exact job/artifact identities, update the execution map, push the
research branch and verify a private Modal handoff backup. Do not restart data
research or stop after each batch. Latest status/receipts supersede this snapshot.
