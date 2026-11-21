# R7 Local Testing on Apple Silicon

21 November 2026. Prepared, **not yet inference-verified**. The owner confirmed
unmetered internet and the pinned base download resumed. HTTP fallback is being
used after Xet stalled; transfer is slow and partial files are retained. Inspect
the downloader before starting another. No Modal job, hosted inference endpoint,
API fallback or public upload is needed for this route.

## Current State

| Item | Verified state |
| --- | --- |
| Original R7/R6 adapters | Downloaded and SHA-256 verified; final R7 recovery state also saved |
| MLX adapter conversion | 496 tensors, 248 target modules and 43,278,336 parameters per adapter; base projection dimensions checked |
| LoRA scaling | Rank 16, alpha 32, MLX scale 2.0; no R6/R7 adapter stacking |
| Numerical conversion check | All module deltas compared with NumPy; maximum absolute difference below 4.3e-9 |
| Local code tests | History isolation, partial-response exclusion, config/key rejection and loopback UI construction passed |
| Public base | Pinned download incomplete; 17,907,663,008 bytes required for the weight file |
| Full-model results | Not run; no local output, speed, quality or memory-fit claim yet |

The adapter checks do not establish full-model inference parity. A 4-bit affine
base (group size 64) is planned for this 16 GiB Mac. Quantization changes the
inference computation; do not pool its scores with prior BF16 results. Both
R7 and R6 must use this same converted base and identical local controls.

## Paths and Runtime

- Worktree: `/Users/welcome/.codex/worktrees/r4-train/ghana-health-ai`.
- Durable local root: `/Users/welcome/Documents/Research-Exports/ghana-health-ai`.
- Isolated Python: `local-runtime/bin/python` under that durable root.
- Versions installed: MLX 0.32.4, mlx-lm 0.32.2, Gradio 6.32.0,
  Transformers 5.23.0. The existing app environment was not changed.
- Converted adapters: `mlx-adapters-v1/` under the durable root, with checksums
  and per-module check receipts in `manifest.json`.
- Download destination: `afrique-base-bf16/` under the durable root. Keep its
  `.cache/huggingface/download/` contents so the client can reuse partial data.
- Current one-shot downloader: `download-base-20261121.py` under the durable
  root, with `download-base-status.json` and `download-base-20261121.log`. It
  has a single-run lock, a 24-hour bound and final weight SHA256 verification.
  It does not auto-start model conversion or inference. Check its recorded PID
  and status before issuing the manual resume command below.
- Frozen tokenizer bundle: `tmp/r7-20261118-local-bundle/` in the worktree.
- Original adapters/recovery: `tmp/r7-20261118-local-export/` in the worktree.

Pinned base: `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1`, revision
`693f12e444a30971ca53889128d7bade9491ecb3`. Weight SHA256:
`6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6`.
Do not substitute a third-party quantization whose source revision is unknown.

## Resume Without Provider Compute

Confirm the connection is unmetered before the large public download. No billable
HF inference is involved, but internet and power are not guaranteed free. Inspect
existing download processes before starting another. From the worktree:

```bash
ROOT=/Users/welcome/Documents/Research-Exports/ghana-health-ai
PY="$ROOT/local-runtime/bin/python"
```

Resume the public pinned files (the original download used the existing Conda
HF client; keep that client for partial-cache compatibility):

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 /Users/welcome/miniconda3/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1", revision="693f12e444a30971ca53889128d7bade9491ecb3", local_dir="/Users/welcome/Documents/Research-Exports/ghana-health-ai/afrique-base-bf16", allow_patterns=["*.json", "*.jinja", "model.safetensors"], max_workers=1, token=False))'
```

Only after the download completes, convert locally. The script verifies frozen
metadata and the entire source-weight hash first, then records converted hashes.
It refuses an existing output directory rather than overwriting artifacts:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "$PY" scripts/prepare_r7_local_mlx.py base \
  --source "$ROOT/afrique-base-bf16" --out "$ROOT/afrique-base-mlx4"
```

Run the first actual inference check and inspect both saved outputs:

```bash
"$PY" scripts/r7_local_mlx.py smoke \
  --base "$ROOT/afrique-base-mlx4" --adapters "$ROOT/mlx-adapters-v1" \
  --bundle tmp/r7-20261118-local-bundle --store "$ROOT/local-smoke-v1"
```

Verify finite outputs, expected adapter changes, peak memory and stop behavior.
Do not treat a successful import or a conversion receipt as a passed inference
check. If the Mac runs out of memory, stop; do not fall back to Modal automatically.
Once the smoke test passes, launch the actual comparison:

```bash
"$PY" scripts/r7_local_mlx.py serve \
  --base "$ROOT/afrique-base-mlx4" --adapters "$ROOT/mlx-adapters-v1" \
  --bundle tmp/r7-20261118-local-bundle --store "$ROOT/local-conversations-v1" \
  --port 7871
```

Intended address is `http://127.0.0.1:7871`, **not currently a verified running demo**.
Use another free port if occupied. Check actual desktop/mobile rendering before
announcing readiness. No public share link is enabled.

## Comparison Contract

- R7 answers first, then R6; one local model allocation with adapter swapping.
- Each model receives its own complete history and a fresh attention cache.
- Partial, cancelled and guard-stopped turns remain visible but are omitted from
  subsequent context. They are never counted as successful complete replies.
- Greedy decoding, 256-token maximum, thinking disabled, 4,096-token context,
  no silent truncation, no hidden answer replacement. Same guard for both.
- Store raw messages, emitted token IDs, prompt hashes, stopping reasons,
  first-text latency and runtime identity locally. Interrupted partial text can
  exist in the saved view; completion status controls history eligibility.
- Local repetitions/meaning errors remain evaluation findings, not reasons to
  silently substitute another model. No claim of clinical validation.
- The frozen R7 behavior/tool/public-task suites are still pending. A casual
  local smoke test is not a replacement for that broader matched evaluation.

Tests without loading full weights:

```bash
PYTHONPATH=scripts "$PY" -m unittest \
  scripts/test_prepare_r7_local_mlx.py scripts/test_r7_local_mlx.py \
  scripts/test_r7_local_mlx_ui.py scripts/test_audit_modal_storage.py
```
