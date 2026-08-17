# ASR R&D Session — 2026-08-15

> Stage 0 (measurement) + Stage 0.5 (local-adaptation probe) of the R&D campaign.
> Context: `docs/ASR_PROJECT_STATUS.md`, `docs/asr-model-decision.md`, `docs/CONTINUE.md`.

## Objective

Decide where the next real training compute goes by (a) producing fair,
comparable measurements of the two model families (Whisper v6 vs DONDO v1) on
benchmark **and** product-domain data, and (b) testing whether the local
recorder corpus moves domain WER at all before scaling data collection.

## Why this session exists (decision context)

- v6 (Whisper-small) serves production: full Waxal test n=1522 greedy **31.49%**,
  beam=5 **30.44%**; promotion gate is Twi WER < 28%.
- DONDO v1 fine-tune reached **35.77% val WER** but was never evaluated on the
  Waxal **test** split, never on English, and never on domain data — the
  comparison with v6 was not apples-to-apples.
- v3–v6 plateaued at ~30–34% on Whisper-small with Waxal-centric data; the
  suspected bottleneck is data (domain match), not architecture.
- The local recorder corpus (40 clips, 182.8s, 2 speakers, buckets
  health_twi / commerce_twi / codeswitch_tw_en) is the only product-domain
  eval/training data we have.

## Code changes this session (branch `feat/local-asr-adaptation`)

1. **`modal/train/train_asr.py` — promotion gate corrected.**
   The gate compared against Round 2 greedy (32.83%), which production (v6,
   30.44% beam5) already beats. A run could have auto-flagged PROMOTE while
   being worse than the serving model. Now:
   - `BASELINE_WER = 0.3149` (v6 greedy, full Waxal test n=1522)
   - `BEAM5_WER = 0.3044` (v6 beam5, serving decode)
   - `ROUND2_WER = 0.3283` kept as historical reference only
   - Summary keys renamed `beats_round2_*` → `beats_v6_*`; card text relabeled.
     (Verified: no script consumes the old keys.)

2. **`modal/train/eval_asr.py` — local-corpus eval mode.**
   New `--local-manifest-path` flag: mounts `tmp/asr-local-train/` into the
   image (`/root/gha_local_asr`), loads the recorder manifest, decodes WAVs,
   and reports overall + **per-bucket** WER/CER. HF-dataset behavior unchanged.

3. **`modal/train/eval_dondo_asr.py` — same local-corpus mode** for the
   DONDO/CTC decode path (language-prefix conditioning preserved).

## Stage 0 — measurement jobs (T4, minutes each)

| # | Job | Model | Eval set | Question answered |
|---|-----|-------|----------|-------------------|
| 0a | `eval_dondo_asr.py` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Waxal `aka_asr` **test** n=300, streaming | True DONDO-vs-v6 gap on the benchmark (v6: 30.16% greedy / 28.76% beam5 on same n=300 streaming subset) |
| 0b | `eval_dondo_asr.py` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Common Voice 22 `en` test n=100, "African English" prefix | English retention (v6: 42.34%; base whisper-small: 11.82%) |
| 0c | `eval_asr.py` | `teckedd/gha-whisper-small-twi-v6` | local corpus, 40 clips, beam=5 | **Domain baseline "before"** — per-bucket health/commerce/code-switch |
| 0d | `eval_dondo_asr.py` | `teckedd/gha-dondo-w2v-bert-twi-v1` | local corpus, 40 clips | DONDO domain baseline, same clips |

Launch: `modal run … --no-wait` (fire-and-forget spawns; results land in the
`akan-speech-eval-results` Modal volume and are pulled with
`pnpm eval:asr-results:pull`).

### Launch log

First spawn attempt (~21:50 UTC) failed silently: `modal run` without
`--detach` stops the app when the local entrypoint exits, which killed the
spawned calls before they ran. Re-launched with `--detach` (~22:25 UTC):

| Job | Modal app | Function call | Status |
|-----|-----------|---------------|--------|
| 0a DONDO Waxal test n=300 | `ap-Sn0gHv6YT14m9mI661uat3` | `fc-01M03RD5YT24YD8KFW5P44W3YJ` | running (detached) |
| 0b DONDO CV22 English n=100 | `ap-sn4ane5iaS7JhbRM5QOrx7` | `fc-01M03RDMESS5RHE7M518V2T0D7` | running (detached) |
| 0c v6 local corpus beam5 | `ap-N4Om0m8Ukf2Ty13aO0y1fe` | `fc-01M03RE2SK7DHZ16Q2M9G7YH2E` | running (detached) |
| 0d DONDO local corpus | `ap-2OPdDwH7POlAf91TEVmZAT` | `fc-01M03REHGN87M13FYYNMAS1JG6` | running (detached) |
| 0.5 v6-local-adapt train | `ap-sC5ZAscmRVwxYba3eLZ4Gc` | `fc-01M043NR0SBFRDMRJ9J30AXW8M` | relaunched 2026-08-16 01:41 UTC with chunked-cast guard. Prior attempts: `ap-R8l7u3q84L4gNQmZQYe1tS` (stopped: full-shard loader), `ap-ApfMFfMqodj7XiJlQmx6aC` (crashed: pyarrow cast bug). |

**Lesson recorded:** all fire-and-forget Modal launches from this repo must use
`modal run --detach`; `--no-wait` alone lets the app stop and kills the call.

**Second launch issue (probe):** the first `v6-local-adapt` run was launched
without `--train-limit`, which selects the **full non-streaming Waxal loader**
(`load_dataset` on all 270 shards). The download crawled at 100–176s/shard with
read timeouts (~7–12h before training). Stopped `ap-R8l7u3q84L4gNQmZQYe1tS` and
relaunched with `--train-limit 4000 --eval-limit 300`, which selects the capped
streaming loader proven in the v7 proof runs. **Rule: never launch
`train_asr.py` without an explicit `--train-limit`.**

**Third issue (DONDO evals):** a typo (`bucket_refs = []` instead of `{}`)
crashed all DONDO evals at row 0. Fixed, validated the manifest-parsing logic
locally (40/40 rows resolve), relaunched 0a/0b/0d at ~22:30 UTC. Final IDs:

| Job | Modal app | Function call |
|-----|-----------|---------------|
| 0a DONDO Waxal test n=300 | `ap-DjoGalCElyxxDZCtbYX9gE` | `fc-01M03RPWGX5JH8GW0MVM6JKJG8` |
| 0b DONDO CV22 English n=100 | `ap-pzdHqfzIVJ3q2jKhIQI7wd` | `fc-01M03RQBKJZ4V5K7FMNA8BZ538` |
| 0c v6 local corpus beam5 | `ap-N4Om0m8Ukf2Ty13aO0y1fe` | `fc-01M03RE2SK7DHZ16Q2M9G7YH2E` |
| 0d DONDO local corpus | `ap-uCOQysZu1nSXUwKFqNP856` | `fc-01M03RQS9QCT6QQRHSS7YMFAVZ` |

## Stage 0.5 — v6-local adaptation probe (A100, ~1h, ~$3–5)

Hypothesis: continuing from the v6 checkpoint with the local corpus at low
weight and low LR adapts the model to product-domain speech (health/commerce/
code-switch, phone audio) without regressing Waxal.

```
modal run --detach modal/train/train_asr.py \
  --base-model teckedd/gha-whisper-small-twi-v6 \
  --run-name v6-local-adapt --max-steps 600 \
  --learning-rate 5e-6 \
  --push-repo teckedd/gha-whisper-small-twi-v6-local \
  --use-local-data --local-weight 0.08 \
  --no-wait
```

Design notes (deviations from the originally proposed run):

- **Continues from v6**, not `openai/whisper-small` — isolates the effect of
  local data; avoids re-deriving v6's 3,000 steps.
- **600 steps, LR 5e-6, local weight 0.08** — 40 clips / 182.8s cannot carry a
  20% weight over 3,000 steps (~480 epochs over 3 minutes of audio =
  memorization; v3–v5 died this way). This is a surgical adaptation run.
- End-of-run full Waxal test (n=1522, greedy + beam5) runs against the
  **corrected v6 gate** (31.49% / 30.44%).

### Success criteria for the probe

| Outcome | Interpretation | Consequence |
|---------|----------------|-------------|
| Domain WER (local corpus) improves vs 0c baseline, Waxal full-test not worse | Local data lever works at small scale | Scale corpus to 200+ clips, then full Stage 1 run |
| Domain WER flat, Waxal flat | 40 clips too small to matter | Data collection before more training |
| Waxal regresses materially | Adaptation too aggressive even at 0.08 | Lower weight/LR further or mix more Waxal replay |

### Probe outcome pivot (checkpoint-recovery evals)

Attempt 4 completed all 600 training steps (validation **30.43% WER / 10.55%
CER**, n=300 Waxal val) and saved the full model to
`akan-speech-checkpoints:/gha-asr/v6-local-adapt_teckedd_gha-whisper-small-twi-v6_s600`
before the gate eval. The gate eval's non-streaming test download then crawled
(132/270 shards in 2h, worsening) while holding an A100, so the app was
stopped. `eval_asr.py` gained `--checkpoint-dir` (mounts the checkpoints
volume, loads model+processor locally), and three T4 evals were launched
against the saved checkpoint (~13:32 UTC):

| Job | Eval | Call |
|-----|------|------|
| after-domain | local corpus n=40, beam5 | `fc-01M05K8F70GX15TT21HG9W3FYZ` |
| gate-greedy | Waxal test stream n=1522, beam1 | `fc-01M05K8WSYF30R9K1X5Y67YY8A` |
| gate-beam5 | Waxal test stream n=1522, beam5 | `fc-01M05K9AT69H3CBKBJHVVZGGBC` |

Note: streaming test eval avoids the 270-shard full resolution that stalled
the trainer's `_run_full_test`; it fetches only test shards on demand.

### Probe Waxal gate (from saved checkpoint, n=1522 streaming, T4)

| Model | Greedy WER | Beam5 WER | Beam5 CER |
|-------|-----------|-----------|-----------|
| v6 (serving) | 31.49% | 30.44% | 10.62% |
| v6-local-adapt-s600 | 31.76% | 30.73% | 10.68% |

→ **No benchmark regression** (+0.27/+0.29 pp — within streaming/eval noise at
n=1522). The 600-step gentle adaptation is non-damaging. Not a promotion
(does not beat v6), but it clears the "do no harm" bar, which was half the
experiment. The other half (domain generalization) is the hold-out experiment
below.

### Probe domain "after" — landed, and it is a train-set contamination lesson

The checkpoint eval on the local corpus returned **0.67% WER / 0.33% CER**
(codeswitch 0.0%, health 0.5%, commerce 1.67%) versus v6's 54.18% baseline.
This is **not** adaptation — the same 40 clips were in the training mix
(8% weight × 600 steps ≈ ~6 epochs per clip). The model memorized the eval
set. v6's 54.18% was true generalization (it never saw the clips); the
candidate's number is train-set recall. **The probe as originally designed is
inconclusive.**

**Corrective experiment (launched ~14:15 UTC):** stratified hold-out split of
the corpus — `manifest.train32.jsonl` (32 clips) / `manifest.holdout8.jsonl`
(8 clips: 5 health, 2 commerce, 1 codeswitch; seed 42). New run
`v6-local-holdout` trains on the 32 only (`--no-full-test-after`; gate numbers
for the recipe already exist from the checkpoint evals), and v6 is being
re-baselined on the same 8 held-out clips for a clean before/after.

**Permanent rule:** local-corpus evals must declare whether the model trained
on the evaluated manifest. The domain eval set for promotion decisions must be
clips the candidate never saw — ideally also unseen speakers.

## Results

### Stage 0 — landed 2026-08-15 ~22:45 UTC

**0a — DONDO v1 on Waxal test n=300 (the first fair benchmark comparison):**

| Model | Decode | WER | CER |
|-------|--------|-----|-----|
| v6 (serving) | greedy | **30.16%** | 10.93% |
| v6 (serving) | beam=5 | **28.76%** | 9.45% |
| DONDO v1 | CTC greedy | **36.47%** | 11.36% |

→ DONDO v1 trails v6 by **6.3–7.7 pp WER** on identical data. Its CER is nearly
equal (11.36% vs 10.93% greedy) — the gap is word-level (spacing/segmentation),
which is exactly what CTC + LM beam decode addresses. **Stage 2 (DONDO v2)
stays on the table only with LM decode + more data; the architecture is not
disqualified, but v6 remains clearly ahead.**

**0b — DONDO v1 on Common Voice 22 English n=100: WER 43.55%, CER 17.11%.**

→ English retention is as bad as v6's (42.34%) and far off base whisper-small
(11.82%). **The separate English ASR route remains mandatory regardless of
which Twi model wins.** No further English-retention investment in DONDO.

**0d — DONDO v1 on the local product corpus (n=40):**

| Bucket | n | WER | CER |
|--------|---|-----|-----|
| health_twi | 25 | **28.00%** | 8.25% |
| commerce_twi | 10 | 35.00% | 14.98% |
| codeswitch_tw_en | 5 | **54.05%** | 15.47% |
| **overall** | 40 | **32.66%** | 10.78% |

→ First-ever domain measurement. Health Twi is DONDO's strongest bucket
(consistent with its GhanaNLP health-adjacent pretraining); code-switching is
the weakest — as predicted for CTC without a decoder LM. Awaiting 0c (v6 on the
same clips) for the head-to-head.

**0c — v6 on the local product corpus (n=40, beam=5):**

| Bucket | n | v6 WER | DONDO v1 WER | Δ (DONDO − v6) |
|--------|---|--------|--------------|----------------|
| health_twi | 25 | 53.96% | **28.00%** | **−25.96 pp** |
| commerce_twi | 10 | 45.00% | **35.00%** | **−10.00 pp** |
| codeswitch_tw_en | 5 | 70.27% | **54.05%** | **−16.22 pp** |
| **overall** | 40 | 54.18% | **32.66%** | **−21.52 pp** |

### ⚑ The headline finding of this session

**The ranking flips on product-domain data.** v6 beats DONDO by 6–8 pp on Waxal
(read speech, the benchmark), but DONDO beats v6 by **~22 pp overall** on our
own recorder clips (phone, quiet, spontaneous product phrases). v6's production
domain WER is 54.18% — barely usable — despite its respectable 30.44% benchmark
number.

Implications:

1. **Waxal WER does not predict product WER.** The promotion gate's Waxal-first
   design measures the wrong thing for our domain. Product-domain evals must
   become first-class gates (this validates the `data/asr-product-eval/` work).
2. **DONDO v2 is now clearly justified** (Stage 2): its domain performance with
   only 1,800 Waxal rows of fine-tuning already beats the serving model where it
   matters. With full Waxal + CV-Twi + local data + higher LR + KenLM decode,
   it has a concrete path below 25% domain WER.
3. **v6-local-adapt probe is now even more important:** if 600 steps on 40 clips
   materially improves v6's domain WER, the data lever is proven for the Whisper
   track too, and Stage 1 proceeds with corpus scaling.
4. **Caveats:** n=40, 2 speakers, quiet environment. Treat as directional, not
   conclusive. Corpus scaling (200+ clips, more speakers, noisy conditions) is
   required before any promotion decision trusts these numbers.

### ✅ The clean hold-out verdict (n=8 truly unseen clips, beam5)

| Bucket | v6 (baseline) | v6-local-holdout (trained on 32) | Δ |
|--------|--------------|----------------------------------|---|
| health_twi (n=5) | 47.50% | **35.00%** | −12.5 pp |
| commerce_twi (n=2) | 53.85% | **30.77%** | −23.1 pp |
| codeswitch_tw_en (n=1) | 28.57% | 42.86% | +14.3 pp (n=1 — noise) |
| **overall (n=8)** | **46.67%** | **35.00%** | **−11.67 pp** |
| memorization check: same model on its 32 train clips | — | 0.42% | (train recall) |

Clean-probe training: val WER **29.70%** / CER 10.31% (n=300 Waxal val), local
weight normalized 0.0625, mix = Waxal + local32 + CV-tw + CV-en + GhanaNLP.

**Interpretation.**

1. **The data lever is real.** 32 clips (~2.5 min of audio) at 6% mix weight
   bought an **11.7 pp absolute** domain WER improvement on clips the model
   never saw, with no Waxal regression (sibling run: 31.76/30.73 vs v6's
   31.49/30.44 on n=1522). This is the first causal evidence that product-domain
   data generalizes, not just memorizes.
2. **Memorization still dominates** (35% holdout vs 0.42% train recall at
   ~6 epochs/clip). With 200+ clips and fresh speakers, the generalization
   share should grow; at 40 clips the model is mostly memorizing voices.
3. **Caveats:** n=8 holdout, 2 speakers only, quiet conditions. Directionally
   strong, statistically thin. Treat as "justify scaling," not "final number."

**Consequences.**

- **Stage 1 is GO:** scale the recorder corpus (200+ clips, 4+ speakers,
  code-switch priority, some noisy/phone conditions), keep the
  train/holdout split discipline, then a full v8 run from v6 with local data
  at 0.10–0.15 weight and a held-out domain gate.
- **Stage 2 (DONDO v2) remains GO:** DONDO v1's 32.66% on the full 40-clip
  corpus (never trained on any of it) already beats the adapted Whisper's
  35% holdout number. With more data + LM decode it is the most promising
  track for domain WER.
- **Data collection is now the highest-leverage work in the project.** Both
  model tracks are gated on it, not on compute.
- Nothing promotes: v6 stays serving; the holdout checkpoint remains a
  research artifact in the checkpoints volume.

### Stage 0 / probe — incidents during collection

- First spawns killed by app stop (`--no-wait` without `--detach`).
- `bucket_refs = []` typo crashed DONDO evals at row 0 (fixed, relaunched).
- 0c crashed on `UnboundLocalError: audio_col` in local mode (fixed, relaunched).
- Probe relaunched with capped streaming loader (see launch-log lessons).
- Probe crashed again at `_to_audio_text`: `datasets` 3.1.0 + current pyarrow
  cannot `cast_column` to Audio on multi-chunk in-memory tables (n=4000 spans
  chunks; the n=64 smoke didn't). Added a guarded skip: when audio rows are
  already decoded 16 kHz arrays (always true for streamed/manifest rows), the
  redundant re-cast is skipped with a log line; lazy undecoded audio still
  raises. Relaunched as `ap-sC5ZAscmRVwxYba3eLZ4Gc`.
- Third probe crash at `interleave_datasets`: the guarded skip left Waxal audio
  as a plain struct while other parts kept the `Audio` feature type — features
  must align to interleave. Fix v2: on cast failure, rebuild the table in
  800-row parts and cast each (single-chunk casts are proven), preserving the
  Audio feature. Relaunched as `ap-ujvZoM3yVDlQHflOyyjpUx` (2026-08-16 ~02:11 UTC).
- Attempt 3 (`ap-ujvZ…`) died on **HF CDN infrastructure flakiness** (read
  timeouts + 503s from `us.aws.cdn.hf.co` xet bridge). Crucially, it had
  **completed all 600 training steps** and only failed during the full-test
  gate download. Its checkpoints persist at
  `akan-speech-checkpoints:/gha-asr/v6-local-adapt_teckedd_gha-whisper-small-twi-v6_s600`
  — recoverable without retraining if ever needed.
  Hardened the loader: `HF_HUB_DOWNLOAD_TIMEOUT=60`, `_with_retries` (4
  attempts, backoff) around Waxal train/val + Common Voice streaming, and a
  retry loop around the full-test `load_dataset` in `_run_full_test`.
  Relaunched as `ap-uz9Tbd7BkJ2jX3orxRFx4t` (~12:23 UTC).

### DONDO serving endpoint deployed (2026-08-16)

`modal/dondo_asr_service.py` (commit after this doc's commits) is a drop-in
clone of the production ASR contract backed by
`teckedd/gha-dondo-w2v-bert-twi-v1`, deployed as a separate Modal app —
production stays on v6.

- Endpoint: `https://createdliving1000--ghana-health-asr-dondo-api.modal.run`
- Health: `{"ok":true,"service":"ghana-health-asr-dondo","engine":"transformers-wav2vec2-bert-ctc"}`
- Live smoke (held-out health clip, 4.4 s): **306 ms latency** (vs seconds
  for Whisper beam-5), transcript near-exact (`koko→kuku` only).
- Local A/B: `MODAL_ASR_URL=https://createdliving1000--ghana-health-asr-dondo-api.modal.run sec -- npm run dev`

### In-app A/B switch (2026-08-16, commit `ee90563`)

The voice UI now has a persisted **ASR v6 / DONDO β** toggle
(`gha:asr-model` in localStorage) flowing `asrModel` through
`/api/voice/transcribe`, `/api/voice/converse`, `/api/voice/converse/stream`,
and `/api/voice/preview` into `modalTranscribe` routing
(`MODAL_ASR_DONDO_URL` env override, default = the DONDO endpoint). English
turns always keep the dedicated English route; default remains v6.

Verified end-to-end via `/api/voice/preview` on a held-out commerce clip
(ref: "Fa aduro no brɛ me wɔ Adenta"): **DONDO** "fa eduro no brɛ me wɔ
adɛnta" (meaning preserved) vs **v6** "fa aduru no barima wɔ adɛnta"
("brɛ me" → "barima" — meaning changed). First in-app evidence that the
domain gap is user-visible, not just metric-visible.

### First real-user corrections + known product issues (2026-08-17)

The switch moved into the side nav (commit `2aaa3d6`; store:
`src/lib/asr-model-store.ts`). First two real corrections landed, both
DONDOβ, both health-domain vocabulary misses:

1. "meteɛ me meyɛ apaa mepɛ **parasita mɔ** atɔ" → "…mepɛ **paracetamol** atɔ"
2. "mete pae me na me **ano soro nso kyie me**" → "meti pae me na **me ho nso
   aye hye**" (fever vocabulary)

User's qualitative read: DONDO is more accurate overall; v6 is close and
sometimes matches. Consistent with the metrics.

**Known issue — response generation (noted for the response-quality pass):**
replies are formulaic — roughly "if you say {transcript} you have to see a
doctor…" in Twi regardless of content. Not an ASR problem; the LLM/reviewer
step needs work (better Twi reply prompting, less templated triage phrasing).
Tracked here so it is not lost.

**Consented audio capture shipped (2026-08-17):** text-only corrections can't
feed audio training, so the correction form now has an explicit opt-in
checkbox ("Ma yɛmfa nne a wokae no nsiɛ model no" / "Share this recording to
improve the model"). With consent, the clip is uploaded with the correction,
stored at `ASR_FEEDBACK_AUDIO_DIR` (default `data/asr-feedback-audio/`,
gitignored), and linked via `AsrFeedback.audioConsent` / `audioPath`
(migration `20260817133000_asr_feedback_consented_audio`). Without consent,
nothing changes — audio is never retained. Export tags consented rows
`consent=user_shared_audio` with real audio paths; unconsented rows keep the
`MISSING_AUDIO_*` marker. Verified e2e: multipart correction → file on disk +
DB row + export row.

## Next decision gates — RESOLVED this session

- **Stage 1 (full Whisper v8 from v6, scaled corpus): GO** — the clean hold-out
  experiment proves domain data generalizes (−11.7 pp on unseen clips from
  32 training clips). Gate condition: corpus scaled to 200+ clips with new
  speakers + kept train/holdout discipline before the full run.
- **Stage 2 (DONDO v2): GO** — DONDO v1's 32.66% on the 40-clip corpus (never
  trained on it) is the best domain number on record; the design below removes
  its handicaps (data, LR, decode).
- **Blocking dependency for both: data collection, not compute.** Corpus
  scaling (200+ clips, 4+ speakers, code-switch priority, noisy conditions) is
  the critical path.
- Nothing promotes without held-out health-domain + code-switch + phone/noise
  evals — the local corpus split (`manifest.train32/holdout8`) is the template.

## Stage 2 design — DONDO v2 (drafted from Stage 0 evidence)

Rationale: DONDO v1 already beats v6 by ~22pp on product-domain audio with only
1,800 Waxal rows, 800 steps, LR 5e-6, and bare greedy CTC decode. Every one of
those was a handicap. v2 removes them:

1. **Data:** full Waxal train (not 1,800-row cap) + Common Voice 22 Twi
   (validated-only) + the local recorder corpus + (later) exported
   `/api/voice/feedback` corrections. Requires extending
   `train_dondo_asr.py` beyond its Waxal-only loader.
2. **Hyperparameters:** LR ~1e-4 with warmup (w2v2/w2v-BERT CTC fine-tune
   territory; 5e-6 was Whisper-style conservatism), 2,500+ steps, batch as
   memory allows on A100.
3. **Decode:** add CTC beam search with a KenLM Twi LM (trained on Waxal
   transcripts + Twi knowledge seed) to the eval path. Expected several pp WER
   gain, zero training cost. Directly targets the word-level gap (0a: WER gap
   6–8pp vs CER gap ~0.4pp) and the code-switch weakness.
4. **Eval design (must change per CONTINUE.md's rule):** report Waxal test
   n=300 AND local-corpus per-bucket WER in the same run summary; promotion
   judged on domain first, benchmark second.
5. ~~**Open question to verify before the run**~~ — **RESOLVED (2026-08-17):**
   the official KhayaAI model card's reference `add_language_prefix` is
   byte-identical to our implementation (`lang_vec[lang_id % D] = 1.0`,
   `prefix_len=1`). Our conditioning was never wrong. Bigger finds from the
   DONDO paper (arXiv:2607.21540, Azunre et al., Apache-2.0):
   - **v1 was undertrained by design**: the paper's recipe is step-1 LR 5e-5
     then anneal 5e-6; our v1 ran a constant 5e-6 (the anneal rate) for 800
     steps on 1,800 rows. v2 runs 2,500 steps at 5e-5 (cosine decay
     approximates the anneal).
   - **Reference baselines (in-domain, religious read speech):** Asante Twi
     monolingual 15.75% WER; multilingual step-2 14.7% — the realistic ceiling
     on clean read Twi. Our Waxal/domain numbers sit on harder data.
   - Paper's own limitation confirms our gap: read religious text → weak on
     spontaneous/code-switched speech "until fine-tuned". That fine-tune is
     exactly what v2 is.

### Data collector upgrades

**Strategy shift (2026-08-17): curated read-speech corpus becomes the primary
data engine.** The user flagged that (a) not all recorder prompt transcripts
are accurate Twi, and (b) free-form collection yields deficient, imbalanced
data (commerce_twi n=10, codeswitch n=5). The agreed pipeline, mirroring how
serious ASR corpora are built:

1. **Source trusted text** — starting with our own reviewed Twi health
   knowledge seed (17 articles in `KnowledgeArticle`).
2. **Extract/correct prompts** — `tmp/asr-collection-pack/prompts.corpus-v1.jsonl`
   (59 utterances, 3–22 words each, health_twi). Translations of medical
   source text get a native-speaker correction pass BEFORE recording; the
   older 432-prompt pack is marked needs-review and must not be trusted as
   gold without that pass.
3. **Read + record in batches** — recorder gets a focused reading-session
   mode (big prompt, minimal chrome, per-bucket progress chips, loadable
   corpus packs).
4. **Validate on import** — `pnpm eval:local-asr-import` gates every batch
   (dedup, duration, holdout flags) before training use.

**Corpus pack v2 (2026-08-17):** `scripts/gen-corpus-pack.ts`
(`pnpm corpus:gen`) translates curated English source lines
(`tmp/corpus-source/*.txt`, authored in-repo for clean licensing) into Twi /
Twi-English drafts via the configured LLM. Generated packs: health_twi 126,
commerce_twi 74, codeswitch_tw_en 50 — all `needs_review: true` with
`en_reference` kept for the correction pass.

**⚑ Translation-quality finding:** both available general models
(gpt-4o-mini AND gpt-4o) produce Twi drafts with **meaning-level errors**
(e.g. "I am pregnant" → "Medi aberewa…" = "I am an old woman"; "stomach" →
"me nan mu" = "my leg"). General LLMs cannot be trusted as Twi translators
for ground-truth data — the human correction pass is load-bearing, not
cosmetic. For larger future packs, GhanaNLP's Khaya translation API
(purpose-built Twi MT) should produce better first drafts; needs an API key.

Read speech ≠ spontaneous speech, and both models need the latter eventually
(pilot feedback loop covers that) — but for domain vocabulary, bucket
balance, and volume, curated read speech is the highest-quality lever per
hour of effort. Solo-recorder constraint acknowledged: content/environment
diversity now, speaker diversity later via pilot corrections.

The browser recorder (`tmp/asr-collection-pack/recorder.html`) now persists
per-speaker progress in localStorage, blocks accidental re-records (explicit
`take_2`/`take_3` via confirm), and shuffles prompt order seeded by speaker
code so speakers cover different prompt subsets. Capture-time quality gates
(live level meter; duration/RMS/clipping/silence checks after stop) block
clips outside 1–20s and require "keep anyway" for soft failures, with instant
playback and re-record before accepting a take. Exported JSONL manifests now
carry dialect/device/environment metadata, a holdout flag (opt-in checkbox
plus automatic every-5th-clip holdout), and a sha256 of each audio file.
`scripts/validate-local-asr-import.ts` (`pnpm eval:local-asr-import`) gates
imports: schema/duration/text checks, exact (sha256) and near (speaker +
reference) duplicate detection against existing manifests, per-bucket and
holdout counts — non-zero exit on schema/dup failures, warnings only for
quality flags.

## Stage 2 launch — DONDO v2 RUNNING (2026-08-17)

```
modal run --detach modal/train/train_dondo_asr.py \
  --run-name dondo-twi-v2 --max-steps 2500 --learning-rate 5e-5 \
  --train-limit 0 --cv-twi-limit 3000 \
  --use-local-data --local-manifest-path /root/gha_local_asr/manifest.train32.jsonl \
  --eval-limit 300 --push-repo teckedd/gha-dondo-w2v-bert-twi-v2 --no-wait
```

App `ap-8UFzfnfrkoH4wiaJc3oMJf`, call `fc-01M08V6614GFXFBE41ZKA27P1X`.
Data: full Waxal train (streamed, cap 20k) + CV22 Twi validated (3k) + local
train32 (holdout8 stays frozen for the domain gate). Image fixes to get
there: cmake pinned 3.31.6 in an early layer + kenlm installed with
`--no-build-isolation` (pip's isolated build env kept pulling cmake 4.x,
which rejects kenlm 0.2.0's CMakeLists). Commit `fe30f3a`.

After completion: eval v2 on Waxal test n=300 AND the frozen holdout8 +
full local corpus via `eval_dondo_asr.py`; LM beam-decode comparison once
the Twi KenLM (`akan-speech-lm` volume) is built.
