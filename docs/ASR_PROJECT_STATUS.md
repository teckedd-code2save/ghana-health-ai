# Ghana Health AI — ASR Research & Improvement Phase

> **Project:** Fine-tune speech recognition for Twi/Akan health and ecommerce use cases.
> **Product:** [ghanahealth.serendepify.com](https://ghanahealth.serendepify.com)
> **Phase:** Research & improvement (active data collection + model iteration)
> **Last updated:** 2026-08-20
>
> **⚑ 2026-08-20 update:** DONDO v2 completed and was pushed to
> `teckedd/gha-dondo-w2v-bert-twi-v2` with a model card. Recovered post-train
> gates: Waxal n=300 **28.12% WER greedy / 27.31% WER with Twi LM**; frozen
> local holdout8 **26.67% greedy / 6.67% with Twi LM**. This makes DONDO v2
> the Twi ASR front-runner, but the local holdout is still too small for final
> promotion. Keep v6 as stable serving default while v2 runs as beta/A-B and
> the product-domain corpus scales. Execution plan:
> [`docs/asr-rd-execution-plan.md`](./asr-rd-execution-plan.md).
>
> **⚑ 2026-08-16 update:** An R&D session completed after this audit. Read
> [`docs/asr-rnd-session-2026-08-15.md`](./asr-rnd-session-2026-08-15.md) first.
> Headlines: v6's product-domain WER is 54.18% (Waxal WER does not predict
> product WER); DONDO v1 leads on domain audio (32.66% on the local corpus);
> a clean hold-out experiment proves domain data generalizes (−11.7pp from
> 32 clips); DONDO v2 and Whisper v8 are both approved, **gated on corpus
> scaling, not compute**. The trainer's promotion gate now compares against
> v6 (31.49%/30.44%), not Round 2. "Immediate next steps" 1–3 below are done.

---

## 1. Mission

Build a production-grade Automatic Speech Recognition (ASR) system for Ghanaian languages (primarily Twi/Akan) that powers health consultations and ecommerce interactions. The system must:

1. Understand **Twi (Akan)** speech accurately in health and commerce domains
2. Handle **Twi-English code-switching** (e.g., "Mepɛ paracetamol na me ti yɛ me ya")
3. **Retain English capability** — do not regress on English ASR
4. Run efficiently in production on Modal cloud GPUs

---

## 2. Model Zoo & Performance Ladder

All checkpoints live at `https://huggingface.co/teckedd/`

### Production Serving Model

| Model | Base | Architecture | WER (greedy) | WER (beam=5) | CER | Status |
|-------|------|-------------|--------------|--------------|-----|--------|
| `gha-whisper-small-twi-v6` | `openai/whisper-small` | Whisper seq2seq | **31.49%** | **30.44%** | 10.62% | ✅ **Current production** |

- Promotion gate: beats Round 2 greedy WER 32.83% ✓
- Production decode: `num_beams=5` in `modal/asr_service.py`
- Full Waxal test n=1522 (immutable)

### Historical Checkpoints (do not promote)

| Model | WER (greedy) | Notes |
|-------|--------------|-------|
| Round 2 (specaug v1) | 32.83% | Pre-v6 baseline |
| v3 | 33.99% | Overfit on validation |
| v4 | 34.96% | Overfit on validation |
| v5 | 34.13% | Overfit on validation |
| v7-lite (frozen encoder) | 40.86% (beam=5) | English-balanced, worse Twi |

### Alternative Base: DONDO (GhanaNLP)

| Model | Base | Architecture | Val WER | Val CER | Status |
|-------|------|-------------|---------|---------|--------|
| `gha-dondo-w2v-bert-twi-v1` | `KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en` | wav2vec2-BERT CTC | 35.77% | 12.19% | ❌ Not promoted |

- Fine-tuned on Waxal (train_limit=1800, 800 steps, lr=5e-6)
- Raw zero-shot WER on Waxal test: **71.91%** → fine-tuned to 35.77% val WER
- **Does not beat v6** (30.44% beam=5). Kept as research reference.
- Language-conditioned: prepend language id `2` for Asante Twi

### Key Benchmarks

| Model | Dataset | n | Beam | WER | CER |
|-------|---------|---|------|-----|-----|
| `openai/whisper-small` (zero-shot) | Waxal test | 300 | 1 | **113.26%** | 92.05% |
| `gha-whisper-small-twi-v6` | Waxal test | 100 | 5 | **32.16%** | 10.74% |
| `gha-whisper-small-twi-v6` | Waxal test | 300 | 5 | **30.44%** | 10.62% |
| `gha-dondo-w2v-bert-twi-v1` | Waxal test | 100 | — | **71.91%** | 26.69% |
| `gha-whisper-small-twi-v7-lite` | Waxal test | 100 | 5 | **40.86%** | 14.10% |

---

## 3. Training Recipe (v6)

### Data Mix

| Source | Config | Split | Weight | Filter | License |
|--------|--------|-------|--------|--------|---------|
| **google/WaxalNLP** | `aka_asr` | train | 0.60 (primary) | text≥2 chars, duration 0.5–28s | Research |
| **Common Voice 22** | `tw` | train | 0.25 | validated votes only (up≥2, down=0), max 5k | CC0 |
| **Common Voice 22** | `en` | train | 0.20 | validated votes only, max 5k | CC0 |
| **GhanaNLP multispeaker** | default | train | 0.15 | text≥2 chars, max 4k | CC BY-NC 4.0 |
| **Local recorder data** *(new)* | `manifest.jsonl` | — | 0.20 | 40 clips, 182.8s | Internal eval |

### Method

- **Full fine-tune** — encoder unfrozen (Whisper-small base is English-pretrained)
- **SpecAugment** — time mask 10% (length 12), feature mask 8% (length 20)
- **Speed perturbation** — {0.9, 1.0, 1.1}x
- **Early stopping** — patience 5, threshold 0.002 WER improvement
- **LR schedule** — cosine, warmup 200 steps, base lr 1e-5
- **Batch** — 8×4 grad accum (32 effective) for small; 4×8 for medium
- **Promotion gate** — auto full Waxal test; promote only if greedy WER < 32.83%

### Hardware

- Modal A100 GPU, 64GB RAM
- Runtime: ~2–3 hours for 3000 steps (small)
- Cost: ~$5–15 per full run

---

## 4. Local Data Collection

### Recorder Tool

- **File:** `tmp/asr-collection-pack/recorder.html`
- **Type:** Static HTML file, runs in browser via `file://`
- **Author:** Built by Codex for this project
- **Features:**
  - Prompts with reference text in Twi/English
  - Browser MediaRecorder (audio/webm;codecs=opus, 64kbps)
  - Per-bucket filtering (health_twi, commerce_twi, codeswitch_tw_en)
  - Speaker code input (e.g., `sp001`, `sp002`)
  - Single download + batch download + manifest export
  - "Mark recorded" for externally-recorded clips
  - Progress bar + status chips

### Collected Corpus (`tmp/asr-local-train/`)

| Bucket | Clips | Duration | Speaker(s) | Domain |
|--------|-------|----------|------------|--------|
| `health_twi` | 25 | 122.6s | sp001, sp002 | Symptoms, fever, malaria, pregnancy, breathing |
| `commerce_twi` | 10 | 39.2s | sp001 | Shopping, produce, medicine, delivery |
| `codeswitch_tw_en` | 5 | 20.9s | sp001 | Medicine names, symptoms, delivery requests |
| **Total** | **40** | **182.8s** | 2 speakers | Health + Commerce + Code-switching |

### Audio Spec

- Format: WAV mono 16-bit PCM
- Sample rate: 16 kHz
- Source: Converted from browser webm/opus via ffmpeg
- Quality: Phone recordings in quiet environment

### Manifest Schema

```json
{
  "id": "health_twi_sp001_u0001",
  "bucket": "health_twi",
  "language": "tw",
  "speaker_label": "speaker_001",
  "reference": "Me yam yɛ me ya na me ho yɛ hyew",
  "audio_path": "/root/gha_local_asr/audio/health_twi_sp001_u0001.wav",
  "duration_s": 5.04,
  "domain_tags": ["symptom", "fever"],
  "recording_tags": ["phone", "quiet", "wav16k"],
  "consent": "internal_eval"
}
```

> **Note:** `audio_path` uses `/root/gha_local_asr/` (Modal image path). `local_audio_path` records the macOS source for traceability.

---

## 5. Training Code State

### Current Patch (uncommitted)

**File:** `modal/train/train_asr.py`
**Status:** Modified on branch `codex/home-session-otp-auth` (wrong branch)

**What changed (+60 lines):**
1. Image packaging — `add_local_dir()` mounts `tmp/asr-local-train/` → `/root/gha_local_asr/`
2. `_load_local_manifest()` — loads manifest, validates audio paths, casts to `datasets.Audio(16000)`
3. Data mix — local gets weight 0.20 when present (normalized with Waxal 0.60 + extras)
4. CLI flags — `use_local_data`, `local_manifest_path`, `local_weight` exposed
5. Model card — documents local recorder data usage

### Smoke Test Command

```bash
modal run modal/train/train_asr.py \
  --smoke --no-wait \
  --run-name local-smoke \
  --train-limit 64 --eval-limit 16 \
  --use-local-data --local-weight 0.20
```

**Expected log:** `[train] local ASR ready n=40 skipped=0 manifest=/root/gha_local_asr/manifest.jsonl`

### Full Run Command

```bash
modal run --detach modal/train/train_asr.py \
  --base-model openai/whisper-small \
  --run-name v6-small-local --max-steps 3000 \
  --push-repo teckedd/gha-whisper-small-twi-v6-local \
  --use-local-data --local-weight 0.20 \
  --no-wait
```

---

## 6. Production Service

**File:** `modal/asr_service.py`

- **Endpoint:** Modal ASGI app with CPU health check + GPU transcription
- **Model:** `teckedd/gha-whisper-small-twi-v6` (fallback to Round 2 if v6 fails)
- **Decode:** `num_beams=5` (best WER), max 45s audio
- **Preprocessing:** ffmpeg webm/opus → 16 kHz wav, silence rejection (RMS < 0.008)
- **Volume:** `ghana-health-asr-models` caches HF downloads

---

## 7. Evaluation & Benchmarking

### Benchmark Tools

- `modal/train/benchmark_asr_ladder.py` — multi-model comparison
- `modal/train/eval_asr.py` — single model evaluation
- `modal/train/eval_dondo_asr.py` — DONDO-specific eval (language id prefix)

### Results Directory

`tmp/asr-results/` contains JSON benchmarks for:
- Baseline (openai/whisper-small zero-shot)
- v3, v4, v6, v7-lite
- DONDO zero-shot and fine-tuned
- English regression tests (Common Voice 22 en test)

### Product Eval Contract

`data/asr-product-eval/manifest.example.jsonl` defines the schema for health/commerce domain evaluation:
- health_twi, commerce_twi, codeswitch_tw_en, health_en, phone_noise
- Consent tracking, domain tags, recording tags

---

## 8. Known Issues & Blockers

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| **Recorder HTML needs refresh for new buttons** | Low | Fixed in file | Static `file://` page; user must reload to get UI updates |
| **train_asr.py patch uncommitted** | Medium | Uncommitted | On wrong branch (`codex/home-session-otp-auth`) |
| **Modal CLI not installed in this env** | Low | Can install | `pip install modal` works; auth via `~/.modal.toml` |
| **Local corpus small (40 clips, 3 min)** | Medium | Acceptable for adaptation | Not enough to train standalone; works as weighted mix |
| **DONDO underperforms vs Whisper** | Low | Documented | 35.77% val WER vs 30.44% beam=5; kept as research |
| **English regression risk** | Medium | Mitigated | CV22 English retention mix (weight 0.20) in v6 recipe |

---

## 9. Next Steps

### Immediate (this session)

1. ✅ **Commit the patch** to a proper feature branch
   ```bash
   git checkout -b feat/local-asr-adaptation
   git add modal/train/train_asr.py
   git commit -m "feat(asr): add local recorder manifest to training mix"
   ```

2. 🔄 **Run smoke test** from Modal-equipped environment
   ```bash
   modal run modal/train/train_asr.py --smoke --no-wait \
     --run-name local-smoke --use-local-data
   ```

3. 🎯 **Launch real run** if smoke passes
   ```bash
   modal run --detach modal/train/train_asr.py \
     --run-name v6-small-local --max-steps 3000 \
     --push-repo teckedd/gha-whisper-small-twi-v6-local \
     --use-local-data --local-weight 0.20 --no-wait
   ```

### Short-term (next 2–4 weeks)

4. **Scale local corpus** — target 200+ clips, 15+ min per bucket
   - Health Twi: 100+ clips (symptoms, medicines, pregnancy, child health)
   - Commerce Twi: 50+ clips (shopping, prices, delivery, bargaining)
   - Code-switch: 50+ clips (Twi-English mixed)
   - Add more speakers (sp003, sp004...) for diversity

5. **Domain-specific eval** — run product eval on health/commerce test set
   - Measure WER per bucket (health vs commerce vs code-switch)
   - Compare before/after local adaptation

6. **English regression check** — run Common Voice 22 en test
   - Baseline: openai/whisper-small on English
   - v6: our fine-tuned model on English
   - Local-adapted v6: verify no regression

7. **DONDO further experiments** (low priority)
   - Try larger train_limit (e.g., 3000)
   - Add local data to DONDO training
   - Compare CTC vs seq2seq for production latency

### Medium-term (1–3 months)

8. **Whisper-medium ladder** — if small beats 28% WER, train medium
   - Expected: ~25–27% WER with more parameters
   - Higher cost (~$10–20 per run)

9. **On-device inference** — investigate ONNX/Core ML export
   - Reduce latency for mobile health app

10. **Community data partnership** — collaborate with GhanaNLP, KNUST linguistics
    - Ethical data collection with proper consent
    - Expand to other Akan dialects (Fante, Akuapem)

---

## 10. Directory Map

```
ghana-health-ai/
├── modal/
│   ├── train/
│   │   ├── train_asr.py              ← v6 Whisper trainer (+local data patch)
│   │   ├── train_dondo_asr.py        ← DONDO w2v-bert trainer
│   │   ├── eval_asr.py               ← Single model eval
│   │   ├── eval_dondo_asr.py         ← DONDO-specific eval
│   │   ├── benchmark_asr_ladder.py   ← Multi-model comparison
│   │   ├── model_card.py             ← HF README generator
│   │   └── push_model_card.py        ← Hub push helper
│   ├── asr_service.py                ← Production Modal ASGI service
│   ├── tts_service.py                ← Text-to-speech service
│   └── embed_service.py              ← Embedding service
├── tmp/
│   ├── asr-local-train/              ← Clean local corpus (40 WAV + manifest)
│   │   ├── audio/
│   │   ├── manifest.jsonl
│   │   └── summary.json
│   ├── asr-collection-pack/          ← Recorder tool + prompts
│   │   ├── recorder.html
│   │   ├── prompts.csv
│   │   └── prompts.jsonl
│   ├── asr-results/                  ← Benchmark JSON outputs
│   ├── hf-readmes/                   ← Generated model cards
│   └── first-sample-asr/             ← Early experiments
├── data/
│   └── asr-product-eval/             ← Product evaluation schema
├── src/                              ← Next.js web app
└── AGENTS.md                         ← Project playbook
```

---

## 11. References

- **Waxal ASR dataset:** `google/WaxalNLP` (aka_asr config)
- **Common Voice 22:** `fsicoli/common_voice_22_0` (tw, en configs)
- **GhanaNLP multispeaker:** `ghananlpcommunity/twi-speech-text-multispeaker-16k`
- **DONDO base:** `KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en`
- **Production model:** `teckedd/gha-whisper-small-twi-v6`
- **Product:** [ghanahealth.serendepify.com](https://ghanahealth.serendepify.com)

---

*Document generated from workspace audit on 2026-08-15. All metrics derived from `tmp/asr-results/` JSON files and `modal/train/train_asr.py` source.*
