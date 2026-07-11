# Model training roadmap — Ghana Health AI

**Status:** Product pipeline works (listen → transcript → understand → speak).  
**Focus now:** Raise model quality until Twi (then Ga, Ewe, Dagbani) is near-native.

Do not spend more cycles on UI gimmicks. Every improvement below should move **WER**, **CER**, **TTS MOS**, or **health-answer faithfulness**.

---

## Current production baselines

| Component | Live model | Known ceiling |
|-----------|------------|---------------|
| **ASR** | `teckedd/whisper-small-waxal-round2-specaug-v1` | ~**32.8% WER** held-out Waxal test |
| **TTS (Twi)** | `facebook/mms-tts-aka` | Generic Akan; not health-domain, not multi-speaker natural |
| **TTS (EN)** | `facebook/mms-tts-eng` | Acceptable English; not Ghana-accent |
| **Understand** | Groq Llama / OpenAI chat | General LLM; not Twi-native; health jargon leakage fixed in prompt only |

Your HF lineage already continues past Round 2 (`serendepify-gsl-asr-ak-…-v0.1` … `v0.6`). Treat **Round 2 as the production ASR baseline** and **v0.6+ as experimental** until they beat Round 2 on a **fixed immutable test**.

---

## North-star metrics

| Metric | Target (Twi MVP) | Stretch |
|--------|------------------|---------|
| ASR WER (held-out multi-domain) | **&lt; 22%** | **&lt; 15%** |
| ASR CER | **&lt; 8%** | **&lt; 5%** |
| Health-domain ASR WER (clinic phrases) | **&lt; 25%** | **&lt; 18%** |
| TTS preference (A/B vs MMS-aka) | **≥ 70% prefer new** | natural + clear numbers |
| Understand: answer grounded on user utterance | **≥ 95%** human pass | Twi-first, no English-only collapse |
| Hallucination rate on silence/noise | **~0%** (already gated in serving) | keep gates after each train |

**Hard rule:** Never promote a checkpoint on training loss alone. Promote only if it wins on:

1. Immutable Waxal-style test (speaker-disjoint)  
2. GhanaNLP / field-phone domain set  
3. Health phrase set (danger signs, maternal, OTC)  
4. Manual listen of 50 random clips  

---

## Phase A — Data (weeks 0–2)

Everything good starts with clean, versioned data.

### A1. ASR corpora (priority order)

| Source | Role | Notes |
|--------|------|--------|
| **google/WaxalNLP `aka_asr`** | Foundation | Already drove Round 2; keep splits frozen |
| **GhanaNLP speech (Twi / Ewe / Ga / Dagbani)** | Domain + dialect | Clean transcripts; phone bandwidth |
| **UGSpeechData** (if license + access) | Scale | Hours/language; may be partially unlabeled |
| **AfriSpeech-200 (GH accents)** | Accent robustness | Medical English + local accents |
| **In-app consented audio** | Production mismatch | Store only with consent; 16 kHz mono; never train on without opt-in |

### A2. Health / understanding text

| Source | Role |
|--------|------|
| GAIN maternal health Q&A (code-mixed) | Reply style + topics |
| GHS / WHO maternal summaries (licensed) | Factual backbone for RAG |
| Parallel GhanaNLP Twi↔EN | Translation + mixed replies |
| Your product logs (consented) | Real intents: fever, market, ANC |

### A3. TTS speech

| Source | Role |
|--------|------|
| GhanaNLP / Bible / open Twi speech-text | Prosody + phonemes |
| Studio or phone read scripts (health phrases) | Domain clarity (numbers, ɛ/ɔ) |
| Single high-quality speaker first | Stability before multi-speaker |

### A4. Data contracts (non-negotiable)

- Unicode NFC, keep `ɛ` `ɔ`, collapse whitespace, strip punctuation for ASR labels (match Round 2).  
- Speaker IDs; **no speaker leak** across train/dev/test.  
- Clip duration 0.4–30 s; drop silence-only.  
- Manifest JSONL: `{id, path, text, language, speaker, domain, split, duration_s, source}`.  
- Version manifests on HF as datasets (e.g. `teckedd/gha-asr-manifest-v1`).

---

## Phase B — ASR (weeks 1–6)

### B1. Scoreboard (full Waxal test n=1522, greedy)

| Checkpoint | WER | CER | Val WER | Status |
|------------|-----|-----|---------|--------|
| **Round 2** greedy | **32.83%** | **11.79%** | — | checkpoint weights |
| **Round 2** `num_beams=5` | **31.52%** | **11.27%** | — | **production decode (shipped)** |
| v3 `gha-whisper-small-twi-v3` | 33.99% | 12.21% | low | do not promote (same-Waxal FT overfit) |
| v4 `gha-whisper-small-twi-v4` | 34.96% | 12.62% | ~27.7% | do not promote (freeze-enc still overfit) |
| v5 `gha-whisper-small-twi-v5` | **34.13%** | **12.29%** | **26.19%** | do not promote (+1.30pp vs R2; multi-source still overfits) |
| v6-small from `openai/whisper-small` | TBD | TBD | TBD | Waxal + **Common Voice 22 Twi validated** + GhanaNLP |
| v6-medium from `openai/whisper-medium` | TBD | TBD | TBD | same data, larger base |

**Hard rule:** Promote only if full-test WER &lt; 32.83% greedy (or better than beam=5 serving bar 31.52% for decode-matched compare). Val WER alone is not enough.

**v5 takeaway:** Continued FT from Round 2 overfits. **v6** retrains from OpenAI bases with:
- `google/WaxalNLP` aka_asr train (foundation; test frozen)
- `fsicoli/common_voice_22_0` config `tw` (CC0) — train split + `up_votes≥2 & down_votes==0`
- optional GhanaNLP Twi multispeaker (lower weight)

```bash
# small first
modal run --detach modal/train/train_asr.py \
  --base-model openai/whisper-small \
  --run-name v6-small --max-steps 3000 \
  --push-repo teckedd/gha-whisper-small-twi-v6 --no-wait

# then medium
modal run --detach modal/train/train_asr.py \
  --base-model openai/whisper-medium \
  --run-name v6-medium --max-steps 2500 \
  --batch-size 4 --learning-rate 8e-6 \
  --push-repo teckedd/gha-whisper-medium-twi-v6 --no-wait
```

### B1b. v5 train (current)

```text
Base: teckedd/whisper-small-waxal-round2-specaug-v1
Data: Waxal train (65%) + ghananlpcommunity/twi-speech-text-multispeaker-16k (35%)
      note: GhanaNLP set is CC BY-NC — research train only
Method: freeze encoder, decoder LR 8e-6, label_smoothing 0.1,
        speed-pert {0.9,1.0,1.1}, strong SpecAug, early-stop on val WER
GPU: Modal A100
Eval: immutable full Waxal test auto-runs after train
```

**Script:** `modal/train/train_asr.py`

```bash
modal run modal/train/train_asr.py \
  --run-name v5 \
  --max-steps 800 \
  --push-repo teckedd/gha-whisper-small-twi-v5
```

### B2. Scaling ladder

1. **v5-mix** — Round 2 + GhanaNLP Twi multispeaker; beat 32.8% Waxal WER *without* regression.  
2. **v5-health** — + health phrase reads + AfriSpeech GH medical.  
3. **v6-multi** — joint Twi + Ewe + Ga + Dagbani with language tags / balanced sampling.  
4. **Larger base only if needed** — `whisper-medium` or `large-v3-turbo` if small plateaus &gt; 25% WER.

### B3. Serving upgrades after each good train

- Point `MODEL_ID` / `DEFAULT_MODEL` in `modal/asr_service.py`.  
- Keep silence/hallucination gates.  
- Log WER on a weekly 100-clip probe set.

---

## Phase C — TTS (weeks 3–8)

### C1. Short term

- Keep **MMS-aka / MMS-eng** for production.  
- Expand acronyms before synth (done).  
- Collect 2–5 hours of **clean Twi health scripts** (one speaker).

### C2. Fine-tune path

| Option | Pros | Cons |
|--------|------|------|
| **Fine-tune MMS-VITS on Twi health reads** | Same stack as prod | Voice quality limited by VITS |
| **XTTS-v2 / StyleTTS2 on Ghana data** | More natural | Heavier; license check |
| **Piper / ONNX for edge** | Offline PWA | Separate train pipeline |

**Script skeleton:** `modal/train/train_tts.py`  
Promote only on **blind A/B** vs `mms-tts-aka` (20 listeners, health sentences + numbers).

### C3. English Ghana-accent (later)

- Optional LoRA on English TTS with GH-accent reads — not blocking Twi.

---

## Phase D — Understanding (weeks 2–8)

Understanding quality is **not only “bigger LLM”**. Split:

### D1. Retrieval (fast win)

- Seed + expand `KnowledgeArticle` with maternal, malaria, danger signs in **Twi + English**.  
- Wire `generateHealthReply` / RAG back into the chat path when intent is HEALTH.  
- Cite sources in metadata (not always in spoken text).

### D2. Preference / SFT data

Build 2–5k rows:

```json
{"lang":"tw","user":"...","assistant":"...","intent":"HEALTH","severity":"MEDIUM"}
```

Sources: GAIN Q&A cleaned, clinician-reviewed templates, synthetic variants from strong teacher models **then human-filtered**.

### D3. Fine-tune options

| Approach | When |
|----------|------|
| Prompt + RAG only | Now–v1.1 |
| LoRA on Llama-3.1-8B / Qwen2.5-7B (Twi-EN health) | After 2k+ clean pairs |
| Smaller on-device SLM | After cloud model is good |

**Script skeleton:** `modal/train/train_understand.py`

Eval: 100 fixed scenarios (danger sign, market price, nonsense ASR, mixed Twi-EN). Score **faithfulness**, **language match**, **no acronym dump**, **escalation correctness**.

---

## Phase E — Languages beyond Twi

Order by data + population impact:

1. **Twi (Asante/Akuapem) + Fante** — primary  
2. **Ewe**  
3. **Ga**  
4. **Dagbani**  

Per language: minimum **50–100 h** labeled speech for competitive ASR, or transfer from multi-lingual Whisper with heavy adapter. Do not claim “perfect” without per-language test sets.

---

## Phase F — Ops for training

| Item | Recommendation |
|------|----------------|
| Compute | Modal H100/A100 jobs; volumes for datasets + checkpoints |
| Secrets | HF write token as Modal secret `huggingface` |
| Experiment log | W&B or simple HF model card metrics table |
| Promotion | Tag `prod-asr`, `prod-tts`, `prod-understand` on HF |
| Cost | Cap concurrent GPU; kill failed crash-loops; short scaledown on serve |

---

## 90-day plan (concrete)

| Week | Deliverable |
|------|-------------|
| 1 | Frozen manifests v1; health phrase list; eval scripts (WER/CER) |
| 2 | ASR train from Round 2 → candidate **v3-twi-clean** |
| 3 | Immutable + health eval; promote only if wins |
| 4–5 | TTS data collection + first fine-tune A/B |
| 5–6 | HEALTH RAG live on chat path; 500 SFT rows |
| 7–8 | Understand LoRA or strong RAG; end-to-end human eval (20 users) |
| 9–12 | Ewe/Ga pilot adapters; production `MODEL_ID` upgrades |

---

## Commands (after scripts are wired)

```bash
# ASR fine-tune (Modal)
modal run modal/train/train_asr.py --help

# Eval only against a checkpoint
modal run modal/train/eval_asr.py --model-id teckedd/whisper-small-waxal-round2-specaug-v1

# Deploy a winning ASR checkpoint
# set MODEL_ID then:
modal deploy modal/asr_service.py
```

---

## What “near perfect” means here

Not zero WER. For rural health voice:

- User **recognizes their own words** in the transcript most of the time  
- Danger phrases are almost never missed  
- Twi TTS is **intelligible and respectful**, not cartoonish  
- Answers stay on-topic and in the right language  

That is the bar. Train until the metrics and human listens say you are there.
