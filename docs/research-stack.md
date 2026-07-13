# Research stack — Ghana Health AI

**Product goal:** A working voice-first health companion that **listens, understands, and speaks Twi** for maternal and everyday health in Ghana — English only when the user asks for it.

This document is the technical north star. Implementation follows published work and public assets, not hand-written banks or English prompt tricks.

---

## 1. Why this stack

| Layer | Wrong path (we left) | Research path (we take) |
|-------|----------------------|-------------------------|
| “Semantics” | 40-row TS bank + char noise | **ABENA** (Ghana-NLP) Twi BERT mean-pool embeddings |
| Understanding | English system prompts + keyword RAG | Twi-first generation grounded on **ABENA retrieval over Twi knowledge** + GhanaNLP parallel for SFT |
| ASR | Single-corpus FT, in-domain only | **Waxal `aka_asr`** + multi-domain mix; promote on **cross-set** WER (Mensah et al. 2025) |
| TTS | Stock MMS forever | MMS-aka → finetune on GhanaNLP / Bible / health reads; A/B |
| Closed SOTA | Ignore | **Khaya** as production oracle/teacher to beat over time |

### Key literature & assets

| Area | Source |
|------|--------|
| Twi BERT (ABENA / DistilABENA / ROBAKO) | Ghana-NLP on HF — mBERT→JW300 Twi→Asante Bible |
| Akan ASR multi-domain benchmark | Mensah, Wiafe et al. (UG / Oulu), 2025 — Whisper vs Wav2Vec2; domain collapse |
| Open Akan ASR scale | Google **WAXAL** (`google/WaxalNLP`, University of Ghana Akan) |
| Community speech | `ghananlpcommunity/twi-speech-text-multispeaker-16k`, Common Voice Twi, Bible speech |
| Parallel MT text | `Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT` (+ Fante/Ewe/Ga/Kusaal) |
| Production Ghanaian speech+MT | **Khaya AI** (GhanaNLP / Algorine) APIs |
| African multilingual LMs | AfriBERTa, AfroXLMR, SERENGETI — Twi often weak; use carefully |

We do **not** claim native Twi competence in the model or the tooling. Native speakers and published eval sets are ground truth.

---

## 2. End-to-end pipeline (research-aligned)

```
Mic (16 kHz)
  → Modal ASR: Whisper fine-tuned on Waxal (+ multi-domain mix)
  → Transcript (noisy OK)
  → ABENA encode (Asante Twi BERT)
  → Retrieve top-k KnowledgeArticle (Twi body) + Product (Twi name) by cosine
  → Twi-first LLM / future LoRA understand model
       · preferredLang === en → English replies only
       · else Twi (light code-mix OK)
  → Modal TTS: MMS-aka (health finetune when ready)
  → Speak
```

Safety (danger signs → escalate) remains a **hard gate**, not the “NLU model.”

---

## 3. Component decisions

### 3.1 ASR

- **Serve:** `teckedd/gha-whisper-small-twi-v6` until a checkpoint beats it on **full Waxal test + held-out domain probes**.
- **Train:** `modal/train/train_asr.py` — OpenAI Whisper base + Waxal train + CV Twi + optional GhanaNLP multispeaker.
- **Promote only if:** full Waxal WER improves **and** no catastrophic regression on a second domain (CV or health phrase set).
- **Literature constraint:** in-domain WER alone is invalid (Mensah et al.).

### 3.2 Text encoder (Twi)

- **Serve:** `Ghana-NLP/abena-base-asante-twi-uncased` via `modal/embed_service.py` (mean-pool last hidden state, L2-norm).
- **Use:** retrieve Twi knowledge / products; log similarity for training data mining.
- **Not used for:** fake intent banks.

### 3.3 Understanding / dialogue

- **Now:** Twi-first LLM generation + ABENA retrieval of **Twi** article text (no English KB dump as primary context).
- **Train:** `modal/train/train_understand.py` on Ghana-NLP parallel + future health SFT (`messages` chat format, Twi assistant default).
- **Later:** LoRA on Llama/Qwen with Twi-majority rows; optional Khaya MT as teacher for back-translation.
- **Never:** English-only collapse when `preferredLang ≠ en`.

### 3.4 TTS

- **Serve:** `facebook/mms-tts-aka` (+ eng only if preferred English); number/chunk prep in `tts_service.py`.
- **Train:** GhanaNLP / Bible / health script finetune; promote on blind A/B.

### 3.5 Khaya

- Optional env keys for oracle ASR/TTS/MT in eval scripts — **not** required for open path.
- Goal remains open, self-hosted Modal weights for the product.

---

## 4. Evaluation (non-negotiable)

| Task | Metric | Gate |
|------|--------|------|
| ASR | WER/CER full Waxal test + 1 OOD set | Beat previous prod |
| Encoder retrieve | Recall@k of correct health article on Twi queries | Human spot-check 50 |
| Understand | Twi language match, faithfulness, escalation | ≥ human pass on 100 scenarios |
| TTS | Preference vs stock MMS-aka | ≥ 70% prefer new |

---

## 5. What we deleted philosophically

- Hand-authored semantic banks as “model intelligence”
- Synthetic char-drop as a substitute for real ASR noise data
- English prompt stuffing of guidelines as “Twi understanding”

Those may still exist as dead code until removed; **runtime path must not depend on them.**

---

## 6. Commands

```bash
# Twi ABENA embeddings (LIVE)
modal deploy modal/embed_service.py
# Infisical: MODAL_EMBED_URL=https://createdliving1000--ghana-health-embed-embed.modal.run

# ASR train (research protocol)
modal run --detach modal/train/train_asr.py \
  --base-model openai/whisper-small \
  --run-name v7-multi --max-steps 3000 \
  --push-repo teckedd/gha-whisper-small-twi-v7

# Understand SFT from GhanaNLP parallel
# Prefer open bases (Qwen2.5-1.5B/7B). Llama needs HF gate access.
# Modal secret huggingface-token must be able to download public weights
# (403 on cas-bridge ⇒ fix token permissions / re-auth).
modal run --detach modal/train/train_understand.py \
  --use-ghananlp-parallel --max-steps 400 \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --push-repo teckedd/gha-understand-twi-v1
```

---

## 7. Success = product goal

Users in Ghana can **speak Twi about pregnancy and everyday health**, get a **clear Twi answer**, and **hear it back** — without being forced through English, and without the system pretending a JSON bank is NLP research.
