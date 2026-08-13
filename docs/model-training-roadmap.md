# Model training roadmap — Ghana Health AI

**North star doc:** [`docs/research-stack.md`](./research-stack.md)  
**Product goal:** Working Twi-first voice health companion (listen → understand → speak).

This roadmap follows **published research and public Ghanaian/African assets**, not heuristics.

---

## Language policy

| Rule | Detail |
|------|--------|
| Default | **Twi** replies + Akan TTS |
| English | **Only** if `preferredLang === en` |
| Never | Collapse to English because ASR looked Englishy |

---

## Stack (research-aligned)

| Layer | Live / target | Research basis |
|-------|---------------|----------------|
| **ASR** | `teckedd/gha-whisper-small-twi-v6` | Waxal Akan (UG/Google); multi-domain caution (Mensah et al. 2025) |
| **Encoder** | `Ghana-NLP/abena-base-asante-twi-uncased` | ABENA — Azunre / Ghana NLP |
| **Retrieve** | ABENA cosine over Twi knowledge/products | Twi-native retrieval, not EN RAG |
| **Understand** | Twi-first LLM → LoRA on GhanaNLP parallel | Ghana-NLP `TWI_ENGLISH_PARALLEL_TEXT` |
| **TTS** | `facebook/mms-tts-aka` | Community Twi TTS data for finetune |
| **Oracle** | Khaya APIs (optional eval) | Production Ghanaian speech+MT bar |

**Removed from the intelligence path:** hand-written `semantic-bank.ts`, char-drop “semantics”, English guideline stuffing.

---

## ASR

### Baseline

| Checkpoint | WER (Waxal full, beam=5) | Status |
|------------|--------------------------|--------|
| Round 2 | 31.52% | previous |
| **v6-small** | **30.44%** | **production** |
| v6-medium | train when scheduled | larger base |

### Protocol (Mensah et al.)

1. Train multi-source (Waxal train + CV Twi + optional GhanaNLP speech).  
2. Eval **full Waxal test** + at least one OOD probe.  
3. Promote only on cross-set improvement — never in-domain loss alone.

```bash
modal run --detach modal/train/train_asr.py \
  --base-model openai/whisper-small \
  --run-name v7-multi --max-steps 3000 \
  --push-repo teckedd/gha-whisper-small-twi-v7
```

---

## Twi encoder (ABENA)

```bash
modal deploy modal/embed_service.py
# MODAL_EMBED_URL = https://…--ghana-health-embed-embed.modal.run
```

Mean-pool last hidden state, L2-norm. Used by `src/lib/twi-retrieve.ts`.

---

## Understanding SFT

```bash
modal run modal/train/train_understand.py --use-ghananlp-parallel --smoke
modal run --detach modal/train/train_understand.py \
  --use-ghananlp-parallel --max-steps 500 \
  --push-repo teckedd/gha-understand-twi-v1
```

Eval: 100 scenarios — **Twi language match**, faithfulness, escalation, no acronym dump.

---

## TTS

Short term: MMS-aka + number/chunk serving.  
Next: finetune on GhanaNLP / Bible / health reads; blind A/B ≥ 70% prefer new.

---

## 90-day research plan

| Week | Deliverable |
|------|-------------|
| 1 | ABENA embed live; product retrieve path; research-stack.md |
| 2 | GhanaNLP parallel SFT smoke + full LoRA |
| 3–4 | ASR multi-domain / medium train; cross-set report |
| 5–6 | Health Twi SFT rows (clinician filter); human eval n=20 |
| 7–8 | TTS finetune A/B |
| 9–12 | Optional Khaya oracle compare; Ga/Ewe only after Twi holds |

---

## Success

Users speak Twi about health, get a clear **Twi** answer, hear it back — grounded in real Akan NLP research, not a JSON bank.
