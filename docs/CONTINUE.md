# Continue here — Ghana Health AI

**Last session focus:** Research-driven Twi stack (ABENA + clean product UI).  
**North star:** [`docs/research-stack.md`](./research-stack.md)  
**Training roadmap:** [`docs/model-training-roadmap.md`](./model-training-roadmap.md)

---

## Product goal (do not lose this)

Voice-first **Twi** health companion for Ghana:

**listen (ASR) → understand (Twi-native path) → speak (TTS)**  

English **only** when the user sets language preference to English.

---

## What’s already in this branch

### App / product
- Clean product UI (home, voice, chat, market, login) — no debug/model chrome in the UI
- `understandUtterance` is **research path**: ABENA retrieve → Twi-first LLM (no semantic bank)
- `src/lib/twi-retrieve.ts` — cosine retrieve over Twi knowledge + products
- `src/lib/embed.ts` — Modal ABENA client (`MODAL_EMBED_URL`)
- Language gate: `resolveReplyLanguage` → EN only if `preferredLang === en`
- Expanded knowledge seed (`prisma/seed.ts`)
- TTS serving upgrades: number expansion, chunking, peak normalize (`modal/tts_service.py`)
- HF model-card helper on train pushes (`modal/train/model_card.py`, `push_model_card.py`)

### Modal (deployed this session)
| Service | URL / note |
|---------|------------|
| **ABENA embed** | `https://createdliving1000--ghana-health-embed-embed.modal.run` |
| Embed health | `https://createdliving1000--ghana-health-embed-health.modal.run` |
| TTS speak | `https://createdliving1000--ghana-health-tts-speak.modal.run` (redeployed with chunking) |
| ASR | Existing prod: v6 Whisper small Twi |

Smoke-tested embed: model `Ghana-NLP/abena-base-asante-twi-uncased`, dim **768**, engine `abena-mean-pool`.

---

## Remaining work (priority order)

### 1. Wire secrets (blocking product path)

In Infisical (dev + prod):

```bash
MODAL_EMBED_URL=https://createdliving1000--ghana-health-embed-embed.modal.run
# keep existing:
# MODAL_ASR_URL=...
# MODAL_TTS_URL=https://createdliving1000--ghana-health-tts-speak.modal.run
```

Redeploy / sync secrets to VPS (`infisical-sync` or hourly job). Confirm `/api/config` shows `abenaEmbed: true`.

Re-seed knowledge if prod DB is stale:

```bash
sec -- npx prisma db seed
```

### 2. Fix HF token for LoRA SFT (blocked)

Modal secret `huggingface-token` returns **403** on Qwen/Llama weight downloads (Xet CDN). ABENA still loads fine.

**Do this:**
1. Create/refresh a HF token with **read** (and **write** if pushing).
2. Update Modal: `modal secret create huggingface-token HF_TOKEN=hf_...` (or edit in dashboard).
3. Re-run understand SFT:

```bash
modal run --detach modal/train/train_understand.py \
  --use-ghananlp-parallel \
  --max-steps 400 \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --push-repo teckedd/gha-understand-twi-v1
```

After push, optionally serve the LoRA (new Modal understand service) or keep Groq/OpenAI for generation until LoRA quality wins eval.

### 3. ASR — multi-domain promote (research protocol)

- Check any in-flight `v6-medium` / train jobs on [Modal dashboard](https://modal.com/apps).
- Promote only if **full Waxal test + OOD probe** beat v6-small (Mensah et al.: in-domain alone is invalid).
- Backfill model cards for bare HF repos:

```bash
modal run modal/train/push_model_card.py \
  --repo teckedd/gha-whisper-small-twi-v6 \
  --wer 0.3044 --cer 0.1062
```

### 4. TTS finetune (not just serving prep)

- Collect 2–5h clean Twi health reads (single speaker first).
- `modal/train/train_tts.py` is still a data-validation skeleton — implement waveform train or use external recipe.
- Blind A/B vs `facebook/mms-tts-aka` before promoting.

### 5. Eval harness (required for “research”)

| Task | Metric | Gate |
|------|--------|------|
| ASR | WER/CER Waxal full + 1 OOD | Beat prod |
| ABENA retrieve | Recall@k on Twi health queries | Human spot-check 50 |
| Understand | Twi language match, faithfulness, escalation | 100 scenarios |
| TTS | Preference vs MMS-aka | ≥ 70% prefer new |

### 6. Optional later

- Khaya API as **oracle** (ASR/TTS/MT) for side-by-side eval — not required for open path.
- OpenMed / clinical plugs — **after** Twi basics hold (`docs/research-stack.md`).
- Ga / Ewe / Dagbani — stay on Twi path until first-class data exists.

---

## Key files

| Path | Role |
|------|------|
| `docs/research-stack.md` | Architecture + literature |
| `docs/CONTINUE.md` | This handoff |
| `src/lib/understand.ts` | Twi-first + retrieve metadata |
| `src/lib/twi-retrieve.ts` | ABENA retrieval |
| `src/lib/embed.ts` | Embed client |
| `modal/embed_service.py` | ABENA Modal service |
| `modal/train/train_understand.py` | GhanaNLP parallel → LoRA |
| `modal/train/train_asr.py` | Waxal multi-source ASR |
| `modal/tts_service.py` | Akan/EN TTS + prep |
| `modal/train/model_card.py` | Required HF cards on push |

---

## Quick verify after secrets

```bash
# local with Infisical
sec -- npm run dev

# embed live
curl -s https://createdliving1000--ghana-health-embed-health.modal.run | jq .

# config flags
curl -s http://localhost:3000/api/config | jq .abenaEmbed,.stack
```

Speak a Twi health phrase on `/voice` — reply should be Twi when language is Twi; message metadata may include `retrieve.engine: "abena"`.

---

## Explicitly abandoned

- Hand-written `semantic-bank.ts` / char-drop “semantics” as product intelligence  
- English knowledge dumps as the primary NLU brain  
- Treating in-domain ASR WER alone as promotion proof  

Continue from **§ Remaining work** above.
