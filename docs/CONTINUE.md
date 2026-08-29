# Continue here — Ghana Health AI

## 2026-08-28 understanding research checkpoint

Current checkpoint:

- Production is live and ready at `https://ghanahealth.serendepify.com`.
- Research workspace: `/research/ase`.
- Latest pushed commit before this checkpoint: `8da6061`.
- New local work adds the first corpus candidate builder and candidate review
  queue; commit/deploy after validation.

Important correction:

- The 50 synthetic benchmark rows are **only** for model comparison.
- The actual corpus comes from licensed datasets, local recordings, opt-in
  contributions, and reviewed synthetic augmentation.
- Models may populate draft transcript/meaning/entity fields, but the project
  must train only from reviewed/finalized exports.

Completed in this pass:

- Fixed Modal NLLB benchmark runner by loading the base NLLB tokenizer against
  the fine-tuned `ninte/twi-en-nllb-v2` weights.
- Ran the 50-row Modal benchmark successfully.
- Built `data/understanding-corpus/candidates.v0.jsonl` with 80 review-ready
  candidates:
  - 30 local-recording rows
  - 50 curated prompt rows
  - 80 draft model annotations
- Added `scripts/build-understanding-corpus-candidates.ts`.
- Added `scripts/summarize-understanding-research.ts`.
- Added report: `docs/understanding-research-report-2026-08-28.md`.

Key result:

- NLLB is fast but not safe as a sole health meaning annotator. It mistranslated
  multiple meaning-critical benchmark rows, including unwellness, pregnancy
  symptoms, eye pain, body-part location, and commerce budget.

## 2026-08-26 understanding corpus workspace

The understanding research direction is **not** to train from the 50 synthetic
benchmark rows. Those rows remain a small probe set for comparing model
behaviour and latency.

New internal workspace:

- `/research/ase`
- API: `/api/research/understanding`
- Local review writes: `tmp/understanding-review/reviews.v0.jsonl`
- Access: open in local development; in production requires an admin/researcher
  account or `RESEARCH_REVIEW_ENABLED=true`

The workspace now separates:

1. **Sources:** GhanaNLP parallel, WAXAL, Common Voice, local recordings,
   opt-in product contributions, and reviewed synthetic TTS.
2. **Storage decisions:** public datasets stay referenced by pinned revisions;
   private audio belongs in private object storage; Postgres stores references,
   hashes, reviews, and experiment metadata.
3. **Benchmark:** the 50 synthetic probes are explicitly labelled as
   non-training, non-gold model-comparison rows.
4. **Review:** native-speaker review can add normalized Twi, faithful English
   meaning, literal gloss, intent, entities, ambiguity, decision, and notes.

Validation run:

```bash
pnpm lint
pnpm build
```

Next research step is Gate 1, not a shortcut:

1. Create the import/audit layer for dataset-derived corpus candidates.
2. Catalogue source IDs, licences, revisions, splits, hashes, speaker metadata,
   and consent scope.
3. Generate model proposals only as drafts.
4. Use `/research/ase` for human review and export only eligible reviewed
   records.

## 2026-08-26 pull/deploy update

Pulled the latest main branch through `9eca525`, then added one safety-path fix
and deployed production via the direct VPS route.

Current live deploy:

- Commit: `32ec6f71df7ed0493d8d386a8d38a73508ec3d98`
- Web image: `ghcr.io/teckedd-code2save/ghana-health-ai:32ec6f71df7ed0493d8d386a8d38a73508ec3d98`
- Deploy path: `bin/deploy-direct`
- Prisma migrations: no pending migrations
- Production readiness: `ready`

Pulled work since the prior handoff:

- Research Preview / Gate 0 cleanup.
- Responsive recent-chat/session drawer.
- Guest conversation registry and server-side conversation loading/deletion.
- Conversation continuity fixes for streamed voice/text.
- Direct response synthesis from the original language.
- Provenance labels for `Live model` versus `Safety fallback`.
- New language-response contract:
  `pnpm eval:language-response`.
- Research direction document:
  `docs/GHANA_LANGUAGE_UNDERSTANDING_RESEARCH.md`.

Local fix added after pull:

- `fix(understanding): ask clearly during fallback outage`
- The no-LLM fallback now asks the user to say the request again in different
  words instead of only saying the model is unavailable.

Validation run:

```bash
pnpm eval:language-response
pnpm eval:understanding:fallback
pnpm lint
pnpm build
pnpm eval:product-readiness:prod
```

Production endpoint checks:

- `/api/config` reports Twi TTS as `stable-twi` /
  `ghananlpcommunity/stable-twi-tts`.
- `/api/tts` with Twi text returned WAV audio, sample rate `22050`, about
  `97k` base64 chars, and about `1.7s` TTS latency.

One local contract still needs a local database or secrets-backed DB before it
can be rerun:

```bash
pnpm eval:conversation:fallback
```

It failed locally with database connection refused, not with an assertion
failure. Production readiness confirms the production DB path is healthy.

Next useful step:

1. Exercise the live UI on production: session drawer, new chat, delete chat,
   streamed transcript preservation, and Twi response style.
2. If UI feels stable, continue Gate 1: canonical research data schema/import
   audit for the Ghana language understanding programme.
3. Keep DONDO v3 spend blocked until the corpus audit shows enough
   speaker-diverse held-out product data.

## 2026-08-21 session controls and response synthesis correction

- The app menu now contains a responsive recent-chat workspace with New chat,
  conversation selection, per-chat context actions, and confirmed deletion.
- Browser guests retain a local registry of their conversation IDs; signed-in
  users receive their server-side conversation list.
- Text responses expose `Live model` versus `Safety fallback` provenance so a
  degraded runtime cannot masquerade as model intelligence.
- Comprehension and answering are now separate calls. The first call sees only
  transcript evidence, recent dialogue, and memory and must either state a
  faithful English meaning or return `understood=false`; it cannot answer.
- Answer generation runs only after the comprehension gate passes and receives
  the accepted meaning, not the raw transcript.
- Hard-coded eye-pain, hospital-choice, migraine, rest/fluids, and generic
  clinic responses have been removed. Deterministic enforcement remains only
  for explicit emergency patterns.

Response path:

`transcript + recent history + scoped memory + ASR evidence -> understood/not-understood gate -> natural answer from accepted meaning -> deterministic emergency enforcement -> stored/streamed response -> optional TTS`

Retrieval remains disabled, so current intelligence comes from the configured
LLM plus conversational context, not an evidence-grounded health knowledge
source. Grounded retrieval is the next response-quality layer after live
model/fallback behavior is verified.

**Last session focus:** Voice-first runtime cleanup, schema-first understanding, agent memory, and DONDO ASR training path.  
**North star:** [`docs/research-stack.md`](./research-stack.md)  
**Training roadmap:** [`docs/model-training-roadmap.md`](./model-training-roadmap.md)
**ASR decision:** [`docs/asr-model-decision.md`](./asr-model-decision.md)
**Model credit plan:** [`docs/model-credit-plan.md`](./model-credit-plan.md)
**Pilot runbook:** [`docs/PILOT.md`](./PILOT.md)

---

## 2026-08-21 handoff

Production is testable at `https://ghanahealth.serendepify.com`.

Current live deploy:

- Commit: `092626068d32d896bf6c9d17fef9767d07487e06`
- Web image: `ghcr.io/teckedd-code2save/ghana-health-ai:092626068d32d896bf6c9d17fef9767d07487e06`
- Twi TTS route: `stable-twi`
- Twi TTS model: `ghananlpcommunity/stable-twi-tts`
- English TTS route: `facebook/mms-tts-eng`
- Stable Twi Modal endpoint:
  `https://createdliving1000--ghana-health-tts-stable-twi-speak.modal.run`

Verification already run:

```bash
curl -sS https://ghanahealth.serendepify.com/api/config
curl -sS https://ghanahealth.serendepify.com/api/tts \
  -H 'Content-Type: application/json' \
  --data '{"text":"Akwaaba, wo ho te sɛn?","language":"tw"}'
```

The TTS smoke test returned `provider=stable-twi`, sample rate `22050`, WAV
audio, and about 2s synthesis latency. Test the service now by using normal
Twi voice chat, then compare:

- Does the voice sound more natural than MMS?
- Does the response stay in Twi when the input is Twi?
- Does the conversation append turns instead of replacing previous turns?
- Does the model picker near the mic route Twi beta/stable as expected?

GitHub Actions deploys are currently blocked by exhausted Actions credits. The
repo now has a direct VPS deploy route:

```bash
bin/deploy-direct
```

That command builds the committed repo on the VPS, applies Prisma migrations,
restarts only `ghana-health-ai-web`, and smoke-tests production config/readiness.
Use `STOP_OTHER_CONTAINERS=1 bin/deploy-direct` only when the VPS needs extra
memory and unrelated containers can be stopped. It does not upload local env
files and does not depend on pulling from GHCR.

Next product/R&D work:

1. Continue P4 live voice polish: interruption handling, streaming states, and
   TTS quality comparison against stable-twi/nano-twi/Qwen research.
2. Continue R1 beta measurement: v6 vs DONDO v2+LM with the same product clips,
   correction logs, and latency.
3. Start P5 commerce tool orchestration only after voice continuity feels
   stable.
4. Build R3 synthetic Twi voice-note generation from reviewed corpus JSONL, but
   keep synthetic audio out of final human held-out evals.

### Later 2026-08-21 continuation update

R3 synthetic Twi voice-note generation is now an executable pipeline, not just a
plan item.

Added:

- `scripts/synthesize-twi-voice-notes.ts`
- `scripts/eval-synthetic-voice-notes.ts`
- `pnpm corpus:synthesize-twi`
- `pnpm eval:synthetic-voice-notes`

The generator reads reviewed Twi/code-switch corpus JSONL, calls the configured
Twi TTS provider, writes audio files, and emits a training/augmentation manifest
with:

- `audio_path`
- `reference`
- `bucket`
- `speaker_label=synthetic_<provider>`
- `source=synthetic_tts`
- `tts_model`
- `voice_id`
- `duration_s`
- `sha256`
- `holdout=false`

Important guardrail: the script refuses `needs_review: true` rows and
`source=llm_translation_draft` rows by default. This is intentional because the
current generated Twi prompt packs still contain poor draft translations and
repetitions. Do not synthesize those into training audio unless using
`--allow-drafts` for a clearly labeled non-training experiment.

Validation completed:

```bash
pnpm eval:synthetic-voice-notes
pnpm eval:tts-routing
pnpm lint
pnpm corpus:synthesize-twi -- \
  --input tmp/asr-collection-pack/prompts.corpus-v2.health_twi.jsonl \
  --dry-run \
  --limit 5
pnpm eval:product-readiness:prod
```

Results:

- Synthetic voice-note contract passed.
- TTS routing contract passed.
- Lint passed.
- Dry-run correctly found `0` eligible rows in the current `health_twi` draft
  prompt pack and skipped all draft rows.
- Production readiness is `ready`.
- Expected degraded items remain:
  - no ASR checkpoint passes all promotion gates yet
  - product eval buckets still need more consented, speaker-diverse clips

One check was intentionally stopped because the user needed to leave:

```bash
pnpm eval:prod:smoke
```

It had already passed production readiness and all JSON live-pipeline fixtures,
then was stopped while waiting on the stream fixture phase. Re-run it next time
if full production stream verification is needed.

Next best step from a new device:

1. Commit/push the R3 synthetic voice-note changes if not already committed.
2. Create or export a reviewed corpus JSONL with `needs_review=false` rows.
3. Run `pnpm corpus:synthesize-twi -- --input <reviewed.jsonl> --dry-run`.
4. If eligible rows look correct, run the same command without `--dry-run`.
5. Validate the resulting manifest with
   `pnpm eval:local-asr-import -- tmp/synthetic-twi-voice-notes/manifest.jsonl`.

### R2 corpus audit update

R2 now has an executable corpus-readiness audit:

```bash
pnpm eval:asr-corpus -- --manifest tmp/asr-local-train/manifest.jsonl
pnpm eval:asr-corpus:strict -- --manifest data/asr-product-eval/manifest.jsonl
```

The audit reports bucket counts, speaker diversity, holdout count, repeated
references, missing audio, synthetic rows, and promotion readiness. It is meant
to answer whether a corpus can support model-credit spend, not just whether the
JSONL shape is valid.

Current local corpus audit:

- rows: `40`
- speakers: `2`
- duration: about `3.0` minutes
- holdout rows: `0`
- synthetic rows: `0`
- missing audio: `0`
- bucket counts:
  - `health_twi`: `25 / 100`
  - `commerce_twi`: `10 / 70`
  - `codeswitch_tw_en`: `5 / 60`
  - `health_en`: `0 / 50`
  - `phone_noise`: `0 / 20`
- promotion-corpus ready: `no`

The same corpus passes import validation:

```bash
pnpm eval:local-asr-import -- tmp/asr-local-train/manifest.jsonl
```

So the immediate interpretation is: the current 40 clips are usable for local
experiments and regression checks, but they are not enough for DONDO promotion
or serious model-credit spend. The next data push needs more speakers, more
commerce/code-switch clips, an English retention bucket, phone/noise clips, and
a real frozen holdout split.

---

## Product goal (do not lose this)

Voice-first **Twi** health companion for Ghana:

**listen (ASR) → understand (Twi-native path) → speak (TTS)**  

English **only** when the user sets language preference to English.

---

## What’s already in this branch

### App / product
- Clean product UI (home, voice, chat, market, login)
- `understandUtterance` runtime path: ASR metadata → schema-first health/commerce intent → memory-aware LLM
- Runtime understanding no longer uses Twi/product retrieval as intelligence. Retrieval metadata is intentionally `engine=none`.
- Focus is inferred when the caller does not pass a tab/focus. Shopping language such as buy/order/market/price, Twi `metɔ` / `atɔ` / `boɔ` / `hwehwɛ`, wins as commerce even when the item is health-related.
- Commerce understanding now extracts structured slots from speech/text:
  - `action`: buy/order/find/price/availability/unknown
  - `item`, `quantity`, `location`, `fulfillment`
  - Slots are stored in assistant metadata and returned by chat/voice APIs as `understanding.commerce`.
  - `src/lib/commerce-plan.ts` adds a deterministic next-action plan:
    - ask missing item/quantity/location
    - search connected local catalog
    - draft order for confirmation
    - explain unsupported when live marketplace is not connected
  - `src/lib/commerce-execute.ts` executes only safe commerce actions:
    - local catalog search with real `Product` rows/prices
    - order draft metadata requiring explicit confirmation
    - no checkout, payment, or external-market mutation
  - `/api/commerce/confirm` is the only commerce voice/chat confirmation endpoint that mutates cart state, and it requires `confirm: true`.
  - APIs return `understanding.commerceExecution`.
  - Voice UI shows a compact commerce action chip for the top matched catalog product; tapping it calls `/api/commerce/confirm`.
  - This is for future search/order agents only; no fake store, price, or availability is invented.
- Health understanding now produces deterministic action-plan metadata in addition to the LLM answer:
  - `src/lib/health-plan.ts` maps transcript quality and clinical severity into `needs_clarification`, `self_care`, `clinic_recommended`, or `urgent_referral`.
  - Chat and voice APIs return `understanding.health`.
  - Assistant audit metadata stores the health plan, so weak ASR can trigger clarification instead of confident advice.
  - `pnpm eval:health-plan` verifies emergency, weak-transcript, and routine pathways without an LLM call.
- `src/lib/agent-memory.ts` — scoped agent memory for profile, health, and commerce continuity
- `src/lib/twi-retrieve.ts` — still present for later experiments, but not part of the live understanding path
- **Passage embedding cache** on `KnowledgeArticle` / `Product` (`embedding_b64`, model, engine, `embedded_at`)
  - Request path embeds **query + cache misses only** (not the full KB every turn)
  - Backfill: `sec -- pnpm db:index-embeddings` (`--force` to rebuild)
- Expanded Twi knowledge seed (maternal, child, malaria, mental, chronic, FP referral)
- Harder danger heuristic (Twi + EN) + force escalate / hotline prefix when urgent
- Config flags: `abenaEmbed` = Modal ABENA only; `embedAny` = ABENA or OpenAI fallback
- TTS serving upgrades; HF model-card helpers on train pushes
- Live chat/voice turn runner with streamed stages and reviewed answer deltas: ASR, retrieval, LLM answer, LLM review, streamed reply, TTS
- Voice UI consumes live backend stage events so the orb/status moves through transcribing, thinking, responding, and speaking instead of showing a generic wait.
- ASR quality metadata now reaches the response model/reviewer so unclear audio can produce a model-generated clarification instead of guessed health advice
- Response pipeline has a deterministic outage fallback:
  - If no LLM key is configured, or the provider times out/returns no draft/review, `understandUtterance` returns a structured fallback instead of throwing.
  - Health fallback uses `src/lib/health-plan.ts` for weak-ASR clarification, urgent referral, clinic recommendation, or conservative self-care.
  - Commerce fallback preserves extracted item/quantity/location and asks the next useful shopping question without inventing stores/prices.
  - `pnpm eval:understanding:fallback` verifies health, weak transcript, and commerce outage behavior.
  - `pnpm eval:conversation:fallback` verifies the full chat/voice turn runner still writes user + assistant messages with fallback metadata when no LLM key is available.
- Voice transcript correction loop:
  - `AsrFeedback` stores corrected transcripts/ratings tied to conversation + user message.
  - `/api/voice/feedback` persists corrections without retaining raw audio.
  - The voice UI exposes a compact inline correction action on the heard transcript.
  - `pnpm eval:asr-feedback:export` exports corrected text rows for review and data-loop planning.
- Understanding eval fixtures cover Twi danger signs, weak ASR clarification, English health, Twi commerce intent, memory, and no invented commerce prices.
- Understanding eval fixtures now include Twi tomato purchase, Twi quantity/delivery/location, and English price-query slot checks.
- `pnpm eval:commerce-plan` covers planner behavior without an LLM call.
- `pnpm eval:commerce-execute` verifies local catalog search/order draft and confirms no cart mutation.
- `pnpm eval:commerce-confirm` verifies cart mutation happens only after explicit confirmation.
- Full live pipeline fixtures pass in JSON and stream mode on `http://localhost:3100`; stream mode verifies staged event order, assistant-before-reply ordering, and reply deltas.
- Product readiness preflight:
  - `/api/readiness` reports runtime readiness for DB, Twi ASR, separate English ASR, TTS, LLM, response fallback, ABENA, ASR promotion, and product speech data.
  - `pnpm eval:product-readiness` fails if any required runtime dependency is blocked.
  - `pnpm eval:product-readiness:prod` runs the same gate against `https://ghanahealth.serendepify.com`.
  - Current readiness is `ready`; ASR model promotion and product eval data remain `degraded`, intentionally not blocking local demo.
- `scripts/audit-asr-promotion.ts` turns the model decision into an executable promotion audit across Twi, English retention, health-domain, code-switch, phone/noise, latency-adjacent evidence, and HF card gates.
- The ASR promotion audit now recognizes HF card evidence verified by `pnpm eval:hf-model-cards`, while still refusing promotion when WER/domain gates fail.
- `data/asr-product-eval/` defines the consent-aware product ASR eval manifest; `scripts/validate-asr-eval-manifest.ts` checks buckets before serious credit spend.

### Explicitly out of scope (validated)
- Electric Sheep Africa **Ghana Country Data Coverage** collection — tabular EN official stats (census, energy, road, facilities). **Not** speech/Twi text for ASR/TTS/SFT. Optional later: facility lat/lon as a referral feature only.

### Modal (deployed)
| Service | URL / note |
|---------|------------|
| **ABENA embed** | `https://createdliving1000--ghana-health-embed-embed.modal.run` |
| Embed health | `https://createdliving1000--ghana-health-embed-health.modal.run` |
| TTS speak | `https://createdliving1000--ghana-health-tts-speak.modal.run` |
| ASR | Prod: v6 Whisper small Twi |

---

## Remaining work (priority order)

### 1. Runtime service wiring

Infisical was **502** last check — retry when up.

Current local config check on `http://localhost:3100/api/config`:

- `modalAsr: true`
- `modalAsrEnglish: true` (`en` route is `english`)
- `modalTts: true`
- `abenaEmbed: true`
- `embedAny: true`
- `appUrl: http://localhost:3100`

The app now has documented non-secret defaults for the public Modal English ASR, ABENA embed, and TTS endpoints. Infisical should still override them when endpoints or tokens change.

Verified public endpoint checks:

- English ASR health: `service=ghana-health-asr-en`, `model=openai/whisper-small`
- ABENA embed health: `service=ghana-health-embed`, `model=Ghana-NLP/abena-base-asante-twi-uncased`
- ABENA embed POST: returns 768-dim embeddings
- TTS speak POST: returns WAV audio from `facebook/mms-tts-aka`

```bash
MODAL_EMBED_URL=https://createdliving1000--ghana-health-embed-embed.modal.run
MODAL_ASR_EN_URL=https://createdliving1000--ghana-health-asr-en-api.modal.run
# keep existing MODAL_ASR_URL / MODAL_TTS_URL
```

Deploy the English endpoint without replacing Twi:

```bash
ASR_APP_NAME=ghana-health-asr-en MODEL_ID=openai/whisper-small modal deploy modal/asr_service.py
```

Verified health: `service=ghana-health-asr-en`, `model=openai/whisper-small`.

Then:

```bash
sec -- npx prisma migrate deploy
sec -- npx prisma db seed
sec -- pnpm db:index-embeddings
# confirm:
curl -s http://localhost:3100/api/config | jq .modalAsrEnglish,.asrRoutes,.abenaEmbed,.modalTts,.appUrl
```

### 2. Fix HF token for LoRA SFT (blocked)

Modal secret `huggingface-token` 403 on Qwen/Llama. Refresh token → re-run `train_understand.py`.

### 3. ASR multi-domain promote

> **2026-08-20 update — see [`docs/asr-rnd-session-2026-08-15.md`](./asr-rnd-session-2026-08-15.md) and the
> [`asr-model-decision.md` addendum](./asr-model-decision.md#2026-08-20-dondo-v2-recovery-addendum).
> Execution path: [`docs/asr-rd-execution-plan.md`](./asr-rd-execution-plan.md).
> Key shifts: (a) v6's product-domain WER is 54.18% — Waxal WER does not
> predict product WER; held-out domain evals are now first-class gates.
> (b) DONDO v1 beats v6 by ~22pp on the local corpus (32.66% vs 54.18%) —
> DONDO v2 completed and is now the Twi ASR front-runner: Waxal n=300
> **28.12% greedy / 27.31% with Twi LM**, frozen local holdout8
> **26.67% greedy / 6.67% with Twi LM**. (c) Clean hold-out experiment proves domain data generalizes
> (−11.7pp on unseen clips from 32 training clips) — Whisper v8 approved once
> the corpus scales. **Critical path: corpus scaling (200+ clips, new
> speakers, code-switch priority), not compute.** v6 stays serving.
> Trainer hardened this session: always pass `--train-limit` (capped
> streaming loader) and launch with `modal run --detach`.

Current evidence says **do not hard-pivot DONDO into default serving yet**:

- DONDO v2 should be the Twi beta/A-B candidate.
- v6 / current Whisper-family path remains the stable default until a larger
  held-out product-domain corpus confirms v2 across speakers/noise/code-switch.
- v6 is **hold-and-validate only**, not a competitive final model. Its Hugging Face model card has been backfilled with real WER/CER, datasets, intended use, limitations, and a medical-device disclaimer.
- DONDO zero-shot was much worse on Waxal. v2 fixed that enough to beat v6 on
  Waxal n=300 and on the tiny product-domain holdout, but the holdout size is
  not enough for final promotion.
- English voice uses the separate English route by default, with `MODAL_ASR_EN_URL` available as an override.
  - 100-sample Common Voice English: `openai/whisper-small` WER **11.82%**, v6 WER **42.34%**.
- `modal/train/train_asr.py` now includes Common Voice English in the extra-data mix for balanced v7 runs.
- `modal/train/benchmark_asr_ladder.py` now launches English-retention evals alongside Twi evals unless `--skip-english` is passed.
- Promote only if full Waxal + health-domain + English-retention + phone/noise eval beat the current production candidate.
- DONDO training now has a working capped streaming smoke path:
  - Modal app `ap-Fllvp76L3nlPCLCCaxW1p3`
  - Function call `fc-01KZSTXGGKQTW5K4YXZZT7SPF0`
  - 24 train / 8 eval rows, 8 steps, WER 53.82%, CER 17.81%
  - This proves wiring only; it is not a promotion candidate.

Initial full balanced v7 jobs were stopped because the original loader resolved large dataset shards before training. The capped proof path now streams Waxal and Common Voice extras, then materializes only the requested samples before feature prep.

Validated proof run:

| Run | Modal app | Function call | Result |
| --- | --- | --- | --- |
| `v7-small-waxal-proof-streamed` | `ap-yz2EwC2h9UncQbBvlRxSVI` | `fc-01KZKRSKJGNSRRNKEM667DB9RJ` | Completed 80 steps on 256 train / 64 eval samples; loader fix proven; do not promote |
| `v7-small-balanced-extra-proof-streamed` | `ap-oQ8n9rH6mSDDvS9vSvFjf7` | `fc-01KZKVZ9BZ52HV5W00V67VT5MM` | Confirmed capped Common Voice extra-data path reaches training; no HF push |

Balanced v7 lite jobs launched on Modal:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-lite-no-en-regression` | `ap-H0H15OWS8H7lxtE5WAuqAe` | `fc-01KZKWQEHHX171AHQ6AY15PF1G` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-5vhppGhhweckMY4HZO4XFP` | `fc-01KZKWQEJSEPMCM73NHNDSMTKV` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |

Current lite settings: 1,200 streamed Waxal train samples, 150 streamed Waxal eval samples, capped streamed Common Voice Twi/English extras, 500 steps. Both completed and are **not promotable**:

- `v7-small-balanced-lite-no-en-regression`: Twi Waxal 100-sample beam-5 WER **40.86%**, CER **14.10%**; English Common Voice 100-sample beam-5 WER **15.10%**, CER **8.10%**. English retention improved versus v6, but Twi regressed too much.
- `v7-small-balanced-lite-frozen-no-en-regression`: train validation WER **53.83%**, CER **22.18%**. Twi regressed badly.
- Both HF repos exist and their downloaded README cards include valid dataset metadata, base model, metrics, intended use, and medical-device disclaimer. Local `modal/train/model_card.py` and `pnpm eval:model-card` validate that future card metadata uses Hub-valid dataset IDs.

The earlier 3,000-sample jobs were too slow to materialize.

Stopped earlier lite jobs:

| Run | Modal app | Function call | Reason |
| --- | --- | --- | --- |
| `v7-small-balanced-lite-no-en-regression` | `ap-pZNnbsEBMIeCBLOh3MmiQ1` | `fc-01KZKRHXNWSAJPWWFKT3ND7A6C` | Stuck on old full-shard loader |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-aPtJKFxoavBNMp3Mv4wHcB` | `fc-01KZKRHXGGKF1Y02SGJGH2YXDY` | Stuck on old full-shard loader |
| `v7-small-balanced-lite-no-en-regression` | `ap-fJaR6pMxKDS2p5BwgolZsf` | `fc-01KZKS8FZ26W6GZZ0BEJTS7CTR` | Stuck downloading full Common Voice English audio shards |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-Dqk2wBvPIQwYZg5SKhLywV` | `fc-01KZKS8FZ73KXQNE1A6YB2227G` | Stuck downloading full Common Voice English audio shards |
| `v7-small-balanced-lite-no-en-regression` | `ap-spYSuQHextHp99xtVEr3Ig` | `fc-01KZKWE7EZ1QVPBB2XXRZQNY6M` | 3,000-sample materialization too slow; replaced with 1,200-sample run |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-jObe1dq2JPlDuoh5KsvZ5O` | `fc-01KZKWE7EZPGT2JW090H80W3H3` | 3,000-sample materialization too slow; replaced with 1,200-sample run |

Stopped full balanced v7 jobs:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-no-en-regression` | `ap-HbD5ihi2bRlVZs3yix7y87` | `fc-01KZKR6FZ2DPZVSET4603KJMBS` | `teckedd/gha-whisper-small-twi-en-balanced-v7` |
| `v7-small-balanced-frozen-no-en-regression` | `ap-hxlw7y8LZ7PyM13hVM0x8s` | `fc-01KZKR6FZ35T88N037AKDZEQAS` | `teckedd/gha-whisper-small-twi-en-balanced-v7-frozen` |
| `v7-medium-balanced-no-en-regression` | `ap-Gn8ykjRwLN9Mc410nFdmFZ` | `fc-01KZKR6G12KMBDSW459Q0JDWFY` | `teckedd/gha-whisper-medium-twi-en-balanced-v7` |

After they complete:

```bash
pnpm eval:asr-results:pull
pnpm eval:asr-results
```

The first real DONDO credit-spend trial has completed:

- HF repo: `teckedd/gha-dondo-w2v-bert-twi-v1`
- Base: `KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en`
- Modal app: `ap-F3x5vbrsPLQh13kBUTvcgA`
- Function call: `fc-01KZSWZY0ES2NWC01N9PA8GQP4`
- Final WER: **35.77%**
- Final CER: **12.19%**
- Decision: **do not promote; keep v6 serving for Twi while English uses the separate English route**
- HF card: pushed and verified with base model, dataset, metrics, intended use, limitations, and promotion gate.

DONDO trial history:

| Run | Modal app | Function call | Target HF repo | Status |
| --- | --- | --- | --- | --- |
| `dondo-waxal-twi-v1` | `ap-hGNVgVb8XAYSA0Vxv2zZXF` | `fc-01KZSV66NG7R2R3F5JSKFMK6WP` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Reached step 217, then CUDA OOM after checkpoint-200 eval |
| `dondo-waxal-twi-v1` resume | `ap-nYJdyzEwxZaXp0ubW7KqUi` | `fc-01KZSWMTN06A7NTRV97X6C68PF` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` H100 resume | `ap-SbPCmY2dgsqSD8zEuNyt44` | `fc-01KZSWTJ4WRAT1ZRSK4YNX26DR` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` GPU-fallback resume | `ap-F3x5vbrsPLQh13kBUTvcgA` | `fc-01KZSWZY0ES2NWC01N9PA8GQP4` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Completed 800/800; pushed model + card; WER 35.77%, CER 12.19%; do not promote |

Monitor it without hand-typing Modal commands:

```bash
pnpm train:dondo:monitor
pnpm train:dondo:results
```

Final DONDO result:

- Step 200 eval WER: **41.04%**
- Step 200 eval CER: **13.99%**
- Final eval WER: **35.77%**
- Final eval CER: **12.19%**
- Final train loss: `0.29205229461193083`
- Final eval loss: `0.47330379486083984`
- Baseline WER to beat: `0.3044`
- Promote: **False**

This is a genuine DONDO improvement over zero-shot and smoke, but not a competitive serving model. Do not spend the next credits blindly on more of the same 800-step Waxal-only run. The next model work should be either a targeted data loop, a larger/cleaner DONDO run with a stronger eval design, or a different architecture only if it has a measurable path below the 30% WER plateau.

### 4. Current local verification

Local dev app verified on `http://localhost:3100` because this session's dev server is running on port 3100. Docker Desktop was started and local Postgres is healthy on `localhost:5437`.

Public production check on `https://ghanahealth.serendepify.com`:

- Home page: `200 text/html`
- `/api/health`: `200 application/json`, body reports `ok: true`
- `/api/readiness`: currently `404 text/html`, which means production is still on an older build and has not received the local readiness/pipeline-gate work yet.

After this branch is shipped, run:

```bash
pnpm eval:product-readiness:prod
```

Passed checks:

```bash
pnpm tsc --noEmit
pnpm lint
pnpm eval:health-plan
pnpm eval:understanding:fallback
pnpm eval:conversation:fallback
pnpm eval:commerce-plan
pnpm eval:commerce-execute
EVAL_BASE_URL=http://localhost:3100 pnpm eval:commerce-confirm
pnpm eval:understanding
EVAL_BASE_URL=http://localhost:3100 pnpm eval:live-pipeline
EVAL_BASE_URL=http://localhost:3100 pnpm eval:live-stream
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-stream
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-feedback
EVAL_BASE_URL=http://localhost:3100 pnpm eval:voice-preview
EVAL_BASE_URL=http://localhost:3100 pnpm eval:product-readiness
pnpm eval:asr-routing
pnpm eval:asr-manifest
pnpm eval:asr-feedback:export
pnpm eval:asr-results:pull
pnpm eval:asr-results
pnpm eval:asr-promotion
pnpm eval:model-card
pnpm eval:hf-model-cards
```

Browser check: desktop and mobile home screens render as the clean light voice-first UI with centered Health/Commerce pills, no market section, no debug text, no console errors.

The local TypeScript eval scripts use `node --import tsx` instead of the TSX CLI so they work under Codex managed sandboxing without an IPC pipe failure.

### 5. TTS finetune + eval harness

See roadmap. TTS train script is still a skeleton.

---

## Key files

| Path | Role |
|------|------|
| `docs/research-stack.md` | Architecture + literature |
| `docs/CONTINUE.md` | This handoff |
| `src/lib/understand.ts` | Twi-first + safety gate |
| `src/lib/health-plan.ts` | Deterministic health action plan from severity + ASR confidence |
| `src/lib/agent-memory.ts` | User/session memory extraction and prompt formatting |
| `src/lib/conversation-turn.ts` | Shared voice/chat turn runner with streamed stages |
| `src/lib/commerce-plan.ts` | Deterministic next-action plan for commerce slots |
| `src/lib/commerce-execute.ts` | Safe local catalog search / order draft execution for commerce plans |
| `src/app/api/commerce/confirm/route.ts` | Explicit confirmed cart-add endpoint for commerce turns |
| `src/app/api/voice/feedback/route.ts` | Captures ASR correction/rating feedback for the data loop |
| `src/lib/twi-retrieve.ts` | ABENA retrieval + passage cache |
| `src/lib/embed.ts` | Embed client + vector b64 helpers |
| `scripts/index-abena-embeddings.ts` | Offline cache warm |
| `modal/embed_service.py` | ABENA Modal service |
| `modal/train/train_understand.py` | GhanaNLP parallel → LoRA |
| `docs/asr-model-decision.md` | ASR evidence, promotion gates, credit-spend order |
| `docs/model-credit-plan.md` | Credit spend order, promotion matrix, product data loop |
| `data/asr-product-eval/README.md` | Product ASR eval manifest and bucket requirements |
| `modal/train/train_dondo_asr.py` | DONDO / w2v-BERT CTC fine-tune path |
| `scripts/eval-understanding.ts` | Runtime understanding regression checks |
| `scripts/benchmark-understanding-llm.ts` | Scores structured LLM meaning extraction on the Twi benchmark |
| `scripts/score-understanding-benchmark.ts` | Scores candidate benchmark artifacts against the project meaning rubric |
| `scripts/export-understanding-training-corpus.ts` | Exports only reviewed understanding rows into train/dev/test manifests |
| `scripts/export-understanding-review-sheet.ts` | Writes a CSV sheet for bulk human review of corpus candidates |
| `scripts/import-understanding-review-sheet.ts` | Imports a corrected review CSV into the local JSONL review fallback |
| `data/understanding-benchmark/rubric.v0.json` | Meaning-preservation rubric for the 50 synthetic benchmark probes |
| `data/understanding-benchmark/scorecard.v0.json` | Current candidate ranking for the benchmark probes |
| `data/understanding-corpus/candidates.v0.jsonl` | 80 draft corpus candidates for human review |
| `scripts/eval-understanding-fallback.ts` | Verifies outage fallback for health, weak ASR, and commerce |
| `scripts/eval-conversation-fallback-contract.ts` | Verifies full conversation turn fallback persists messages |
| `scripts/eval-commerce-execute.ts` | Verifies commerce execution does not mutate cart/checkout |
| `scripts/eval-commerce-confirm-contract.ts` | Verifies confirmation is required before cart mutation |
| `scripts/eval-voice-feedback-contract.ts` | Verifies feedback endpoint persists correction + ASR metadata |
| `scripts/eval-product-readiness.ts` | Verifies shareable runtime readiness checks |
| `scripts/validate-asr-eval-manifest.ts` | Validates ASR product-eval manifest shape and bucket readiness |
| `scripts/export-asr-feedback.ts` | Exports captured transcript corrections to product-eval JSONL |
| `scripts/audit-asr-promotion.ts` | Product-level ASR promotion gate audit |
| `scripts/eval-hf-model-cards.ts` | Verifies pushed HF model cards have base model, datasets, metrics, and limitations |

---

## Explicitly abandoned

- Hand-written `semantic-bank.ts` as product intelligence  
- English knowledge dumps as primary NLU  
- In-domain ASR WER alone as promotion proof  
- Official Ghana tabular CSVs as voice-model training data  

Continue from **§ Remaining work** above.

---

## 2026-08-29 Understanding Research Status

Current best draft-understanding candidate:

| Candidate | Meaning score | Exact cases | Notes |
| --- | ---: | ---: | --- |
| `openai:gpt-5.4-mini` | 94.2% | 42/50 | Best current draft annotator and product understanding candidate. Still misses some health/commerce meanings. |
| `ninte/twi-en-nllb-v2` | 77.5% | 30/50 | Fast translation baseline only; unsafe alone for health meaning. |
| `facebook/nllb-200-distilled-600M` + `mclanorjeff/NLLB-Twi-Human-Aligned` | 76.1% | 28/50 | Adapter path works on Modal but did not beat v2 on this rubric. |

New commands:

```bash
pnpm eval:understanding:model:human-aligned
pnpm eval:understanding:llm
pnpm eval:understanding:score
pnpm corpus:understanding:export
pnpm corpus:understanding:export:strict
```

Corpus state:

- `80` candidate rows exist.
- `80` have draft annotations.
- `30` reference local audio artifacts.
- `0` saved human reviews exist in the local review file.
- `0` rows are currently training-eligible.
- Review decisions now persist in Postgres through
  `research_understanding_reviews`; the JSONL review file is only a local
  fallback.

The research workbench now exposes the benchmark scorecard in `/research/ase`.
It also shows how many reviewed rows are training-ready, the current
train/dev/test counts, and a direct link to
`/api/research/understanding/export`.

The corpus is ready for review, not training. The next real step is to review
rows in the workbench and run `pnpm corpus:understanding:export:strict`; only
that export should feed training. The export endpoint and strict command must
remain unready until a human has corrected and accepted enough corpus rows to
pass the readiness gate: at least 20 reviewed rows, train/dev/test coverage,
health-domain coverage, no duplicate meaning keys, and consent scope on every
row. Commerce-domain coverage is visible as a warning so the product track can
follow without blocking the first health-focused run.

For faster offline review, use the workbench **Download 20-row training pack**
action or run
`pnpm corpus:understanding:review-sheet -- --scope minimum-training --out tmp/understanding-corpus/minimum-training-review.v0.csv`.
Use **Download assisted pack** or add `--prefill draft` to prefill review
columns from model drafts without approving them. Rows still import for
training only when `decision` is changed to `reviewed`; untouched `unreviewed`
rows are skipped. That pack is selected to cover row count, train/dev/test,
health, and commerce before the remaining queue. Fill the `review_*`,
`decision`, `review_notes`, and `reviewer` columns, and preserve
`proposed_split`. Then upload the corrected CSV through the workbench **Upload
reviewed CSV** action. Production uploads persist to Postgres and immediately
update the export readiness gate. For local fallback work, import with
`pnpm corpus:understanding:import-review-sheet -- --input <sheet.csv>`, then run
the strict export gate.
