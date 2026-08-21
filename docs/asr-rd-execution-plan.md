# ASR R&D Execution Plan

Last updated: 2026-08-21

## North Star

Deliver a voice pipeline that understands Twi health and commerce speech well
enough for real product conversations. The product should default to stability,
while R&D keeps pushing model quality through controlled A/B releases and
larger held-out product-domain evaluation.

## Current Decision

DONDO v2 is the Twi ASR front-runner.

| Model | Gate | Result | Decision |
|-------|------|--------|----------|
| v6 Whisper | Stable production default | Waxal n=300 beam5 28.76%; local holdout8 46.67% | Keep as stable fallback |
| v6-local-holdout | Whisper adaptation proof | local holdout8 35.00% | Proves local data helps; not serving |
| DONDO v2 greedy | Twi beta candidate | Waxal n=300 28.12%; local holdout8 26.67% | Strong beta |
| DONDO v2 + Twi LM | Best measured Twi path | Waxal n=300 27.31%; local holdout8 6.67% | Front-runner; needs larger holdout |

Do not make DONDO v2 the default until the product-domain holdout is larger
and speaker-diverse. Do expose it as the Twi beta/A-B model.

English stays on a separate English ASR route. Do not spend cycles trying to
force Twi fine-tunes to retain English until the Twi path is stable.

## Priority Order

User-approved order as of 2026-08-21:

Split the work into two lanes so product improvements can move while model
quality research stays rigorous.

Product enhancement lane:

1. **P1 — DONDO v2 beta serving:** wire `teckedd/gha-dondo-w2v-bert-twi-v2`
   into the Twi beta route with Twi LM decoding enabled.
2. **P2 — chat continuity and controls:** keep conversations continuous within
   a guest/browser session, add a compact model picker near the record control,
   and keep previous sessions/settings in a collapsible side nav.
3. **P3 — response understanding:** fix intent/entity extraction and
   language-matched response behavior once transcripts are usable.
4. **P4 — live voice and TTS:** make the conversation feel real through
   streaming state, TTS, interruption handling, and correction flow. Evaluate
   Twi-native TTS candidates first; keep Qwen3-TTS 12Hz as a research
   candidate that must prove Twi pronunciation.
5. **P5 — commerce/tool orchestration:** evaluate HF SmolAgents for commerce
   search/order and internal model-ops automation, not medical authority.

Model-quality research lane:

1. **R1 — beta measurement loop:** compare v6 vs DONDO v2 on the same product
   fixtures, log model choice/latency/corrections, and preserve consented
   audio for training.
2. **R2 — held-out corpus expansion:** build the 250+ clip product-domain gate
   with more speakers, code-switching, and phone/noise coverage.
3. **R3 — synthetic Twi voice-note generation:** use a Twi TTS model to turn
   reviewed synthetic Twi text into labeled audio for augmentation and stress
   testing. Synthetic audio must not be used as final promotion evidence.
4. **R4 — DONDO v3:** return to training only after the beta loop and larger
   validated corpus produce enough signal.
5. **R5 — Whisper hedge:** keep broader Whisper as a later comparison lane,
   not the immediate product bet.

## Execution Tracks

### Track 1 — Evaluation Corpus

This is the critical path.

Targets for the next promotion gate:

- 250+ held-out product-domain clips.
- At least 6 speakers, with gender and age variation where available.
- Health, commerce, and Twi-English code-switch buckets.
- Phone/laptop mic mix, quiet/noisy environment labels.
- No candidate may train on the promotion split.

Required buckets:

| Bucket | Minimum held-out clips | Notes |
|--------|------------------------|-------|
| health_twi | 100 | symptoms, pregnancy, fever, pain, medication, triage |
| commerce_twi | 70 | item search, quantities, price, location, delivery |
| codeswitch_tw_en | 60 | English product/medicine names inside Twi |
| noise_phone | 20 | can overlap with the above buckets |

Every batch must pass `pnpm eval:local-asr-import` before training use.

### Track 2 — DONDO Serving Beta

Product lane. Ship DONDO v2 as the Twi beta route, not the default.

Work items:

- Point the DONDO beta service at `teckedd/gha-dondo-w2v-bert-twi-v2`.
- Enable Twi LM decoding in the service path, matching the eval path.
- Keep v6 available as fallback and comparison.

Promotion condition:

- DONDO v2 must beat v6 by at least 10pp WER on the larger held-out
  product-domain set.
- It must not regress live latency beyond a product-acceptable threshold.
- It must pass safety and correction-review checks in health flows.

### Track 3 — Data Scaling

Research lane.

Prioritize data quality over more random fine-tunes.

Collection rules:

- Use curated, reviewed Twi prompts for read speech.
- Capture spontaneous user corrections with explicit audio consent.
- Keep speaker IDs, dialect, device, environment, bucket, prompt ID, and
  holdout/train flags attached to every clip.
- For code-switch, include real medicine/product words users actually say:
  paracetamol, malaria test, ORS, tomatoes, rice, delivery, MoMo, price.

Do not train on unreviewed LLM-generated Twi as gold text.

### Track 3B — Synthetic Twi Voice Notes

Research lane.

Explore a streamlined flow where reviewed synthetic Twi text is converted into
voice notes by a Twi TTS model, producing paired `audio + transcript + metadata`
without manual recording for every utterance.

Use cases:

- Pretraining/augmentation for domain vocabulary and phrase coverage.
- Stress tests for health, commerce, code-switch, numbers, medicine names, and
  place names.
- Regression fixtures for the ASR service and response pipeline.
- Bootstrapping prompts before collecting real speaker audio.

Rules:

- Synthetic text must be reviewed Twi, not raw LLM output.
- Synthetic audio must be labeled `source=synthetic_tts` with model, voice,
  generation date, text source, bucket, and language metadata.
- Do not mix synthetic audio into the final held-out human evaluation set.
- Do not let synthetic audio dominate DONDO v3; use it as a controlled
  augmentation slice with Waxal/CV/local human replay.
- Always validate gains on real held-out human speech.

Candidate TTS generators:

- `ghananlpcommunity/stable-twi-tts` for Twi/code-switch voice generation.
- `ghananlpcommunity/nano-twi` for fast Asante Twi generation.
- Qwen3-TTS 12Hz only if it proves usable Twi pronunciation.

Target artifact:

- A script that reads reviewed corpus JSONL and writes a manifest:
  `audio_path`, `reference`, `bucket`, `speaker_label=synthetic_<model>`,
  `source=synthetic_tts`, `tts_model`, `voice_id`, `duration_s`, and
  `holdout=false`.

### Track 4 — Next Training Spend

Research lane.

Spend compute only after the next data batch lands.

Recommended order:

1. Beta measurement loop with DONDO v2 + LM enabled.
2. DONDO v3 after 250+ new reviewed/consented clips.
3. Whisper v8 from v6 only if the new data shows Whisper still has a path
   to close the product-domain gap.
4. Response-understanding training after ASR transcripts are stable enough to
   give the LLM real input.

DONDO v3 recipe should start from v2, mix Waxal replay, CV Twi, and reviewed
local product data, and preserve a frozen product-domain holdout.

### Track 5 — Product Response Pipeline

Product lane.

Do not let retrieval or templated responses mask ASR failures.

Chat continuity is part of the voice product, not optional polish. The app must
not clear the previous conversation after every recording inside the same
browser session. For logged-out users, maintain a familiar guest-session
experience with local/in-memory continuity; for logged-in users, persist turns
to account sessions.

The response path should be:

1. ASR transcript with model confidence and language route.
2. Lightweight intent/entity extraction for health or commerce.
3. LLM response in the user's spoken language.
4. Safety review for health.
5. TTS response.
6. User correction loop.

UI requirements:

- Keep the home/chat UI voice-first and minimal.
- Put the model picker close to the record control, not buried in debug UI.
- Keep the picker compact and explicit enough for beta testing: stable/v6 vs
  DONDO beta. The picker is an intentional escape hatch when DONDO performs
  badly repeatedly.
- Previous sessions and settings belong in a collapsible side nav.
- The side nav should collapse cleanly on web and mobile.
- A new recording should append to the current conversation, not replace it.
- Refresh should not destroy guest chat context within the active browser
  session unless the user intentionally starts over.

Retrieval is optional and task-specific. For commerce, the goal is item
understanding and shopping action. For health, the goal is safe symptom
clarification and referral guidance. Retrieval should not be the primary
compensation for weak transcription.

### Track 5B — Live Voice, TTS, and Agents

Product lane.

This track improves the product feel after ASR is usable.

Current production TTS is `facebook/mms-tts-aka`, which is intelligible but not
natural enough for the product. Replace it.

TTS candidates:

- First evaluate GhanaNLP `stable-twi-tts` for natural Twi and Twi-English
  code-switching. It is IPA-driven and explicitly built for Twi plus Ghanaian
  English spans.
- Evaluate GhanaNLP `nano-twi` as a fast, offline Asante Twi fallback. It is
  single-voice and not voice-cloning, but the latency profile is attractive.
- Evaluate Qwen3-TTS 12Hz as a research candidate for naturalness, streaming,
  and consented voice-clone experiments. It does not officially list Twi among
  supported languages, so it must prove Twi pronunciation before becoming a
  product TTS candidate.
- Start Qwen checks with the smaller Qwen3-TTS 12Hz 0.6B path for feasibility,
  then test 1.7B only if quality clearly needs it.
- Measure first-audio latency, real-time factor, Twi pronunciation, English
  medicine/product words, interruption behavior, and Modal cost.
- Do not ship cloned voices or custom voice features without an explicit
  consent and safety policy.

Runtime routing knobs:

- `TTS_TWI_PROVIDER=mms|stable-twi|nano-twi|qwen` selects the Twi voice route.
- Twi defaults to the deployed `stable-twi` candidate; set
  `TTS_TWI_PROVIDER=mms` to force the old MMS route.
- `MODAL_TTS_URL` remains the MMS fallback route.
- `STABLE_TWI_TTS_URL`, `NANO_TWI_TTS_URL`, and `QWEN_TTS_URL` point to
  candidate `/speak` endpoints with the same response contract.
- The checked-in stable Twi default is
  `https://createdliving1000--ghana-health-tts-stable-twi-speak.modal.run`.
- Deploy/update the stable Twi service with `pnpm deploy:tts:stable-twi`.
- `/api/config`, `/api/readiness`, `/api/tts`, and voice-turn metadata expose
  the active provider/model so live tests can compare voice quality without UI
  changes.
- `pnpm eval:tts-routing` verifies provider routing without calling remote
  synthesis services.

Agent candidates:

- Evaluate Hugging Face SmolAgents for tool orchestration around commerce
  search/order flows, eval automation, and model-ops workflows.
- Do not use an agent as the medical reasoning authority. Health safety stays
  in deterministic policy checks plus reviewed LLM prompting.
- Keep tool contracts explicit: `intent`, `entities`, `risk_flags`,
  `recommended_next_action`, `language`, and `confidence`.

Reference candidates:

- GhanaNLP stable Twi TTS:
  `https://huggingface.co/ghananlpcommunity/stable-twi-tts`
- GhanaNLP nano Twi TTS:
  `https://huggingface.co/ghananlpcommunity/nano-twi`
- WAXAL MMS Twi fine-tuning baseline:
  `https://huggingface.co/rnjema-unima/mms-tts-twi-baseline`
- Qwen3-TTS 12Hz HF collection:
  `https://huggingface.co/collections/Qwen/qwen3-tts`
- Qwen3-TTS 12Hz 0.6B Base:
  `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- Qwen3-TTS 12Hz 1.7B CustomVoice:
  `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- HF SmolAgents docs:
  `https://huggingface.co/docs/smolagents/en/index`

## Weekly R&D Rhythm

Every R&D cycle should produce:

- One frozen eval manifest.
- One training manifest.
- One model card or model-card update for any pushed model.
- One summary JSON in `tmp/asr-results`.
- One doc update with exact WER/CER, sample counts, buckets, and caveats.
- One product recommendation: default, beta, hold, or reject.

No model gets promoted from trainer validation alone.

## Immediate Next Actions

1. Wire DONDO v2 + Twi LM into the beta ASR service.
2. Fix chat continuity: recordings append to the current conversation, and
   logged-out users keep guest-session continuity across refresh.
3. Add a compact model picker near the record control and keep sessions/settings
   in a collapsible side nav.
4. Repair response understanding and language-matched replies around the
   improved transcript path.
5. Prototype live voice/TTS quality checks with Twi-native voices first; keep
   Qwen3-TTS 12Hz as a research candidate.
6. Evaluate SmolAgents for commerce/tool orchestration and R&D automation.
7. Run the beta measurement loop: v6 vs DONDO v2 greedy vs DONDO v2+LM on the
   same fixtures and consented correction stream.
8. Build the synthetic Twi voice-note generator from reviewed corpus JSONL.
9. Expand the held-out product corpus to 250+ clips with speaker diversity.
10. Launch DONDO v3 only after the expanded corpus is validated.
