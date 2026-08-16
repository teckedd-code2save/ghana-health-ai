# ASR model decision

Date: 2026-08-09

## Product direction

Keep the live product on the best validated Whisper-family ASR while we spend training credits on measured improvement. Do not hard-pivot the serving path to DONDO until a fine-tuned DONDO checkpoint beats the current Whisper baseline on our own eval sets.

The product should assume ASR can be wrong. Transcript quality metadata must travel into the response model and reviewer so unclear speech leads to a model-generated clarification instead of guessed health advice.

English speech should not be forced through a Twi-specialized checkpoint once English regression is observed. The app supports a separate English ASR route:

- `MODAL_ASR_URL`: default Twi/Ga product ASR endpoint.
- `MODAL_ASR_EN_URL`: English ASR endpoint: `https://createdliving1000--ghana-health-asr-en-api.modal.run`
- `MODAL_ASR_EN_TOKEN`: optional token for the English endpoint.

If `MODAL_ASR_EN_URL` is missing, English turns use the documented default English endpoint in `src/lib/modal-asr.ts`. Environment values still override the default, but English should not fall back to the Twi-specialized checkpoint because the Common Voice English check confirms v6 is far worse than base Whisper on English.

Deploy a separate English ASR endpoint without replacing the Twi endpoint:

```bash
ASR_APP_NAME=ghana-health-asr-en MODEL_ID=openai/whisper-small modal deploy modal/asr_service.py
```

Then set `MODAL_ASR_EN_URL` to that deployment URL in Infisical if the endpoint changes from the documented default.

Verified health:

```json
{"ok":true,"service":"ghana-health-asr-en","model":"openai/whisper-small"}
```

## Current evidence

| Model | Eval | WER | CER | Decision |
| --- | ---: | ---: | ---: | --- |
| `teckedd/whisper-small-waxal-round2-specaug-v1` | full Waxal test, beam 5 | 31.52% | 11.27% | Hold and validate |
| `teckedd/gha-whisper-small-twi-v6` | Waxal, beam 5 | 30.44% | 10.62% | Hold-and-validate only; not competitive enough |
| `teckedd/gha-whisper-small-twi-v6` | 100-sample Waxal, beam 5 | 32.16% | 10.74% | No clear promotion over prior |
| `teckedd/gha-whisper-small-twi-v6` | 100-sample Waxal, greedy | 32.87% | 11.20% | No clear promotion |
| `openai/whisper-small` | 100-sample Common Voice English, beam 5 | 11.82% | 6.00% | English baseline route |
| `teckedd/gha-whisper-small-twi-v6` | 100-sample Common Voice English, beam 5 | 42.34% | 32.74% | English regression confirmed |
| `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` | 100-sample Common Voice English, beam 5 | 15.10% | 8.10% | English retained better than v6 |
| `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` | 100-sample Waxal, beam 5 | 40.86% | 14.10% | Twi regressed; do not promote |
| `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` | 150-sample validation during train | 53.83% | 22.18% | Twi regressed badly; do not promote |
| `KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en` | 100-sample Waxal zero-shot | 71.91% | 26.69% | Not serving-ready |
| DONDO smoke fine-tune | Waxal smoke, 24 train / 8 eval, 8 steps | 53.82% | 17.81% | Wiring only; not quality evidence |
| DONDO real fine-tune checkpoint | Waxal eval subset at step 200/800 | 41.04% | 13.99% | Improved from zero-shot/smoke, still not promotable |
| `teckedd/gha-dondo-w2v-bert-twi-v1` | DONDO Waxal fine-tune final eval | 35.77% | 12.19% | Pushed to HF with model card; do not promote |
| `openai/whisper-small` | 100-sample Waxal zero-shot | 114.54% | 92.17% | Not usable for Twi |

DONDO is still worth exploring because its own in-domain card reports strong Southern Ghana multilingual WER, including Asante Twi. Our zero-shot Waxal result shows that product use needs domain adaptation before DONDO can replace Whisper. The DONDO Modal training path now reaches actual training/evaluation with capped streaming rows, so the next step is a real trial, not more theory.

The v6 Hugging Face model card has been backfilled with the real hold-and-validate positioning, WER/CER, base model, datasets, intended use, limitations, and medical-device disclaimer. It should not be described publicly as a final or competitive checkpoint.

## Promotion gates

A checkpoint can replace the current production ASR only if it passes:

- Twi WER below 28% on the same held-out Waxal split and decode settings.
- No material English regression versus the English baseline route.
- Separate health-domain eval: symptoms, pregnancy, malaria, medicines, emergencies, ecommerce health items.
- Code-switch eval for Twi-English turns.
- Phone/noisy-audio eval.
- Latency acceptable for the live voice loop.
- Hugging Face push includes a model card with base model, data, eval sets, WER/CER, intended use, limitations, and medical-device disclaimer.

## Credit-spend order

The detailed credit plan is in [`docs/model-credit-plan.md`](./model-credit-plan.md). The short version: build product eval data first, then spend credits only on runs that can pass the product gates.

1. Balanced Whisper continuation from the best current checkpoint.
   - Mix Twi, English, and code-switching.
   - Use conservative learning rates and freeze/partial-freeze trials to reduce English regression.
   - Stop runs early when Twi WER stays above the current 30% range or English regresses.
   - Current train script mixes Waxal Twi, Common Voice Twi, Common Voice English, and optional GhanaNLP Twi multispeaker data.

2. DONDO fine-tune trial.
   - Completed: `teckedd/gha-dondo-w2v-bert-twi-v1`, final WER **35.77%**, CER **12.19%**.
   - Result: useful research signal, not a serving candidate.
   - Further DONDO spend should be targeted: better data, health-domain eval, phone/noise eval, and a concrete path below the current 30% Whisper plateau.

3. Data loop.
   - Collect consented health utterances from real product categories.
   - Prioritize examples where ASR causes the response model to ask for clarification.
   - Add corrected transcripts to the next ASR training batch.

## Live pipeline requirement

The serving path remains:

`audio -> ASR -> transcript quality metadata -> schema-first understanding -> memory-aware LLM answer -> LLM safety review -> streamed reviewed reply -> optional TTS`

No static health response bank or hand-authored answer templates should be used as product intelligence. Retrieval is disabled in runtime understanding until ASR and intent quality are strong enough for retrieval to help rather than mask weak comprehension.

Commerce must remain action-intent first. A request to buy, price, find, order, or search for a health item is ecommerce unless the user is asking for clinical advice. This is covered by live JSON and SSE pipeline fixtures.

## Credit-spend commands

Print the full benchmark ladder:

```bash
python modal/train/benchmark_asr_ladder.py
```

Run Twi + English-retention eval jobs first:

```bash
python modal/train/benchmark_asr_ladder.py --execute --max-samples 500 --no-wait
```

Only after those evals land, launch controlled balanced fine-tunes. The first training command in the ladder is a small no-push proof run that verifies capped streaming before larger credit spend:

```bash
python modal/train/benchmark_asr_ladder.py --execute --include-train --max-samples 500 --no-wait
```

Training-loader status:

| Run | Modal app | Function call | Result |
| --- | --- | --- | --- |
| `v7-small-waxal-proof-streamed` | `ap-yz2EwC2h9UncQbBvlRxSVI` | `fc-01KZKRSKJGNSRRNKEM667DB9RJ` | Completed 80 steps; streamed 256 train / 64 eval Waxal samples; no HF push |
| `v7-small-balanced-extra-proof-streamed` | `ap-oQ8n9rH6mSDDvS9vSvFjf7` | `fc-01KZKVZ9BZ52HV5W00V67VT5MM` | Confirmed capped Common Voice extra-data path reaches training; no HF push |

Balanced v7 lite fine-tunes:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-lite-no-en-regression` | `ap-H0H15OWS8H7lxtE5WAuqAe` | `fc-01KZKWQEHHX171AHQ6AY15PF1G` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-5vhppGhhweckMY4HZO4XFP` | `fc-01KZKWQEJSEPMCM73NHNDSMTKV` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |

Current lite settings: 1,200 streamed Waxal train samples, 150 streamed Waxal eval samples, capped streamed Common Voice Twi/English extras, 500 steps. Both completed and are **not promotable**. The non-frozen run improved English retention versus v6 but regressed Twi to **40.86% WER** on the 100-sample Waxal beam-5 check. The frozen run ended at **53.83% validation WER**. The Hugging Face repos exist with README model cards, valid dataset metadata, base model, metrics, intended use, and medical-device disclaimer. `pnpm eval:model-card` validates that future card metadata strips dataset config suffixes for Hub compatibility.

Stopped earlier balanced v7 fine-tunes:

| Run | Modal app | Function call | HF repo |
| --- | --- | --- | --- |
| `v7-small-balanced-no-en-regression` | `ap-HbD5ihi2bRlVZs3yix7y87` | `fc-01KZKR6FZ2DPZVSET4603KJMBS` | `teckedd/gha-whisper-small-twi-en-balanced-v7` |
| `v7-small-balanced-frozen-no-en-regression` | `ap-hxlw7y8LZ7PyM13hVM0x8s` | `fc-01KZKR6FZ35T88N037AKDZEQAS` | `teckedd/gha-whisper-small-twi-en-balanced-v7-frozen` |
| `v7-medium-balanced-no-en-regression` | `ap-Gn8ykjRwLN9Mc410nFdmFZ` | `fc-01KZKR6G12KMBDSW459Q0JDWFY` | `teckedd/gha-whisper-medium-twi-en-balanced-v7` |
| `v7-small-balanced-lite-no-en-regression` | `ap-pZNnbsEBMIeCBLOh3MmiQ1` | `fc-01KZKRHXNWSAJPWWFKT3ND7A6C` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-aPtJKFxoavBNMp3Mv4wHcB` | `fc-01KZKRHXGGKF1Y02SGJGH2YXDY` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |
| `v7-small-balanced-lite-no-en-regression` | `ap-fJaR6pMxKDS2p5BwgolZsf` | `fc-01KZKS8FZ26W6GZZ0BEJTS7CTR` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-Dqk2wBvPIQwYZg5SKhLywV` | `fc-01KZKS8FZ73KXQNE1A6YB2227G` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |
| `v7-small-balanced-lite-no-en-regression` | `ap-spYSuQHextHp99xtVEr3Ig` | `fc-01KZKWE7EZ1QVPBB2XXRZQNY6M` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite` |
| `v7-small-balanced-lite-frozen-no-en-regression` | `ap-jObe1dq2JPlDuoh5KsvZ5O` | `fc-01KZKWE7EZPGT2JW090H80W3H3` | `teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen` |

Pull and summarize results:

```bash
pnpm eval:asr-results:pull
pnpm eval:asr-results
pnpm eval:asr-promotion
```

## DONDO trial commands

Smoke test the DONDO training path before spending credits:

```bash
pnpm train:dondo:smoke
```

Latest smoke result:

| Run | Modal app | Function call | Result |
| --- | --- | --- | --- |
| `dondo-waxal-twi-v1` smoke | `ap-Fllvp76L3nlPCLCCaxW1p3` | `fc-01KZSTXGGKQTW5K4YXZZT7SPF0` | Completed 8 steps; WER 53.82%, CER 17.81%; no HF push |

Launch the first real DONDO trial detached:

```bash
pnpm train:dondo:modal
```

That command pushes to `teckedd/gha-dondo-w2v-bert-twi-v1` only after training and writes a model card with base model, data, metrics, language-prefix note, promotion gate, and limitations.

Real trial history:

| Run | Modal app | Function call | Target HF repo | Status |
| --- | --- | --- | --- | --- |
| `dondo-waxal-twi-v1` | `ap-hGNVgVb8XAYSA0Vxv2zZXF` | `fc-01KZSV66NG7R2R3F5JSKFMK6WP` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Reached step 217, then CUDA OOM after checkpoint-200 eval |
| `dondo-waxal-twi-v1` resume | `ap-nYJdyzEwxZaXp0ubW7KqUi` | `fc-01KZSWMTN06A7NTRV97X6C68PF` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` H100 resume | `ap-SbPCmY2dgsqSD8zEuNyt44` | `fc-01KZSWTJ4WRAT1ZRSK4YNX26DR` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Stopped after staying silent/no logs |
| `dondo-waxal-twi-v1` GPU-fallback resume | `ap-F3x5vbrsPLQh13kBUTvcgA` | `fc-01KZSWZY0ES2NWC01N9PA8GQP4` | `teckedd/gha-dondo-w2v-bert-twi-v1` | Completed 800/800; pushed model + card; WER 35.77%, CER 12.19%; do not promote |

Final DONDO evidence: the resumed run completed 800/800 steps. It improved from **71.91%** zero-shot WER and **53.82%** smoke WER to **35.77%** final WER / **12.19%** CER, then pushed model files and a valid model card to Hugging Face. That is real progress, but it still fails the baseline-to-beat value of **30.44%** and should not replace the current Whisper-family serving path.

The DONDO training script now includes explicit phase logs and row-prep progress logs. Keep it available for controlled follow-up experiments, but do not run another expensive DONDO trial without changing the data or evaluation design.

---

## 2026-08-16 R&D session addendum

Full writeup: [`docs/asr-rnd-session-2026-08-15.md`](./asr-rnd-session-2026-08-15.md).

### New evidence (same-split comparisons, streaming)

| Model | Eval | WER | CER | Decision |
| --- | ---: | ---: | ---: | --- |
| `teckedd/gha-dondo-w2v-bert-twi-v1` | Waxal test n=300, CTC greedy | 36.47% | 11.36% | Trails v6 (30.16/28.76) on benchmark |
| `teckedd/gha-dondo-w2v-bert-twi-v1` | Common Voice 22 English n=100 | 43.55% | 17.11% | English route stays separate, permanently |
| `teckedd/gha-whisper-small-twi-v6` | Local product corpus n=40, beam5 | **54.18%** | 20.92% | Benchmark WER does not predict product WER |
| `teckedd/gha-dondo-w2v-bert-twi-v1` | Local product corpus n=40 | **32.66%** | 10.78% | **DONDO beats v6 by ~22pp on domain audio** |
| v6-local-adapt-s600 (checkpoint) | Waxal test n=1522 greedy / beam5 | 31.76% / 30.73% | 10.68% | No regression; not a promotion |
| v6-local-holdout-s600 (checkpoint) | Held-out 8 domain clips, beam5 | **35.00%** vs v6 46.67% | 12.79% | **Domain data generalizes: −11.7pp from 32 clips** |

### Updated direction

1. **Domain evals become first-class promotion gates.** v6's 30.44% Waxal
   beam5 hides a 54.18% product-domain WER. Waxal-first promotion measured the
   wrong thing. Gates must add held-out product-domain clips (health/commerce/
   code-switch) — candidates must never train on the gate set
   (`tmp/asr-local-train/manifest.train32.jsonl` / `manifest.holdout8.jsonl`
   is the split template).
2. **DONDO v2 is approved for spend** with the changed design: full Waxal +
   CV-Twi + local corpus, LR ~1e-4, KenLM Twi beam decode, domain-first eval.
   Rationale: best domain number on record (32.66%) achieved with every
   handicap (1,800 rows, LR 5e-6, bare CTC greedy).
3. **Whisper v8 (Stage 1) is approved for spend** gated on corpus scaling:
   the hold-out experiment proves the data lever works for the Whisper track.
4. **Data collection is the critical path** (200+ clips, 4+ speakers,
   code-switch priority, noisy/phone conditions). Both tracks are gated on
   data, not compute.
5. **v6 remains the serving model.** No checkpoint from this session promotes.
