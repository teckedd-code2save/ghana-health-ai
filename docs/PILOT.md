# Ghana Health AI Pilot Runbook

Public pilot URL:

```text
https://ghanahealth.serendepify.com
```

## What is ready

- Voice-first home screen with Health / Commerce focus tabs.
- Twi and English runtime routes are separate so English does not fall back to the Twi ASR checkpoint.
- Replies stream through the live response pipeline and use the selected/spoken language path.
- ASR correction feedback is captured without retaining raw audio by default.
- Commerce can understand shopping intent, search the connected local catalog, and require explicit confirmation before cart mutation.

## Before sharing

Run:

```bash
pnpm eval:prod:smoke
```

This checks public readiness, live JSON responses, streamed responses, ASR feedback capture, and commerce confirmation.

Expected non-blocking degradations:

- `asr_model_promotion`: no current ASR checkpoint beats all promotion gates.
- `product_eval_data`: pilot speech buckets are not full yet.

## Pilot ask

Ask testers to try short, natural voice turns:

- Twi health symptoms and urgent warnings.
- Twi shopping requests such as buying medicine, rice, soap, tomatoes, or ORS.
- English health questions.
- Twi-English code-switching, especially medicine and product names.
- Mobile phone recordings with normal background noise.

When the transcript is wrong, testers should use the correction action under the heard transcript. Those corrections are the most valuable output of the pilot.

## Data loop

Build the same-day recording pack:

```bash
pnpm asr:collection-pack
```

Open `tmp/asr-collection-pack/recorder.html` to record prompts locally, or record on any phone and keep filenames as the prompt ID. Then attach the recordings:

```bash
ASR_AUDIO_DIR=/path/to/recordings ASR_SPEAKER_LABEL=speaker_001 pnpm asr:attach-audio
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-audio-ready.jsonl pnpm eval:asr-manifest
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-audio-ready.jsonl pnpm eval:asr-manifest:score
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-scored.jsonl pnpm eval:asr-quality
```

Export corrected transcripts:

```bash
pnpm eval:asr-feedback:export
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-feedback-export.jsonl pnpm eval:asr-manifest
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-feedback-export.jsonl pnpm eval:asr-quality
```

Use the export to identify high-frequency ASR errors and to fill the product eval buckets in `data/asr-product-eval/README.md`.

Do not spend major model credits until the eval buckets have enough consented examples to measure whether a new checkpoint actually improves Twi health, Twi commerce, code-switching, English retention, and phone/noise speech.
