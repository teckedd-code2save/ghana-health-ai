# ASR Product Eval Manifest

This folder defines the product-level ASR eval set. Keep private audio out of git unless it is explicitly consented and safe to share. Prefer storing audio outside the repo and committing only a manifest with stable IDs, labels, and references.

## Required Buckets

| Bucket | Minimum before serious training spend | Examples |
| --- | ---: | --- |
| `health_twi` | 100 | symptoms, malaria, pregnancy, medicine, emergency |
| `commerce_twi` | 50 | buy, price, find, order, cart, delivery |
| `codeswitch_tw_en` | 50 | Twi sentence with English medicine/product names |
| `health_en` | 50 | English health questions for retention |
| `phone_noise` | 50 | mobile mic, background noise, low bandwidth |

## JSONL Schema

Each line in `manifest.jsonl` should follow:

```json
{
  "id": "health_twi_0001",
  "bucket": "health_twi",
  "language": "tw",
  "reference": "me yam yɛ me ya na me ho yɛ hyew",
  "audio_path": "/secure/eval/audio/health_twi_0001.wav",
  "speaker_label": "speaker_001",
  "consent": "internal_eval",
  "domain_tags": ["symptom", "fever"],
  "recording_tags": ["phone", "quiet"],
  "notes": "No diagnosis label; transcript only."
}
```

Rules:

- `reference` is the human transcript used for WER/CER.
- `audio_path` may be absolute, repo-relative, or a remote private URI.
- Do not include names, phone numbers, addresses, or personally identifying health stories.
- Use `speaker_label`, not real names.
- Keep clinical labels out unless a qualified reviewer created them.

## Validation

```bash
pnpm eval:asr-manifest
```

Use a custom path:

```bash
ASR_PRODUCT_EVAL_MANIFEST=/path/to/manifest.jsonl pnpm eval:asr-manifest
```

## Collection Pack

Generate a read-aloud pack for same-day pilot recording:

```bash
pnpm asr:collection-pack
```

This writes:

- `tmp/asr-collection-pack/prompts.jsonl`
- `tmp/asr-collection-pack/prompts.csv`
- `tmp/asr-collection-pack/README.md`
- `tmp/asr-collection-pack/recorder.html`

Ask speakers to record files using the prompt ID as the filename, for example:

```text
health_twi_sp001_u0001.wav
commerce_twi_sp003_u0004.webm
```

For one-machine collection, open `tmp/asr-collection-pack/recorder.html` in a browser and download each recording. The recorder uses the correct prompt ID as the filename.

Attach recorded audio into an eval manifest:

```bash
ASR_AUDIO_DIR=/path/to/recordings ASR_SPEAKER_LABEL=speaker_001 pnpm asr:attach-audio
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-audio-ready.jsonl pnpm eval:asr-manifest
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-audio-ready.jsonl pnpm eval:asr-manifest:score
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-product-eval-scored.jsonl pnpm eval:asr-quality
```

Use `pnpm eval:asr-manifest:strict` only after the bucket minimums are full. Strict mode now also rejects missing or tiny local audio files.

## Feedback Export

Voice transcript corrections are stored in `asr_feedback`. Export them for review:

```bash
pnpm eval:asr-feedback:export
```

Default output:

```bash
tmp/asr-feedback-export.jsonl
```

Then validate the export:

```bash
ASR_PRODUCT_EVAL_MANIFEST=tmp/asr-feedback-export.jsonl pnpm eval:asr-manifest
```

Important: correction exports use `MISSING_AUDIO_FOR_FEEDBACK_*` placeholders until matching consented audio is available. These rows are useful for transcript review, ASR error analysis, commerce/health phrase mining, and prompt/eval construction. They are not full acoustic ASR training rows until real audio paths are attached.
