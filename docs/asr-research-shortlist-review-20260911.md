# ASR Research Shortlist Review - 2026-09-11

User scope correction after this review: understanding/response LLM ONLY.
The ASR experiment order here is parked, not approved active work. Use relevant
language-adaptation research principles for the LLM; see
`docs/response-research-reset-20260911.md` and `docs/CONTINUE.md`.

## Scope

User shared [Weekly ASR Research Shortlist](https://chatgpt.com/c/6a862c01-8b38-83ea-b1e3-c80212dbeedf).
Read the conversation, including the Kumawood WS25 plan and subsequent weekly
shortlists. Checked the main candidate papers and dataset/model cards against
primary sources. This is a research review, not authorization inferred from the
shared conversation to run its commands or schedule its historical reminders.
No new ASR job, public upload, model promotion or production change was made.

ASR improvements address hearing. They cannot explain or repair the typed
`wo ho te s3n` response failure. Keep response-model research active separately.

## What Is New And Useful

1. **Qwen3-ASR Twi challengers.** Compare the
   [Asante](https://huggingface.co/dicksonsarpong9/qwen3-asr-asante-twi) and
   [Akuapem](https://huggingface.co/dicksonsarpong9/qwen3-asr-akuapem-twi)
   checkpoints before training them. Asante's card reports 1.15 seconds on one
   4.48-second clip on H200, not held-out WER. This is not comparable to our
   product latency without matched hardware and serving conditions. The
   GhanaNLP organization Akuapem copy is not an independent architecture.
2. **[BuzzASR](https://arxiv.org/html/2609.09554v1).** The paper combines
   language-specific tokenization, text-only decoder training and ASR training.
   It reports better CER than Whisper-large-v3 on 77/102 evaluated languages;
   this is not a measured Twi result or a promised improvement for Whisper-small.
   New token embeddings AND output projections are warm-started from old
   subtoken representations, with consistent IDs and special tokens. Merely
   swapping a tokenizer file would be wrong. Measure Twi fragmentation first;
   byte-level tokenizers' unknown-token counts are not a useful proxy. Run an
   explicit English-retention comparison for any Twi specialization.
3. **[Kumawood speech](https://huggingface.co/datasets/ghananlpcommunity/kumawood-speech-transcriptions).**
   Useful conversational acoustic material, but `twi_text` is Google STT and
   `text` is English subtitle OCR, not a verified translation pair. Subtitles
   may condense dialogue and their timing is not exact speech segmentation.
   Preserve film grouping, label provenance and noisy-label status. Do not
   turn character dialogue into assistant self-descriptions for LLM training.
4. **Task and failure-specific evaluation.**
   [SpeakPay](https://arxiv.org/abs/2609.01737) motivates scoring quantities and
   transaction success alongside WER, but its reported 33.33% transaction success
   is not product readiness or a Twi result. The
   [hallucination-projection paper](https://arxiv.org/abs/2609.04561) also reports
   a genuine-speech WER penalty for its gated method, +0.33 to +4.39 percentage
   points in its experiments. Evaluate quiet speech and false rejection before
   considering such a change; do not simply suppress more audio.

## Do Not Repeat Completed Work As A New Discovery

`docs/asr-rd-execution-plan.md` already records DONDO v2 with a Twi language
model: 27.31% WER on its Waxal n=300 slice, versus 28.12% greedy and 28.76%
Whisper v6 beam5. Its 6.67% local result is on only eight clips. These are
historical, limited-slice results, not a fresh comprehensive comparison.
DONDO and language-model rescoring are existing baselines, not new proposals.

## Corrections Before The WS25 Experiment

- 25 hours weak plus 5 hours clean is 83.3/16.7 by duration, not 80/20. State
  whether the sampler balances duration, utterances or tokens. An 80/20 duration
  mix would require 6.25 clean hours for 25 weak hours.
- 1,200 updates is not automatically three epochs. Calculate effective passes
  from the actual selected utterances, batch size and sampling policy.
- Twi Common Voice replay does not protect English. Include English replay,
  held-out English and code-switch evaluation, with a declared regression gate.
- Teacher agreement is not truth. Calibrate agreement thresholds against a
  blinded human sample, track shared errors, and compare an independent strong
  teacher instead of assuming v6 pseudo-labels improve v6.
- An `eval` split name does not establish independence from historical model
  training. [Ghana Speech Eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
  includes Ghana Speech and WAXAL-derived sources. Audit audio/source overlap
  for every candidate; keep a separate consented product holdout.
- Report strict WER/CER plus explicitly versioned orthographic normalization,
  dialect, speaker, device, domain, silence and critical-entity slices. Keep
  spelling tolerance from hiding changed quantities or negation.
- Keep research-use restrictions/provenance attached to the data and resulting
  model; research eligibility is not automatic commercial deployment clearance.

## Revised Order

1. Freeze and overlap-check a common evaluation set. Compare current DONDO v2
   greedy and LM, Whisper v6, and the Qwen3-ASR Twi challengers. Pin revisions.
2. Use observed errors to choose either a filtered conversational-audio pilot
   or a tokenizer/text-adaptation ablation. Do not launch both at corpus scale
   before establishing the relevant failure mode and compute budget.
3. Expand only the method that improves held-out accuracy while satisfying
   English, code-switch, quiet-speech and latency constraints.
4. Publish reproducible evaluation artifacts and a complete model card,
   including negative findings, source restrictions and remaining limitations.

This review verifies the main sources above, not every incidental news item or
reference in the shared multi-week conversation. There is no evidence here
that 30% is an immutable Twi WER ceiling, nor a guarantee of a particular gain.
