# Foundation and Voice Experiments, 2026-09-09

Status: in progress. No new model has passed the Twi delivery gate. No proprietary
teacher calls, corpus publication or production model replacement in these runs.

## Direct instruction baselines

`modal/train/benchmark_general_understanding.py` keeps hidden reference answers
and rubrics local. Only messages, test IDs and read-only tool schemas go to Modal.
The expanded manifest contains 74 cases: 30 project/product regression checks,
12 held-out source-reading cases (WAXAL, GhanaNLP and AfriHealth), and the same
32 locked medical response cases used to test Pilot v1. Synthetic fixtures are
tests, not new training data. Raw predictions and immutable manifests are in
`tmp/general-foundation-comparison/`.

Measured first runs:

- AfriqueQwen3.5-4B-Instruct-v1, revision
  `b714a660a3d8b643dc02a36f6c44471cf9229946`, failed Twi understanding: repeats
  prompts, invents tomato meaning, switches languages and loops. It is the
  post-trained original Afrique checkpoint, NOT the 50-language Twi CPT variant.
  Run `ap-NZ1J6mpaz8MvMKRA3vSB2c`, raw output
  `afrique-20260909T203916Z.json`. This preliminary 38-case run accidentally
  described four English AfriHealth excerpts as Twi. Exclude those four from
  conclusions; the errors on separate Twi regressions already reject the model.
  The corrected manifest includes GhanaNLP and correct source-language labels.
- Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, completes
  English instructions but fails elementary Twi meaning. For example, eye pain
  becomes "I am your father" and the negated tomato purchase becomes eating a
  ripe tomato. Run `ap-gBkU5hJWFgug8Ess4iaMWF`, output
  `qwen9-20260909T204832Z.json`; 74 cases, 327.810s loading, 127.979s batched
  generation, actual A100-SXM4-40GB. Greedy non-thinking decoding, max 640 tokens.
  This is not the publisher's preferred sampling configuration; no claim is made
  that it exhausts all possible settings. These gross meaning failures still
  mean it is not accepted as a direct Twi assistant or annotation teacher.

No percentages of clinical accuracy or native-language quality are inferred from
lexical checks, model size, loss, JSON validity or reference similarity.

## Instruction-transfer experiment

`modal/train/build_twi_instruction_bridge.py` builds two private CPU candidates:

`AfriqueQwen3.5-4B-50Langs + alpha * (Qwen3.5-4B - Qwen3.5-4B-Base)`

Alpha is 0.5 or 1.0. Tokenizer IDs, architecture dimensions, tensor names, shapes
and finite values must agree. Only text weights are produced; original instruction
tokenizer and chat template are retained. Source revisions, tensor counts and
shard hashes are preserved. Existing Modal public-model cache is used; no private
corpus or speaker recording participates. This is merging, NOT gradient SFT.
Candidates are not promoted merely because they load.

Both CPU merges completed in 178.47 seconds, with 426 compatible language
tensors and 8,411,502,592 bytes per candidate. Build run
`ap-DELN0bvwhutmFgqymKCkKT`; local report `bridge-build.json`. Both candidates
failed generation checks. Alpha 1.0 produced corrupted, repetitive completions
on the 74-case run (`bridge10-20260909T211414Z.json`). Alpha 0.5 had relevant
initial content, but failed to end its turn. All 17 follow-up cases reached the
640-token limit even with the publisher's sampling and repetition settings
(`bridge05-20260909T212523Z.json`). It invents later chat turns and search results.
Do not trim these failures away and present the opening sentences as a success.
Matching shapes and vocabulary do not prove common weight ancestry or validate
instruction-vector transfer. No more merge promotion is justified by these runs.
The shared cache now retains compiled inference kernels to reduce repeated setup.

A separate public foundation, `google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, completed the same 74 cases in
`ap-cwPFBrnaxOGcuwARu1K4kZ` (`gemma12-20260909T211525Z.json`). It handles
several English and simple arithmetic tasks but misreads Twi: buying becomes
eating, a corrected quantity reverts, and breathing becomes a homepage question.
It is not accepted as a Twi assistant. Apache-2.0, ungated, no hosted Google API.
No new fine-tune is claimed.
[Official model card](https://huggingface.co/google/gemma-4-12B-it).

Gemma 4 31B, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, completed
30 project/product cases in `ap-bLI87Y1Xw81l0VEChdCsaI`
(`gemma31-20260909T212359Z.json`). A generic Twi/English language hint was added,
so this is NOT a size-only controlled comparison against the 12B run. All turns
stopped normally. Negation, arithmetic, a named-person reference and native tool
requests were materially more useful. Twi health understanding still failed:
breathing became tight clothing, and a young baby's age became prematurity.
Twi wording also remained poor. The isolated eye phrase was interpreted as
sleepiness; its ambiguity needs native review rather than an invented certainty.
31B is a possible training foundation, not a production winner or an unchecked
Twi annotation teacher. No held-out corpus results exist for this 30-case run.

Method motivation, not evidence that it works for Twi:
[Merge and Conquer](https://arxiv.org/abs/2603.28263) and
[EstLLM](https://arxiv.org/abs/2603.02041).
The latter uses CPT, multilingual replay and post-training; some English
regressions remain, so both languages need evaluation here.

After a foundation demonstrates meaning preservation, apply a balanced,
source-separated training mixture from the general research roadmap. The 95-source
pilot is not the full corpus; do not run more epochs on it and call that progress.

## External Direct-Answer Reference

After the translation-heavy project adapter failed conversational checks, a
bounded comparison was prepared for
[`DariusTheGeek/mhqa-itu-adapters`](https://huggingface.co/DariusTheGeek/mhqa-itu-adapters),
revision `16247b4983ed169c5cfc18f2099701cf31425500`, subfolder `gen7454`.
This is explicitly an external author's model, not another project fine-tune.
It tests whether direct Akan response supervision helps the same Gemma 31B
foundation. The publisher does not pin the original base revision; our comparison
uses the existing pinned base and identical base/adapter prompts and decoding.

The model card's 0.728509 leaderboard score belongs to an entire retrieval,
generation, sampling and judge ensemble, NOT this adapter's standalone accuracy.
Training is competition sexual/reproductive-health QA; no clinical, general-chat
or commerce competence is assumed. No answer bank, retrieval, ensemble, external
judge or proprietary response fallback is included in our comparison. These
development checks are separate from AfriHealth corpus scoring; overlap with the
external model's competition data would invalidate a naive AfriHealth holdout claim.

Runner: `modal/train/compare_public_response_adapter.py`. One A100-80GB, 1,800s
ceiling, 30 paired prompts, 384 output tokens maximum. Raw replies are saved as
they arrive. There is no training, production endpoint, model publication, owner
audio upload or private conversation input. Only pinned public weights are cached.

CPU preflight `ap-AbEi9NjyHJGTcPUi1L7nW2` passed exact names/shapes for 820
text LoRA tensors. The published checkpoint includes 378 additional image-encoder
tensors, which are unused by text-only inference and explicitly excluded. No
other unmatched tensors are silently dropped. The first CPU preflight correctly
rejected those image tensors before any GPU use. Published weight SHA256:
`9f3a8e6d3982e6c673c9bd5f533a93102bb32e6da588466537e55a2b86894245`.
The paired GPU run `ap-gR867dkg5P1W9tWwOwgsVN` completed all 60 replies.
Artifacts: `tmp/general-foundation-comparison/external-gen7454-20260909T232755Z.jsonl`
and its `.summary.json`; raw-input SHA256
`56d37b954882ee840f14df4f1fac23c54bb98ea0705aa322b0573645630b1c41`.

Direct-answer supervision improved the specific breathing/chest-pain and
pregnancy examples: they now preserve the reported symptoms instead of confusing
breathing with clothing. It also retains corrected 1kg, arithmetic, English
switching and the English web-search call. However, the tomato tool request
becomes an invented `fa_brɛ_wɔ_adenta(tomato, 2)` function, not an available API.
Budgeting and child-fever replies loop; the code-switched fever reply also repeats
and hits the length limit. Two cases trigger the mechanical eight-word repetition
flag; three reach the token limit. The base has one non-repetitive length stop.
The newborn-malaria follow-up loses important conversational context, while vague
unwell prompts receive generic support advice. No native/clinical score is invented.

Decision: not a product replacement or general-purpose model release. This gives
limited positive evidence for direct Twi answer supervision, not a complete fix.
The next project SFT should compare direct-response mixtures with retained English
conversation and valid tool trajectories, against the base and saved language
checkpoint. Do not extend the translation-only recipe or train more copies of the
same narrow medical examples. The comparison remains development evidence.

## Additional Source Audit

More rows are not automatically more direct-response supervision. Public cards
were inspected before adding any of these sources to training:

- [ghana-chat-corpus-ak](https://huggingface.co/datasets/michsethowusu/ghana-chat-corpus-ak),
  revision `56cb228a73f2b1e9226d20bca8f2fa9f6fc9b94b`, has 226,300 rows but
  describes machine translation through an Amharic pivot. Its source extraction
  joins assistant responses while keeping the first user question. This can
  flatten unrelated turns; it is not native-reviewed conversation gold.
- [Aya Twi](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split/viewer/twi/train)
  has 7,320 rows in the viewer. The first inspected rows are AfriSenti sentiment
  classification with English instructions/labels, not Twi assistant answers.
  This is a partial inspection, not a claim about every row in that split.
- [VoxCPM2 Akan](https://huggingface.co/FarmerlineML/voxcpm2-akan-sft) is another
  possible speech comparator. Its card leaves exact dataset provenance incomplete;
  reported validation loss is not a native listening result. No run was started.

None of these sources was bulk-imported or used to inflate accepted corpus counts.

## Speech work

Production `/api/config` was read on 2026-09-09: Twi uses stable-Twi, English MMS.
Two actual routing defects were fixed locally, not yet deployed:

- ASCII Twi words incorrectly triggered the mixed-English voice. Only explicit
  English language or bracketed English spans now change the stable-Twi mode.
- `GHS 25` became `Ghana Health Service 25`; prices now preserve currency meaning.
  Explicit bracketed English spans are preserved only for stable-Twi.

`modal/train/compare_twi_voices.py` produced nine matched-text CPU samples:
stable-Twi speakers 1 and 6, plus nano-Twi four-step. Three texts, pinned model
revisions. `scripts/prepare_voice_listening.py` creates constant-gain matched
listening copies while preserving raw audio and recording hashes. No pitch,
timing or denoising edits. All artifacts: `tmp/twi-voice-comparison/`.
Run `ap-KLHnZIbXzhbOk7G5TFKK23` completed. Nano's lower runtime does not prove
better pronunciation, and the user has not yet rated these samples.

`modal/train/compare_cosy_twi_voice.py` tests Kasanoma Twi v0.4 (CosyVoice3)
using the owner's previously authorized speaker-1 reference. The reference is a
temporary inference input, not a volume artifact or published voice. No other
speaker is used. Model/runtime pinned; three same-text samples. First image build
failed on an old package's `pkg_resources` dependency, before inference. The build
environment was pinned and retried. Two subsequent runs exposed missing imports
(`matplotlib`, then `wget`); these were fixed, and imports are now checked during
the CPU image build. Run `ap-L3ZtYhYMMLTGnaI3S8faHs` successfully produced three
24kHz samples in `cosy-manifest.json`: greeting 4.16s, question 6.32s, shopping
7.00s. Synthesis took 7.74s, 5.50s and 5.95s respectively, excluding model loading.
The ONNX frontend fell back to CPU because its CUDA-11 library was unavailable;
the main Torch synthesis ran on the L4. This is not optimized streaming latency.
`cosy-listening.json` contains matched-level listening copies. Originals remain.

`scripts/check_voice_roundtrip.py` passed all twelve generated samples through
the existing DONDO v2 private method. Same three sentences, 28 reference words
per voice. Word-edit counts: stable voice 1 = 18, stable voice 6 = 20, nano = 16,
CosyVoice with owner reference = 10. Raw recognitions, actual model/decode path,
audio hashes and normalization are recorded in `asr-roundtrip.json`.
This is only a diagnostic: tiny sample, spelling/segmentation sensitivity, ASR
errors, and possible speaker bias because DONDO was adapted using owner speech.
It does NOT establish naturalness, clinical correctness, speaker similarity or a
general TTS accuracy percentage. The owner has not rated the listening samples.
No production voice replacement has been made.

Additional local speech-completeness fixes: `speakableText` no longer truncates
at 500 characters. Supported text up to 2,000 cleaned characters is retained;
oversized input gets an explicit no-audio result. Stable-Twi synthesizes bounded
chunks, keeps bracketed English spans intact, verifies PCM sample rates/finite
audio, and returns the complete concatenated utterance. MMS fallback no longer
silently cuts at 600 characters. Optional TTS failure now leaves the completed
conversation text intact. These changes are not deployed. Unit checks cover
text preservation and chunking. A subsequent real service check completed in
`ap-F23dVSkiOolW8nZVw3PoYk`: 664 characters delivered through five actual synthesis
calls, 40.474 seconds of valid PCM audio, 10.370 seconds synthesis time. An explicit
English span stayed intact. `completeness-check.json` records the transport result;
it is not a pronunciation score.

### Private Streaming Voice Measurements

The owner explicitly authorized temporary private Modal synthesis with their
reference in the current chat. It is passed as an argument and temporary WAV,
never written to the model volume or published. Generated audition audio stays
local; this is not a trained owner-voice checkpoint or a public voice endpoint.

`ap-ZjeaCWjd5beWf1emH9opod` exposed mutable buffering in the pinned upstream
CosyVoice model: `token_hop_len` grows 25 -> 50 -> 100 and remains enlarged for
later utterances. The sequential comparison now resets the initial value before
each utterance. This is not a validated concurrency-safe production patch.
`ap-cTJs8SPpJ4cxiFxc4rZAWC` confirmed warm first-audio improvements on the same
two phrases: approximately 6.4-6.9 seconds became 3.2 seconds.

The optimized runtime uses Torch/Torchaudio 2.5.1, ONNX Runtime 1.20.2, Whisper
20240930 feature extraction and FP16. The old Whisper package constrained Triton
to an incompatible version; the upgraded package passes `pip check`. CUDA 12 /
cuDNN 9 library discovery is explicit, and `ldd` is checked during the CPU build.
Inference refuses an optimized result if the speech-tokenizer session lacks the
CUDA provider. Shape operators and the speaker-embedding frontend can remain on
CPU; this is not a claim that every operation runs on GPU.

Failed intermediate runs are not hidden: one unavailable package version, one
isolated-build `pkg_resources` failure, and two runtime library-loading failures.
Final run `ap-eNG5gprZKwzY7fqmN6Iyj5` completed with CUDA enabled:

| Text | First audio | Synthesis total | Audio duration |
| --- | ---: | ---: | ---: |
| Greeting, first utterance | 5.308s | 8.773s | 5.20s |
| Feeling unwell, warm | 2.500s | 7.170s | 6.68s |
| Shopping, warm | 2.498s | 7.376s | 7.08s |

Times exclude model loading/network and are three single measurements, not p95.
Later chunks still arrive too slowly for uninterrupted immediate playback. Do not
call this live-ready. Raw audio and chunk timings: `cosy-streaming-fp16.json`.
Gain-only audition copies: `cosy-streaming-listening.json`. FP16 changes generated
audio, so the earlier pronunciation diagnostic cannot be transferred to it.

The local Gradio tester has a collapsed five-voice comparison. Each sample is
allowlisted and hash-checked; only generated audio bytes are served. Ratings reset
on selection changes and are stored against the exact audio hash in local SQLite,
never automatically marked training-eligible. Original chats and reviews remain.
Desktop/mobile browser tests verify actual audio metadata and switching, not
pronunciation. Native listening feedback is still required before voice selection.

### Owner Feedback And Tail Comparison

The owner listened to the optimized feeling-unwell sample and wrote:
"this sounded more like me, just became a bit muffled at the end."
The exact observation is saved locally in `voice_observations`, linked to
`ad416b3eb6806228e81e332f2a8efa7bf27f9f92318487fb20138d7a531ba5c7`.
It is a positive speaker-likeness observation with an ending defect, not a blanket
pronunciation pass or an invented naturalness score. This supersedes the earlier
"unrated" status for that one sample only; original manifests remain unchanged.

The delivered WAV contains the full 6.68 seconds returned by the three synthesis
chunks. It has no digitally clipped samples, and the audition copy differs only
by constant gain plus PCM quantization. The final second is quieter than the
first three seconds; that measurement alone does not diagnose muffling.

Private test `ap-0eQoLZNCy24g0J5IMQY4Zr` completed three variants with identical
reference, text, seed, pinned weights and optimized runtime. The FP16 streaming
control reproduced the original WAV byte-for-byte. FP32 streaming produced a
6.32-second version; whole-utterance FP16 produced 6.68 seconds and retained a
quiet tail. These modes may change sampled speech tokens, so this is not an
isolated vocoder-precision experiment. No equalizer or ending-only gain was used.

Raw report: `tmp/twi-voice-comparison/cosy-tail-check.json`. Gain-only listening
report: `cosy-tail-listening.json`. New inline samples offered to the owner:

- A: `listen-tail-fp32-stream-question.wav`, SHA256
  `efdcdf5441a8475ba615cfd5b9a4d2332ebd8def0344ec8896ba5c6231296a70`.
- B: `listen-tail-fp16-whole-question.wav`, SHA256
  `f00b8aa414804d65f09d6a06a06b6c56dff6038bfb5707cc409285ef109d764a`.

On September 10, the owner chose **A** in response to the A/B/original comparison.
The exact `A` reply is stored in local SQLite `voice_preferences` with the comparison
question and hashes of all offered samples. This is a relative one-sentence
preference, not a pronunciation pass or numeric naturalness score. The private
audition starts on `CosyVoice A`; A/B options are available only for this sentence.
No production voice replacement or chat inference change. Longer-utterance quality
and uninterrupted streaming remain unverified.
The source reference was again a temporary Modal argument/file, not a model-volume
artifact, saved speaker embedding or public voice. Platform-level processing and
retention are not represented as nonexistent. The completed test has stopped.

`modal/train/train_tts.py` is an old preparation skeleton, NOT a working speech
fine-tuning loop. Do not run it or publish its copied weights as a trained model.
Qwen3-TTS does not list Twi among its supported languages; voice cloning alone is
not a demonstrated Twi-language adaptation.

## Checks

TTS routing TypeScript checks and full TypeScript type checking passed earlier
in this work. The latest Python run passes 37 tests across voice routing,
audio review provenance, language-data preparation, training contracts, benchmark
isolation, playground, pilot core, paired-checkpoint diagnostics and external
adapter mapping. Run the Python suites with
`PYTHONPATH=scripts` because the playground test imports its sibling directly.
Python 3.13 reports existing SQLite connection ResourceWarnings; assertions pass.
No production deployment or Next UI changes have
been made in this experiment. Preserve the existing private playground and all
local reviews while changing its eventual model choices.
