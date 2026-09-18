# Native Answer Instruction Synthesis

## Native Question Translation

`scripts/translate_native_questions.py` adds Twi question candidates and a separate
English back-translation, without overwriting source answers or English questions.
Inputs come from the 71 retained sources, with 117 candidate questions, not 117
independent answers. The existing Afrique 9B language-task runner is reused for
translation only, not chat generation or paired evaluation.

Forward run `afrique9_20260910T153004Z`, call
`fc-01M25Z1JZT3KHHWCPJWZAQMY8D`, completed all 117 in 269.20 seconds with no
empty/length-limited outputs. Inspection still flagged 56 possible meaning
changes, including percentages, question focus, named roles and units. Zero
format failures is NOT semantic verification. Three separate agent-proposed
corrections are offered for review, not silently applied or human-approved.
Findings are checksum-bound to the actual inference result in
`data/response-adaptation/native-question-review-findings.json`.

The completed back stage also translates all 71 ORIGINAL Twi answers to English for source
alignment inspection. This is additional evidence, not a replacement English
reference. The model never receives the matching original English answer in that
request. Same-model back-translation cannot independently prove correctness.
Run `afrique9_20260910T153628Z`, call `fc-01M25ZDA12KG1BRX3P2YKX67H0`, completed
188 outputs in 423.00 seconds. There were no empty/length-limited translations.
Actual failures include Tuesday becoming Thursday (66021), a borehole instruction
becoming boiling water (96944), fire brigade becoming a gang (67778), and sport
becoming market (64446). These are model-output errors, not evidence to silently
rewrite or reject an otherwise correct original. Source and translator errors
must be assessed separately.

Question/answer source IDs, revisions, unchanged strings, source hashes and
training splits stay attached. New demonstrations are unchanged training sources
66150, 71191 and 67230, chosen outside all candidate source groups and held-out
groups. They differ from the earlier translation benchmark's demonstrations;
do not import that benchmark's chrF++ result as this run's accuracy.

Artifacts: `tmp/native-question-translation-v1/`. Both stages and export are
complete; do not resubmit or overwrite. The exporter offers question translations,
back-translations, evidence/reviewer findings and original answers together.
Every record remains in review with no automatic training/gold acceptance.
`review.json` file SHA256:
`0e15ac5f5695a46125fb1ddae20e7865781e298b1e3435099cbbd7d320ba324b`.
Logical content hash is separately recorded in `summary.json`; it is not the
byte hash of the indented JSON file. The complete folder and correction findings
are backed up on the private Modal volume under the original native-answer run.
News/event statements require source context; an unchanged parallel sentence
is not automatically a justified standalone conversational response.

```sh
uv run scripts/translate_native_questions.py status --stage back
```

Six focused translation-preparation tests cover immutable answers/snapshots,
complete result identity, keeping original English references out of back-translation
inputs, review-only exports, and refusing findings from a different model result.

## Latest Evidence-Grounded Review

An additional completed stage, `ground`, preserves the original generation and
boolean-review results. Call `fc-01M25VP9J0KZ3T8XPBT01RZHXT` used the same pinned
Gemma English reviewer on 105 sources plus 14 paired calibration controls.
It completed 119 inputs in 272.08 seconds. Every claimed supporting passage
is checked as an exact substring of the source answer; incomplete evidence is
not accepted. Separate fields assess answerability, missing context, speaker
identity, and suitability. These remain model judgments, not certification.

The model passed 26/28 calibration judgments. It now rejects the actual missing
reason case 67762 and context-dependent 69190. It still falsely accepts one
unidentified-pronoun control and rejects one properly grounded pronoun control.
This does NOT pass the automatic batch gate. Controls test known failure
categories and are not a new unseen benchmark.

Inspection found nine additional source defects, including 79 becoming 70,
200,000 becoming 100,000, and a legal sentence becoming a grammatical sentence.
There are now 14 agent flags in the source-review inventory; no original source
text was rewritten. Some are possible wording/terminology defects requiring
native review, not asserted human corrections.

Current review artifact: `tmp/native-answer-synthesis-v1/ground-review.v2.json`.
All 128 original sources remain, with 71 having a conservatively retained
English-compatible review candidate after source flags. This is not 71 gold
training rows. The earlier `ground-review.json` mistakenly hid all candidates
when a global calibration check failed; v2 separates visibility for review from
automatic acceptance. The original export is preserved, not overwritten.

Additional unresolved issue: many source answers describe a particular news
event, government decision or unnamed organization. A generated question that
matches that answer does not make the assertion valid in a context-free chat.
Those need supplied source context or exclusion from standalone response SFT.
Do not silently train them as current facts. English matching also cannot
certify the Twi side, which is why source flags override positive English votes.

Grounding result logical hash:
`202d197477fd71bbb20eeeb741a82885e5e75de24b5b985ee11de78c233b034f`.
New tests cover verbatim evidence, required judgments and hidden control labels.
No training or model promotion resulted from this calibration run.

The original experiment and completed earlier stages follow.

## Reason For This Experiment

V5's source-task gains did not become good Twi replies. Its Twi-labeled response
tokens were dominated by health text; no broad Twi conversation task was present.
Larger Qwen models also made clear Twi meaning errors. They are not accepted as
Twi response teachers. Repeating their generated Twi at larger scale would not
solve that measured failure.

This bounded experiment starts with EXISTING Twi source answers, not new
machine-translated replies. It generates suitable English questions from the
existing English side, then independently checks question/answer compatibility.
The original Twi answer, source identity, revision, hash and source split remain
unchanged. It does not fabricate new Twi transcripts or medical answers.

The research motivation is [X-Instruction](https://arxiv.org/abs/2405.19744):
constructing instructions around native-language text can avoid relying entirely
on a teacher's weak target-language generation. Our English parallel-source
question generation is an adaptation of that idea, not a reproduction of its
iterative evaluator training. The paper does not establish Twi results or
guarantee success here.

## Data

Pinned source: Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT,
revision `5f59d16167c9432a6fa0dac5e7e6a7e48e161bdb`.
Parent: `tmp/general-language-corpus/adaptation-v1`, whose manifest and files
are checksum-verified. Training source groups only; no validation/test leakage.

From 5,516 Twi-target rows, this pilot's filters retain 4,495 candidates and
exclude 269 length/structure cases, 492 health-related cases, 258 repeated
English or Twi texts, and two agent-flagged meaning mismatches. Filtering is NOT
semantic certification of the retained pool. The first deterministic 128 source
answers are used to validate the new method before a larger synthesis run.
Two questions about one source remain one source, not two new conversations.

The two flagged source records, 66498 and 96852, remain unchanged in the original
corpus. Their review flags are in
`data/response-adaptation/source-review-findings.jsonl`. Upstream research and
attribution restrictions are retained; no commercial clearance or public upload
is implied by this work.

## Models And Boundaries

- Question generation: Qwen/Qwen3.8-27B, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Independent English compatibility review: google/gemma-4-31B-it, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Only English source text and questions are passed to these calls. No private
  recordings, real chats, database content, or Twi target rewriting.
- English-only reviewer controls test 14 judgments about time, quantity,
  uncertainty, relevance, missing context and false AI biography. These controls are never training
  rows. Their expected labels are not supplied to the reviewer.
- Grammar-constrained JSON is validated; incomplete or malformed outputs are
  retained as failures. Model agreement does not mark a source human reviewed.
- Each stage has one H100, a 1,200-second ceiling, zero automatic retries,
  no public serving, and a fresh container. CPU preflight verifies pinned cached
  tokenizers before allocating a GPU. No new hosted-model API calls.

## Run And Artifacts

Run: `native_answers_20260910T134538Z`.
Generation call: `fc-01M25S5S8C3REXTFK6AFJW9W26`.
App: `ghana-native-answer-synthesis`.
Local folder: `tmp/native-answer-synthesis-v1/`.
Private Modal volume: `ghana-health-understand-train`,
`/native-answers/<run_id>/<generate|review>/`.

Generation, independent review and export are COMPLETE. Generation took 244.76
seconds (138.14 loading), producing questions for 105 sources and rejecting 23.
Independent review `fc-01M25ST8NRP2J6AAZE3QCAC614` took 227.16 seconds
(159.05 loading), covering 105 sources and seven two-question controls. It passed
all 14 controls and approved at least one question for 104 sources, 194 distinct
questions in total. These are model votes, NOT accepted training examples.

Inspection caught false approvals despite the perfect control score. In row
67762, a vague statement about 'some reasons' does not answer WHY people are
not marrying. Rows 69776 and 65047 have source-translation mismatches that the
English reviewer cannot detect. These two sources and an additional polarity
case, 68178, were flagged, bringing the source flag inventory to five.

The generator's first version allowed generic first-person answers. That rule
was revised after inspecting its actual outputs: no real AI possessions/family
claims without fictional framing. The independent reviewer used the revised
rule and rejected the cattle-ownership answer. Every model result retains its
actual prompt, so earlier generation is not relabeled as using the new wording.

`scripts/gate_native_answers.py` adds conservative triage for source flags,
context-dependent questions, vague quantities and missing reasons. Its separate
`gated-review.json` keeps 85 sources with at least one remaining review candidate.
All 128 originals and all model judgments remain available. There are zero
gold/automatic training acceptances. The gate can flag valid examples too;
it prioritizes review, not semantic accuracy. Do NOT scale directly from 104
model-positive sources as if English agreement certified their Twi.

Use existing receipts, not duplicate submissions:

```sh
uv run scripts/build_native_answers.py status --stage generate
uv run scripts/build_native_answers.py status --stage review
```

`review.json` contains original Twi answer, original English meaning, two
candidate questions where possible, rejection explanations, independent review,
and source provenance. It is separate from the old research UI and production
corpus. Exports are immutable; both export commands refuse overwrite. No new
training, automatic gold acceptance, or model promotion occurred.

The method still needs native alignment checks, natural dialogue coverage,
multi-turn and tool trajectories, and measured transfer after training.
