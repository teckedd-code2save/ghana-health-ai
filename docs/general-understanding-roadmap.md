# General Understanding Research

Updated 2026-09-09 following the user's scope clarification.

## Product target

One independently serving Twi-English LLM that understands and answers general
questions, follows instructions, preserves conversational context and corrections,
handles code-switching, and asks for missing information. Health education and
ecommerce are specializations. Web search and commerce APIs are tools the model
can request; they are not knowledge the model should pretend it already has.
The final response comes from our model, not a hidden proprietary response model.

Speech recognition and TTS remain separate components. Better transcription is
necessary but cannot repair a text model that misunderstands an accurate transcript.
The private text playground deliberately isolates the latter problem.

## What the pilot actually established

305 annotated records were available, but only 120 had model consensus; 185
unresolved records were excluded. One of those 120 was similar to locked evaluation
data. The remaining 119 were split into 95 training sources and 24 held-out sources.
Three tasks per source made 285 training examples, not 285 independent records.
These are uncalibrated model annotations, not native-speaker gold.

The split and filtering were defensible for an exploratory pilot. Treating that
pilot as the user's large, general-purpose training program would not be.
The selected data was narrow health-education QA with heavily imbalanced intents.
The adapter learned much of the output format but failed to demonstrate better
semantics. Its 13/24 intent agreement was below a 14/24 majority-class baseline.
English reference similarity improved; Twi did not. These are small, proxy-based
measurements, not general understanding or clinical accuracy.

See [the measured pilot report](./medical-response-pilot.md). No more epochs on
the same 95-source mix are justified by these results.

## Better foundation choices

Recommendation: compare a capable instruction model with a genuinely Twi-adapted
foundation, then train the winner on a balanced task mix. Do not repeat expensive
Twi pretraining from scratch before checking existing foundations.

| Candidate | Role | Evidence and remaining uncertainty |
| --- | --- | --- |
| Qwen/Qwen3.5-9B | Instruction and tool-use baseline; potential delivery foundation | Official documentation supports tool calling and direct, non-thinking replies. Twi performance on our data is not established. Our previous 35B annotation calibration failure is a warning, not a reason to assume this smaller model is a good Twi teacher. |
| McGill-NLP/AfriqueQwen3.5-4B-50Langs | Twi language-foundation challenger | Its card explicitly includes Akan/Twi, reports African-language continued pretraining and English replay. It is a pretrained base, not an instruction assistant. Needs broad SFT and tool training before chat comparison is fair. |
| MiniCPM5-1B-Twi and our pilot | Cheap reference, not preferred next foundation | Already measured locally. The upstream author also documents weak reasoning, factual unreliability, alignment erosion and dependence on anti-repetition decoding. |

Sources checked 2026-09-09:
[Qwen card](https://huggingface.co/Qwen/Qwen3.5-9B),
[Afrique 50-language card](https://huggingface.co/McGill-NLP/AfriqueQwen3.5-4B-50Langs),
[MiniCPM Twi card](https://huggingface.co/ghananlpcommunity/MiniCPM5-1B-Twi).

Afrique's published additional-language improvements pool Twi with other
languages; they are not Twi-specific accuracy or our product results. Its card
declares CC-BY-4.0; preserve upstream terms and provenance before redistribution.
Candidate selection above is a research recommendation, not a measured winner.

Tiny Aya Earth was investigated but is not the default alternative: Akan/Twi
is absent from its listed focus languages, and its weights are noncommercial.
It can remain a research comparator, not an assumed production shortcut.
[Official card](https://huggingface.co/CohereLabs/tiny-aya-earth).

## Corpus and curriculum

The existing 12,858 source-training records remain useful, but are not 12,858
accepted instruction examples. They contain 4,407 Twi and 4,402 English AfriHealth
QA sources plus 2,249 GhanaNLP and 1,800 WAXAL speech-text sources. Do not fabricate
medical answers for general speech transcripts or silently relabel them as QA.

Build a versioned mixture with distinct tasks and source-group splits:

1. General language and conversation: everyday instructions, explanations,
   passage-grounded answers, summarization, corrections and multi-turn reference
   resolution. Include Twi, English and code-switched dialogue.
2. Meaning preservation: existing WAXAL/GhanaNLP text, clean parallel text and
   reviewed errors. Supervise negation, time, speaker, quantities, uncertainty and
   exact entity spans. Translation is an auxiliary task, not the whole model.
3. Tool trajectories: request, tool schema, model call, executed result, and final
   language-matched answer. Include no-tool cases, failed search, empty inventory,
   changed quantities, interrupted checkout and explicit purchase confirmation.
4. Health education: source-backed explanations and questions, health literacy,
   appropriate uncertainty, escalation and carefully distinguished patient context.
   Academic policy questions must not dominate patient education.
5. Commerce: product, quantity, budget, delivery area, comparison and order state.
   Asking to search for tomatoes is not authorization to charge or place an order.

An initial ablation can sample supervised tokens at 50% general/meaning,
25% tools/commerce and 25% health education. This is a starting experiment,
not an empirically optimal ratio. Track Twi and English exposure independently
within each task; preserve substantial English instruction replay. Report distinct
sources, examples, tokens and duplication separately. Do not duplicate ten prompts
to claim a larger corpus.

For general parallel data, the roughly 999k-row pristine-twi-english resource is
an available research source, but its card says English is machine-translated and
the license is CC-BY-NC. Inspect diversity, alignment and source overlap first;
neither its size nor its name establishes correctness or commercial suitability.
[Dataset card](https://huggingface.co/datasets/ghananlpcommunity/pristine-twi-english).

For executable tool supervision, APIGen/xLAM provides 60k English function-call
examples with format, execution and semantic checks, under a CC-BY-4.0 dataset
card. Access acknowledgement may be required. These are synthetic tool examples,
not human Twi conversations. Reuse the schemas and executable validation approach;
translate/localize user language without translating API names or changing numbers.
Add actual multi-step result-to-answer trajectories, not just JSON calls.
[Dataset card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).

## Training sequence

1. Inventory task coverage and isolate fixed evaluation groups before selecting
   training rows. Retain existing locked medical sets; add held-out general,
   commerce and tool episodes. Do not treat reviewed playground examples as both
   future training inputs and a permanent test set.
2. Test instruction-model semantics directly; probe pretrained language candidates
   with suitable language/reading tasks rather than treating raw base completions
   as chat-quality evidence. Compare full assistant behavior after equivalent SFT.
3. Run bounded LoRA/QLoRA SFT on the balanced mixture, evaluating by task and
   language during training. Stop if Twi or English regress. More steps or lower
   training loss alone are not promotion criteria.
4. Use saved reviewer preferences for targeted correction training. Use DPO only
   after there are reliable preference pairs; do not let the model validate itself.
5. Publish an honest model card and evaluation artifacts for a successful research
   release, then test through an explicit private model choice. Product promotion
   is a separate decision, not an automatic consequence of a completed GPU job.

The current training obstacle is task coverage and trustworthy supervision,
not access to another 95-row training run. Use Modal for open-model inference,
candidate evaluation and training; do not restart OpenAI requests while funding
is unavailable. Resume saved annotation work rather than regenerate paid results.

## Evaluation and runtime

Report general instruction following, multi-turn consistency, language matching,
meaning preservation, entity accuracy, response usefulness and latency separately.
Tool evaluation must measure argument correctness, success after execution,
result-grounded answers, no-tool selection and authorization compliance. Assess
health education with source correctness and native review, not only keyword hits.
General-purpose capability does not remove the need for health safety evaluation.

At runtime: conversation state -> our model -> optional validated tool request ->
tool result -> our model's answer. Web search results are untrusted evidence, not
instructions. Allowlist tools, validate arguments, limit retries/time/cost, redact
sensitive search terms where needed, and require confirmation for consequential
commerce actions. No arbitrary code execution or unbounded browsing from model text.
Only display citations from retrieved results actually supplied to the model.

## Private pilot testing

The isolated playground is `scripts/research_playground.py`, backed by the private
Modal app `ghana-understanding-pilot-private`. It offers the actual pilot and
untouched base, both using the documented sampling settings, without reference
answers, prompt examples, a medical system template or proprietary fallback.
There is no web-search or commerce execution in this pilot tester; it evaluates
text generation only. Do not interpret claims of searching or ordering as real actions.

Start with `python3 scripts/research_playground.py --port 7863`, then open
`http://127.0.0.1:7863/?__theme=light`. It requires the existing authenticated
Modal profile. Gradio 6.17.3 and Modal 1.4.3 were used locally.
Deploy the private class with `modal deploy modal/research_pilot_service.py`.
The adapter mount is read-only and its checksum is verified on cold load.
No corpus archive, public endpoint, HF upload or production route is added.

The active conversation survives refresh in this browser. Ratings and optional
corrections are in `tmp/research-playground/reviews.sqlite3`, with actual model
revision and adapter checksum beside each completed turn. No automatic training
eligibility is assigned to these reviews. Application logs/corpus are not uploaded;
test inputs are sent to Modal and remain subject to Modal's platform retention.

Costs are bounded per request: one L4 container, zero minimum containers, 60-second
idle scale-down, 180-second function timeout, 90-second decoding limit and 384
output tokens. No background annotation job runs. Stop cancels a known Modal call;
if cancellation occurs before the call ID reaches the browser bridge, the bounded
remote request can finish in the background. Requests are not free, and no current
Modal credit balance is asserted here.

## Verification checkpoint

On 2026-09-09, both model choices returned actual Modal inference. Conversation
refresh, server-restart persistence, rating/correction saving, stop recovery,
desktop framing and 390x844 mobile framing were checked. Eight Python tests
passed across the playground and existing pilot core. The final short inference
streamed 44 text chunks and stopped normally at 61 tokens, but answered in Twi
despite an explicit English-language instruction. Serving works; language
instruction following still fails. No broad semantic improvement is claimed.
