# Reference-Backed Annotation

The owner reviewed the HTTPS corpus, reported that the data generally looks good,
and approved the next collection steps. This is not blanket per-row approval or
authorization to start another response-model fine-tune.

## Source Boundary

Parent: `tmp/corpus-releases/twi-stage-v1-20260911`, unchanged and sealed.
Parent manifest SHA-256:
`ddc88cb47302905d98d74d7e08f66856dd397836479619691c13e40a12e83866`.

The reference-backed plan selects 11,557 existing screened sentence pairs:
11,330 train and 227 validation. It accounts for all 32,667 original annotation
work-plan records; 21,110 require other work rather than English-reference
analysis. Dictionary entries, missing English translations, health answers and
conversation derivatives are not silently included in this task.

Four saved HTTPS review actions cover three distinct source records. Their source
hashes were checked against the ledger and the journal was copied unchanged into
the research work directory. The VPS journal remains authoritative and writable;
the source release and old reviews were not changed. Earlier approval of an
English reference does not approve a newly generated entity/intent annotation.

## Qualification

Only existing English references are supplied to the annotator. Original Twi,
expected control labels and reference answers for scoring are not sent as model
instructions. Outputs classify record function, conversational intent, entities,
negation, time, quantity, uncertainty, experiencer and unresolved ambiguity.
All evidence quotes must be literal spans of the English reference. They are
explicitly NOT Twi character offsets. No translation, advice or final response
is requested, and no clinical certification follows from this annotation.

Qualification uses 32 source-validation records plus ten development controls.
Controls cannot enter training. Source-validation records retain their split.

The first 9B run failed before generation because a newer Hub cache check expected
an unneeded `.gitattributes` file. The loader now requests only the same cached
artifact patterns used during CPU preparation. This failure is preserved at
`tmp/corpus-reference/v1-20260915`.

The repaired Qwen3.5-9B run completed all 42 outputs:
`tmp/corpus-reference/v1-20260915-load2`, call
`fc-01M2H71GRR8NYXBWN8KX8ZZB59`. Forty-one passed mechanical checks, but manual
assistant inspection found semantic category errors (including a tyre as a body
part and dishonesty as negation). The full method was rejected. No bulk run or
training used these outputs. The age control retained the age as an entity but
omitted it from the required quantities list; do not describe that as lost age.

Qwen3-235B also completed all 42 outputs, ONLY for English analysis with explicit
field definitions. Work directory: `tmp/corpus-reference/v2-235-20260915`, call
`fc-01M2H7NZ1GEQ3ZMY1ZQRVQYEX3`. It took 712.56 seconds including 637.45 seconds
loading. Thirty-nine outputs passed mechanical checks. It omitted a negation,
inferred a direct purchasing intent from quoted speech in a narrative, and
misclassified indefinite quantity as uncertainty. This method also failed; no
bulk use or training. Its earlier failed Twi translation gate remains closed.

A final English-only comparison is in `tmp/corpus-reference/v3-oss-20260915`,
using cached open-weight `openai/gpt-oss-120b`, revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, on Modal H100. This does not use the
OpenAI API or select a product response model. Qualification call:
`fc-01M2H8WRCREYB9KPXHBS29YSJN`. Consult the receipt and completion section before
resuming; an existing call must never be submitted twice.

## Resumption And Outputs

Use `/Users/welcome/miniconda3/bin/python3` and existing Modal credentials.
The current private deployment is `ghana-corpus-reference-annotation-oss`. One call is
bounded at 900 seconds on one H100 GPU, no automatic retries. Completed requests
are checkpointed remotely and receipts are persisted locally before waiting.
No proprietary API fallback, public dataset/model upload, ASR or TTS work occurs.

```sh
python scripts/corpus_reference.py qualify --teacher oss120 --out tmp/corpus-reference/v3-oss-20260915 --wait 60
# Only after qualification and the source-reference semantic audit pass:
python scripts/corpus_reference.py run --teacher oss120 --out tmp/corpus-reference/v3-oss-20260915 --max-batches 1
python scripts/corpus_reference.py export --teacher oss120 --out tmp/corpus-reference/v3-oss-20260915
```

Batch layout is fixed at 256 records. `--max-batches` is the maximum batch index
to reach; completed batches are reused, not resubmitted. The first batch establishes
actual throughput. More than 10% mechanically flagged rows stops further scaling.
Semantic checks are still required because exact quotes do not prove correct labels.

Exports retain source IDs, hashes, revisions, terms and shared splits. Structured
research SFT views exclude dictionary/fragment/uncertain records and use English
analysis targets for original Twi inputs. They are not general dialogue training
examples. Wrong/incomplete outputs and pending work have separate files.

The review UI can read a separate, source-hash-bound annotation database; source
rows and the review journal remain separate. Reviews from the new UI identify the
annotation version as well as the source. HTTPS deployment of this extension is
not implied by a local build. See completion notes added after verification.

## Completed Comparisons

The OSS comparison completed 42 outputs in 521.76 seconds, including 206.84
seconds loading. Thirty-seven passed mechanical checks. Three had no parsed
final annotation, one changed an evidence quote's capitalization, and the buying
control disagreed with a problematic discourse-function rubric. The buying
quantity was preserved: a source statement can express a buying goal without
being a direct request. Do not misreport that as lost quantity or inability to
understand shopping. Existing v1 scoring is preserved, not retroactively passed.

Assistant inspection also found unsupported entity categories and unnecessary
ambiguity annotations. The full method remains unqualified. No bulk annotation
or training ran. The new contract in `corpus_reference_contract_v2.py` separates
`expressed_goal` from `conversational_intent`; it needs fresh qualification and
is not used to change old results. A next test must use fresh source-validation
cases and distinguish detected/quarantined format errors from undetected meaning
errors; do not keep tuning against only these same 42 cases.

A four-case diagnostic, call `fc-01M2H9MTXWG2AZEGK6N7F9CAPQ`, returned all four
final answers with tracing enabled. It did not reproduce the missing finals and
does not replace the original qualification. It uses a different batch shape;
sampling and numerical differences remain possible. The original failed raw
Harmony outputs were not retained, so their cause cannot be recovered or claimed
as proven. Future calls retain raw Harmony output, parsed-field names and stop
metadata privately. None of that reasoning is exposed in the review UI or SFT.

The pinned tokenizer maps message end to token 200007, separate from end-of-text
199999. The initial suspicion that message boundaries were automatically treated
as EOS was not supported. No unproven stop-token override was applied.
Primary references: [pinned model generation configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/generation_config.json),
[Harmony](https://github.com/openai/harmony), and
[vLLM 0.28 sampling implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/sampling_params.py).

Correction projection: `tmp/corpus-reference/review-v2-20260915`.
It contains the two latest candidates on 32 REAL source-validation records, not
the ten synthetic controls. The builder checks each source against the parent
ledger. Missing final outputs are not selectable options. Existing source hashes
and model-version identities are retained, so reviews can be matched to the
exact candidates. These 32 checks are not a new training corpus or 32 approved
rows. The 11,557-row missing-field plan is still pending bulk annotation.

Meanwhile, the full source-backed bilingual handoff has been compiled and
privately stored without any teacher-generated targets. See
`alignment-handoff-20260915.md` for its actual counts and next-stage limitations.
