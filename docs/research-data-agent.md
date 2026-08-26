# Research data agent — evidence-first programme

**Status:** product direction; implementation begins with a small internal benchmark.  
**Internal workspace:** `/research/ase` (unlisted and role-protected; not yet implemented).  
**Public product:** Ghana Health AI remains unchanged while research experiments run.

## Purpose

Build a reproducible data and model-development loop for Ghanaian-language
understanding without pretending that every component must be invented here.
The programme must first discover, reproduce, evaluate, and reuse credible
public work. We train a new model only where measured evidence shows a gap.

The first target is deliberately narrow:

> Twi or Twi-English transcript → faithful English meaning, normalized Twi,
> ambiguity, and structured intent/entities.

ASR, understanding, response generation, and TTS remain separately replaceable
components. An external model name must not become the name of a model developed
by this project.

## Two delivery tracks

### Track A — useful now

1. Assemble a small, human-verified benchmark from corpus records we may legally
   evaluate.
2. Run existing Twi/Akan translation and understanding checkpoints unchanged.
3. Compare them with an open multilingual instruction model and the current
   hosted reference model.
4. Select the best deployable checkpoint for an internal experiment.
5. Expose it only through the protected research workspace until it passes the
   product promotion gate.

This track should take days, not become a multi-month prerequisite for learning.

### First decision round

The first round is intentionally smaller than the future research platform. It
tests three distinct roles instead of declaring one universal winner:

| Candidate | Role in the test | Initial constraint |
| --- | --- | --- |
| `ninte/twi-en-nllb-v2` | Twi → English meaning baseline | 615M parameters; CC BY-NC, so internal research only |
| `Qwen/Qwen3-8B` | Structured meaning, context, intent, and ambiguity baseline | Larger runtime; must prove enough Twi competence to justify serving cost |
| Current hosted response model | Product-quality reference after meaning recovery | Closed dependency and not the future owned checkpoint |

Kasawa, ABENA, Opani, and other discovered work stay in the registry, but they
are not forced into the same task: Kasawa is a small Twi language-model base,
ABENA is an encoder, and the published Opani checkpoint is English → Twi. They
may become useful bases, retrievers, teachers, or reverse-translation tools
after the first result.

The first internal deployment should therefore be a **meaning adapter**, not a
new public chat model. It receives a transcript, returns the benchmark contract,
and passes the recovered meaning plus original transcript to the existing
response model. The public response path changes only after side-by-side review.

### Track B — improve over time

1. Expand reviewed Twi-English, code-switch, ASR-noise, health, and commerce
   corpora.
2. Reproduce the strongest public training recipes.
3. Compare specialist bases with Qwen/Gemma-class multilingual bases.
4. Train an independent structured-understanding checkpoint.
5. Publish versioned datasets, evaluations, limitations, and model cards where
   licences and consent allow.

Track B must not block Track A.

## Reuse before invention

For every candidate, record:

- repository and immutable revision;
- model lineage and base checkpoint;
- supported task and direction;
- datasets and claimed metrics;
- licence and commercial/research restrictions;
- tokenizer and runtime requirements;
- reproduced results on our benchmark;
- decision: deploy, teacher, baseline, training base, hold, or reject.

Initial model families to evaluate include GhanaNLP ABENA, available Twi↔English
NLLB adaptations, Opani, Kasawa, a current open multilingual instruction model,
and the hosted production model as a reference. Model-card claims are hypotheses
until reproduced.

## Existing corpus and contamination policy

Existing assets have different roles:

| Asset | Immediate use | Constraint |
| --- | --- | --- |
| GhanaNLP Twi-English parallel | Translation benchmark and training candidate | Pin revision and audit licence/splits |
| WAXAL Akan transcripts | Twi text and ASR-noise experiments | Needs verified English meaning for understanding evaluation |
| WAXAL audio/test | ASR evaluation | Not an understanding gold set by itself |
| Product understanding fixtures | Smoke and regression tests | Too small for model-selection claims |
| Local recorded corpus | End-to-end probes | Existing holdout is small; never train on promotion rows |
| Consented future turns | Domain expansion | Consent, retention, deletion, and human review required |

Every record must carry source identity, source split, immutable revision, and
content hash. A benchmark row is ineligible if the candidate or its adapter was
trained on that row and the result is presented as held-out performance.

## Benchmark contract

The first benchmark is intentionally small. It should contain verified examples
covering standard Twi, informal spelling, code-switching, health, commerce,
negation, tense, ambiguity, and short contextual follow-ups.

Canonical prediction shape:

```json
{
  "understood": true,
  "normalized_twi": "Me ti yɛ me yaw fi nnora.",
  "literal_english": "My head has hurt me since yesterday.",
  "natural_english": "I have had a headache since yesterday.",
  "intent": "report_symptom",
  "entities": [
    { "type": "body_part", "value": "head" },
    { "type": "symptom", "value": "pain" },
    { "type": "onset", "value": "yesterday" }
  ],
  "ambiguities": []
}
```

Specialist models are scored only on fields they support. Translation, semantic
extraction, ambiguity, contextual resolution, latency, memory, and licence
fitness are reported separately; there is no misleading single score.

Promotion requires human review of meaning-critical errors, especially body
parts, negation, time, quantity, medicine names, and emergency language.

## Reference-first storage

Do not copy large public datasets or model weights into the application.

| Material | System of record |
| --- | --- |
| Public datasets | Original Hugging Face repository, pinned revision |
| Public and trained model weights | Hugging Face model repository, pinned revision |
| Benchmark jobs and annotations | PostgreSQL |
| Small predictions/reports | JSON or Parquet artifact referenced by job |
| Consented private audio | Private object storage with retention/deletion controls |
| Downloaded weights/data | Disposable Modal cache |

Remote compute receives only an approved, versioned benchmark export. Private
corpus rows, participant recordings, and production chat turns are not sent to
third-party compute merely because a repository reference exists. External job
submission must record the provider, payload class, model revision, and approval.

PostgreSQL stores references, hashes, annotations, reviews, and experiment
metadata—not model weights, audio blobs, or base64 media.

## Data agent responsibilities

The agent may:

- discover public medical, public-health, and commerce resources;
- record provenance, licence, jurisdiction, date, and permitted uses;
- extract source passages and propose faithful Twi translations;
- propose normalized Twi, English meaning, entities, and metadata;
- identify duplicates, conflicts, missing fields, and uncertain samples;
- prepare review queues and versioned training exports;
- retrieve or recommend existing models and datasets before new training.

The agent may not:

- mark its own translation as human-verified;
- turn copyrighted or restricted material into a training corpus without a
  compatible permission basis;
- silently treat synthetic Twi or synthetic audio as gold evaluation data;
- overwrite raw evidence;
- generate medical doctrine or claim clinical authority;
- expose participant data or internal research routes publicly.

For medicine, prefer authoritative public-health sources and retain passage-level
citations. For commerce, use public, licensed, or project-owned product language.
Synthetic translations are training candidates only after review and remain
labelled with model, prompt, source, and revision provenance.

## Internal workspace

The intended `/research/ase` workspace will be unlisted, excluded from indexing,
and authorized for `ADMIN`/`RESEARCHER` roles. It will eventually provide:

- corpus and provenance browser;
- model registry;
- benchmark launch and status;
- side-by-side predictions;
- human correction/review;
- model leaderboard by task;
- approved dataset export;
- experiment history and reproducibility metadata.

An unlisted URL is not a security boundary; authorization is mandatory.

## Immediate milestone

1. Replace the existing repetition-oriented understanding SFT assumptions with
   the structured benchmark contract above.
2. Create a small locked benchmark and model registry.
3. Run existing deployable checkpoints before spending training credits.
4. Select one candidate for an internal deployment experiment.
5. Continue corpus enrichment and model refinement only after reviewing the
   failure report.

The first milestone succeeds when we can answer, with evidence: **which existing
model best recovers the meaning of our Twi transcripts, what it gets wrong, and
whether it is safe and licensable to test internally.**

## Deliberate pace

Each phase must leave a useful artifact before the next begins:

1. **Discover:** registry with provenance, licence, and role.
2. **Reproduce:** predictions from unchanged public checkpoints.
3. **Review:** native-speaker corrections and an error report.
4. **Deploy privately:** one pinned candidate behind `/research/ase`.
5. **Collect carefully:** consented corrections and domain examples.
6. **Refine:** adapter or fine-tune only for measured gaps.
7. **Train independently:** only when the reviewed corpus and evaluation justify it.

No broad medical/commerce ingestion, synthetic-data factory, or new base-model
training begins before the first candidate has produced reviewed predictions.
