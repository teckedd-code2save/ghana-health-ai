# Ghana Language Understanding Research Programme

**Status:** Approved research direction; implementation planning follows the
Research Preview cleanup.

**Initial language:** Asante Twi/Akan

**Initial product laboratory:** Ghana Health AI

## 1. Research thesis

Ghana Health AI currently has separate speech-recognition and response models,
but no dedicated layer whose sole responsibility is to recover what a speaker
meant. Asking one general model to interpret a noisy Twi transcript, assess
health risk, and write an answer in one operation hides misunderstanding behind
fluent language.

This programme will build an uncertainty-aware Ghanaian-language understanding
layer that converts speech evidence into a faithful, machine-actionable meaning
representation before any health, commerce, or general response is produced.

The central research question is:

> Can a context-aware semantic recovery model convert imperfect,
> code-switched Ghanaian-language ASR transcripts into faithful English meaning
> and structured intent while knowing when it has not understood enough to act?

The intended architecture is:

```text
speech
  -> language-specific ASR
  -> Ghana language understanding
       - corrected/normalised source transcript
       - faithful English meaning
       - semantic frame
       - uncertainty and evidence
  -> health, commerce, or general reasoning
  -> answer in the user's preferred language
  -> language-specific TTS
```

The English meaning is an interlingua for downstream models; it is not a claim
that English is intrinsically better. The original language, corrected source,
code-switching, uncertainty, and conversational references remain first-class
evidence so translation cannot silently erase meaning.

## 2. Two-front data strategy

### Front A: enrich existing research data

Recover more value from every licensed dataset and local corpus already used in
ASR work, including Waxal, GhanaNLP, Common Voice, and consented local research
recordings.

The existing evidence is typically:

```text
audio -> source-language transcript
```

The research data agent will propose additional layers:

```text
audio
  -> original transcript
  -> normalised/corrected source transcript
  -> faithful English meaning
  -> semantic frame
  -> uncertainty, linguistic, acoustic, licence, and provenance metadata
  -> human review decision
```

Model-generated translations and annotations are drafts, never gold labels.
Only reviewed records may enter verified understanding training or evaluation
sets.

### Front B: consented Ghana Health AI interactions

Ghana Health AI will act as a real-world research preview. Participants who
explicitly opt in may contribute a voice turn and/or corrections.

Each contributed turn can preserve:

```text
consented audio
  -> ASR model output and confidence
  -> system interpretation
  -> corrected source transcript
  -> corrected English meaning
  -> corrected semantic metadata
  -> review and provenance history
```

Two corrections must remain distinct:

1. **Transcription correction:** what words were spoken?
2. **Meaning correction:** what did the speaker intend to communicate?

A transcript can be correct while its interpretation is wrong. Collapsing the
two would corrupt both ASR and understanding research.

## 3. Canonical research record

The durable record should preserve raw evidence, model proposals, corrections,
and review state rather than overwriting earlier values.

```json
{
  "record_id": "uuid",
  "source": "waxal|ghananlp|common_voice|local_research|product_contribution|synthetic_tts",
  "source_record_id": "immutable upstream identifier",
  "language": "tw",
  "dialect": "asante|akuapem|fante|unknown",
  "domain": "health|commerce|general|other",
  "audio": {
    "artifact_id": "nullable",
    "human_or_synthetic": "human",
    "speaker_id": "pseudonymous",
    "device": "phone",
    "environment": "quiet|noise|unknown",
    "duration_seconds": 4.2,
    "consent_scope": "dataset_license|research_audio|text_only",
    "retention_state": "retained|deleted|withdrawn"
  },
  "asr_observations": [
    {
      "model": "teckedd/gha-dondo-w2v-bert-twi-v2",
      "transcript": "raw model output",
      "confidence": 0.81,
      "created_at": "ISO-8601"
    }
  ],
  "source_transcript": {
    "original": "upstream or participant text",
    "model_normalisation": "draft",
    "human_correction": null,
    "review_status": "needs_review"
  },
  "meaning": {
    "model_english": "draft faithful meaning",
    "human_english": null,
    "understood_meaning": null,
    "review_status": "needs_review"
  },
  "semantics": {
    "intent": "report_symptom",
    "entities": {},
    "negations": [],
    "time": [],
    "context_references": [],
    "code_switched_terms": [],
    "uncertain_segments": [],
    "requires_clarification": false
  },
  "provenance": {
    "dataset_version": "immutable version",
    "licence": "SPDX or source licence",
    "annotation_models": [],
    "reviewers": [],
    "eligible_for_training": false,
    "eligible_for_final_evaluation": false
  }
}
```

## 4. Research data agent responsibilities

The agent is an evidence-management system, not an unrestricted synthetic-data
generator. It will operate through typed, auditable stages.

### Import and catalogue

- Preserve upstream IDs, versions, licences, splits, and checksums.
- Never silently merge incompatible licences or consent scopes.
- Keep original transcripts immutable.
- Detect duplicate audio, transcript, speaker, and meaning records.

### Propose annotations

- Normalise the Ghanaian-language transcript.
- Produce a faithful English meaning rather than a literal gloss.
- Extract intent, entities, negation, time, quantity, location, and discourse
  references.
- Preserve English words used naturally inside Ghanaian-language speech.
- Mark every uncertain segment and alternate interpretation.
- Attach model, prompt, version, temperature, and generation timestamp.

### Route human review

- Prioritise high-disagreement, safety-relevant, code-switched, and
  low-confidence records.
- Present audio, ASR output, corrected transcript, English meaning, and semantic
  frame together.
- Support correct, transcript-wrong, meaning-wrong, both-wrong, unclear-audio,
  second-review, exclude, and sensitive-data decisions.
- Require dual review or adjudication for frozen safety-critical evaluation.

### Audit corpus readiness

- Report language, dialect, speaker, gender/age where voluntarily supplied,
  device, noise, domain, intent, and code-switch coverage.
- Measure repeated meanings and synthetic dominance.
- Prevent speaker or meaning leakage across train and evaluation splits.
- Recommend the highest-value next collection batch before compute spend.

### Generate controlled synthetic evidence

- Generate source-language text and English meaning proposals for human review.
- Send only reviewed source text to the configured Ghanaian-language TTS.
- Produce audio/ASR-error pairs for augmentation and stress testing.
- Label the TTS model, voice, text source, generation date, and checksums.
- Never count synthetic speech as final human evaluation evidence.

## 5. Consent and participant rights

The application must not silently record every user. Research contribution must
be separate from ordinary product use and based on explicit, informed consent.

The Research Preview consent experience must explain:

- what audio, transcript, correction, meaning, and metadata may be collected;
- the research purpose and foreseeable model-training use;
- whether the participant is sharing text only or text plus audio;
- retention, security, de-identification, and access controls;
- that Ghana Health AI is not a medical device or substitute for clinical care;
- how to decline without losing ordinary access;
- how to withdraw future participation and request removal where feasible;
- that synthetic or model-generated annotations will be human-reviewed before
  becoming verified research data.

Consent must be versioned, timestamped, revocable, and scoped. Audio collection
defaults to off. Existing correction-only consent must not be interpreted as
broad research-programme consent.

Before formal participant recruitment or publication, the project should seek
the appropriate institutional ethics review and data-protection guidance.

## 6. Dataset separation

Maintain physically and logically distinct evidence classes:

| Class | Can train? | Can decide promotion? | Human audio? |
| --- | --- | --- | --- |
| Unreviewed model drafts | No | No | Mixed |
| Reviewed synthetic augmentation | Yes, controlled | No | No |
| Consented human training | Yes | No | Yes |
| Frozen consented human validation | No | During development | Yes |
| Frozen consented human test | No | Yes | Yes |

No candidate model, prompt optimiser, translation teacher, or reviewer may see
the frozen test labels during development.

## 7. Model programme

The verified corpus can support separate, composable models:

1. **ASR:** audio to source-language transcript.
2. **Transcript recovery:** noisy ASR output to corrected source language.
3. **Semantic recovery:** transcript plus dialogue context to faithful English
   meaning, semantic frame, uncertainty, and clarification need.
4. **Domain reasoning:** verified meaning to safe health guidance or commerce
   action planning.
5. **Response generation:** response plan to natural target-language text.
6. **TTS:** target-language text to speech.
7. **Uncertainty/routing:** decide whether to answer, clarify, use another model,
   or request human review.

The first instruction-tuning task is semantic recovery. Its output must never be
a user-facing answer.

```text
Instruction:
Recover the speaker's intended meaning from the Ghanaian-language ASR evidence
and recent dialogue. Return corrected source text, faithful English meaning,
structured semantics, uncertainty, and clarification need. Do not answer the
speaker and do not add unsupported facts.
```

## 8. Evaluation

The understanding layer is evaluated independently from response fluency.

| Capability | Primary measure |
| --- | --- |
| Corrected transcript | WER/CER against human source transcript |
| Meaning preservation | Native-speaker adequacy and critical omission rate |
| Intent/entities | Macro F1 and per-domain F1 |
| Negation | Exact correctness |
| Time, quantity, location | Normalised exact/F1 |
| Context resolution | Correct antecedent/referent accuracy |
| Code-switching | Preserved-term accuracy |
| Uncertainty | Calibration error and selective accuracy |
| Clarification | Precision/recall for answer-versus-clarify decision |
| Safety semantics | Unsupported addition and critical omission rate |

The decisive question for a health record is:

> Did the interpretation add, remove, negate, or alter anything that could
> change the downstream health action?

Direct ASR-to-answer and translate-then-answer remain baselines. The proposed
semantic recovery layer must demonstrate improvement on the same frozen human
test set.

## 9. Academic and grant outputs

Potential durable outputs include:

- a consented Ghanaian spoken-language understanding corpus;
- a Twi ASR-to-meaning benchmark with realistic model errors;
- an uncertainty-aware semantic recovery model;
- a native-speaker correction and adjudication interface;
- a provenance-preserving intelligent data agent;
- evidence on synthetic TTS augmentation versus real human speech;
- a deployed Research Preview demonstrating the full feedback loop;
- later Ga, Ewe, Fante, and other Ghanaian-language extensions.

Candidate research questions:

1. Does explicit semantic recovery outperform direct ASR-to-answer pipelines?
2. How much reviewed synthetic data transfers to real Ghanaian speech?
3. Can model disagreement reduce native-speaker annotation cost?
4. Does conversation context improve short follow-up interpretation?
5. How should uncertainty be calibrated for safety-sensitive low-resource SLU?
6. Can one multilingual model expand beyond Twi without degrading the initial
   language?

## 10. Execution gates

The project proceeds in gated stages:

### Gate 0 — Research Preview honesty

- Mark the product as a Research Preview.
- Preserve live-model versus fallback provenance.
- Remove canned health advice that can masquerade as understanding.
- Add usable session creation, history, selection, and deletion.

### Gate 1 — Data model and import audit

- Freeze the canonical schema and provenance rules.
- Inventory licences, versions, splits, and available fields for every existing
  dataset.
- Produce an import-only audit before generating annotations.

### Gate 2 — Review prototype

- Build the aligned audio/transcript/meaning/semantics review workflow.
- Pilot with a small native-reviewed batch.
- Measure reviewer agreement and refine guidelines.

### Gate 3 — Intelligent data agent

- Implement typed import, proposal, disagreement, routing, audit, and export
  stages.
- Keep every mutation and model proposal reproducible and reversible.

### Gate 4 — First understanding benchmark

- Create speaker- and meaning-safe train/validation/test splits.
- Benchmark general LLM, translation pipeline, and fine-tuned semantic recovery
  approaches.

### Gate 5 — Research Preview contribution loop

- Complete ethics/privacy review appropriate to deployment and publication.
- Ship versioned research consent and withdrawal controls.
- Collect and review participant contributions without coupling contribution to
  ordinary product access.

### Gate 6 — Train and publish responsibly

- Train only on eligible reviewed records.
- Publish model/data cards with limitations, licences, consent scope, evaluation
  design, and safety boundaries.
- Promote no model on synthetic or leaked evidence.

## 11. Immediate boundary

The current implementation pass completes Gate 0 only. It does not begin broad
audio retention, automated annotation, or participant recruitment. Those start
after the data model, consent design, review workflow, and research governance
are planned cleanly.
