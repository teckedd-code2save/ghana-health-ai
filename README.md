# Ghana Health AI

Research and application code for Twi/Akan speech, language understanding, and conversational generation. Ghana-facing health and commerce workflows provide an application testbed, not evidence of clinical readiness.

**Research question:** Which data and adaptation strategies improve faithful Twi understanding and useful multi-turn responses while retaining general and English capabilities?

**Public repository:** [teckedd-code2save/ghana-health-ai](https://github.com/teckedd-code2save/ghana-health-ai). Public code does **not** imply public release of private datasets, conversation logs, evaluation bundles, or trained adapters.

## Start Here

- [Research overview: results, provenance, and reproduction limits](docs/research/overview.md)
- [R4 continuation: consumed data, comparisons and current limitations](docs/TWI_UNDERSTANDING_R4_20261107.md)
- [R4 companion model card](docs/TWI_R4_MODEL_CARD.md)
- [Language understanding research programme](docs/GHANA_LANGUAGE_UNDERSTANDING_RESEARCH.md)
- [ASR model decision record](docs/asr-model-decision.md)
- [R3 application access](docs/R3_GUEST_CHAT.md)
- [Contributing](CONTRIBUTING.md) and [repository operating guide](AGENTS.md)

Dated documents retain historical plans and decisions; they are not live deployment-status reports.

## Research Tasks

| Task | Measurements | Boundary |
| --- | --- | --- |
| Speech recognition | WER/CER and cross-domain transcription evaluation | In-domain scores do not establish general reliability |
| Meaning preservation | Bilingual and structured meaning checks, contextual corrections | Fluency can hide changed meaning |
| Twi conversation | Fixed histories, scripted rollouts, instruction-following review | Completion and formatting are not semantic correctness |
| Language retention | Token-weighted negative log-likelihood and perplexity | Lower loss is not clinical or conversational quality |
| Speech synthesis | Language routing, text preservation, listening review | Routing checks are not naturalness certification |
| Data accountability | Versioned admission, source lineage, consumed rows/tokens | Inventory is not training consumption or human approval |

## Recorded Development Results

**6 November 2026:** completed comparison of a pinned multilingual foundation, R2 meaning adaptation, and R3 response adaptation. These aggregates come from locally held verified receipts. The underlying private artifacts are **not published by this README**.

| Measure | Base | R2 meaning | R3 response |
| --- | ---: | ---: | ---: |
| Language perplexity | 15.450 | 5.229 | 5.162 |
| Strict literal JSON match with EOS | 6/8 | 8/8 | 8/8 |
| EOS-completed replies / 75 chat slots | 70 | 71 | 67 |
| Repeated four-word sequence present | 9/75 generated | 9/75 generated | 13/73 generated |
| Error / blocked slots | 0 / 0 | 0 / 0 | 1 / 1 |

Scope: 46 fixed histories plus 10 scripted rollouts comprising 29 turns per checkpoint; 343 language-loss rows and 9,315 predicted tokens per checkpoint. All 1,254 slots were finalized, including the error and blocked slot. This is **development evaluation, not a blind final benchmark**.

R3's modest language-loss improvement does not establish reliable overall conversational improvement over R2. Repeated four-word sequences are not all pathological loops. Native-human quality review and clinical certification remain pending. See the [overview](docs/research/overview.md) for protocol and data boundaries.

### R4 Continuation

**9 November 2026:** R4 continued the final R3 adapter on 1,544 versioned dataset
records, expanded into 1,642 distinct training views and 1,997 presentations.
The run consumed 161,773 effective target tokens. New and spelling-augmented
material accounts for 7.4% of response-target tokens; this is a focused
continuation, not a full-inventory run. Both checkpoints and consumed data are
documented in the [R4 model card](docs/TWI_R4_MODEL_CARD.md).

| Fresh matched development run | Base | R3 | R4 final |
| --- | ---: | ---: | ---: |
| Raw EOS replies / 75 planned | 71 | 67 | 73 |
| Raw replies with repeated four-word sequences | 5/75 generated | 12/73 generated | 8/75 generated |
| Guard pauses / 75 guarded requests | 1 | 5 | 1 |
| Held-out Twi perplexity | 15.4532 | 5.1590 | 5.1968 |

The same 343 held-out text rows and 75 development chat slots were evaluated
again under a pinned runtime; these fresh outputs are separate from the table
above. Raw R3 has one error and one blocked follow-up. Lower repetition and some
better follow-ups coexist with slightly worse perplexity, arithmetic failures
and a name-retention regression. See [exact denominators, observations and local
comparison instructions](docs/TWI_UNDERSTANDING_R4_20261107.md). The public
application remains R3; R4 is available in the private local comparison only.

## Application and Code

Try the [Ghana Health AI application](https://ghanahealth.serendepify.com/chat). R3 is a research model option; live output can contain meaning errors and repetition.

The application uses Next.js 16, React, TypeScript, Prisma 7, and Postgres. It includes real speech-recognition and synthesis clients, conversation orchestration, and separate research tooling. It is no longer accurately described as a voice-stub-only MVP.

| Area | Entry point |
| --- | --- |
| Application | [src/app](src/app) |
| Conversation orchestration | [conversation-turn.ts](src/lib/conversation-turn.ts) |
| Speech clients | [ASR](src/lib/modal-asr.ts), [TTS](src/lib/modal-tts.ts) |
| Research and evaluation | [scripts](scripts) |
| Model services and training | [modal](modal) |
| Command inventory | [package.json](package.json) |

Endpoint health, deployment availability, and scientific validation are separate. Do not infer a production-model promotion from research results.

## Local Development

Use Node.js 20 and the repository-pinned pnpm version. Configure an authorized **development** database and required service settings through approved secrets tooling. Do not copy production credentials or commit environment files.

```bash
corepack enable
pnpm install --frozen-lockfile
sec -- pnpm db:generate
sec -- pnpm dev
```

The application runs at `http://localhost:3000`. `sec` is the project's configured secret-injection helper, not a dependency installed by this repository. Database migration/seeding and remote model services require separate setup; see [AGENTS.md](AGENTS.md). A fresh clone alone does not reproduce private model experiments.

Training, live evaluation, and deployment commands can invoke paid remote services. They are not part of this quick start; review their inputs, access requirements, and budget first.

## Safety and Availability

This is research software, not a clinically validated medical system. Generated health responses are not a substitute for qualified care. Private conversations, review journals, credentials, and restricted source material must not be published with code or results.

Repository visibility is public. The checkout currently has no standalone license file; upstream model and dataset terms remain separate. No new dataset or checkpoint publication is announced here.
