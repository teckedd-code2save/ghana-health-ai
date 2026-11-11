# Twi understanding: what changed from base to R4 and R5?

11 November 2026. The adapter-disabled AfriqueQwen 9B base and the final R4
and R5 adapters were compared on the same tasks. OpenAI did not generate or
grade their answers. Only aggregates and methodology are published; source
records, model outputs and weights remain outside this release.

| Task | Base | R4 | R5 |
| --- | ---: | ---: | ---: |
| Twi sentence: published English meaning preferred in both answer orders | 636/1,092 | 890/1,092 | 900/1,092 |
| Both Twi sentences in a pair matched correctly in both orders | 184/546 | 367/546 | 376/546 |
| Both changed and unchanged values recalled through a Twi conversation | 11/24 | 23/24 | 24/24 |
| Same conversation measure in English | 18/24 | 18/24 | 18/24 |

**Recognition.** From the [Twi-English translation corpus by Zakaria et al.
(2026)](https://data.mendeley.com/datasets/x3f8w84s7h/1), 1,213 sentence pairs
were reserved by shared normalized source side. The final diagnostic kept
1,092 after screening against project training packs, protected texts and the
proposed training pool, and forming disjoint same-domain distractor pairs.
Each checkpoint scored the same A/B choice with the options in both positions;
ties and order-sensitive answers are not successes. No translation generation
or human judgment is implied. The source contains incorrect references, and
base-model pretraining exposure is unknown.

Compared with R4, R5 gained 13 sentences and lost three: +0.92 percentage
points (paired bootstrap interval, 0.18 to 1.65 points). Versus the base, R5
gained 271 and lost seven: +24.18 points (21.52 to 26.83). Intervals resample
546 pairs, not individual answer orderings. They describe this diagnostic.

**Conversation.** Four authored task families (journey, roles, quantities and
object location), six assignments each, in Twi and English. Each model generated
its own four-turn history with greedy decoding and a 48-new-token limit per
reply. The last two replies must each include the expected value without another
controlled value. Missing turns stay in the denominator. An invalid-Unicode
continuation for R4 was preserved. Recognition and conversation results are
separate from IFEval, perplexity, and earlier open-ended agent judgments.

R5's Twi recall gain does not establish that all of its replies were useful:
it remembered a corrected destination but volunteered a distance the user had
not supplied. All three models returned zero for the untouched quantity in
the six English egg-count follow-ups. The wording of that instruction matters.
Natural multi-turn usefulness, clinically appropriate responses, wrongful
refusals and broader dialect coverage still need separate evaluation.

## Before R6

The 16,085-row bilingual intake yielded 11,385 provisional training rows and
1,213 reserved rows. In a targeted review of 218 training-side pairs, a
reference-aware model judged 130 faithful, 84 different and four unclear.
These counts are not a corpus-wide error estimate. A separate blind reader
failed the qualification check on definite half versus unspecified part.
The current increment has 57 research-only alignment rows. A 4,973-ID worklist
awaits privacy/suitability clearance and qualified semantic validation.

Separately, 2,489 complete English source histories were screened. Of 36
source-bound Twi adaptations, a second agent review marked 27 conditional
candidates and nine for repair. These are not human-certified Twi conversations.
R6 had not been trained at this checkpoint. Preserve provenance, adjudicate
meaning disagreements, and freeze the training mixture and holdouts before
launching it.

[Aggregate counts, paired intervals and hashes](twi-scale-evaluation-20261111.json)
| [R4 model card](TWI_R4_MODEL_CARD.md) | [R5 model card](TWI_R5_MODEL_CARD.md)
| [Evaluation workers and scorer](../research/r5-scale/README.md)
