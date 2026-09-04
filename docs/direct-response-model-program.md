# Direct Twi Response Model Program

## Decision

The next research checkpoint is a direct-response LLM, not another hidden hint
to a hosted model. For each turn it must produce both a machine-checkable
semantic record and its own short Twi reply.

```text
Twi speech/text + short history
  -> normalized Twi, faithful English meaning, intent, entities, uncertainty
  -> safety level, clarification decision, Twi reply
```

The response model must not depend on `gpt-5.6-sol` at inference time. A
deterministic safety gate remains around the model for known red flags and for
cases where the model declares uncertainty.

## Corpus contract

Only rows with all of the following can enter response SFT:

- reviewed normalized Twi;
- reviewed faithful English meaning and intent;
- reviewed Twi response;
- reviewed safety level: `routine`, `same_day`, `urgent`, or `emergency`;
- source, consent scope, and stable split.

`pnpm corpus:response:export` writes the candidate JSONL files to
`tmp/response-capable-corpus/v1/`. `pnpm corpus:response:export:strict` blocks
unless there are at least 100 reviewed response rows distributed across train,
dev, and test.

The existing large semantic corpus is still valuable: it teaches meaning
recovery. It is not silently converted into response examples, because that
would teach generic advice without reviewed target replies.

## Review loop

The chat review control opens a one-at-a-time sample card. It starts with
medical rows that already contain source-linked Twi response drafts. A reviewer
can correct the meaning, intent, Twi reply, and safety level, then mark the row
reviewed. A consented recording is attached to the exact row with its speaker
ID, allowing several speakers to record the same phrase.

Recordings are private review data stored in Postgres for this first research
lane. They are not public assets and are not included in a model or dataset
push without a separate export, consent, and licence review.

## Promotion gate

A direct-response checkpoint can be exposed as a research choice only after:

1. It emits parseable structured output and a Twi reply on every held-out case.
2. It preserves held-out meanings, including negation, person, symptom, time,
   and code-switching.
3. It makes no critical emergency-routing errors on the locked safety set.
4. Native review finds the Twi reply useful and faithful rather than templated.
5. Its Hugging Face card includes source licences, corpus counts, the split
   policy, evaluation results, limitations, and its non-production status.

Until then, `Research v1` remains an interpreter experiment. It should not be
described as a better assistant simply because the product can obtain a good
final reply elsewhere.
