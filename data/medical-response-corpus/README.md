# Medical Twi response corpus

This folder contains source-linked medical voice-response rows for review.

`seed.v0.jsonl` contains English source utterances, faithful English meanings,
intents, entities, safety level, safe answer text, and source URLs.

`twi-drafts.v0.jsonl` adds model-produced Twi user utterance and Twi answer
drafts generated in chunks. These rows are ready for review, not final gold
labels.

Current source anchors:

- CDC urgent maternal warning signs.
- WHO maternal and newborn counselling danger signs via NCBI Bookshelf.
- CDC flu emergency warning signs.
- WHO malaria fact sheet.
- NHS red-eye guidance.

Validation:

```bash
pnpm eval:medical:twi
```

Regeneration:

```bash
pnpm corpus:medical:twi -- --chunk-size 4
pnpm corpus:understanding:candidates
cp tmp/understanding-corpus/candidates.v0.jsonl data/understanding-corpus/candidates.v0.jsonl
```

Do not train on these rows until a reviewer has checked the Twi, English
meaning, intent, and answer safety.
