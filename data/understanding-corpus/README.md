# Understanding corpus candidates

`candidates.v0.jsonl` is a review queue, not a training corpus.

Each row is derived from project-owned prompt packs or existing local recordings
and keeps source identity, text, review status, draft model proposals, consent
scope, and training eligibility flags.

Important boundaries:

- `model_proposal` fields are drafts for human correction.
- `eligible_for_training=false` until reviewed and exported through a split-safe
  process.
- `eligible_for_final_evaluation=false`; these rows must not become promotion
  evidence without frozen split policy.
- `local-research://...` values are portable artifact identifiers, not embedded
  audio.

Next step: review rows in `/research/ase`, correct the Twi/English meaning and
semantic fields, then export only reviewed rows into a versioned training or
evaluation manifest.
