# Source-Backed Alignment Handoff

The owner reported that the corpus generally looks good and approved the next
collection steps. This is not blanket row-level certification or permission to
skip the response-data gap. No new model was trained in this pass.

## Delivered Data

Local release: `tmp/corpus-ready/twi-alignment-v2-20260915` (95 MB).
Parent: the unchanged sealed `twi-stage-v1-20260911` corpus.

| Separate View | Unique Sources | Train Examples | Validation Examples |
| --- | ---: | ---: | ---: |
| Sentence alignment | 11,557 | 22,660 | 454 |
| Dictionary alignment | 18,734 | 36,782 | 686 |
| Grounded-QA question alignment | 113 | 222 | 4 |
| Total alignment | 30,404 | 59,664 | 1,144 |

Each pair produces English-to-Twi and Twi-to-English tasks. The 60,808 examples
are two views of 30,404 sources, NOT 60,808 unique utterances or conversations.
Question alignment translates the existing questions, not their answers.
Dictionary entries are not disguised as sentences. Both directions preserve the
same source ID, hash, group and split. No references were generated or replaced.

English retention is copied byte-for-byte into its own directory: 2,131 complete
training conversation paths and 44 validation paths. Speaker roles and histories
are unchanged. It is not added to the alignment count or called new Twi data.

The sentence training view has 1,819,585 content tokens, of which 405,534 are
assistant targets across BOTH languages. Dictionary training has 216,283 target
tokens and question training has 3,420. The large JSON file size and templated
prompt tokens must not be mistaken for a large native-language pretraining set.
Counts use pinned Qwen3.5-4B tokenization only; the training base is not selected.

## Storage And Verification

The handoff contains source accounting, per-record provenance and terms, a
dataset card, checksums, a saved review snapshot and an explicit training config.
All 30,404 alignment sources were accounted for. No source was blocked by the
four saved review actions in the snapshot. Those actions cover three distinct
source records overall, not the entire corpus and not new generated labels.

The independent verifier reconstructs every expected example from the unchanged
source pairs, verifies both directions, checks source/group splits and confirms
unchanged English retention. The parent corpus also passed its full verification
again: 1,178,118 accounted source rows, 49,922 unique sources in accepted task
views, and zero cross-collection split conflicts.

Fifteen handoff files were uploaded and read back with matching SHA-256 hashes
from private Modal volume `ghana-health-understand-train`:

`/corpus-training-releases/twi-alignment-v2-20260915/0d59772b4b49`

Receipt: `tmp/corpus-ready/twi-alignment-v2-20260915.storage.json`.
No public Hugging Face upload, new training, production-chat change, ASR or TTS
work was performed. This private backup is the derived alignment handoff, not
an archive of the entire original 3.4 GB corpus.

```sh
python scripts/prepare_source_alignment.py verify --out tmp/corpus-ready/twi-alignment-v2-20260915
# Reproduce from the parent into a NEW destination, preserving the live reviews:
python scripts/prepare_source_alignment.py build --out tmp/corpus-ready/NEW_VERSION
# Upload only a verified handoff; existing matching files are reused:
python scripts/store_source_alignment.py --release tmp/corpus-ready/NEW_VERSION
```

Use `/Users/welcome/miniconda3/bin/python3`. A new build snapshots current HTTPS
reviews, so its review journal/hash may legitimately differ. The incomplete
`twi-alignment-v1-20260915` directory contains no finished handoff and must not be
used; its first build rejected the previously unhandled grounded-QA record type.

## Next Training Stage

The next defensible stage is a bounded bilingual-alignment adaptation experiment,
not another claim of a finished Twi conversational assistant. Keep the untouched
foundation as a control, preserve English replay, and evaluate both translation
directions on the held-out groups. Do not select a base purely from parameter
count or tokenizer coverage; use measured held-out comprehension and generation.
Before a launch, select explicit mixture weights, apply the foundation's native
chat template, verify assistant-only loss masks and define English-retention
and dialogue-regression tests. The current config deliberately cannot choose
these implicitly. Dictionary examples must not silently dominate the mixture.

Generated structured annotation remains unqualified. Three English-only teacher
methods were tested, and the third exposed problems in both outputs and the task
contract. See `reference-annotation-20260915.md`. These failures do not invalidate
the existing English references or prove that the models cannot understand Twi.
No failed teacher label entered this source-backed handoff.

Native multi-turn Twi responses remain the major missing collection for direct
response training. Source-backed alignment, English replay and 110 earlier
grounded Twi QA examples do not substitute for that. Health semantic/clinical
review and Twi commerce/tool trajectories also remain unfinished.
