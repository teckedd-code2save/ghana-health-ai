# R7 Source Coverage and Review Queues

17 November 2026. Private research inventory, not training admission, semantic
certification, a clinical judgment, or evidence that a teacher is qualified.
No new Twi, API requests, GPU work, source rewrites, or training were performed.

## Reproduce and Verify

From the R7 worktree, with Python, PyArrow and the original source files present:

```bash
python scripts/prepare_r7_source_inventory.py prepare --out tmp/r7-20261117/source-inventory/new-version
python scripts/prepare_r7_source_inventory.py verify --out tmp/r7-20261117/source-inventory/v4
python -m unittest discover -s scripts -p 'test_prepare_r7_source_inventory.py' -v
```

The default input project is
`/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai`.
The additional R6 history input is
`/Users/welcome/ghana-health-twi-r6/work/responses-r6/records.jsonl`.
The default output is `tmp/r7-20261117/source-inventory/v4/` in the current
worktree. Earlier `v1/`, `v2/` and `v3/` inventories are retained, not overwritten.

`manifest.json` binds every input file by SHA-256 and size, configuration, code,
and every output artifact. Inputs are hashed again before and after output
creation. `SEALED.json` binds the output manifest. Existing output directories,
linked inputs, changed source files, duplicate JSON keys, changed pack artifacts,
and broken source seals fail closed.

## Original Corpora

| Source | Original rows/messages | Split or scope |
| --- | ---: | --- |
| OASST training | 84,437 | 9,846 original conversation trees; 39,283 English messages |
| OASST validation | 4,401 | 518 protected trees; 2,022 English messages |
| AfriHealth training | 29,538 | Eight language/country variants |
| AfriHealth validation | 6,673 | Protected, including answers |
| AfriHealth test holdout | 2,612 | Protected questions; answers empty |
| GhanaNLP public CSV | 6,090 | 6,085 distinct raw bilingual pairs |
| Community parallel Parquet | 124,445 | 124,445 distinct raw bilingual pairs |
| R6 original adapted histories | 997 | Exactly matched to R6 registered response messages |

Full origin, language, source-label, distinct-target, duplicate-target, lexical
pattern and historical scheduled-token counts are in `summary.json`. Historical
scheduled exposure is not a new training-consumption receipt. Source rows,
assistant nodes, complete paths, prefix views and repeated exposures are distinct
counting units throughout.

### Source Hashes

The manifest contains every hash, including pack artifacts and inherited split
evidence. Selected root identities:

| Input | SHA-256 |
| --- | --- |
| R5 pack manifest | `29e23dfa6d363a30b9d3f457963513d3f6777d3c0e88b8d62ed24d30d7340147` |
| R6 pack manifest | `f8cdda46dc7418060bdad254add35048d6eebdb1b49aa3c5aad47050ba2830d0` |
| R6 original histories | `cbac18d36bd33742f5e2b37084e0d6deb0e3ff6a9cdce3df52449f7b64a986ee` |
| OASST train Parquet | `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e` |
| OASST validation Parquet | `24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416` |
| GhanaNLP CSV | `1d17fa9729ac06d9864c2f2c4c6dd4523c8a4ffb5e199f9f1a40fe0ed90cc84a` |
| Community parallel Parquet | `d2454f40726f13d21e589c5b7053d4aeda231ffa803096fc7ffb632987c1db1b` |
| Inherited alignment release | `98d9f62aebfd45c81c8c2a12373e07bad22c231145862f9befc83d2e558f47cd` |

## Queues and Boundaries

`eligible` means eligible for this source-review queue only. Every candidate
retains `training_eligible: false`. `pending` has unresolved evidence or review
requirements; `excluded` is not eligible for this queue. Reasons are explicit
and nonexclusive, so reason counts must not be summed as unique record counts.

### Original English Conversations

`oasst-candidates.jsonl` contains 4,087 complete, unchanged original-English
histories from 2,938 train trees. Of these histories, 2,424 are from source
families absent from R6. The queue has 38 correction-cue histories, 1,111 other
follow-up histories, 288 other category/list-cue histories, and 2,650 other
histories. These disjoint lexical buckets are sampling facets, not validated
behavioral categories or measured quality improvements.

All assistant ancestors must have original rank zero. Every ancestor must be
English, positively reviewed upstream, nondeleted, and annotated `pii=0`.
Contact-pattern findings and control-token literals prevent queue export. The
script does not load or export OASST contributor `user_id` values. This screen
does not prove that all personal information is absent; downstream privacy
review remains necessary. No redaction silently changes source wording.

The complete root-to-target chain, message IDs, parent IDs, original tree,
original split and source-row references remain attached. No preferred sibling
is substituted into a real history. `oasst-tree-index.jsonl` preserves ancestry
metadata for all 88,838 original train/validation nodes without their text or
contributor identities. Decisions cover every original assistant node.

Queue order uses deterministic round-robin coverage buckets, not the first few
records in a file. The explicit 20,000-history budget did not remove eligible
histories in this run. Earlier assistant targets recur as masked context in
later paths; expanded-path occurrences are not independent answers.

### Parallel Meaning: Progress Without New Generation

`parallel-meaning-candidates.jsonl` contains 28,025 **reference-only** unchanged
source pairs, routed separately:

| Route | Candidate pairs | Intended boundary |
| --- | ---: | --- |
| Sentences | 5,352 | GhanaNLP original parallel meaning; no new translation |
| Utterances | 5 | Recognized complete short utterances, not arbitrary short terms |
| Lexicon | 22,668 | Dictionary/glossary entries, not sentence or conversation targets |

The 28,025 total must not be called 28,025 new general-purpose answers. Of the
candidate pairs, 27,627 source IDs are absent from R6; most are lexical entries.
Prioritize the sentence/utterance routes independently of the lexicon route.
Of the sentence candidates, 4,954 source IDs are absent from R6 and 398 already
occur in R6. All five utterances and 22,668 lexicon source IDs are absent from R6.
Each candidate's `semantic_reference` identifies the original English/Twi
columns, original pair-content hash, inherited pair-content hash and source hash.
It explicitly declines to certify meaning correctness.

Original schemas and mappings:

- GhanaNLP CSV: `id`, `text` (English), `label` (Twi), `Unnamed: 3`. IDs are
  `ghana_nlp_parallel:<original id>`. References retain zero-based row index and
  one-based CSV record number, not a guessed physical line for multiline CSV.
- Community Parquet: `eng`, `twi`, `count`, `source_files`. IDs are
  `community_parallel:<original row index>`. `source_files` determines the
  inherited source genre; `count` is not independent-answer multiplicity.

Routing reuses the pinned `source-content-types-v2` policy, whose SHA-256 is
`2b0eb38b13af06df10d71e8b3c2e85617b471a62006e8f108e632ad6d799f019`.
The sealed alignment release's source accounting, assessments, view records and
original-file hashes supply inherited split and provenance evidence. Raw file
names containing `train` do not override a previously assigned validation split.

Across both full raw sources, 28,025 pairs are queue-eligible, 851 are excluded,
and 101,659 remain pending. There are 100,244 raw pairs without inherited split
assignments, 570 inherited validation sources, and 887 document-like structures
requiring separate review; these reasons overlap. Long/multisentence prose is
not silently treated as a sentence or assembled into invented documents.
Mixed-source and unknown source types remain pending. Original English and Twi
text are not copied into these queues: hashes and original row references permit
exact retrieval. Source terms, attribution and noncommercial restrictions remain
in force; no new commercial clearance is asserted.

### Health

`health-candidates.jsonl` contains 10,683 original-English **pending** train QA
rows, not clinically cleared targets. They are grounded only in their original
dataset QA record, not in an independently verified clinical reference.

All original validation/test rows and observed normalized-question families that
touch them are excluded. Exact and lexical-near overlaps are also screened.
AfriHealth identifier suffixes are reused across splits: all 6,673 validation
suffixes and all 2,612 test suffixes also occur in training. They do **not** prove
cross-language alignment. Neither row position nor suffix is used to invent
translation pairs. Cross-language family alignment, medical grounding, source
terms and clinical review remain unresolved; every health candidate is pending.

### Earlier R5 Replay

`r5-replay-candidates.jsonl` is reference-only. It preserves original record,
file and prior-review hashes and family identities; no prior review is promoted
to a new certification. Both exact-history differences and absent answer text
for the same source are measured. Changed system instructions alone must not be
described as loss of an answer. `r5-replay-decisions.jsonl` includes retained,
protected and pending records, including unchanged language replay.

| R5 class | Registered records | Pending replay | Protected or already present |
| --- | ---: | ---: | ---: |
| Responses | 657 | 645 | 12 |
| Meaning | 1,047 | 1,047 | 0 |
| English retention | 217 | 156 | 61 |
| Language text | 1,187 | 0 | 1,187 |

There are 1,848 pending replay records, not 1,848 newly verified answers. Source
comparison resolves connected families, including OASST message/tree aliases;
raw identifier spelling is reported separately. Across the entire registered R5
inventory, 926 response, 1,047 meaning and 302 English assistant occurrences have
no exact target text in R6 for the same connected source family. These totals
include protected records, which must remain excluded, and are not queue sizes.

## Inherited R6 Review Findings

The 997 histories contain 2,038 assistant occurrences, 2,038 exact distinct
targets, and 2,036 normalized distinct targets. Their original origins are 961
`openai_gpt-5.6-sol_authored_adaptation`, 27
`ai_authored_adaptation_second_review_conditional`, and nine
`ai_authored_adaptation_openai_repaired`. R6 registered source records omit these
origin fields; the original history file preserves them. Do not infer origin
from a stripped record or treat all adapted histories as new independent sources.

Review selectors include 55 short targets, 77 repeated-sentence/line occurrences,
81 repeated-eight-gram occurrences, seven repeated-list-item occurrences, and
23 literal examples reused across different category labels. These can include
legitimate repetition, code and repeated templates. They are not certified
mistakes. One exact generic phrase matched; no exact sympathy-only template
matched. A zero exact-template count does not establish absence of unhelpful
sympathy-only responses. Likewise, the one Twi-history correction-cue match is
not an estimate of all actual corrections. Raw word mentions are not category
coverage or evidence of distinct correct examples.

The source-backed search-claim concern is independently located at original
history line 188, assistant message index 1:
`r6-conversation:db4fd67cea63816e500908dd7801bee6cf595fc7548d507319d6435c6561591b`.
It claims earlier online-source discovery without tool-execution evidence in
the history. Both the preserved English source and Twi target receive
`past_search_claim_without_tool_evidence`, with `repair_review_required: true`.
Faithfully translating an unsupported action claim does not validate that claim.
No replacement answer was generated. All findings retain source hashes and
indices in `r6-review-flags.jsonl` for targeted review.

## Protection and Remaining Gates

The original R6 pack has no pack-local held-content artifact and reports zero
protected units while setting `overlap_free: true`. This inventory does not
reuse that flag as clearance. It rechecks both pack references, R5's 18,358
held units and explicit exclusions, the bound fresh holdout, original OASST
validation, all AfriHealth validation/test questions and available answers,
and the inherited parallel validation split.

The matcher uses NFC/casefold word-normalized exact text and eight-word shingle
Jaccard at least 0.85. Protection propagates across connected source/group/tree
identities and through all original OASST branches, not just selected paths.
Candidate and original/derived source views are scanned. This conservative
lexical screen can exclude benign overlaps and cannot certify semantic or
unknown-pretraining independence. Re-run protection against the final R7
evaluation freeze before any admission.

All source queues require the parent's final registry/admission, exact native
encoding, budget and GPU authorization gates. Review fresh families, actual
corrections and useful follow-ups separately against R6. No corpus count,
lexical flag, or source-faithful translation establishes a model improvement.

## Verification

The full script completed and the v4 seal verified. Nineteen local tests cover
source hashes, immutable output, original references, ancestor reconstruction,
rank and split exclusions, privacy projection, family aliases, scoped routing,
missing split assignments, literal review flags and accurate replay accounting.
No test claims semantic correctness of all candidate pairs.

## Registered Alignment Candidates

The complete export is `tmp/r7-20261117/alignment-candidates-v1/`, bound to the
sealed source inventory v4. Its manifest SHA-256 is
`a4f86025d6c4e5b81a6608c291af20ca8bde0f0f6eff5fa27f9c716650dcf8f8`.
Both seals verified; 15 alignment tests and 19 inventory tests passed.

| Class | Original pairs | Derived views | Pairs with flags | Without automatic flags, not gold |
| --- | ---: | ---: | ---: | ---: |
| Sentences | 5,352 | 10,704 | 568 | 4,784 |
| Dictionary entries | 22,668 | 45,336 | 15,388 | 7,280 |
| Utterances | 5 | 10 | 0 | 5 |
| Total | 28,025 | 56,050 | 15,956 | 12,069 |

These are 28,025 source pairs, not 56,050 independent examples. Every original
pair is registered in both directions, English to Twi and Twi to English.
Neither source nor target is normalized, corrected, truncated or retranslated.
No conversation, intent or dictionary explanation is invented. All 56,050 views
remain `training_eligible: false`, `admission_status: pending_quality_decision`,
`semantic_certification: false` and `generated_target: false`.
They are not scheduled: the total exceeds the later 50,000-exposure schedule cap.

### Registry Interface

- `source-pairs.jsonl`: complete original English/Twi strings, pair identity,
  class, exact source file/row references, both inherited view references,
  original/inherited pair-content hashes, source-family keys, type assessment,
  lengths, review flags and canonical record hash.
- `datasets/{sentences,lexicon,utterances}/views.jsonl`: two plain alignment
  records per pair, with `source_language`, `target_language`, `source_text`,
  `target_text`, text hashes, source-pair hash and full provenance. These are
  alignment views, **not chat messages** or admitted trainer inputs.
- `datasets/<class>/manifest.json` and `source-manifests.json`: per-class
  registry identities, record counts, relative artifact paths and hashes.
- `mapping-review-groups.jsonl`: per-class duplicate and bidirectional
  one-to-many mapping groups, with all affected pair IDs; no automatic verdict.
- `representative-examples.jsonl`: 18 original examples selected by a
  deterministic class sample plus flag examples. This is an inspection aid,
  not the dataset or a limit on export size.
- `summary.json`, `manifest.json`, `SEALED.json`: counts, limitations, all bound
  inputs/artifacts and the output seal.

### Review Counts

Flags below are overlapping pair counts, not proven mistakes:

| Flag | Sentence pairs | Dictionary pairs | Utterance pairs |
| --- | ---: | ---: | ---: |
| English maps to multiple normalized Twi values | 542 | 11,922 | 0 |
| Twi maps to multiple normalized English values | 8 | 1,519 | 0 |
| Duplicate normalized pair | 0 | 112 | 0 |
| Word-length ratio greater than four | 1 | 4,974 | 0 |
| Numeric-literal difference | 12 | 1 | 0 |
| Unicode format character | 9 | 0 | 0 |

The sentence mapping report contains 271 English-to-multiple-Twi groups and
four reverse groups. The dictionary report contains 4,252 English-to-multiple-Twi
groups, 690 reverse groups, and 56 normalized duplicate groups. There are no
exact duplicate pairs within these selected classes. Polysemy, synonyms, spelling,
OCR artifacts and real source errors require different decisions; these flags
do not resolve them. For example, the numeric selector can flag `1965` versus
a number joined to a word, and ordinal `15th` versus `15`, without a meaning error.

Representative review cases include `ghana_nlp_parallel:66701`, which shares
its Twi target with another English reference, and `ghana_nlp_parallel:65992`,
which has a large word-length mismatch. Their complete unchanged references and
original row evidence are retained. Neither flagged nor unflagged originals
receive a native-speaker or clinical certification.

Character and word lengths were measured, not tokenizer lengths. The longest
sentence references are 428 English/502 Twi characters and 76 English/97 Twi
normalized words; dictionary maxima are 146/133 characters and 22/24 words;
utterance maxima are 19/19 characters and four/four words. These maxima may
come from different pairs. No sequence-fit claim is made by this export.

The exporter independently reconstructs all 22,565 distinct normalized held-text
fingerprints from bound prior-pack and original validation/test sources, checks
the fingerprint set exactly against inventory v4, and scans both languages and
their concatenation. Existing source-family exclusions remain inherited. The
final R7 evaluation freeze still requires the parent's boundary check.

```bash
python scripts/prepare_r7_alignment_candidates.py prepare --out tmp/r7-20261117/alignment-candidates-new
python scripts/prepare_r7_alignment_candidates.py verify
python -m unittest discover -s scripts -p 'test_prepare_r7_alignment_candidates.py' -v
```
