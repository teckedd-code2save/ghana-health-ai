# R7 Research Data: Source and Training Views

This is a working private research release, not a public dataset or a claim of
native-language or clinical certification. No R7 training has completed at this
checkpoint. Refer to `R7-STATUS.md` for the current execution state.

## Collections

| Collection | Sources | Derived training views | Purpose |
| --- | --- | --- | --- |
| Sentence alignment | 4,782 original bilingual pairs | 9,564 directional views | Bind Twi sentences to their existing English references |
| Lexical alignment | 7,280 original lexical pairs | 14,560 directional views | Lexical grounding, kept distinguishable from prose |
| Short-utterance alignment | 5 original pairs | 10 directional views | Preserve useful short utterances |
| Controlled translation follow-ups | Reuses the 4,782 sentence pairs above | 4,782 final-target views in 2,391 paired groups | Latest-request changes and wrong-language correction |
| Executed English tools | 32 simulated source groups | 129 histories, 378 eligible assistant prefixes | Structured actions, result grounding, corrections and recovery |
| Conversation replay | 633 retained R5 and 868 retained R6 response histories | 1,501 complete histories before prefix expansion/deduplication | Restore useful earlier targets and retain scoped corrections |
| Meaning replay | Previously admitted R5/R6 references and annotations | 3,758 full records before combined deduplication | Retain existing semantic alignment |
| English retention | Previously admitted English conversation sources | 1,475 full records before prefix expansion/deduplication | Preserve English capability |
| Language replay | Previously admitted text | 1,187 text records | Maintain language modeling exposure |

Do not sum the first and fourth rows into new unique source counts. The fourth
row is an augmentation of the first. Three paraphrased prompt forms in the tool
collection do not create three independent underlying scenarios.

## Source Provenance

The bilingual material preserves original English and Twi fields from pinned
`Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` and
`ghananlpcommunity/twi-english-parallel-text` snapshots. Original source files,
row hashes, inherited split decisions and source revisions are retained in each
record. These sources carry noncommercial/research restrictions; this admission
does not authorize commercial use or raw-data redistribution.

The controlled follow-ups use English instruction scaffolding authored in code.
Every Twi assistant target is byte-for-byte the original admitted reference.
They are not naturally collected conversations or LLM-generated Twi responses.
Wrong-language prior replies are explicitly counterfactual context and receive
no training loss. Both paired source identities travel with every derivative.

Tools use project-authored synthetic English requests and an executable local
sandbox. There are four fictional catalogue products. Calculator, search,
comparison and draft-order results are actual sandbox outputs, never invented
web evidence. No purchase, payment or real external action occurs. Deliberately
invalid calls are context-only; successful recovery and truthful result reporting
are supervised. English tool competence does not establish Twi tool competence.

## Screening and Admission

- Original references are unchanged. Suggested repairs live in separate versions.
- Source IDs, hashes and inherited evaluation boundaries are checked.
- Duplicate mappings, potential polysemy and other unresolved alignment flags
  remain outside the current no-flag admission. They are not automatically errors.
- Full histories use the pinned student's native chat template with no truncation.
- Supervision covers only the selected final assistant target, including its native
  end-of-turn marker; user text, tool results and earlier replies are masked.
- Exact tokenized duplicates within a collection are removed during assembly.
- Source groups and full-history content are checked against actual held-out text,
  not a placeholder clearance flag.
- Each original row receives an admission, exclusion or review-needed decision.

Before combined assembly, alignment contains 369,545 one-pass target tokens;
controlled follow-ups add 197,002 and tool views add 15,168. These are component
counts, not additional independent sources or completed training.

The sealed `pack-v1` combines 36,966 source records into 39,462 full-history
prefix/text views, removes 855 exact within-collection duplicates and retains
**38,607 views / 1,849,881 effective target tokens**. Its one-pass schedule contains
2,413 optimizer updates. Direct response targets contribute 735,911 tokens;
meaning/alignment 603,289; English retention 466,685; language 31,856; and executed
tools 12,140. Native encoding and the inherited/fresh/public-task holdout checks
passed. These are planned exposures until a completed consumption receipt exists.
The protected public set preserves the exact 2,004 corrected-R6 benchmark cases;
foundation-model exposure remains unknown.

## Limits and Evaluation

Automatically screened source references are not semantic gold. Existing
translations can still contain mistakes. The current teachers failed critical
Twi meaning controls, so their outputs are excluded from new Twi supervision.
Medical-source expansion remains separate pending grounding and review; no new
clinical-readiness claim follows from this release.

R6 and R7 will be compared using matched decoding on protected fresh chats,
fixed-history corrections and independent rollouts. Report category correctness,
constraint updates, correction uptake and repetition separately from fluent
wording, request completion and provenance verification. Existing public-task
benchmarks remain separate. Development scenarios are disclosed as development
evidence; foundation-model exposure is unknown.

## Private Artifacts

Working outputs are under `tmp/r7-20261117/` in the R7 worktree. Relevant releases:
`alignment-admission-v1`, `alignment-followups-v2`, `replay-admission-v2`,
and `tool-admission-v2`.
Each includes source manifests, checksums, admission decisions and token counts.
Private Modal backups use volume `ghana-health-understand-train`, prefix
`/research-handoffs/twi-understanding/2026-11-17-r7/`.

Reproduce with `scripts/admit_r7_alignment.py`,
`scripts/prepare_r7_alignment_followups.py`, `scripts/admit_r7_tools.py`, then
`scripts/prepare_understanding_r7.py`. Each command requires original pinned
inputs and a new output directory; existing sealed releases are never overwritten.
