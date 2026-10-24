# Sealed Instruction Research Export

Date: 24 October 2026. Scope: the selected 699-source instruction correction
queue, not certification of the complete corpus or four training collections.

## Final Result

I/J is complete for this queue. Release `release-v6-20261024` is sealed after
all **1,520,077 comparisons across 62 sources** and a separate cold reconstruction
produced identical bytes for all **18 export artifacts**. The normal seal verifier
also passes. All 699 originals are accounted for: **333 eligible research sources,
366 excluded**. Original text, protected data and saved reviews are unchanged.

| Partition | Sources | Assistant tokens | Sequence tokens |
| --- | ---: | ---: | ---: |
| Twi prose | 86 | 18,752 | 26,208 |
| Language exercises | 220 | 25,045 | 43,893 |
| Code exercises | 17 | 5,161 | 6,813 |
| Literal labels | 9 | 373 | 1,049 |
| Literal structure | 1 | 62 | 197 |
| Total | 333 | 49,393 | 78,160 |

The accepted sources form 332 related families. This is not 333 natural
conversations: task categories remain separate and no new multi-turn trajectories
or English retention rows were added. Source/agent-derived material remains
labelled synthetic. This release has no whole-example human approvals and
does not certify native fluency, clinical safety or future model performance.

Exclusions are 363 protected-overlap/related-family sources, one specialist
admission hold and two unverifiable linked sources. Of the 363, 362 have direct
protected-text witnesses and one is excluded through its related family.
Conservative lexical matches can be false positives; their evidence is retained,
not waived to raise the row count. Exported holdout conflicts, duplicate final
messages and truncated targets are all zero under the recorded rules.

Manifest SHA-256:
`0cd8d8ddb3b63e6cc3cbd92efeb2ce55d1c1c1e17baa8d7f3bf5bf52142c501c`.
Report SHA-256:
`dc9d911303136aa987c13237970057a0e8e52ec48f9165668afc0b15ec54009c`.

## Admission

D51/E50 preserve the 699 originals: 669 replaced, 28 retained, two unavailable
sources held. F/G accounts for every original with 696 provisional research
admissions and three exclusions. The extra exclusion is an explicitly recorded
specialist-assessment hold; a meaning pass does not waive that restriction.

Provisional task counts: 355 language exercises, 287 Twi prose, 39 code
exercises, 14 literal labels and one literal structure. The five categories
are kept separate. They are not all ordinary Twi conversations. Whole-example
human approvals, independent native certification and clinical certification
are not claimed.

## Resource Failure And Fix

The previous full-scan attempt failed before sealing after more than 1.4 million
comparisons. Its generic guard combined elapsed time, witness count and evidence
count without identifying which bound was exceeded. Restarting that process
would lose already-completed comparisons.

The replacement commits source cursor and losslessly compressed findings in
one SQLite transaction every 250 records. Resume verifies all input and code
hashes, runtime versions, source inventory, batch checksums and cursor continuity.
It rejects changed inputs, corrupted state and concurrent writers. An interrupted
uncommitted batch can repeat; committed batches do not.

Exact, near and eight-word-span matching rules remain unchanged. All evidence
is disk-backed, without sampling or a cumulative witness-count cutoff. Time
slices checkpoint; a disk bound reports the limit and retains committed state.
Related-source protection is resolved after the entire comparison set, never
inferred from a train label alone.

The deliberate 1,000-record pause resumed from exactly 1,000. Consistent partial
and complete snapshots were backed up and verified by reading and hashing every
remote artifact. The isolated HTTPS execution map distinguishes completed
comparisons from the still-pending independent reconstruction and seal.

## Release Contract

V5's completed comparison scan exposed a second checker defect before sealing:
436 original input fields contain a known literal `nan` missing-value marker,
with empty pinned English input. Its spurious exact matches inflated evidence
and could exclude corrected records. V5 finalization was stopped unsealed.
V6 reuses verified unaffected batches and rechecks affected raw rows after
excluding only that non-content field from the candidate index. It does not
drop real NaN content, short answers, genuine overlaps or any holdout rule.
Deleting old marker findings without rechecking would be unsafe: the strongest
false match could have hidden a weaker genuine overlap. Eleven focused tests
cover that case and equality with cold results. Full independent reconstruction
subsequently passed before V6 was sealed.

The frozen inventory contains 98 path/role entries and requires all 1,520,077
comparison records across 62 sources. The existing 1,178,118-row legacy ledger
is comparison material, not newly admitted training data.

After the scan, inherited-family and holdout exclusions apply before rendering.
The exporter validates complete message schemas, original source identities,
target hashes and assistant-only loss masks. It measures compatibility with
the pinned local tokenizer, uses a 4,096-token limit and never truncates targets.
That tokenizer measurement does not select the next training base.

A fresh process reconstructs every output from the same pinned inputs into a
separate directory and compares the complete artifact set and byte hashes.
This second scan is independently checkpointed. Only successful reconstruction
permits a seal. Unsealed attempts remain evidence of work, not training releases.

## Verified Software And Hosting

- 522 instruction tests pass, including 15 resumability cases, nine chunked-backup cases, 22 exporter cases and 11 marker/recheck cases.
- Six research-source recovery tests pass separately.
- The initial full-suite failures were missing jsdom/Mermaid and a stopped Docker
  runtime for the pinned PHP check. Restoring the prerequisites fixed all four;
  no tests were skipped or weakened.
- Seven research/article Mermaid diagrams parse.
- HTTPS checks pass for authentication, full queue/asset hashes, CSRF rejection,
  desktop/mobile layout and unchanged public chat. No review decisions are written.
- All 601 allowlisted research source files have a verified private recovery
  archive, separate from corpus data. Credentials and environment files are excluded.
- The final export, completed reconstruction snapshot and final review projection
  are privately backed up: all 26 files matched remote SHA-256 reads. The final
  source/document archive has its own receipt; these are not public downloads.

## Handoff And Next Stage

The derived V6 build reused 1,443,497 verified parent comparisons and rechecked
76,580 affected rows. Its cold reconstruction used no parent results. Both are
complete, separately resumable, and agree. V1-V5 remain unsealed historical
attempts, not alternative training releases.

The private release contains separate message/token partitions, source and
admission ledgers, full overlap evidence, schema, dataset card, configuration,
checksums and `SEALED.json`. Reproduce from the pinned inventory, bound source
code and runtime:

```bash
python scripts/instruction_training_export_v3.py verify \
  --out tmp/instruction-export/release-v6-20261024 --reconstruct
```

Recommended next stage: include this as an explicitly weighted instruction
partition in a controlled adaptation experiment, with separately admitted
English retention and independent held-out meaning, naturalness, repetition
and English-regression checks. Do not train a purported general conversational
model on this partition alone or concatenate its task types blindly. Selecting
the base and mixture remains a separate decision. The measured tokenizer is
not that selection. Refresh source protections and review decisions before a run.

No training, public dataset upload, ASR/TTS change or new production model
deployment occurred. The wider corpus and its four collections retain their
own readiness requirements.

The [authenticated review](https://ghanahealth.serendepify.com/research/corpus/?collection=responses)
and [execution map](https://ghanahealth.serendepify.com/research/corpus/progress)
display the current published checkpoint. Private storage receipts and exact
resume instructions remain in the research workspace.
