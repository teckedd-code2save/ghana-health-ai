# R6 and R7 Checkpoint Backups

Updated 21 November 2026. **Prepared and locally verified; HF uploads pending
write-capable local authentication.** A model card alone is not a weight backup.
Do not report a successful upload without a committed revision and verification
receipt. No paid compute, paid storage upgrade or public dataset upload is authorized.

## Contents

| Checkpoint | Verified local package | Scope |
| --- | ---: | --- |
| R6 | 193,187,626 bytes before checksum manifest | Final PEFT adapter, configuration, frozen tokenizer assets, identity, model card |
| R7 | 714,146,552 bytes before checksum manifest | Same, plus full final training state at update 2,413 and its committed manifest |

Durable package root:
`/Users/welcome/Documents/Research-Exports/ghana-health-ai/hf-backups-20261121-v2/`.
Use its `r6/` and `r7/` directories. Earlier prepared versions remain local;
this v2 adds explicit benchmark/data tables and stronger identity verification.

Planned private HF repositories: `teckedd/gha-twi-understanding-r6` and
`teckedd/gha-twi-understanding-r7`. Neither is confirmed uploaded by this record.

| Artifact | SHA256 |
| --- | --- |
| R6 trained adapter | `6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c` |
| R7 trained adapter | `f6e6b484ca404146016eefabd170738e66791880775238e435e31577300e57a7` |
| R6 package CHECKSUMS.json | `19edab3b516305965a083ad1b22d515185f0e74c0021feaa2ec68698255687c0` |
| R7 package CHECKSUMS.json | `de3bd768830fb3a9270fe63f686a25a32d785d5e995774c61fc4b7d68f5998df` |

These are adapters, not standalone 9B models. Both require the exact foundation
`McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` at revision
`693f12e444a30971ca53889128d7bade9491ecb3`. The base remains protected on Modal;
its local download is incomplete. Do not stack adapters: load R6 OR R7.

## Upload and Verify

From the research worktree, after the owner signs in locally with an HF write
token (never pass a token in chat, command arguments or Git):

```bash
ROOT=/Users/welcome/Documents/Research-Exports/ghana-health-ai
PY=/Users/welcome/miniconda3/bin/python
"$PY" scripts/prepare_r6_r7_hf_backup.py upload \
  --folder "$ROOT/hf-backups-20261121-v2/r6" \
  --receipt "$ROOT/hf-r6-upload-20261121.json"
"$PY" scripts/prepare_r6_r7_hf_backup.py upload \
  --folder "$ROOT/hf-backups-20261121-v2/r7" \
  --receipt "$ROOT/hf-r7-upload-20261121.json"
```

The uploader checks the account, refuses public repositories, enforces a file
allowlist, verifies original weight/config/tokenizer hashes and the R7 recovery
binding, and requires each package to be below 1 GiB. It never changes plans or
starts HF inference. If the free private-storage quota rejects the upload, stop;
do not buy storage. If account plan status has changed, reassess storage billing
before proceeding. Existing source licenses and consent restrictions still apply.

After upload it reads the exact committed file inventory. Large files are checked
against server-reported LFS SHA256 and size; small files are downloaded and hashed.
The receipt explicitly distinguishes that verification from a full weight
round-trip download or inference restore, neither of which has been claimed.

If upload succeeded but verification was interrupted, use the printed HF commit
with `verify --folder ... --receipt ... --revision <exact-40-character-commit>`.
Inspect the remote commit before retrying an upload; do not create blind retries.
Never overwrite existing receipts. Preserve the exact committed revision for restore.

## Recovery Boundaries

- R6 can be restored for inference with its adapter and pinned base. Its full
  optimizer/RNG checkpoint remains on Modal and is not included in this package.
- R7 includes its final adapter/optimizer/scheduler/RNG state, but training resume
  also requires the original sealed data and code in the existing private archive.
  This is preservation, not permission to launch another paid job.
- No raw corpus, recordings, reviews, credentials or private billing records are
  included. Do not upload the combined private recovery archive as a model repo.
- Preserve all existing Modal and local model/corpus/recovery copies. HF upload
  verification alone does not authorize deletion of any of them.
- R6's card reports the corrected benchmark and limitations. R7's card reports
  training and data coverage, with quality evaluation still pending.

Public R4/R5 repositories inspected on 21 November contained cards only, not
weights. Their separate original trained adapters remain protected; these R6/R7
packages do not certify the completeness of every historical model backup.
