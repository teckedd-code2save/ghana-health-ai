# Private Corpus Review Service

This service packages the review workbench separately from public chat. It
listens on loopback behind the existing HTTPS reverse proxy, authenticates
pages/assets/API, and stores source/proposal-bound decisions in an append-only
journal. Do not expose its port directly or copy secrets into this repository.

## Build And Test Without Corpus Data

Install the repository's locked Node dependencies and Python 3.11 or later.
The browser harness additionally needs Playwright and Chrome; it writes only
disposable fixture reviews, never production reviews.

```bash
node scripts/corpus-review/build.mjs tmp/review-host-test twi-stage-v1-20260911
python3 -m unittest discover -s scripts -p test_instruction_review_projection.py
CORPUS_HOST_PACKAGE=tmp/review-host-test node scripts/test-corpus-review-host.cjs
```

No corpus, credentials or real user conversations are included in the public
test fixtures. `check-corpus-review-host.cjs` is a separate, read-only operational
check for the authorized host; it reads the existing review credential into
memory without printing or persisting it and never writes reviews.

## Package A Private Checkpoint

Use verified, immutable D and E artifacts and the original source inventories
from private research storage. Run their full reconstruction verifiers first.
`package_instruction_review.py` checks artifact integrity and exact source,
target and assessment bindings; it is not an independent semantic verifier.

```bash
python3 scripts/package_instruction_review.py \
  --corrections <D-directory> --corrections-sha256 <D-manifest-sha256> \
  --meaning <E-directory> --meaning-sha256 <E-manifest-sha256> \
  --originals <sealed-original-queue.jsonl> \
  --originals <sealed-inspection-handoff.jsonl> \
  --checkpoints 'D49 / E48' --as-of 2026-10-20 \
  --out <new-private-projection-directory>
```

The original inventories must cover exactly the D/E union, with no duplicates.
Original text, full proposals, corrected references and assessments stay private.
The projection never creates training admissions or claims human approval.

The deployment package needs the existing read-only source ledger and annotation
sidecars. Add the new `queue.json` and `manifest.json` under
`tmp/corpus-response-review/<release>/`; add generated `progress.html` under
`public/`. The build includes the renderer, stylesheet and read-only row reader,
not any data. Do not package `tmp/corpus-review` or credentials.

## Update The Existing Service

Use a new staged release under `/opt/ghana-corpus-review/stage-<version>`.
Verify transfer checksums and the existing journal hash before switching.
`update.sh <version> <expected-current-version>` refuses a changed current
release, a different source ledger, an existing target directory or a bundled
review journal. It reconnects the existing journal directory, preserves old
browser assets and rolls back if the health check fails.

The service requires `CORPUS_RELEASE`, `REVIEW_ORIGIN` and a strong
`REVIEW_PASSWORD` through the existing protected server environment file.
Keep the existing loopback bind, service user, file permissions and proxy route.
Do not run this update against the public chat service or change its database.

After switching, verify authentication, queue and asset hashes, corrected-source
views, pending examples, the execution map, responsive layout and unchanged
review history. Retain the prior release for rollback.
