# Private HTTPS Corpus Review

## Access And Scope

URL: https://ghanahealth.serendepify.com/research/corpus/

This is an isolated review service, not a new public-chat deployment. It reuses
the tested React workbench and source-hash-bound review handlers. A standalone
gateway authenticates every page, asset and API request with HTTP Basic over
HTTPS. Username: `reviewer`. The password is stored only in the root-readable
`/opt/ghana-corpus-review/.env` on the VPS and was delivered to the owner; never
copy it into this repository. This is a separate review login, not Google auth.

Only `/research/corpus` and `/research/corpus/*` are routed to the new service.
The public application still uses `127.0.0.1:13020`. The review server listens on
`127.0.0.1:13121`, behind Caddy's existing valid HTTPS certificate. Its direct
health check is not publicly routed. Unauthenticated requests receive 401;
cross-origin saves receive 403. Source files are not downloadable by path.

## Data And Corrections

The private projection contains 71,831 non-synthetic, non-protected review rows
from `twi-stage-v1-20260911`, around 69 MB including the application. It is not a
training export. The parent manifest hash is
`ddc88cb47302905d98d74d7e08f66856dd397836479619691c13e40a12e83866`.
The projection ledger hash is
`567144f0dac984be08993789f4a38a0020de0afb2e7f5942ded541232e259bda`.
The original 3.4 GB sealed release and all raw data remain unchanged locally.

No saved local review journal existed at packaging time (zero copied decisions).
Visual inspection and unsaved browser drafts are not persisted approvals. Local
browser drafts do not transfer between localhost and the HTTPS origin.

Use the HTTPS address for subsequent reviews. The service writes append-only,
source-hash-bound decisions to:

`/opt/ghana-corpus-review/reviews/twi-stage-v1-20260911/decisions.jsonl`

This directory survives service restarts and is separate from deployed code.
The frozen projection is read-only to the service. No production database
credentials are supplied, and no production reviews are modified. Export this
journal before building a corrected dataset release; do not overwrite it with
an older laptop copy. Local and HTTPS journals are not automatically merged.

## Operation

- Unit: `ghana-corpus-review.service`, non-login user `gha-corpus-review`.
- Active code (2026-09-15): `/opt/ghana-corpus-review/current` -> `releases/reference-20260915-r2`.
- Root-owned password file: `/opt/ghana-corpus-review/.env`, mode 0600.
- Service restrictions: read-only system, only review journal writable,
  private temporary directory, no privilege escalation, 768 MB memory limit.
- Caddy site: `/etc/caddy/sites/55-ghana-health-ai.caddy`; a timestamped
  `.before-corpus-*.bak` preserves the previous route configuration.
- Stop/start: `systemctl stop ghana-corpus-review` / `systemctl start ghana-corpus-review`.
- Logs: `journalctl -u ghana-corpus-review`; no authorization headers logged by
  the Node service. Caddy retains its existing sensitive-header redaction.

Password rotation changes only REVIEW_PASSWORD in the protected VPS file,
followed by a service restart. Never include the value in shell arguments or
committed configuration. Current access is shared owner review access; named
reviewer accounts and per-user attribution remain future work.

## Verification

Standalone fixture tests passed authentication, dataset isolation, cross-origin
save rejection, stale source rejection, durable idempotent saves and mobile/
desktop rendering. Public HTTPS smoke checks confirmed authenticated real data,
401 for unauthenticated pages/assets/API, 403 for cross-origin saves and an
unchanged 200 response from the public application. No corpus review was
fabricated by these tests. TypeScript and the existing local review tests pass.

Rebuild with `package_corpus_review.py` into a NEW directory, then
`node scripts/corpus-review/build.mjs OUTPUT RELEASE`. The installation script
is intentionally first-install-only and refuses changed Caddy configuration,
an occupied port or an existing deployment. Future releases must preserve the
independent review journal and inspect the active configuration before rolling.

## Model Loading Work

CPU-only I/O probe: `fc-01M2ETK68C8WWWT5YJ1NK6YQ2M`.
Four ~10 GB shards staged in 20.19 seconds with matching SHA-256 hashes; a local
read of one shard took 7.76 seconds versus 15.61 seconds from the volume.
The estimated full staging time was 119.35 seconds, not a GPU load measurement.

The revised runner verifies and stages the pinned checkpoint onto ephemeral
storage, then uses lazy reads from that local path. Sequential calls can reuse
the loaded engine; the container scales down after 60 idle seconds. No public
model endpoint or new training was started. Qualification remains bounded at
1,200 seconds with no automatic retry.

New qualification receipt: `tmp/corpus-teacher/qwen235-v3-staged`;
call `fc-01M2ETZCZMG4XMF5Q7DYNMYW26` completed with all 40 outputs.
Actual staging took 162.18 seconds. Engine readiness, including staging and
kernel warmup, took 568.05 seconds. Total qualification took 1,078.51 seconds,
inside the unchanged 1,200-second bound. Shard-loading logs measured about
2.17 seconds per shard, versus roughly 150 seconds before staging.

The loading problem is resolved for this run, but teacher qualification FAILED.
Four dialogue outputs hit the output limit. Of eight meaning controls, an
assistant comparison against the fixed references found five critical errors:
child fever became an unsettled matter, delivery became residence, tomorrow
became later, eye pain became a hand-ownership claim, and uncertain arrival became
negated departure. The keyword checker caught only three of those five. See
`semantic-audit.json`; it is not a native-speaker review. The gate remains closed
and none of these outputs entered training. No unchanged bulk run is warranted.
The prior sealed dataset's failed-loading receipt is preserved, not rewritten.

Static assets now have content-versioned names. Upload new assets and server
support first, then atomically switch index.html; leave older versioned assets
available for open pages. Tests verify the exact served asset hashes. The mobile
footer is in document flow so it cannot hide either correction field.

References: [Modal weight lifecycle](https://modal.com/docs/guide/model-weights)
and [vLLM loading strategies](https://docs.vllm.ai/en/stable/api/vllm/config/load/).

## 2026-09-15 Review Update

The source projection remains unchanged. A separate 32-record correction queue
now compares Qwen235 and OSS120 English-reference annotations on real validation
sources. Open `/research/corpus/?collection=teacher`, labelled Annotation checks.
The existing full collections remain in the same dropdown. Controls and failed
raw reasoning are not exposed as corpus rows. This is not a new training corpus.

The queue artifact is `tmp/corpus-reference/review-v2-20260915`; the deployed copy
is under `current/tmp/corpus-reference-review/twi-stage-v1-20260911/`.
Each candidate set has its own hash. New review records include that version;
earlier source approvals do not certify new labels, and stale choices are not
silently restored. Intent/entity corrections default to the appropriate field.
URL state now preserves the current collection and position across refreshes.

`scripts/corpus-review/update.sh` rolls only this isolated service. It requires
an explicit new version and expected previous version, verifies the unchanged
ledger, retains older browser bundles and reuses the independent review journal
and credential file. Caddy and the main application are not modified.

The four original saved actions survived both review updates with the same SHA:
`0f23994e213b1d0b5bc966361be1ef55403cc3307c5ae0f1f2f06bf1d873caeb`.
Service-user write permission was checked without adding a test review.
Authenticated HTTPS checks verified all 32 source/candidate identities, the four
full collections, bundle hashes, access controls, and mobile/desktop layouts.
Screenshots were inspected after compacting the annotation fields. Null intent
now displays as no direct request, not as a missing annotation. Empty and absent
fields remain distinct. No real review was submitted by the tests.
