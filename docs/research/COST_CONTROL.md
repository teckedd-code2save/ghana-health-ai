# Research Cost Controls

Applies to Modal, paid model APIs, hosted storage and deployed research services.
AGENTS.md makes this a required operating procedure for every agent.

## Before Any New Paid Work

1. Read the provider's current billing page or supported billing API. Record the
   observation time, billing period, gross usage, credits actually applicable to
   the planned resource, net charges, invoices and usage/spend limits. Do not
   infer available funds from an old balance or confuse Codex, API and Modal credits.
2. Inventory active jobs and scheduled/warm deployments. Check detached receipts
   before any launch: never pay for an accidental duplicate or retry an unknown
   submission. Check storage separately even when no containers are running.
3. Estimate GPU/CPU/memory cost from rate and bounded runtime, plus new storage
   in GiB-days/month, checkpoint growth, downloads/egress and explicit retry count.
   Include the wider evaluation plan, not only the training job. Show uncertainty.
4. Record the user's dollar ceiling and stop condition in a private cost receipt.
   Bound timeout, concurrency and retries accordingly. A general research goal
   does not authorize charges beyond that ceiling. Unknown billing or insufficient
   headroom blocks the launch, not read-only inspection or local preparation.
5. Give persistent artifacts an owner, purpose, size estimate, review date and
   retention decision before creating large caches or checkpoint collections.

## While Work Runs

- Recalculate after the first measured batch. Check at least hourly during active
  paid work, and before retries, resumes, scale-ups or additional downloads.
- Alert at 80% of the authorized budget. If projected remaining work exceeds
  headroom, stop adding work and arrange a safe checkpoint before the cap. Do
  not wait for a provider charge or hard kill. Ask for any budget increase.
- Save actual consumption and costs, not just the estimate. Provider billing can
  lag: maintain a conservative local estimate too. Report unknowns explicitly.
- Do not invent a scheduled monitor: this is an operating rule until a verified
  automated guard is implemented. Provider settings are a separate protection.

## Storage Retention and Cleanup

- Distinguish irreplaceable source data/reviews/results from reproducible public
  model caches and redundant intermediate artifacts. Inventory references before
  deleting; current models, evaluation baselines and recovery states can depend
  on apparently old files.
- New training must have a bounded checkpoint-retention plan compatible with its
  tested recovery implementation. Prefer current/previous recoverable state plus
  explicitly selected best/final artifacts over unlimited full-state copies.
- Keep pinned revision/hash metadata when retiring a reproducible cache. Verify
  availability and access rights before assuming it can be downloaded again.
- Do not duplicate terabyte caches into a second paid backup to save storage.
  Preserve essential private artifacts in verified off-provider storage before
  removing their last provider copy. Use content-addressed/incremental backups.
- Prefer read-only Volume metadata and direct file downloads for inventories and
  model exports. Hash locally rather than launching a paid container solely to
  checksum artifacts. Check the remaining egress allowance first. A folder named
  "cache" can contain original datasets; it is not blanket deletion permission.
- Present exact deletion candidates, expected savings and dependencies for user
  approval. Never delete datasets, reviews, adapters or recovery checkpoints
  based solely on age or a file-name pattern.

## Current Restriction: 18 November 2026

The user reported a billing problem and authorized a **US$0 monthly out-of-pocket
spend limit** in Modal. It was saved and visibly verified in the billing UI.
No new paid research jobs or automatic resumes are authorized until the user
explicitly changes this restriction. R7 training already completed; matched
paid evaluation and deployment must wait for budget clearance. Read-only checks
and local work may continue. No invoice was paid.

The user subsequently approved exactly five public-cache directories totaling
533.425 GiB. Their deletion completed on 18 November, with before/after metadata
checks confirming the retained cache entries and five protected research volumes
were unchanged. Corpus, trained weights and R7 recovery files remain protected.
The private receipt is `tmp/modal-storage-audit-20261118/cleanup-receipt.json`.
This is not permission for further deletion or for re-downloading retired models
into paid storage. Check whether the user's connection is metered before a large
local download; local inference avoids provider charges, not internet or power costs.

Modal's UI warns that **volume storage charges still accrue** when the spend limit
stops workloads. A zero spend limit does not clear existing charges or provide
a zero-total-bill guarantee. Its default allowance must not be treated as a user
budget. Reducing stored bytes is a separate, approved cleanup task.

## Provider References

### 21 November Operational Update

- Stopped and readback-verified all 49 approved Ghana Health AI and akan-speech
  deployed apps. No unrelated app or data volume was removed. Private receipt:
  `tmp/cost-hold-20261121/app-shutdown.json`.
- Fresh storage inventory and a further cache-only proposal are retained in
  `tmp/modal-storage-audit-20261121/`. The proposal is not approved for deletion.
  Corpus, owner-trained models and R7 recovery are explicitly protected.
- No new paid compute or API work was launched. The owner confirmed unmetered
  local internet for the pinned public base download.
- HF R6/R7 private backup preparation is complete; upload is pending local write
  authentication. No plan upgrade or paid private storage is authorized. Stop
  on quota rejection; see [backup procedure](R6_R7_HF_BACKUPS.md).
- Browser access failed during this audit, so the billing UI/spend-limit setting
  was not reverified on 21 November. Its last verified setting remains 18 November.
  CLI app/storage readback is separate evidence, not a billing-page readback.

- [Modal billing and incremental charges](https://modal.com/docs/guide/billing)
- [Usage budgets versus net spend limits](https://modal.com/docs/guide/budgets)
- [Current compute and storage rates](https://modal.com/pricing)
- [Volume billing and deletion lag](https://modal.com/docs/guide/volumes#pricing)

Rates were checked on 18 November 2026; verify them again for future estimates.
Modal documents daily storage snapshots and possible billing for up to four
days after deletion. Cleanup does not retrospectively cancel incurred charges.
