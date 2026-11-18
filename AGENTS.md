# Agent Playbook — ghana-health-ai

This is the operating manual for anyone (human or AI) picking up an issue in this repo. Read it before opening a PR.

## Stack at a glance

- **Next.js 16** · TypeScript
- **Postgres** (Hetzner VPS, Docker) · **Prisma 7**
- **Secrets:** self-hosted Infisical at `https://secrets.serendepify.com` — never commit `.env*` files
- **Hosting:** Hetzner VPS `128.140.12.62`, Docker Compose at `/opt/ghana-health-ai/`, Caddy reverse proxy
- **Public URL:** `https://ghanahealth.serendepify.com`

## Running locally

```bash
# Inject all secrets from Infisical — never make a .env file
sec -- npm run dev

# Other useful ones:
sec -- npm run build               # production build
sec -- npm run lint
```

## Picking up an issue

1. **Read the issue end-to-end.** If acceptance criteria are unclear, leave a comment before writing code.
2. **Create a branch** off `main`: `<type>/<issue-number>-<short-slug>` — e.g. `feat/42-admin-form`, `fix/57-retry`, `chore/61-sentry`.
3. **Self-assign:** `gh issue edit <N> --add-assignee @me`.
4. **Touch only what the issue asks for.** No drive-by refactors.

## Commit convention

Conventional commits, with the issue number in the trailer:

```
<type>(<scope>): <imperative summary, <72 chars>

<optional body — why, not what>

Refs #<N>            ← partial progress
Closes #<N>          ← fully resolves the issue
```

Types: `feat` · `fix` · `chore` · `refactor` · `docs` · `test` · `perf` · `security` · `build` · `ci`

Use `Closes #N` in the **PR description**, not just the commit, so squash-merges keep the link.

## Pull requests

- **Title:** same shape as the commit (`feat(scope): summary`).
- **Body:** fill in the PR template — Summary, Test plan, Closes #N.
- **Size:** under ~400 lines of diff when you can.
- **CI must be green** before requesting review.
- **One issue per PR.**

## Deploy flow

Push to `main` → GitHub Actions builds the Docker image, pushes to `ghcr.io/teckedd-code2save/ghana-health-ai`, SSHes the VPS, runs `prisma migrate deploy`, and rolls the container. No manual steps.

If a deploy needs a new secret:

1. Add the key to Infisical at https://secrets.serendepify.com.
2. Reference it in code as `process.env.MY_KEY`.
3. For build-time `NEXT_PUBLIC_*` vars: also add to `Dockerfile` ARG/ENV and `.github/workflows/deploy.yml` build-args.
4. Automatic Infisical-to-VPS sync was removed at the repository owner's request. Infisical changes no longer update `/opt/ghana-health-ai/.env` or GitHub Variables automatically. Apply production secret changes through the deployment environment tooling, and update matching GitHub Variables for `NEXT_PUBLIC_*` values before building. Existing deployed values remain in place.

## Operating

```bash
bin/logs              # tail web container, last 200 lines
bin/logs 500          # tail last 500 lines
bin/logs --since 10m  # last 10 minutes (no follow)
bin/logs db           # tail postgres
bin/rollback          # roll back to :bootstrap (original known-good image)
bin/rollback <sha>    # roll back to a specific SHA
```

## Things that are NOT okay

- Committing `.env*` files (gitignored, but double-check)
- Hardcoding API keys, even test ones
- Editing existing migration files
- Pushing directly to `main` (the workflow assumes PR-then-merge)
- Wide refactors mixed into a bugfix PR

## Required Cost Controls

These apply to every agent, including research handoffs. See
[the cost-control checklist](docs/research/COST_CONTROL.md).

- Before paid compute, model downloads, bulk annotation, evaluation or deployment,
  check current provider usage, applicable unexpired credits, invoices, active
  jobs, storage size/cost, and both usage and out-of-pocket spend limits.
- Record a bounded job estimate covering compute, memory, storage growth,
  checkpoints, retries and network transfer. Obtain an explicit dollar ceiling
  and stop condition. Access to an API key or a prior "continue" is not unlimited
  spending authorization. Unknown billing/headroom means no new paid launch.
- Recheck after the first batch, at least hourly while paid work runs, and before
  any resume, retry, scale-up or new model download. Report material changes.
- Preserve recoverability, but do not retain unlimited checkpoints, duplicate
  model caches or repeated full backups without a storage budget and retention
  plan. Storage can keep billing with all compute stopped.
- If the user reports inability to pay or exhausted credits, hold new paid jobs
  and automatic resumes. Keep read-only inspection/local work moving. Ask before
  changing account billing settings or deleting any stored data; never pay,
  increase limits, top up or change plans without explicit authorization.
- Handoffs must include the current spend restriction, last cost check, active
  jobs, retention decisions and next authorized action. A documented policy is
  not proof of an automatic spending monitor or provider-enforced storage cap.

## Where to find things

- **Issues / roadmap:** https://github.com/teckedd-code2save/ghana-health-ai/issues
- **Secrets UI:** https://secrets.serendepify.com/project/4499072b-2a0e-44b3-a067-c27020409c48/secrets/dev
- **Production logs:** `bin/logs` (or SSH the VPS and `docker logs ghana-health-ai-web -f`)
- **Caddy site config:** `/etc/caddy/sites/{{caddy_priority}}-ghana-health-ai.caddy` on the VPS
- **Compose file:** `/opt/ghana-health-ai/docker-compose.yml` on the VPS
