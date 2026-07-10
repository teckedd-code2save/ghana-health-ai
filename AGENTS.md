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
4. The `infisical-sync.yml` workflow projects secrets to `/opt/ghana-health-ai/.env` on the VPS automatically (hourly + on dispatch).

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

## Where to find things

- **Issues / roadmap:** https://github.com/teckedd-code2save/ghana-health-ai/issues
- **Secrets UI:** https://secrets.serendepify.com/project/4499072b-2a0e-44b3-a067-c27020409c48/secrets/dev
- **Production logs:** `bin/logs` (or SSH the VPS and `docker logs ghana-health-ai-web -f`)
- **Caddy site config:** `/etc/caddy/sites/{{caddy_priority}}-ghana-health-ai.caddy` on the VPS
- **Compose file:** `/opt/ghana-health-ai/docker-compose.yml` on the VPS
