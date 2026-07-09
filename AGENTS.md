# Agent Playbook — ghana-health-ai

This is the operating manual for anyone (human or AI) picking up work in this repo. Read it before opening a PR.

## Stack at a glance

- **Next.js 16 (App Router)** · TypeScript · Tailwind CSS 4
- **Postgres 18** (Docker local / Hetzner VPS) · **Prisma 7** (`@prisma/adapter-pg`)
- **Secrets:** self-hosted Infisical at `https://secrets.serendepify.com` — never commit `.env*` files
- **GPU voice (Phase 1+):** Modal (`modal/` Parakeet / Whisper fine-tune stubs)
- **Hosting target:** Hetzner VPS, Docker Compose at `/opt/ghana-health-ai/`, Caddy reverse proxy
- **Public URL (planned):** `https://ghanahealth.serendepify.com` (set when shipping)

## Spec source

`../Ghana_Health_AI_Technical_Spec.md` — MVP is Twi maternal health + voice chat stub + market ecommerce.

## Running locally

```bash
# 1. Postgres
docker compose up -d

# 2. Secrets from Infisical (preferred) — never make a committed .env
sec -- pnpm dev

# Or local gitignored .env.local with DATABASE_URL etc.

# Useful:
sec -- pnpm run build
sec -- pnpm run lint
sec -- pnpm db:migrate
sec -- pnpm db:seed
```

Demo user after seed: `demo@ghanahealth.ai` / `demo1234`

## Product surfaces

| Route | Purpose |
|-------|---------|
| `/` | Landing |
| `/chat` | Health RAG + ecommerce intent companion |
| `/voice` | ASR stub + Voice ID enroll/verify |
| `/market` | Catalog, cart, mock MoMo checkout |
| `/login` | Register / session auth |

All UI data comes from `/api/*` → Prisma → Postgres. No hardcoded business catalogs in components.

## Picking up an issue

1. Read the issue end-to-end.
2. Branch off `main`: `<type>/<issue-number>-<short-slug>`.
3. Touch only what the issue asks for.

## Commit convention

```
<type>(<scope>): <imperative summary, <72 chars>

Refs #<N>
Closes #<N>
```

Types: `feat` · `fix` · `chore` · `refactor` · `docs` · `test` · `perf` · `security` · `build` · `ci`

## Deploy flow

Push to `main` → (after `ship-to-vps` wiring) GitHub Actions builds image, pushes GHCR, migrates, rolls container.

New secrets: add in Infisical, reference as `process.env.KEY`. For `NEXT_PUBLIC_*`, also add Dockerfile ARG/ENV and deploy workflow build-args.

## Things that are NOT okay

- Committing `.env*` files
- Hardcoding product/health data in React components
- Claiming clinical diagnosis without disclaimers
- Shipping without Prisma migrations for schema changes
