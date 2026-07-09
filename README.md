# Ghana Health AI

Voice-first health companion for Ghana — **Twi-first** maternal health guidance, speaker-aware voice pipeline stubs, and everyday market ecommerce.

Built from [`Ghana_Health_AI_Technical_Spec.md`](../Ghana_Health_AI_Technical_Spec.md).

## MVP scope (this repo)

- Next.js PWA UI (home, chat, voice, market, account)
- Postgres + Prisma 7 data platform
- Health RAG over seeded maternal knowledge articles
- Intent routing (health / ecommerce / general)
- Cart + mock MoMo checkout
- Voice ASR + Voice ID **stubs** (swap for Modal Parakeet in Phase 1)
- Infisical project `ghana-health-ai` for secrets
- Docker Compose Postgres + shippable Dockerfile

## Quick start

```bash
cd Documents/SoftwareEngineering/serendepify/ghana-health-ai

docker compose up -d

# Preferred: Infisical injection
sec -- pnpm db:migrate
sec -- pnpm db:seed
sec -- pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

**Demo login:** `demo@ghanahealth.ai` / `demo1234`

### Local env fallback (gitignored)

```bash
cat > .env.local <<'EOF'
DATABASE_URL=postgresql://ghana_health:ghana_health_dev@localhost:5437/ghana_health_ai?schema=public
JWT_SECRET=dev-local-jwt-secret-change-me-32
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:3000/api
VOICE_MODE=stub
HEALTH_DISCLAIMER_ENABLED=true
EOF
```

## Architecture

```
Browser / PWA
  → Next.js App Router (UI + /api)
      → Prisma 7 + Postgres
      → Health RAG (keyword MVP) / Cart / Orders / Voice stubs
  → modal/  (GPU ASR/TTS jobs — deploy separately on Modal)
```

## Safety

This product provides **general information only**. It is not a doctor. Emergency language is injected for high-severity symptom patterns; users should contact CHWs, clinics, or emergency services when needed.

## Roadmap hooks

| Phase | Status in repo |
|-------|----------------|
| 0 Foundation | Schema, seed KB, stub voice, Modal skeletons |
| 1 Twi health chat | RAG chat + voice UI live (LLM/ASR upgrade pending) |
| 2 Multi-speaker + ecommerce | Market live; Parakeet/Sortformer pending Modal |
| 3 Scale languages | Schema supports `ga` / `ee` / `dag` |

## License

UNLICENSED — Serendepify private.
