-- Explicit agent memory for profile, health, and commerce continuity.
-- Memory is scoped and auditable; it is not a hidden replacement for the
-- current user transcript.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'MemoryScope') THEN
    CREATE TYPE "MemoryScope" AS ENUM ('HEALTH', 'COMMERCE', 'PROFILE');
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS "agent_memories" (
    "id" UUID NOT NULL,
    "user_id" UUID,
    "session_id" TEXT,
    "scope" "MemoryScope" NOT NULL,
    "key" TEXT NOT NULL,
    "value" JSONB NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    "source" TEXT NOT NULL DEFAULT 'conversation',
    "last_observed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "agent_memories_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "agent_memories_user_id_session_id_scope_key_key"
  ON "agent_memories"("user_id", "session_id", "scope", "key");
CREATE INDEX IF NOT EXISTS "agent_memories_user_id_scope_idx" ON "agent_memories"("user_id", "scope");
CREATE INDEX IF NOT EXISTS "agent_memories_session_id_scope_idx" ON "agent_memories"("session_id", "scope");

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.table_constraints
    WHERE constraint_name = 'agent_memories_user_id_fkey'
      AND table_name = 'agent_memories'
  ) THEN
    ALTER TABLE "agent_memories"
      ADD CONSTRAINT "agent_memories_user_id_fkey"
      FOREIGN KEY ("user_id") REFERENCES "users"("id")
      ON DELETE CASCADE ON UPDATE CASCADE;
  END IF;
END $$;
