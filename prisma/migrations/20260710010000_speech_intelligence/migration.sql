-- Speech intelligence + payments + audit
ALTER TABLE "orders" ADD COLUMN IF NOT EXISTS "paystack_ref" TEXT;
ALTER TABLE "orders" ADD COLUMN IF NOT EXISTS "authorization_url" TEXT;
ALTER TABLE "orders" ADD COLUMN IF NOT EXISTS "paid_at" TIMESTAMP(3);

CREATE UNIQUE INDEX IF NOT EXISTS "orders_paystack_ref_key" ON "orders"("paystack_ref");

CREATE TABLE IF NOT EXISTS "audit_logs" (
    "id" UUID NOT NULL,
    "action" TEXT NOT NULL,
    "actor_id" UUID,
    "entity_type" TEXT,
    "entity_id" TEXT,
    "ip" TEXT,
    "meta" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "audit_logs_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "audit_logs_action_created_at_idx" ON "audit_logs"("action", "created_at");
CREATE INDEX IF NOT EXISTS "audit_logs_actor_id_idx" ON "audit_logs"("actor_id");

CREATE TABLE IF NOT EXISTS "offline_queue_items" (
    "id" UUID NOT NULL,
    "session_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "synced_at" TIMESTAMP(3),
    CONSTRAINT "offline_queue_items_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "offline_queue_items_session_id_status_idx" ON "offline_queue_items"("session_id", "status");
