-- Cache ABENA (or fallback) passage embeddings for Twi retrieval.
-- Avoid re-encoding the full knowledge base on every voice/chat turn.

ALTER TABLE "knowledge_articles" ADD COLUMN IF NOT EXISTS "embedding_b64" TEXT;
ALTER TABLE "knowledge_articles" ADD COLUMN IF NOT EXISTS "embed_model" TEXT;
ALTER TABLE "knowledge_articles" ADD COLUMN IF NOT EXISTS "embed_engine" TEXT;
ALTER TABLE "knowledge_articles" ADD COLUMN IF NOT EXISTS "embedded_at" TIMESTAMP(3);

ALTER TABLE "products" ADD COLUMN IF NOT EXISTS "embedding_b64" TEXT;
ALTER TABLE "products" ADD COLUMN IF NOT EXISTS "embed_model" TEXT;
ALTER TABLE "products" ADD COLUMN IF NOT EXISTS "embed_engine" TEXT;
ALTER TABLE "products" ADD COLUMN IF NOT EXISTS "embedded_at" TIMESTAMP(3);
