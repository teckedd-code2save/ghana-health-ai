/**
 * Backfill / refresh ABENA passage embeddings on KnowledgeArticle + Product.
 *
 * Usage (with Infisical):
 *   sec -- pnpm db:index-embeddings
 *   sec -- pnpm db:index-embeddings -- --force
 *
 * Requires MODAL_EMBED_URL (research path) or OPENAI_API_KEY (emergency fallback).
 */

import "../src/config/load-env";
import { ensurePassageEmbeddings } from "../src/lib/twi-retrieve";
import { isAbenaConfigured, isEmbedConfigured } from "../src/lib/embed";
import { prisma } from "../src/db/prisma";

async function main() {
  const force = process.argv.includes("--force");

  if (!isEmbedConfigured()) {
    console.error(
      "No embed path configured. Set MODAL_EMBED_URL (preferred) or OPENAI_API_KEY.",
    );
    process.exit(1);
  }

  console.log("Indexing passage embeddings…", {
    abena: isAbenaConfigured(),
    force,
  });

  const result = await ensurePassageEmbeddings({ force });
  console.log("Done:", result);

  const [articles, products] = await Promise.all([
    prisma.knowledgeArticle.count({
      where: { isActive: true, embeddingB64: { not: null } },
    }),
    prisma.product.count({
      where: { isActive: true, embeddingB64: { not: null } },
    }),
  ]);
  console.log("Cached rows:", { articles, products });
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
