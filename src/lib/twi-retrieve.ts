/**
 * Twi-native retrieval via ABENA embeddings (Ghana-NLP research path).
 *
 * Embeds the ASR/transcript query and ranks KnowledgeArticle / Product
 * rows by cosine similarity on **Twi** text fields — not English dumps,
 * not a hand-written semantic bank.
 *
 * Passage vectors are cached on the row (embedding_b64). Request path only
 * embeds the query + any missing/stale passages (lazy warm).
 */

import { prisma } from "@/db/prisma";
import {
  cosineSimilarity,
  embedOne,
  embedTexts,
  isEmbedConfigured,
  vectorFromB64,
  vectorToB64,
  type EmbedEngine,
} from "@/lib/embed";

export type RetrievedArticle = {
  slug: string;
  titleTw: string;
  titleEn: string;
  bodyTw: string;
  bodyEn: string;
  source: string | null;
  score: number;
};

export type RetrievedProduct = {
  id: string;
  nameTw: string;
  nameEn: string;
  priceGhs: number;
  score: number;
};

export type TwiRetrieveResult = {
  engine: "abena" | "openai" | "lexical" | "none";
  model?: string;
  articles: RetrievedArticle[];
  products: RetrievedProduct[];
  /** How many passages were re-embedded this call (cache misses). */
  cacheMisses?: number;
};

type PassageRow = {
  id: string;
  text: string;
  embeddingB64: string | null;
  embedModel: string | null;
  embedEngine: string | null;
};

/** Fallback lexical rank when ABENA is offline — temporary, not the product brain. */
function lexicalScore(query: string, doc: string): number {
  const q = query
    .toLowerCase()
    .normalize("NFC")
    .split(/[\s,.;!?]+/)
    .filter((t) => t.length > 2);
  if (!q.length) return 0;
  const d = doc.toLowerCase().normalize("NFC");
  return q.reduce((s, t) => s + (d.includes(t) ? 1 : 0), 0) / q.length;
}

function cacheUsable(
  row: { embeddingB64: string | null; embedModel: string | null; embedEngine: string | null },
  wantEngine: EmbedEngine,
  wantModel: string,
): boolean {
  if (!row.embeddingB64 || !row.embedEngine || !row.embedModel) return false;
  if (row.embedEngine !== wantEngine) return false;
  // Allow minor model id drift if engine is abena and both mention abena
  if (wantEngine === "abena") {
    return (
      row.embedModel === wantModel ||
      row.embedModel.includes("abena") ||
      wantModel.includes("abena")
    );
  }
  return row.embedModel === wantModel;
}

/**
 * Ensure passage embeddings exist for articles/products.
 * Writes cache back to Postgres. Safe to call from seed / index script / lazy retrieve.
 */
export async function ensurePassageEmbeddings(opts?: {
  force?: boolean;
  articleLimit?: number;
  productLimit?: number;
}): Promise<{
  engine: EmbedEngine;
  model: string;
  articlesUpdated: number;
  productsUpdated: number;
}> {
  const force = opts?.force ?? false;
  const [articles, products] = await Promise.all([
    prisma.knowledgeArticle.findMany({
      where: { isActive: true },
      take: opts?.articleLimit ?? 200,
      select: {
        id: true,
        titleTw: true,
        bodyTw: true,
        embeddingB64: true,
        embedModel: true,
        embedEngine: true,
      },
    }),
    prisma.product.findMany({
      where: { isActive: true },
      take: opts?.productLimit ?? 200,
      select: {
        id: true,
        nameTw: true,
        nameEn: true,
        descriptionTw: true,
        embeddingB64: true,
        embedModel: true,
        embedEngine: true,
      },
    }),
  ]);

  // Probe engine with a tiny encode so we know which model/engine we're caching under
  const probe = await embedOne("akwahosan", "passage");
  if (!probe) {
    return { engine: "none", model: "none", articlesUpdated: 0, productsUpdated: 0 };
  }

  const articlePassages: PassageRow[] = articles.map((a) => ({
    id: a.id,
    text: `${a.titleTw}. ${a.bodyTw}`,
    embeddingB64: a.embeddingB64,
    embedModel: a.embedModel,
    embedEngine: a.embedEngine,
  }));
  const productPassages: PassageRow[] = products.map((p) => ({
    id: p.id,
    text: `${p.nameTw}. ${p.nameEn}. ${p.descriptionTw ?? ""}`,
    embeddingB64: p.embeddingB64,
    embedModel: p.embedModel,
    embedEngine: p.embedEngine,
  }));

  const missing = [...articlePassages, ...productPassages].filter(
    (r) => force || !cacheUsable(r, probe.engine, probe.model),
  );

  if (!missing.length) {
    return {
      engine: probe.engine,
      model: probe.model,
      articlesUpdated: 0,
      productsUpdated: 0,
    };
  }

  const emb = await embedTexts(
    missing.map((m) => m.text),
    { mode: "passage" },
  );
  if (emb.engine === "none" || emb.vectors.length !== missing.length) {
    return {
      engine: emb.engine,
      model: emb.model,
      articlesUpdated: 0,
      productsUpdated: 0,
    };
  }

  const now = new Date();
  const articleIds = new Set(articlePassages.map((a) => a.id));
  let articlesUpdated = 0;
  let productsUpdated = 0;

  await Promise.all(
    missing.map(async (row, i) => {
      const vector = emb.vectors[i]!;
      const data = {
        embeddingB64: vectorToB64(vector),
        embedModel: emb.model,
        embedEngine: emb.engine,
        embeddedAt: now,
      };
      if (articleIds.has(row.id)) {
        await prisma.knowledgeArticle.update({ where: { id: row.id }, data });
        articlesUpdated += 1;
      } else {
        await prisma.product.update({ where: { id: row.id }, data });
        productsUpdated += 1;
      }
    }),
  );

  return {
    engine: emb.engine,
    model: emb.model,
    articlesUpdated,
    productsUpdated,
  };
}

export async function retrieveTwiContext(
  query: string,
  opts?: {
    articleLimit?: number;
    productLimit?: number;
    focus?: "health" | "commerce";
  },
): Promise<TwiRetrieveResult> {
  const focus = opts?.focus ?? "health";
  const articleLimit = opts?.articleLimit ?? (focus === "commerce" ? 0 : 2);
  const productLimit = opts?.productLimit ?? (focus === "commerce" ? 4 : 0);
  const text = (query || "").trim();
  if (!text) {
    return { engine: "none", articles: [], products: [] };
  }

  const [articles, products] = await Promise.all([
    prisma.knowledgeArticle.findMany({
      where: { isActive: true },
      take: 80,
      select: {
        id: true,
        slug: true,
        titleTw: true,
        titleEn: true,
        bodyTw: true,
        bodyEn: true,
        source: true,
        tags: true,
        embeddingB64: true,
        embedModel: true,
        embedEngine: true,
      },
    }),
    prisma.product.findMany({
      where: { isActive: true },
      take: 80,
      select: {
        id: true,
        nameTw: true,
        nameEn: true,
        descriptionTw: true,
        priceGhs: true,
        tags: true,
        embeddingB64: true,
        embedModel: true,
        embedEngine: true,
      },
    }),
  ]);

  if (isEmbedConfigured()) {
    try {
      const q = await embedOne(text, "query");
      if (q?.vector.length) {
        // Resolve passage vectors from cache; embed only misses
        type WithVec<T> = T & { vector: number[] | null };
        const articleRows: WithVec<(typeof articles)[0]>[] = articles.map((a) => ({
          ...a,
          vector:
            a.embeddingB64 && cacheUsable(a, q.engine, q.model)
              ? vectorFromB64(a.embeddingB64)
              : null,
        }));
        const productRows: WithVec<(typeof products)[0]>[] = products.map((p) => ({
          ...p,
          vector:
            p.embeddingB64 && cacheUsable(p, q.engine, q.model)
              ? vectorFromB64(p.embeddingB64)
              : null,
        }));

        const misses: { kind: "article" | "product"; id: string; text: string; idx: number }[] =
          [];
        articleRows.forEach((a, idx) => {
          if (!a.vector) {
            misses.push({
              kind: "article",
              id: a.id,
              text: `${a.titleTw}. ${a.bodyTw}`,
              idx,
            });
          }
        });
        productRows.forEach((p, idx) => {
          if (!p.vector) {
            misses.push({
              kind: "product",
              id: p.id,
              text: `${p.nameTw}. ${p.nameEn}. ${p.descriptionTw ?? ""}`,
              idx,
            });
          }
        });

        if (misses.length) {
          const emb = await embedTexts(
            misses.map((m) => m.text),
            { mode: "passage" },
          );
          if (emb.engine !== "none" && emb.vectors.length === misses.length) {
            const now = new Date();
            await Promise.all(
              misses.map(async (m, i) => {
                const vector = emb.vectors[i]!;
                const data = {
                  embeddingB64: vectorToB64(vector),
                  embedModel: emb.model || q.model,
                  embedEngine: emb.engine,
                  embeddedAt: now,
                };
                if (m.kind === "article") {
                  articleRows[m.idx]!.vector = vector;
                  await prisma.knowledgeArticle.update({ where: { id: m.id }, data });
                } else {
                  productRows[m.idx]!.vector = vector;
                  await prisma.product.update({ where: { id: m.id }, data });
                }
              }),
            );
          }
        }

        const scoredArticles = articleRows
          .filter((a) => a.vector?.length)
          .map((a) => ({
            slug: a.slug,
            titleTw: a.titleTw,
            titleEn: a.titleEn,
            bodyTw: a.bodyTw,
            bodyEn: a.bodyEn,
            source: a.source,
            score: cosineSimilarity(q.vector, a.vector!),
          }))
          .sort((a, b) => b.score - a.score);

        const scoredProducts = productRows
          .filter((p) => p.vector?.length)
          .map((p) => ({
            id: p.id,
            nameTw: p.nameTw,
            nameEn: p.nameEn,
            priceGhs: Number(p.priceGhs),
            score: cosineSimilarity(q.vector, p.vector!),
          }))
          .sort((a, b) => b.score - a.score);

        if (scoredArticles.length || scoredProducts.length) {
          return {
            engine: q.engine === "abena" ? "abena" : "openai",
            model: q.model,
            articles: takeConfident(scoredArticles, articleLimit, q.engine),
            products: takeConfident(scoredProducts, productLimit, q.engine),
            cacheMisses: misses.length,
          };
        }
      }
    } catch (e) {
      console.error("[twi-retrieve]", e);
    }
  }

  // Lexical fallback on Twi fields only
  const scoredA = articles
    .map((a) => ({
      slug: a.slug,
      titleTw: a.titleTw,
      titleEn: a.titleEn,
      bodyTw: a.bodyTw,
      bodyEn: a.bodyEn,
      source: a.source,
      score: lexicalScore(text, `${a.titleTw} ${a.bodyTw} ${a.tags.join(" ")}`),
    }))
    .sort((a, b) => b.score - a.score);

  const scoredP = products
    .map((p) => ({
      id: p.id,
      nameTw: p.nameTw,
      nameEn: p.nameEn,
      priceGhs: Number(p.priceGhs),
      score: lexicalScore(text, `${p.nameTw} ${p.nameEn} ${p.tags.join(" ")}`),
    }))
    .sort((a, b) => b.score - a.score);

  return {
    engine: "lexical",
    articles: scoredA.filter((a) => a.score > 0).slice(0, articleLimit),
    products: scoredP.filter((p) => p.score > 0).slice(0, productLimit),
  };
}

function takeConfident<T extends { score: number }>(
  rows: T[],
  limit: number,
  engine: EmbedEngine,
): T[] {
  if (limit <= 0 || !rows.length) return [];
  const top = rows[0]?.score ?? 0;
  const floor = engine === "abena" ? 0.32 : 0.33;
  const minScore = Math.max(floor, top - 0.055);
  return rows.filter((row) => row.score >= minScore).slice(0, limit);
}
