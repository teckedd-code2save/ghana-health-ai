/**
 * Twi-native retrieval via ABENA embeddings (Ghana-NLP research path).
 *
 * Embeds the ASR/transcript query and ranks KnowledgeArticle / Product
 * rows by cosine similarity on **Twi** text fields — not English dumps,
 * not a hand-written semantic bank.
 */

import { prisma } from "@/db/prisma";
import { cosineSimilarity, embedOne, embedTexts, isEmbedConfigured } from "@/lib/embed";

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

export async function retrieveTwiContext(
  query: string,
  opts?: { articleLimit?: number; productLimit?: number },
): Promise<TwiRetrieveResult> {
  const articleLimit = opts?.articleLimit ?? 4;
  const productLimit = opts?.productLimit ?? 4;
  const text = (query || "").trim();
  if (!text) {
    return { engine: "none", articles: [], products: [] };
  }

  const [articles, products] = await Promise.all([
    prisma.knowledgeArticle.findMany({ where: { isActive: true }, take: 40 }),
    prisma.product.findMany({ where: { isActive: true }, take: 40 }),
  ]);

  if (isEmbedConfigured()) {
    try {
      const q = await embedOne(text, "query");
      if (q?.vector.length) {
        const passageTexts = [
          ...articles.map((a) => `${a.titleTw}. ${a.bodyTw}`),
          ...products.map((p) => `${p.nameTw}. ${p.nameEn}. ${p.descriptionTw ?? ""}`),
        ];
        const emb = await embedTexts(passageTexts, { mode: "passage" });
        if (emb.engine !== "none" && emb.vectors.length === passageTexts.length) {
          const scoredArticles = articles.map((a, i) => ({
            slug: a.slug,
            titleTw: a.titleTw,
            titleEn: a.titleEn,
            bodyTw: a.bodyTw,
            bodyEn: a.bodyEn,
            source: a.source,
            score: cosineSimilarity(q.vector, emb.vectors[i]!),
          }));
          scoredArticles.sort((a, b) => b.score - a.score);

          const offset = articles.length;
          const scoredProducts = products.map((p, i) => ({
            id: p.id,
            nameTw: p.nameTw,
            nameEn: p.nameEn,
            priceGhs: Number(p.priceGhs),
            score: cosineSimilarity(q.vector, emb.vectors[offset + i]!),
          }));
          scoredProducts.sort((a, b) => b.score - a.score);

          return {
            engine: emb.engine === "abena" ? "abena" : "openai",
            model: emb.model,
            articles: scoredArticles.slice(0, articleLimit).filter((a) => a.score > 0.25),
            products: scoredProducts.slice(0, productLimit).filter((p) => p.score > 0.25),
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
