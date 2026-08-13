import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";
import type { CommerceUnderstanding } from "@/lib/understand";

export type CommerceExecutionResult = {
  mode: "none" | "local_catalog_search" | "order_draft";
  status: "not_applicable" | "needs_clarification" | "ready" | "no_matches";
  products: Array<{
    id: string;
    sku: string;
    nameEn: string;
    nameTw: string;
    priceGhs: number;
    unit: string;
    stock: number;
  }>;
  draft?: {
    item: string;
    quantity?: string;
    location?: string;
    fulfillment: "delivery" | "pickup" | "unknown";
    requiresConfirmation: true;
  };
  note: string;
};

export async function executeCommercePlan(
  commerce: CommerceUnderstanding | undefined,
): Promise<CommerceExecutionResult | undefined> {
  const plan = commerce?.plan;
  if (!commerce || !plan) return undefined;

  if (plan.status === "needs_clarification") {
    return {
      mode: "none",
      status: "needs_clarification",
      products: [],
      note: plan.messageHint,
    };
  }

  if (plan.nextAction === "search_local_catalog" && plan.searchQuery) {
    const products = await searchLocalCatalog(plan.searchQuery);
    return {
      mode: "local_catalog_search",
      status: products.length ? "ready" : "no_matches",
      products,
      note: products.length
        ? "Matched connected local catalog products only."
        : "No connected local catalog match; do not invent external stores or prices.",
    };
  }

  if (plan.nextAction === "draft_order" && plan.orderDraft) {
    const products = await searchLocalCatalog(plan.orderDraft.item);
    return {
      mode: "order_draft",
      status: "ready",
      products,
      draft: {
        item: plan.orderDraft.item,
        quantity: plan.orderDraft.quantity,
        location: plan.orderDraft.location,
        fulfillment: plan.orderDraft.fulfillment,
        requiresConfirmation: true,
      },
      note: "Order draft only; require explicit user confirmation before cart, checkout, or payment.",
    };
  }

  return {
    mode: "none",
    status: "not_applicable",
    products: [],
    note: plan.messageHint,
  };
}

async function searchLocalCatalog(query: string) {
  const tokens = tokenize(query);
  if (!tokens.length) return [];

  const where: Prisma.ProductWhereInput = {
    isActive: true,
    OR: tokens.flatMap((token) => [
      { nameEn: { contains: token, mode: "insensitive" as const } },
      { nameTw: { contains: token, mode: "insensitive" as const } },
      { descriptionEn: { contains: token, mode: "insensitive" as const } },
      { descriptionTw: { contains: token, mode: "insensitive" as const } },
      { sku: { contains: token, mode: "insensitive" as const } },
      { tags: { has: token.toLowerCase() } },
    ]),
  };

  const rows = await prisma.product.findMany({
    where,
    orderBy: [{ stock: "desc" }, { nameEn: "asc" }],
    take: 5,
  });

  return rows.map((product) => ({
    id: product.id,
    sku: product.sku,
    nameEn: product.nameEn,
    nameTw: product.nameTw,
    priceGhs: Number(product.priceGhs),
    unit: product.unit,
    stock: product.stock,
  }));
}

function tokenize(query: string) {
  const stop = new Set(["a", "an", "the", "of", "for", "me", "ma", "sɛ", "se", "kg"]);
  const raw = query
    .toLowerCase()
    .replace(/[^\wɛɔƐƆŋŊ\s-]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !stop.has(token));
  const synonyms = new Set(raw);
  if (raw.includes("oral") && raw.includes("rehydration")) synonyms.add("ors");
  if (raw.includes("salts")) synonyms.add("ors");
  return Array.from(synonyms).slice(0, 8);
}
