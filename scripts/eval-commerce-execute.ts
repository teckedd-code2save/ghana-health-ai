import "../src/config/load-env";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";
import { executeCommercePlan } from "../src/lib/commerce-execute";
import type { CommerceUnderstanding } from "../src/lib/understand";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const suffix = randomUUID().slice(0, 8);
  const tomatoSku = `TEST-TOMATO-${suffix}`;
  const riceSku = `TEST-RICE-${suffix}`;

  await prisma.product.createMany({
    data: [
      {
        sku: tomatoSku,
        nameEn: `Tomato ${suffix}`,
        nameTw: `Tomato ${suffix}`,
        category: "produce",
        priceGhs: 18.5,
        unit: "kg",
        stock: 12,
        tags: ["tomato", "tomatoes", "produce", suffix],
      },
      {
        sku: riceSku,
        nameEn: `Rice ${suffix}`,
        nameTw: `Ɛmo ${suffix}`,
        category: "staples",
        priceGhs: 90,
        unit: "bag",
        stock: 6,
        tags: ["rice", "ɛmo", "staples", suffix],
      },
    ],
  });

  try {
    const searchCommerce: CommerceUnderstanding = {
      action: "find",
      item: `tomato ${suffix}`,
      location: "Madina",
      fulfillment: "unknown",
      confidence: 0.9,
      source: "deterministic",
      plan: {
        status: "ready_for_search",
        nextAction: "search_local_catalog",
        missing: [],
        searchQuery: `tomato ${suffix}`,
        messageHint: "Search local catalog",
      },
    };

    const search = await executeCommercePlan(searchCommerce);
    assertOk(search?.mode === "local_catalog_search", `expected search mode, got ${search?.mode}`);
    assertOk(search.status === "ready", `expected ready search, got ${search.status}`);
    assertOk(
      search.products.some((product) => product.sku === tomatoSku && product.priceGhs === 18.5),
      "expected tomato match with catalog price",
    );

    const draftCommerce: CommerceUnderstanding = {
      action: "order",
      item: `rice ${suffix}`,
      quantity: "1 bag",
      location: "Adenta",
      fulfillment: "delivery",
      confidence: 0.92,
      source: "deterministic",
      plan: {
        status: "ready_for_order_draft",
        nextAction: "draft_order",
        missing: [],
        searchQuery: `rice ${suffix}`,
        orderDraft: {
          item: `rice ${suffix}`,
          quantity: "1 bag",
          location: "Adenta",
          fulfillment: "delivery",
        },
        messageHint: "Draft order",
      },
    };

    const draft = await executeCommercePlan(draftCommerce);
    assertOk(draft?.mode === "order_draft", `expected draft mode, got ${draft?.mode}`);
    assertOk(draft.draft?.requiresConfirmation === true, "draft must require confirmation");
    assertOk(draft.draft.item === `rice ${suffix}`, "draft item mismatch");
    assertOk(draft.products.some((product) => product.sku === riceSku), "expected rice match");

    const cartItems = await prisma.cartItem.findMany({
      where: { productId: { in: search.products.concat(draft.products).map((product) => product.id) } },
    });
    assertOk(cartItems.length === 0, "commerce execution must not mutate cart");

    console.log("ok commerce-execute");
  } finally {
    await prisma.product.deleteMany({ where: { sku: { in: [tomatoSku, riceSku] } } });
    await prisma.$disconnect();
  }
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

export {};
