import "../src/config/load-env";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");

async function main() {
  const suffix = randomUUID().slice(0, 8);
  const sku = `TEST-CONFIRM-${suffix}`;
  const product = await prisma.product.create({
    data: {
      sku,
      nameEn: `Confirm Tomato ${suffix}`,
      nameTw: `Confirm Tomato ${suffix}`,
      category: "produce",
      priceGhs: 22,
      unit: "kg",
      stock: 10,
      tags: ["tomato", "confirm", suffix],
    },
  });

  try {
    const rejected = await fetch(`${baseUrl}/api/commerce/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        productId: product.id,
        quantity: 2,
        source: "voice",
      }),
    });
    assertOk(!rejected.ok, "confirmation endpoint must reject missing confirm=true");

    let cartItems = await prisma.cartItem.findMany({ where: { productId: product.id } });
    assertOk(cartItems.length === 0, "rejected confirmation must not mutate cart");

    const confirmed = await fetch(`${baseUrl}/api/commerce/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: true,
        productId: product.id,
        quantity: 2,
        source: "voice",
      }),
    });
    const bodyText = await confirmed.text();
    assertOk(confirmed.ok, `confirmation HTTP ${confirmed.status}: ${bodyText}`);
    const body = JSON.parse(bodyText) as {
      action?: string;
      cart?: { items?: Array<{ productId?: string; quantity?: number }> };
    };
    assertOk(body.action === "cart_add_confirmed", `unexpected action ${body.action}`);
    assertOk(
      body.cart?.items?.some((item) => item.productId === product.id && item.quantity === 2),
      "confirmed cart response missing product",
    );

    cartItems = await prisma.cartItem.findMany({ where: { productId: product.id } });
    assertOk(cartItems.length === 1, `expected one cart item, got ${cartItems.length}`);
    assertOk(cartItems[0]?.quantity === 2, `expected quantity 2, got ${cartItems[0]?.quantity}`);

    console.log("ok commerce-confirm-contract");
  } finally {
    await prisma.cartItem.deleteMany({ where: { productId: product.id } });
    await prisma.product.delete({ where: { id: product.id } }).catch(() => undefined);
    await prisma.$disconnect();
  }
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

export {};
