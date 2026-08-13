import "../src/config/load-env";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");
const isLocalBase = ["localhost", "127.0.0.1"].includes(new URL(baseUrl).hostname);

type ProductResponse = {
  products?: Array<{
    id?: string;
    stock?: number;
    nameEn?: string;
  }>;
};

async function resolveProductId() {
  if (isLocalBase) {
    const suffix = randomUUID().slice(0, 8);
    const product = await prisma.product.create({
      data: {
        sku: `TEST-CONFIRM-${suffix}`,
        nameEn: `Confirm Tomato ${suffix}`,
        nameTw: `Confirm Tomato ${suffix}`,
        category: "produce",
        priceGhs: 22,
        unit: "kg",
        stock: 10,
        tags: ["tomato", "confirm", suffix],
      },
    });
    return { productId: product.id, cleanup: true };
  }

  const res = await fetch(`${baseUrl}/api/products?q=paracetamol`);
  const text = await res.text();
  assertOk(res.ok, `products HTTP ${res.status}: ${text}`);
  const body = JSON.parse(text) as ProductResponse;
  const product = body.products?.find((item) => item.id && (item.stock ?? 0) >= 2);
  assertOk(product?.id, "no public product available for confirmation contract");
  return { productId: product.id, cleanup: false };
}

async function main() {
  const product = await resolveProductId();

  try {
    const rejected = await fetch(`${baseUrl}/api/commerce/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        productId: product.productId,
        quantity: 2,
        source: "voice",
      }),
    });
    assertOk(!rejected.ok, "confirmation endpoint must reject missing confirm=true");

    if (isLocalBase) {
      const cartItems = await prisma.cartItem.findMany({ where: { productId: product.productId } });
      assertOk(cartItems.length === 0, "rejected confirmation must not mutate cart");
    }

    const confirmed = await fetch(`${baseUrl}/api/commerce/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: true,
        productId: product.productId,
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
      body.cart?.items?.some((item) => item.productId === product.productId && item.quantity === 2),
      "confirmed cart response missing product",
    );

    if (isLocalBase) {
      const cartItems = await prisma.cartItem.findMany({ where: { productId: product.productId } });
      assertOk(cartItems.length === 1, `expected one cart item, got ${cartItems.length}`);
      assertOk(cartItems[0]?.quantity === 2, `expected quantity 2, got ${cartItems[0]?.quantity}`);
    }

    console.log(`ok commerce-confirm-contract base=${baseUrl}`);
  } finally {
    if (isLocalBase) {
      await prisma.cartItem.deleteMany({ where: { productId: product.productId } });
      if (product.cleanup) {
        await prisma.product.delete({ where: { id: product.productId } }).catch(() => undefined);
      }
      await prisma.$disconnect();
    }
  }
}

main().catch(async (error) => {
  console.error(error);
  if (isLocalBase) {
    await prisma.$disconnect();
  }
  process.exit(1);
});

export {};
