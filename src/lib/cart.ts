import { cookies } from "next/headers";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";

const CART_COOKIE = "gha_cart_session";

export async function getOrCreateCart() {
  const user = await getSessionUser();
  const jar = await cookies();
  let sessionId = jar.get(CART_COOKIE)?.value;

  if (user) {
    let cart = await prisma.cart.findFirst({
      where: { userId: user.id },
      include: { items: { include: { product: true } } },
    });
    if (!cart) {
      cart = await prisma.cart.create({
        data: { userId: user.id },
        include: { items: { include: { product: true } } },
      });
    }
    return cart;
  }

  if (!sessionId) {
    sessionId = crypto.randomUUID();
    jar.set(CART_COOKIE, sessionId, {
      httpOnly: true,
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 14,
    });
  }

  let cart = await prisma.cart.findFirst({
    where: { sessionId },
    include: { items: { include: { product: true } } },
  });
  if (!cart) {
    cart = await prisma.cart.create({
      data: { sessionId },
      include: { items: { include: { product: true } } },
    });
  }
  return cart;
}

export function serializeCart(cart: Awaited<ReturnType<typeof getOrCreateCart>>) {
  const items = cart.items.map((i) => ({
    id: i.id,
    productId: i.productId,
    quantity: i.quantity,
    product: {
      id: i.product.id,
      sku: i.product.sku,
      nameEn: i.product.nameEn,
      nameTw: i.product.nameTw,
      priceGhs: Number(i.product.priceGhs),
      unit: i.product.unit,
    },
    lineTotal: Number(i.product.priceGhs) * i.quantity,
  }));
  const totalGhs = items.reduce((s, i) => s + i.lineTotal, 0);
  return { id: cart.id, items, totalGhs };
}

export async function addProductToCart(input: { productId: string; quantity: number }) {
  const cart = await getOrCreateCart();
  const product = await prisma.product.findFirst({
    where: { id: input.productId, isActive: true },
  });
  if (!product) {
    return { ok: false as const, error: "Product not found", status: 404 };
  }
  if (product.stock < input.quantity) {
    return { ok: false as const, error: "Insufficient stock", status: 400 };
  }

  await prisma.cartItem.upsert({
    where: { cartId_productId: { cartId: cart.id, productId: input.productId } },
    create: { cartId: cart.id, productId: input.productId, quantity: input.quantity },
    update: { quantity: { increment: input.quantity } },
  });

  const updated = await prisma.cart.findUniqueOrThrow({
    where: { id: cart.id },
    include: { items: { include: { product: true } } },
  });
  return { ok: true as const, cart: serializeCart(updated) };
}
