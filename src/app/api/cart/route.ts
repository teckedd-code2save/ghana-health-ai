import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { cookies } from "next/headers";

const CART_COOKIE = "gha_cart_session";

async function getOrCreateCart() {
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

function serializeCart(cart: Awaited<ReturnType<typeof getOrCreateCart>>) {
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

export async function GET() {
  const cart = await getOrCreateCart();
  return jsonOk({ cart: serializeCart(cart) });
}

const addSchema = z.object({
  productId: z.string().uuid(),
  quantity: z.number().int().min(1).max(99).default(1),
});

export async function POST(req: Request) {
  try {
    const body = addSchema.parse(await req.json());
    const cart = await getOrCreateCart();
    const product = await prisma.product.findFirst({
      where: { id: body.productId, isActive: true },
    });
    if (!product) return jsonError("Product not found", 404);
    if (product.stock < body.quantity) return jsonError("Insufficient stock");

    await prisma.cartItem.upsert({
      where: { cartId_productId: { cartId: cart.id, productId: body.productId } },
      create: { cartId: cart.id, productId: body.productId, quantity: body.quantity },
      update: { quantity: { increment: body.quantity } },
    });

    const updated = await prisma.cart.findUniqueOrThrow({
      where: { id: cart.id },
      include: { items: { include: { product: true } } },
    });
    return jsonOk({ cart: serializeCart(updated) });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Cart update failed", 500);
  }
}

const patchSchema = z.object({
  itemId: z.string().uuid(),
  quantity: z.number().int().min(0).max(99),
});

export async function PATCH(req: Request) {
  try {
    const body = patchSchema.parse(await req.json());
    const cart = await getOrCreateCart();
    const item = await prisma.cartItem.findFirst({
      where: { id: body.itemId, cartId: cart.id },
    });
    if (!item) return jsonError("Item not found", 404);

    if (body.quantity === 0) {
      await prisma.cartItem.delete({ where: { id: item.id } });
    } else {
      await prisma.cartItem.update({
        where: { id: item.id },
        data: { quantity: body.quantity },
      });
    }

    const updated = await prisma.cart.findUniqueOrThrow({
      where: { id: cart.id },
      include: { items: { include: { product: true } } },
    });
    return jsonOk({ cart: serializeCart(updated) });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Cart update failed", 500);
  }
}
