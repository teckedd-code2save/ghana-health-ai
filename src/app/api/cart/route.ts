import { z } from "zod";
import { prisma } from "@/db/prisma";
import { jsonError, jsonOk } from "@/lib/api";
import { addProductToCart, getOrCreateCart, serializeCart } from "@/lib/cart";

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
    const result = await addProductToCart(body);
    if (!result.ok) return jsonError(result.error, result.status);
    return jsonOk({ cart: result.cart });
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
