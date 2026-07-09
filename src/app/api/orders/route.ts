import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonCreated, jsonError, jsonOk } from "@/lib/api";
import { cookies } from "next/headers";

const schema = z.object({
  paymentMethod: z.enum(["MOMO", "USSD", "CASH", "CARD", "MOCK"]).default("MOCK"),
  phone: z.string().optional(),
  notes: z.string().max(500).optional(),
});

export async function GET() {
  const user = await getSessionUser();
  if (!user) return jsonError("Unauthorized", 401);
  const orders = await prisma.order.findMany({
    where: { userId: user.id },
    include: { items: true },
    orderBy: { createdAt: "desc" },
    take: 20,
  });
  return jsonOk({
    orders: orders.map((o) => ({
      ...o,
      totalGhs: Number(o.totalGhs),
      items: o.items.map((i) => ({ ...i, unitPrice: Number(i.unitPrice) })),
    })),
  });
}

export async function POST(req: Request) {
  try {
    const body = schema.parse(await req.json());
    const user = await getSessionUser();
    const jar = await cookies();
    const sessionId = jar.get("gha_cart_session")?.value;

    const cart = await prisma.cart.findFirst({
      where: user ? { userId: user.id } : { sessionId: sessionId ?? "__none__" },
      include: { items: { include: { product: true } } },
    });

    if (!cart || cart.items.length === 0) return jsonError("Cart is empty");

    for (const item of cart.items) {
      if (item.product.stock < item.quantity) {
        return jsonError(`Insufficient stock for ${item.product.nameEn}`);
      }
    }

    const total = cart.items.reduce(
      (s, i) => s + Number(i.product.priceGhs) * i.quantity,
      0,
    );

    const order = await prisma.$transaction(async (tx) => {
      const created = await tx.order.create({
        data: {
          userId: user?.id,
          status: body.paymentMethod === "MOCK" ? "PAID" : "AWAITING_PAYMENT",
          paymentMethod: body.paymentMethod,
          totalGhs: total,
          phone: body.phone ?? user?.phone,
          notes: body.notes,
          momoRef: body.paymentMethod === "MOCK" ? `MOCK-${Date.now()}` : null,
          items: {
            create: cart.items.map((i) => ({
              productId: i.productId,
              quantity: i.quantity,
              unitPrice: i.product.priceGhs,
              nameSnap: i.product.nameEn,
            })),
          },
        },
        include: { items: true },
      });

      for (const item of cart.items) {
        await tx.product.update({
          where: { id: item.productId },
          data: { stock: { decrement: item.quantity } },
        });
      }

      await tx.cartItem.deleteMany({ where: { cartId: cart.id } });
      return created;
    });

    return jsonCreated({
      order: {
        ...order,
        totalGhs: Number(order.totalGhs),
        items: order.items.map((i) => ({ ...i, unitPrice: Number(i.unitPrice) })),
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Checkout failed", 500);
  }
}
