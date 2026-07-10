import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonCreated, jsonError, jsonOk } from "@/lib/api";
import { cookies } from "next/headers";
import { initializePayment, isPaystackConfigured } from "@/lib/paystack";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";
import { randomUUID } from "node:crypto";

const schema = z.object({
  paymentMethod: z.enum(["MOMO", "USSD", "CASH", "CARD", "MOCK"]).default("MOMO"),
  phone: z.string().optional(),
  email: z.string().email().optional(),
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
    const ip = clientIp(req);
    const rl = rateLimit(`order:${ip}`, 15, 60);
    if (!rl.allowed) return jsonError("Too many checkout attempts", 429);

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

    const usePaystack =
      (body.paymentMethod === "MOMO" || body.paymentMethod === "CARD") &&
      isPaystackConfigured();

    // MOCK always, or CASH, or Paystack not configured
    if (!usePaystack) {
      const order = await prisma.$transaction(async (tx) => {
        const created = await tx.order.create({
          data: {
            userId: user?.id,
            status: body.paymentMethod === "CASH" ? "AWAITING_PAYMENT" : "PAID",
            paymentMethod: body.paymentMethod === "MOMO" ? "MOCK" : body.paymentMethod,
            totalGhs: total,
            phone: body.phone ?? user?.phone,
            notes: body.notes,
            momoRef:
              body.paymentMethod === "CASH" ? null : `MOCK-${Date.now()}`,
            paidAt: body.paymentMethod === "CASH" ? null : new Date(),
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

      await writeAudit({
        action: "order.mock_checkout",
        actorId: user?.id,
        entityType: "order",
        entityId: order.id,
        ip,
        meta: { totalGhs: total, method: body.paymentMethod },
      });

      return jsonCreated({
        order: {
          ...order,
          totalGhs: Number(order.totalGhs),
          items: order.items.map((i) => ({ ...i, unitPrice: Number(i.unitPrice) })),
        },
        payment: { mode: "mock", authorizationUrl: null },
      });
    }

    // Paystack MoMo / card
    const reference = `gha_${randomUUID().replace(/-/g, "").slice(0, 20)}`;
    const email =
      body.email || user?.email || `${(body.phone || "guest").replace(/\W/g, "")}@ghanahealth.local`;
    const appUrl = process.env.NEXT_PUBLIC_APP_URL || "https://ghanahealth.serendepify.com";

    const order = await prisma.$transaction(async (tx) => {
      return tx.order.create({
        data: {
          userId: user?.id,
          status: "AWAITING_PAYMENT",
          paymentMethod: body.paymentMethod,
          totalGhs: total,
          phone: body.phone ?? user?.phone,
          notes: body.notes,
          paystackRef: reference,
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
    });

    const init = await initializePayment({
      email,
      amount: Math.round(total * 100),
      reference,
      callback_url: `${appUrl}/market?paid=${order.id}`,
      metadata: {
        orderId: order.id,
        phone: body.phone ?? user?.phone,
        custom_fields: [
          { display_name: "Order", variable_name: "order_id", value: order.id },
        ],
      },
      channels: body.paymentMethod === "MOMO" ? ["mobile_money", "ussd"] : ["card", "mobile_money"],
    });

    if (!init.status || !init.data?.authorization_url) {
      await prisma.order.update({
        where: { id: order.id },
        data: { status: "CANCELLED", notes: `Paystack init failed: ${init.message}` },
      });
      return jsonError(init.message || "Payment provider rejected the request", 502);
    }

    await prisma.order.update({
      where: { id: order.id },
      data: { authorizationUrl: init.data.authorization_url, momoRef: reference },
    });

    // Clear cart only after successful init
    await prisma.cartItem.deleteMany({ where: { cartId: cart.id } });
    // Reserve stock
    for (const item of cart.items) {
      await prisma.product.update({
        where: { id: item.productId },
        data: { stock: { decrement: item.quantity } },
      });
    }

    await writeAudit({
      action: "order.paystack_init",
      actorId: user?.id,
      entityType: "order",
      entityId: order.id,
      ip,
      meta: { reference, totalGhs: total },
    });

    return jsonCreated({
      order: {
        ...order,
        authorizationUrl: init.data.authorization_url,
        totalGhs: Number(order.totalGhs),
        items: order.items.map((i) => ({ ...i, unitPrice: Number(i.unitPrice) })),
      },
      payment: {
        mode: "paystack",
        authorizationUrl: init.data.authorization_url,
        reference,
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Checkout failed", 500);
  }
}
