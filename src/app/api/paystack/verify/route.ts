import { z } from "zod";
import { prisma } from "@/db/prisma";
import { verifyPayment } from "@/lib/paystack";
import { jsonError, jsonOk } from "@/lib/api";
import { writeAudit } from "@/lib/audit";

const schema = z.object({
  reference: z.string().min(3),
});

export async function POST(req: Request) {
  try {
    const body = schema.parse(await req.json());
    const verified = await verifyPayment(body.reference);
    if (!verified.status || verified.data?.status !== "success") {
      return jsonError("Payment not successful", 400, { data: verified.data });
    }

    const order = await prisma.order.findFirst({
      where: { paystackRef: body.reference },
      include: { items: true },
    });
    if (!order) return jsonError("Order not found", 404);

    if (order.status !== "PAID") {
      await prisma.order.update({
        where: { id: order.id },
        data: { status: "PAID", paidAt: new Date(), momoRef: body.reference },
      });
      await writeAudit({
        action: "order.verified",
        entityType: "order",
        entityId: order.id,
        meta: { reference: body.reference },
      });
    }

    return jsonOk({
      order: {
        ...order,
        status: "PAID",
        totalGhs: Number(order.totalGhs),
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Verify failed", 500);
  }
}
