import { prisma } from "@/db/prisma";
import { verifyPayment, verifyWebhookSignature } from "@/lib/paystack";
import { writeAudit } from "@/lib/audit";
import { jsonError, jsonOk } from "@/lib/api";

export async function POST(req: Request) {
  const raw = await req.text();
  const signature = req.headers.get("x-paystack-signature");

  if (!verifyWebhookSignature(raw, signature)) {
    return jsonError("Invalid signature", 401);
  }

  let event: {
    event?: string;
    data?: { reference?: string; status?: string; amount?: number };
  };
  try {
    event = JSON.parse(raw);
  } catch {
    return jsonError("Invalid JSON", 400);
  }

  if (event.event === "charge.success" && event.data?.reference) {
    const reference = event.data.reference;
    const verified = await verifyPayment(reference);
    if (verified.status && verified.data?.status === "success") {
      const order = await prisma.order.findFirst({ where: { paystackRef: reference } });
      if (order && order.status !== "PAID") {
        await prisma.order.update({
          where: { id: order.id },
          data: {
            status: "PAID",
            paidAt: new Date(),
            momoRef: reference,
          },
        });
        await writeAudit({
          action: "order.paid",
          entityType: "order",
          entityId: order.id,
          meta: { reference, amount: verified.data.amount },
        });
      }
    }
  }

  return jsonOk({ received: true });
}
