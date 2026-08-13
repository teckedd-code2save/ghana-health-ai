import { z } from "zod";
import { addProductToCart } from "@/lib/cart";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  confirm: z.literal(true),
  productId: z.string().uuid(),
  quantity: z.number().int().min(1).max(99).default(1),
  source: z.enum(["voice", "chat", "manual"]).default("voice"),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`commerce-confirm:${ip}`, 20, 60);
    if (!rl.allowed) return jsonError("Too many commerce confirmations — wait a minute", 429);

    const body = schema.parse(await req.json());
    const result = await addProductToCart({
      productId: body.productId,
      quantity: body.quantity,
    });
    if (!result.ok) return jsonError(result.error, result.status);

    return jsonOk({
      ok: true,
      action: "cart_add_confirmed",
      source: body.source,
      cart: result.cart,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return jsonError(error.issues[0]?.message ?? "Invalid confirmation");
    }
    console.error("[commerce/confirm]", error);
    return jsonError("Commerce confirmation failed", 500);
  }
}
