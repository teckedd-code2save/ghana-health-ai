import crypto from "node:crypto";
import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { sendLoginCode } from "@/lib/email";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  email: z.string().email(),
});

function hashCode(email: string, code: string) {
  return crypto
    .createHash("sha256")
    .update(`${email.toLowerCase()}:${code}:${getEnv().JWT_SECRET}`)
    .digest("hex");
}

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`otp:${ip}`, 8, 60);
    if (!rl.allowed) return jsonError("Too many code requests. Try again in a minute.", 429);

    const body = schema.parse(await req.json());
    const email = body.email.toLowerCase();
    const code = String(crypto.randomInt(100000, 1000000));
    const user = await prisma.user.findUnique({ where: { email }, select: { id: true } });

    await prisma.emailOtp.create({
      data: {
        email,
        codeHash: hashCode(email, code),
        expiresAt: new Date(Date.now() + 10 * 60 * 1000),
        userId: user?.id,
      },
    });

    await sendLoginCode(email, code);
    return jsonOk({ ok: true });
  } catch (error) {
    if (error instanceof z.ZodError) return jsonError(error.issues[0]?.message ?? "Invalid email");
    const message = error instanceof Error ? error.message : "Could not send sign-in code";
    return jsonError(message, message.includes("configured") ? 503 : 500);
  }
}
