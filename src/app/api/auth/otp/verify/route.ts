import crypto from "node:crypto";
import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getEnv } from "@/config/env";
import { createSessionToken, setSessionCookie } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  email: z.string().email(),
  code: z.string().regex(/^\d{6}$/),
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
    const rl = rateLimit(`otp:verify:${ip}`, 12, 60);
    if (!rl.allowed) return jsonError("Too many attempts. Try again in a minute.", 429);

    const body = schema.parse(await req.json());
    const email = body.email.toLowerCase();
    const otp = await prisma.emailOtp.findFirst({
      where: {
        email,
        codeHash: hashCode(email, body.code),
        usedAt: null,
        expiresAt: { gt: new Date() },
      },
      orderBy: { createdAt: "desc" },
    });
    if (!otp) return jsonError("Invalid or expired code", 401);

    const user = await prisma.user.upsert({
      where: { email },
      update: {},
      create: {
        email,
        preferredLang: "tw",
        consentHealth: false,
        consentVoice: false,
      },
    });

    await prisma.emailOtp.update({
      where: { id: otp.id },
      data: { usedAt: new Date(), userId: user.id },
    });

    const token = await createSessionToken(user.id);
    await setSessionCookie(token);
    return jsonOk({ user: { id: user.id, email: user.email, preferredLang: user.preferredLang } });
  } catch (error) {
    if (error instanceof z.ZodError) return jsonError(error.issues[0]?.message ?? "Invalid code");
    console.error("[auth/otp/verify]", error);
    return jsonError("Could not verify code", 500);
  }
}
