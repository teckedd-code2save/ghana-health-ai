import { z } from "zod";
import { prisma } from "@/db/prisma";
import { createSessionToken, hashPassword, setSessionCookie } from "@/lib/auth";
import { jsonCreated, jsonError } from "@/lib/api";

const schema = z.object({
  email: z.string().email().optional(),
  phone: z.string().min(9).optional(),
  password: z.string().min(8),
  displayName: z.string().min(1).max(80).optional(),
  preferredLang: z.enum(["tw", "en", "ga"]).optional(),
  consentVoice: z.boolean().optional(),
  consentHealth: z.boolean().optional(),
});

export async function POST(req: Request) {
  try {
    const body = schema.parse(await req.json());
    if (!body.email && !body.phone) {
      return jsonError("Email or phone required");
    }

    if (body.email) {
      const exists = await prisma.user.findUnique({ where: { email: body.email } });
      if (exists) return jsonError("Email already registered", 409);
    }
    if (body.phone) {
      const exists = await prisma.user.findUnique({ where: { phone: body.phone } });
      if (exists) return jsonError("Phone already registered", 409);
    }

    const user = await prisma.user.create({
      data: {
        email: body.email,
        phone: body.phone,
        displayName: body.displayName,
        preferredLang: body.preferredLang ?? "tw",
        passwordHash: await hashPassword(body.password),
        consentVoice: body.consentVoice ?? false,
        consentHealth: body.consentHealth ?? false,
      },
    });

    const consents = [];
    if (body.consentVoice) consents.push({ userId: user.id, kind: "voice", granted: true });
    if (body.consentHealth) consents.push({ userId: user.id, kind: "health", granted: true });
    if (consents.length) await prisma.consentRecord.createMany({ data: consents });

    const token = await createSessionToken(user.id);
    await setSessionCookie(token);

    return jsonCreated({
      user: {
        id: user.id,
        email: user.email,
        phone: user.phone,
        displayName: user.displayName,
        preferredLang: user.preferredLang,
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Registration failed", 500);
  }
}
