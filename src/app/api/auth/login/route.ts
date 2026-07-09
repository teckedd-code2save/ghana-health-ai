import { z } from "zod";
import { prisma } from "@/db/prisma";
import { createSessionToken, setSessionCookie, verifyPassword } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";

const schema = z.object({
  email: z.string().email().optional(),
  phone: z.string().optional(),
  password: z.string().min(1),
});

export async function POST(req: Request) {
  try {
    const body = schema.parse(await req.json());
    const user = body.email
      ? await prisma.user.findUnique({ where: { email: body.email } })
      : body.phone
        ? await prisma.user.findUnique({ where: { phone: body.phone } })
        : null;

    if (!user?.passwordHash) return jsonError("Invalid credentials", 401);
    const ok = await verifyPassword(body.password, user.passwordHash);
    if (!ok) return jsonError("Invalid credentials", 401);

    const token = await createSessionToken(user.id);
    await setSessionCookie(token);

    return jsonOk({
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
    return jsonError("Login failed", 500);
  }
}
