import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { embeddingToB64, stubEmbeddingFromPassphrase } from "@/lib/voice-stub";
import { jsonCreated, jsonError, jsonOk } from "@/lib/api";

const schema = z.object({
  passphrase: z.string().min(8).max(200),
  language: z.enum(["tw", "en", "ga", "ee", "dag"]).default("tw"),
  sampleDurationS: z.number().positive().optional(),
});

export async function POST(req: Request) {
  try {
    const user = await getSessionUser();
    if (!user) return jsonError("Login required for Voice ID enrollment", 401);
    if (!user.consentVoice) {
      return jsonError("Voice consent required before enrollment", 403);
    }

    const body = schema.parse(await req.json());
    const embedding = stubEmbeddingFromPassphrase(body.passphrase);

    // Deactivate previous enrollments
    await prisma.voiceEnrollment.updateMany({
      where: { userId: user.id, isActive: true },
      data: { isActive: false },
    });

    const enrollment = await prisma.voiceEnrollment.create({
      data: {
        userId: user.id,
        embeddingB64: embeddingToB64(embedding),
        passphraseHint: body.passphrase.slice(0, 12) + "…",
        language: body.language,
        sampleDurationS: body.sampleDurationS ?? 30,
        isActive: true,
      },
    });

    return jsonCreated({
      enrollment: {
        id: enrollment.id,
        language: enrollment.language,
        enrolledAt: enrollment.enrolledAt,
        threshold: enrollment.threshold,
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Enrollment failed", 500);
  }
}

export async function GET() {
  const user = await getSessionUser();
  if (!user) return jsonError("Unauthorized", 401);
  const enrollment = await prisma.voiceEnrollment.findFirst({
    where: { userId: user.id, isActive: true },
    select: {
      id: true,
      language: true,
      enrolledAt: true,
      lastVerifiedAt: true,
      threshold: true,
      passphraseHint: true,
    },
  });
  return jsonOk({ enrollment });
}
