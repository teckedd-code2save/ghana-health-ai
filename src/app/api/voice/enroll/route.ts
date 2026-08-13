import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import {
  embeddingFromPcm,
  embeddingToB64,
  pcmFromB64,
} from "@/lib/voice-embed";
import { jsonCreated, jsonError, jsonOk } from "@/lib/api";
import { writeAudit } from "@/lib/audit";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  /** Base64 little-endian float32 mono PCM (preferred — from Web Audio). */
  pcmB64: z.string().min(100),
  sampleRate: z.number().int().positive().default(16000),
  language: z.enum(["tw", "en", "ga"]).default("tw"),
  sampleDurationS: z.number().positive().optional(),
  phraseHint: z.string().max(80).optional(),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice-enroll:${ip}`, 10, 60);
    if (!rl.allowed) return jsonError("Too many enroll attempts", 429);

    const user = await getSessionUser();
    if (!user) return jsonError("Login required for Voice ID enrollment", 401);
    if (!user.consentVoice) {
      return jsonError("Voice consent required before enrollment", 403);
    }

    const body = schema.parse(await req.json());
    const samples = pcmFromB64(body.pcmB64);
    const embedding = embeddingFromPcm(samples, body.sampleRate);

    await prisma.voiceEnrollment.updateMany({
      where: { userId: user.id, isActive: true },
      data: { isActive: false },
    });

    const enrollment = await prisma.voiceEnrollment.create({
      data: {
        userId: user.id,
        embeddingB64: embeddingToB64(embedding),
        passphraseHint: body.phraseHint
          ? body.phraseHint.slice(0, 40) + (body.phraseHint.length > 40 ? "…" : "")
          : "audio-enrollment",
        language: body.language,
        sampleDurationS: body.sampleDurationS ?? samples.length / body.sampleRate,
        isActive: true,
        threshold: 0.82,
      },
    });

    await writeAudit({
      action: "voice.enroll",
      actorId: user.id,
      entityType: "voice_enrollment",
      entityId: enrollment.id,
      ip,
      meta: { dims: embedding.length, durationS: enrollment.sampleDurationS },
    });

    return jsonCreated({
      enrollment: {
        id: enrollment.id,
        language: enrollment.language,
        enrolledAt: enrollment.enrolledAt,
        threshold: enrollment.threshold,
        mode: "audio-pcm",
      },
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    if (e instanceof Error && /too short|Not enough voiced/i.test(e.message)) {
      return jsonError(e.message, 400);
    }
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
      sampleDurationS: true,
    },
  });
  return jsonOk({ enrollment });
}
