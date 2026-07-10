import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import {
  cosineSimilarity,
  embeddingFromB64,
  embeddingFromPcm,
  pcmFromB64,
} from "@/lib/voice-embed";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";

const schema = z.object({
  pcmB64: z.string().min(100),
  sampleRate: z.number().int().positive().default(16000),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice-verify:${ip}`, 20, 60);
    if (!rl.allowed) return jsonError("Too many verify attempts", 429);

    const user = await getSessionUser();
    if (!user) return jsonError("Unauthorized", 401);

    const body = schema.parse(await req.json());
    const enrollment = await prisma.voiceEnrollment.findFirst({
      where: { userId: user.id, isActive: true },
    });
    if (!enrollment) return jsonError("No active voice enrollment", 404);

    const samples = pcmFromB64(body.pcmB64);
    const probe = embeddingFromPcm(samples, body.sampleRate);
    const enrolled = embeddingFromB64(enrollment.embeddingB64);
    const score = cosineSimilarity(probe, enrolled);
    const verified = score >= enrollment.threshold;

    if (verified) {
      await prisma.voiceEnrollment.update({
        where: { id: enrollment.id },
        data: { lastVerifiedAt: new Date() },
      });
    }

    await writeAudit({
      action: "voice.verify",
      actorId: user.id,
      entityType: "voice_enrollment",
      entityId: enrollment.id,
      ip,
      meta: { verified, score: Number(score.toFixed(4)) },
    });

    return jsonOk({
      verified,
      score: Number(score.toFixed(4)),
      threshold: enrollment.threshold,
      mode: "audio-pcm",
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    if (e instanceof Error && /too short|Not enough voiced/i.test(e.message)) {
      return jsonError(e.message, 400);
    }
    console.error(e);
    return jsonError("Verification failed", 500);
  }
}
