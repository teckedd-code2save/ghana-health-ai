import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import {
  cosineSimilarity,
  embeddingFromB64,
  stubEmbeddingFromPassphrase,
} from "@/lib/voice-stub";
import { jsonError, jsonOk } from "@/lib/api";

const schema = z.object({
  passphrase: z.string().min(1),
});

export async function POST(req: Request) {
  try {
    const user = await getSessionUser();
    if (!user) return jsonError("Unauthorized", 401);

    const body = schema.parse(await req.json());
    const enrollment = await prisma.voiceEnrollment.findFirst({
      where: { userId: user.id, isActive: true },
    });
    if (!enrollment) return jsonError("No active voice enrollment", 404);

    const probe = stubEmbeddingFromPassphrase(body.passphrase);
    const enrolled = embeddingFromB64(enrollment.embeddingB64);
    const score = cosineSimilarity(probe, enrolled);
    const verified = score >= enrollment.threshold;

    if (verified) {
      await prisma.voiceEnrollment.update({
        where: { id: enrollment.id },
        data: { lastVerifiedAt: new Date() },
      });
    }

    return jsonOk({ verified, score: Number(score.toFixed(4)), threshold: enrollment.threshold });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("Verification failed", 500);
  }
}
