import { createHash } from "node:crypto";
import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { rateLimit } from "@/lib/rate-limit";

const schema = z.object({ conversationId: z.string().uuid(), messageId: z.string().uuid(),
  assessment: z.enum(["useful", "incorrect", "unsure"]), note: z.string().trim().max(4000).default("") }).strict();

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user) return jsonError("Sign in to review research replies.", 401);
  if (!rateLimit(`r1-feedback:${user.id}`, 40, 60).allowed) return jsonError("Too many reviews. Please wait a minute.", 429);
  try {
    const body = schema.parse(await request.json());
    const message = await prisma.message.findFirst({ where: { id: body.messageId,
      conversationId: body.conversationId, role: "ASSISTANT", conversation: { userId: user.id } } });
    const metadata = message?.metadata as { research?: { executionSha?: string; variant?: string; complete?: boolean } } | null;
    if (!message || !metadata?.research?.executionSha) return jsonError("Research reply not found.", 404);
    const generation = metadata.research.variant === "r3_response" ? "r3" : "r1";
    const saved = await prisma.auditLog.create({ data: { action: `research.${generation}.reply_feedback`, actorId: user.id,
      entityType: "message", entityId: message.id, meta: { conversationId: body.conversationId,
        assessment: body.assessment, note: body.note, research: metadata.research,
        outputSha256: createHash("sha256").update(message.content).digest("hex"),
        trainingConsent: false, source: `app-${generation}-review` } } });
    return jsonOk({ saved: true, reviewId: saved.id });
  } catch (error) {
    if (error instanceof z.ZodError) return jsonError("Choose an assessment and a note under 4,000 characters.");
    return jsonError("Your review could not be saved. Please try again.", 500);
  }
}
