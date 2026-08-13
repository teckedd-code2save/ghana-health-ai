import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { writeAudit } from "@/lib/audit";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  conversationId: z.string().uuid(),
  messageId: z.string().uuid().optional(),
  originalTranscript: z.string().min(1).max(4000),
  correctedTranscript: z.string().trim().min(1).max(4000).optional(),
  rating: z.number().int().min(1).max(5).optional(),
  language: z.enum(["tw", "en", "ga"]).optional(),
  focus: z.enum(["health", "commerce"]).optional(),
  notes: z.string().trim().max(800).optional(),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice-feedback:${ip}`, 30, 60);
    if (!rl.allowed) return jsonError("Too many feedback updates — wait a minute", 429);

    const body = schema.parse(await req.json());
    const user = await getSessionUser();

    const conversation = await prisma.conversation.findUnique({
      where: { id: body.conversationId },
      select: { id: true, userId: true },
    });
    if (!conversation) return jsonError("Conversation not found", 404);
    if (conversation.userId && user?.id !== conversation.userId) {
      return jsonError("Conversation not found", 404);
    }

    const message = body.messageId
      ? await prisma.message.findFirst({
          where: {
            id: body.messageId,
            conversationId: body.conversationId,
            role: "USER",
          },
          select: {
            id: true,
            language: true,
            metadata: true,
          },
        })
      : null;
    if (body.messageId && !message) return jsonError("Transcript message not found", 404);

    const metadata = isRecord(message?.metadata) ? message.metadata : {};
    const feedback = await prisma.asrFeedback.create({
      data: {
        userId: user?.id,
        conversationId: body.conversationId,
        messageId: message?.id,
        language: body.language ?? message?.language ?? user?.preferredLang ?? "tw",
        focus: body.focus,
        originalTranscript: body.originalTranscript,
        correctedTranscript: body.correctedTranscript,
        rating: body.rating,
        notes: body.notes,
        asrModel: stringValue(metadata.asrModel),
        asrRoute: stringValue(metadata.asrRoute),
        metadata: {
          source: "voice-ui",
          hasCorrection: Boolean(body.correctedTranscript),
          rating: body.rating ?? null,
        },
      },
    });

    await writeAudit({
      action: "voice.asr_feedback",
      actorId: user?.id,
      entityType: "asr_feedback",
      entityId: feedback.id,
      ip,
      meta: {
        conversationId: body.conversationId,
        messageId: message?.id,
        language: feedback.language,
        focus: body.focus,
        hasCorrection: Boolean(body.correctedTranscript),
        rating: body.rating ?? null,
      },
    });

    return jsonOk({ ok: true, feedbackId: feedback.id });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return jsonError(error.issues[0]?.message ?? "Invalid feedback");
    }
    console.error("[voice/feedback]", error);
    return jsonError("Feedback failed", 500);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function stringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value : undefined;
}
