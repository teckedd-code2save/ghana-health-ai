import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { runConversationTurn } from "@/lib/conversation-turn";
import { publicFailure } from "@/lib/public-errors";

const schema = z.object({
  message: z.string().min(1).max(4000),
  conversationId: z.string().uuid().optional(),
  language: z.enum(["tw", "en", "ga"]).optional(),
  speak: z.boolean().optional(),
});

/**
 * Text chat — LLM understanding is the product.
 * Optional TTS when speak=true and Modal TTS is configured.
 */
export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`chat:${ip}`, 40, 60);
    if (!rl.allowed) return jsonError("Too many messages — wait a minute", 429);

    const body = schema.parse(await req.json());
    const user = await getSessionUser();
    const language = body.language ?? user?.preferredLang ?? "tw";

    const turn = await runConversationTurn({
      user,
      ip,
      text: body.message,
      conversationId: body.conversationId,
      language,
      channel: "WEB",
      speak: body.speak,
    });

    return jsonOk({
      conversationId: turn.conversationId,
      userMessage: {
        id: turn.userMessageId,
        content: body.message,
      },
      message: {
        id: turn.assistantId,
        role: "ASSISTANT",
        content: turn.reply,
        intent: turn.understanding.intent,
        latencyMs: turn.stage.totalLatencyMs,
        metadata: {
          intent: turn.understanding.intent,
          severity: turn.understanding.severity,
          escalate: turn.understanding.escalate,
          engine: turn.understanding.engine,
          replyLanguage: turn.understanding.replyLanguage,
          comprehension: turn.understanding.comprehension ?? null,
          health: turn.understanding.health ?? null,
          commerce: turn.understanding.commerce ?? null,
          commerceExecution: turn.commerceExecution ?? null,
          retrieve: turn.understanding.retrieve ?? null,
          review: turn.understanding.review ?? null,
          synthesis: turn.understanding.synthesis ?? null,
        },
      },
      understanding: {
        intent: turn.understanding.intent,
        severity: turn.understanding.severity,
        escalate: turn.understanding.escalate,
        engine: turn.understanding.engine,
        health: turn.understanding.health ?? null,
        commerce: turn.understanding.commerce ?? null,
        commerceExecution: turn.commerceExecution ?? null,
        synthesis: turn.understanding.synthesis ?? null,
        comprehension: turn.understanding.comprehension ?? null,
      },
      tts: turn.tts,
      stage: turn.stage,
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error("[chat]", e);
    const failure = publicFailure(e, "chat");
    return jsonError(failure.message, failure.status);
  }
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const conversationId = searchParams.get("conversationId");
  const user = await getSessionUser();
  if (!conversationId) {
    const ids = searchParams
      .get("ids")
      ?.split(",")
      .filter((id) => z.string().uuid().safeParse(id).success)
      .slice(0, 50) ?? [];
    const conversations = await prisma.conversation.findMany({
      where: user ? { userId: user.id } : { userId: null, id: { in: ids } },
      orderBy: { updatedAt: "desc" },
      take: 50,
      select: {
        id: true,
        title: true,
        language: true,
        intent: true,
        createdAt: true,
        updatedAt: true,
        _count: { select: { messages: true } },
      },
    });
    return jsonOk({ conversations });
  }
  const conversation = await prisma.conversation.findUnique({
    where: { id: conversationId },
    select: { id: true, userId: true },
  });
  if (!conversation) return jsonError("Conversation not found", 404);
  if (conversation.userId && conversation.userId !== user?.id) {
    return jsonError("Conversation not found", 404);
  }
  const messages = await prisma.message.findMany({
    where: { conversationId },
    orderBy: { createdAt: "asc" },
  });
  return jsonOk({ messages });
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const conversationId = searchParams.get("conversationId");
  if (!conversationId || !z.string().uuid().safeParse(conversationId).success) {
    return jsonError("Valid conversationId required");
  }
  const user = await getSessionUser();
  const conversation = await prisma.conversation.findUnique({
    where: { id: conversationId },
    select: { id: true, userId: true },
  });
  if (!conversation || (conversation.userId && conversation.userId !== user?.id)) {
    return jsonError("Conversation not found", 404);
  }
  await prisma.conversation.delete({ where: { id: conversationId } });
  return jsonOk({ deleted: true, conversationId });
}
