import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { runConversationTurn } from "@/lib/conversation-turn";

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
          health: turn.understanding.health ?? null,
          commerce: turn.understanding.commerce ?? null,
          commerceExecution: turn.commerceExecution ?? null,
          retrieve: turn.understanding.retrieve ?? null,
          review: turn.understanding.review ?? null,
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
      },
      tts: turn.tts,
      stage: turn.stage,
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    if (
      e instanceof Error &&
      (e.message.includes("ECONNREFUSED") || e.message.includes("Can't reach database"))
    ) {
      return jsonError("Service is starting — database is not reachable yet", 503);
    }
    return jsonError("Chat failed", 500);
  }
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const conversationId = searchParams.get("conversationId");
  if (!conversationId) return jsonError("conversationId required");
  const messages = await prisma.message.findMany({
    where: { conversationId },
    orderBy: { createdAt: "asc" },
  });
  return jsonOk({ messages });
}
