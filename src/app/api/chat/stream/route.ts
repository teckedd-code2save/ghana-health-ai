import { z } from "zod";
import { getSessionUser } from "@/lib/auth";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { runConversationTurn } from "@/lib/conversation-turn";

const schema = z.object({
  message: z.string().min(1).max(4000),
  conversationId: z.string().uuid().optional(),
  language: z.enum(["tw", "en", "ga"]).optional(),
  speak: z.boolean().optional(),
});

function streamHeaders() {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
  };
}

function formatEvent(event: string, data: unknown) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function publicTurnPayload(turn: Awaited<ReturnType<typeof runConversationTurn>>) {
  return {
    conversationId: turn.conversationId,
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
  };
}

export async function POST(req: Request) {
  const encoder = new TextEncoder();

  return new Response(
    new ReadableStream({
      async start(controller) {
        const send = (event: string, data: unknown) => {
          controller.enqueue(encoder.encode(formatEvent(event, data)));
        };

        try {
          const ip = clientIp(req);
          const rl = rateLimit(`chat:stream:${ip}`, 40, 60);
          if (!rl.allowed) {
            send("error", { error: "Too many messages — wait a minute", status: 429 });
            return;
          }

          const body = schema.parse(await req.json());
          const user = await getSessionUser();
          const language = body.language ?? user?.preferredLang ?? "tw";

          send("stage", { name: "accepted", at: Date.now() });
          const turn = await runConversationTurn({
            user,
            ip,
            text: body.message,
            conversationId: body.conversationId,
            language,
            channel: "WEB",
            speak: body.speak,
            onStage: (stage) => {
              if (stage.name === "reply_delta") {
                send("reply_delta", { chunk: stage.chunk, at: stage.at });
                return;
              }
              send("stage", stage);
            },
          });

          send("final", publicTurnPayload(turn));
        } catch (e) {
          console.error("[chat/stream]", e);
          if (e instanceof z.ZodError) {
            send("error", { error: e.issues[0]?.message ?? "Invalid input", status: 400 });
          } else if (
            e instanceof Error &&
            (e.message.includes("ECONNREFUSED") || e.message.includes("Can't reach database"))
          ) {
            send("error", {
              error: "Service is starting — database is not reachable yet",
              status: 503,
            });
          } else {
            send("error", { error: "Chat failed", status: 500 });
          }
        } finally {
          controller.close();
        }
      },
    }),
    { headers: streamHeaders() },
  );
}
