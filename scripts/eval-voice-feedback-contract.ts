import "../src/config/load-env";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const baseUrl = process.env.EVAL_BASE_URL || "http://localhost:3000";
const isLocalBase = ["localhost", "127.0.0.1"].includes(new URL(baseUrl).hostname);

type ChatResponse = {
  conversationId?: string;
  userMessage?: {
    id?: string;
    content?: string;
  };
};

async function createConversationViaApi() {
  const res = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: "mepɛ sɛ meto tomato",
      language: "tw",
      speak: false,
    }),
  });
  const text = await res.text();
  assertOk(res.ok, `chat setup HTTP ${res.status}: ${text}`);
  const body = JSON.parse(text) as ChatResponse;
  assertOk(body.conversationId, "chat setup missing conversationId");
  assertOk(body.userMessage?.id, "chat setup missing userMessage.id");
  return { conversationId: body.conversationId, messageId: body.userMessage.id };
}

async function main() {
  const localSetup = isLocalBase
    ? await (async () => {
        const marker = `feedback-contract-${randomUUID()}`;
        const conversation = await prisma.conversation.create({
          data: {
            language: "tw",
            channel: "VOICE",
            title: marker,
            intent: "UNKNOWN",
          },
        });
        const message = await prisma.message.create({
          data: {
            conversationId: conversation.id,
            role: "USER",
            content: "mepɛ sɛ meto tomato",
            language: "tw",
            metadata: {
              mode: "voice",
              asrModel: "contract-asr",
              asrRoute: "contract-route",
            },
          },
        });
        return { conversationId: conversation.id, messageId: message.id };
      })()
    : await createConversationViaApi();

  try {
    const res = await fetch(`${baseUrl}/api/voice/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversationId: localSetup.conversationId,
        messageId: localSetup.messageId,
        originalTranscript: "mepɛ sɛ meto tomato",
        correctedTranscript: "mepɛ sɛ metɔ tomato",
        language: "tw",
        focus: "commerce",
        rating: 3,
      }),
    });

    const responseText = await res.text();
    assertOk(res.ok, `feedback HTTP ${res.status}: ${responseText}`);
    const body = JSON.parse(responseText) as { feedbackId?: string };
    assertOk(body.feedbackId, "feedbackId missing");

    if (isLocalBase) {
      const feedback = await prisma.asrFeedback.findUnique({
        where: { id: body.feedbackId },
      });
      assertOk(feedback, "feedback row missing");
      assertOk(feedback.correctedTranscript === "mepɛ sɛ metɔ tomato", "correction mismatch");
      assertOk(feedback.asrModel === "contract-asr", "asr model metadata missing");
      assertOk(feedback.asrRoute === "contract-route", "asr route metadata missing");
    }

    console.log(`ok voice-feedback-contract base=${baseUrl}`);
  } finally {
    if (isLocalBase) {
      await prisma.conversation
        .delete({ where: { id: localSetup.conversationId } })
        .catch(() => undefined);
      await prisma.$disconnect();
    }
  }
}

main().catch(async (error) => {
  console.error(error);
  if (isLocalBase) {
    await prisma.$disconnect();
  }
  process.exit(1);
});

export {};
