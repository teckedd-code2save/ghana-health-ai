import "../src/config/load-env";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const baseUrl = process.env.EVAL_BASE_URL || "http://localhost:3000";

async function main() {
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

  try {
    const res = await fetch(`${baseUrl}/api/voice/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversationId: conversation.id,
        messageId: message.id,
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

    const feedback = await prisma.asrFeedback.findUnique({
      where: { id: body.feedbackId },
    });
    assertOk(feedback, "feedback row missing");
    assertOk(feedback.correctedTranscript === "mepɛ sɛ metɔ tomato", "correction mismatch");
    assertOk(feedback.asrModel === "contract-asr", "asr model metadata missing");
    assertOk(feedback.asrRoute === "contract-route", "asr route metadata missing");

    console.log("ok voice-feedback-contract");
  } finally {
    await prisma.conversation.delete({ where: { id: conversation.id } }).catch(() => undefined);
    await prisma.$disconnect();
  }
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

export {};
