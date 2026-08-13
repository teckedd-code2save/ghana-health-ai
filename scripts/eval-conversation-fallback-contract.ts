import "../src/config/load-env";
import { prisma } from "../src/db/prisma";
import { runConversationTurn } from "../src/lib/conversation-turn";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const previousGroq = process.env.GROQ_API_KEY;
  const previousOpenai = process.env.OPENAI_API_KEY;
  delete process.env.GROQ_API_KEY;
  delete process.env.OPENAI_API_KEY;

  let conversationId: string | undefined;
  try {
    const turn = await runConversationTurn({
      user: null,
      text: "mepɛ sɛ metɔ tomatoes",
      language: "tw",
      focus: "commerce",
      channel: "WEB",
      speak: false,
    });
    conversationId = turn.conversationId;

    assertOk(turn.understanding.engine === "fallback", `expected fallback, got ${turn.understanding.engine}`);
    assertOk(turn.stage.llm === false, "fallback turn should not mark llm stage true");
    assertOk(turn.stage.review === false, "fallback turn should not mark model review true");
    assertOk(turn.understanding.intent === "ECOMMERCE", `expected commerce intent, got ${turn.understanding.intent}`);
    assertOk(turn.reply.length > 10, "expected fallback reply");

    const messages = await prisma.message.findMany({
      where: { conversationId: turn.conversationId },
      orderBy: { createdAt: "asc" },
    });
    assertOk(messages.length === 2, `expected user+assistant messages, got ${messages.length}`);
    assertOk(messages[1]?.metadata && typeof messages[1].metadata === "object", "expected assistant metadata");
    console.log(`ok conversation-fallback conversation=${turn.conversationId}`);
  } finally {
    if (conversationId) {
      await prisma.conversation.delete({ where: { id: conversationId } }).catch(() => undefined);
    }
    if (previousGroq) process.env.GROQ_API_KEY = previousGroq;
    if (previousOpenai) process.env.OPENAI_API_KEY = previousOpenai;
    await prisma.$disconnect();
  }
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

export {};
