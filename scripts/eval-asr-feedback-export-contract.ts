import "../src/config/load-env";
import fs from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { prisma } from "../src/db/prisma";
import { exportAsrFeedback } from "./export-asr-feedback";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const marker = `feedback-export-contract-${randomUUID()}`;
  const outPath = path.join(process.cwd(), "tmp", `${marker}.jsonl`);
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
  await prisma.asrFeedback.create({
    data: {
      conversationId: conversation.id,
      messageId: message.id,
      language: "tw",
      focus: "commerce",
      originalTranscript: "mepɛ sɛ meto tomato",
      correctedTranscript: "mepɛ sɛ metɔ tomato",
      rating: 3,
      asrModel: "contract-asr",
      asrRoute: "contract-route",
    },
  });

  try {
    const result = await exportAsrFeedback({ outPath, limit: 25 });
    assertOk(result.rows.length >= 1, "expected at least one exported feedback row");
    const exported = result.rows.find((row) => row.reference === "mepɛ sɛ metɔ tomato");
    assertOk(exported, "expected corrected transcript in export");
    assertOk(exported.bucket === "commerce_twi", `unexpected bucket ${exported.bucket}`);
    assertOk(exported.audio_path.startsWith("MISSING_AUDIO"), "expected missing audio marker");

    const raw = await fs.readFile(outPath, "utf8");
    assertOk(raw.includes("mepɛ sɛ metɔ tomato"), "export file missing corrected transcript");
    console.log("ok asr-feedback-export-contract");
  } finally {
    await prisma.conversation.delete({ where: { id: conversation.id } }).catch(() => undefined);
    await fs.unlink(outPath).catch(() => undefined);
    await prisma.$disconnect();
  }
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

export {};
