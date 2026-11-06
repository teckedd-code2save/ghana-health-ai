import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";
import type { ConversationTurnInput, ConversationTurnResult } from "@/lib/conversation-turn";
import type { UnderstandResult } from "@/lib/understand";
import { ResearchR1Error, type ResearchR1Mode } from "@/lib/model-modes";
import { R1_VARIANTS, isContinuableResearchReply, researchR1Reply, validateResearchHistory, type R1Message } from "@/lib/research-r1";
import { rateLimit } from "@/lib/rate-limit";
import { isModalTtsConfigured, modalSpeak } from "@/lib/modal-tts";

const running = new Set<string>();

export async function runResearchR1Turn(input: ConversationTurnInput, mode: ResearchR1Mode): Promise<ConversationTurnResult> {
  if (!input.user) throw new ResearchR1Error("Sign in to test the research models.", 401);
  const key = input.user.id;
  if (running.has(key)) throw new ResearchR1Error("A research reply is still running. Please wait for it to finish.", 409);
  if (!rateLimit(`r1-user:${key}`, 30, 3600).allowed) throw new ResearchR1Error("Research testing limit reached. Please try again later.", 429);
  running.add(key);
  try {
    return await run(input, mode);
  } finally {
    running.delete(key);
  }
}

async function run(input: ConversationTurnInput, mode: ResearchR1Mode): Promise<ConversationTurnResult> {
  const startedAt = Date.now();
  const text = input.text.trim();
  if (!text) throw new ResearchR1Error("Enter a message.", 400);
  const userId = input.user!.id;
  const language = input.language ?? input.user!.preferredLang ?? "tw";
  let conversation = input.conversationId ? await prisma.conversation.findUnique({ where: { id: input.conversationId } }) : null;
  if (input.conversationId && (!conversation || conversation.userId !== userId)) throw new ResearchR1Error("Conversation not found.", 404);
  if (!conversation) conversation = await prisma.conversation.create({ data: {
    userId, language, channel: input.channel, title: text.slice(0, 60), intent: "UNKNOWN",
  } });
  const conversationId = conversation.id;
  const prior = await prisma.message.findMany({ where: { conversationId }, orderBy: { createdAt: "asc" }, take: 81 });
  if (prior.some((message) => {
    const meta = message.metadata as { research?: { variant?: string; complete?: boolean } } | null;
    return (message.role !== "USER" && message.role !== "ASSISTANT") ||
      meta?.research?.variant !== R1_VARIANTS[mode] ||
      (message.role === "ASSISTANT" && (!message.content.trim() || !isContinuableResearchReply(meta.research, mode)));
  })) throw new ResearchR1Error("Start a new conversation for this research checkpoint. Foreign or unverified replies will not be mixed into it.", 409);
  const messages: R1Message[] = [...prior.map((message) => ({
    role: (message.role === "USER" ? "user" : "assistant") as R1Message["role"], content: message.content,
  })), { role: "user", content: text }];
  validateResearchHistory(messages);
  await input.onStage?.({ name: "conversation", detail: conversationId, at: Date.now() });
  const userMessage = await prisma.message.create({ data: { conversationId, role: "USER", content: text, language,
    metadata: { research: { variant: R1_VARIANTS[mode] }, transcript: input.transcript ?? null, focus: input.focus ?? null } as Prisma.InputJsonValue } });
  await input.onStage?.({ name: "user_message", at: Date.now() });
  await input.onStage?.({ name: "understanding", detail: "research_model", at: Date.now() });
  const generated = await researchR1Reply(messages, mode, async (chunk) => {
    await input.onStage?.({ name: "reply_delta", chunk, at: Date.now() });
  });
  const understandLatencyMs = Date.now() - startedAt;
  const understanding: UnderstandResult = {
    reply: generated.result.output, intent: "UNKNOWN", severity: "LOW", escalate: false,
    engine: "llm", model: `${generated.research.model} / ${generated.research.variant}`,
    replyLanguage: language === "en" ? "en" : "tw", research: generated.research,
    synthesis: { mode: "live_model", provider: "modal-r1-direct", model: `${generated.research.model} / ${generated.research.variant}`,
      usedHistory: prior.length > 0, usedMemory: false, safetyEnforced: false,
      understandingModel: { mode, used: true, model: generated.research.model, latencyMs: understandLatencyMs } },
  };
  const assistant = await prisma.message.create({ data: {
    conversationId, role: "ASSISTANT", content: understanding.reply, language, intent: "UNKNOWN", disclaimer: true,
    latencyMs: understandLatencyMs,
    metadata: { research: generated.research, outputSha256: generated.outputSha256, model: understanding.model,
      synthesis: understanding.synthesis, engine: "llm", replyLanguage: understanding.replyLanguage,
      evidence: generated.result, researchOnly: true, clinicalAssessment: "not_performed", trainingConsent: false } as Prisma.InputJsonValue,
  } });
  await prisma.conversation.update({ where: { id: conversationId }, data: { updatedAt: new Date() } });
  await input.onStage?.({ name: "assistant_message", at: Date.now() });
  let tts: ConversationTurnResult["tts"] = null;
  if (input.speak && generated.research.complete && isModalTtsConfigured()) {
    await input.onStage?.({ name: "tts", detail: "started", at: Date.now() });
    try {
      const spoken = await modalSpeak(understanding.reply, understanding.replyLanguage);
      if (spoken.audio_base64) tts = { audioBase64: spoken.audio_base64, sampleRate: spoken.sample_rate,
        format: spoken.format || "wav", model: spoken.model, provider: spoken.provider, latencyMs: spoken.latency_ms };
    } catch { /* Preserve the model's completed text when speech is unavailable. */ }
    await input.onStage?.({ name: "tts", detail: tts ? "complete" : "unavailable", at: Date.now() });
  }
  return { conversationId, reply: understanding.reply, userMessageId: userMessage.id, assistantId: assistant.id,
    understanding, tts, stage: { startedAt, totalLatencyMs: Date.now() - startedAt, llm: true,
      understandLatencyMs, review: false, retrieveEngine: "none", tts: Boolean(tts), ttsLatencyMs: tts?.latencyMs } };
}
