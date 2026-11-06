import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";
import type { ConversationTurnInput, ConversationTurnResult } from "@/lib/conversation-turn";
import type { UnderstandResult } from "@/lib/understand";
import { ResearchR3Error } from "@/lib/model-modes";
import { verifiedR3Config, researchR3Reply, validateR3History, verifyR3StoredReply, type R3Message } from "@/lib/research-r3";
import { rateLimit } from "@/lib/rate-limit";
import { isModalTtsConfigured, modalSpeak } from "@/lib/modal-tts";

const running = new Set<string>();

export async function runResearchR3Turn(input: ConversationTurnInput): Promise<ConversationTurnResult> {
  if (!input.user) throw new ResearchR3Error("Sign in to test the research model.", 401);
  const key = input.user.id;
  if (running.has(key)) throw new ResearchR3Error("A research reply is still running.", 409);
  if (!rateLimit(`r3-user:${key}`, 30, 3600).allowed) throw new ResearchR3Error("Research testing limit reached. Please try again later.", 429);
  running.add(key);
  try { return await run(input); } finally { running.delete(key); }
}

async function run(input: ConversationTurnInput): Promise<ConversationTurnResult> {
  const startedAt = Date.now();
  const text = input.text.trim();
  if (!text) throw new ResearchR3Error("Enter a message.", 400);
  validateR3History([{ role: "user", content: input.text }]);
  const userId = input.user!.id;
  const language = input.language ?? input.user!.preferredLang ?? "tw";
  let conversation = input.conversationId ? await prisma.conversation.findUnique({ where: { id: input.conversationId } }) : null;
  if (input.conversationId && (!conversation || conversation.userId !== userId)) throw new ResearchR3Error("Conversation not found.", 404);
  const config = await verifiedR3Config();
  if (!config) throw new ResearchR3Error("R3 is not available. No other model was used.");
  const prior = conversation ? await prisma.message.findMany({ where: { conversationId: conversation.id }, orderBy: { createdAt: "asc" }, take: 131 }) : [];
  const messages: R3Message[] = [];
  for (const [i, message] of prior.entries()) {
    const meta = message.metadata as { research?: { variant?: string; bundleSha?: string } } | null;
    if (message.role !== (i % 2 ? "ASSISTANT" : "USER") || meta?.research?.variant !== "r3_response" ||
        meta.research.bundleSha !== config.identity.manifest_sha256) throw new ResearchR3Error("Start a new conversation for this R3 checkpoint.", 409);
    if (message.role === "ASSISTANT") verifyR3StoredReply(message.metadata, message.content, config.identity, messages);
    messages.push({ role: message.role === "USER" ? "user" : "assistant", content: message.content });
  }
  messages.push({ role: "user", content: text });
  validateR3History(messages);
  input.signal?.throwIfAborted();
  let conversationId = conversation?.id;
  let userMessageId: string | undefined;
  await input.onStage?.({ name: "understanding", detail: "research_model", at: Date.now() });
  const generated = await researchR3Reply(messages, config, async chunk => {
    await input.onStage?.({ name: "reply_delta", chunk, at: Date.now() });
  }, input.signal, async () => {
    input.signal?.throwIfAborted();
    if (!conversation) conversation = await prisma.conversation.create({ data: { userId, language,
      channel: input.channel, title: text.slice(0, 60), intent: "UNKNOWN" } });
    conversationId = conversation.id;
    await input.onStage?.({ name: "conversation", detail: conversationId, at: Date.now() });
    const message = await prisma.message.create({ data: { conversationId, role: "USER", content: text, language,
      metadata: { research: { variant: "r3_response", bundleSha: config.identity.manifest_sha256 },
        transcript: input.transcript ?? null } as Prisma.InputJsonValue } });
    userMessageId = message.id;
    await input.onStage?.({ name: "user_message", at: Date.now() });
  });
  if (!conversationId || !userMessageId) throw new ResearchR3Error("R3 admission was not recorded. No other model was used.");
  input.signal?.throwIfAborted();
  const understandLatencyMs = Date.now() - startedAt;
  const understanding: UnderstandResult = { reply: generated.result.output, intent: "UNKNOWN", severity: "LOW", escalate: false,
    engine: "llm", model: `${generated.research.model} / r3_response`, replyLanguage: language === "en" ? "en" : "tw", research: generated.research,
    synthesis: { mode: "live_model", provider: "modal-r3-direct", model: `${generated.research.model} / r3_response`,
      usedHistory: prior.length > 0, usedMemory: false, safetyEnforced: false,
      understandingModel: { mode: "r3_response", used: true, model: generated.research.model, latencyMs: understandLatencyMs } } };
  const assistant = await prisma.message.create({ data: { conversationId, role: "ASSISTANT", content: understanding.reply,
    language, intent: "UNKNOWN", disclaimer: true, latencyMs: understandLatencyMs,
    metadata: { research: generated.research, outputSha256: generated.outputSha256, evidenceSha256: generated.evidenceSha256, evidence: generated.result,
      model: understanding.model, synthesis: understanding.synthesis, engine: "llm", replyLanguage: understanding.replyLanguage,
      researchOnly: true, clinicalAssessment: "not_performed", trainingConsent: false } as Prisma.InputJsonValue } });
  await prisma.conversation.update({ where: { id: conversationId }, data: { updatedAt: new Date() } });
  await input.onStage?.({ name: "assistant_message", at: Date.now() });
  let tts: ConversationTurnResult["tts"] = null;
  if (input.speak && generated.research.complete && isModalTtsConfigured()) {
    try {
      const spoken = await modalSpeak(understanding.reply, understanding.replyLanguage);
      if (spoken.audio_base64) tts = { audioBase64: spoken.audio_base64, sampleRate: spoken.sample_rate,
        format: spoken.format || "wav", model: spoken.model, provider: spoken.provider, latencyMs: spoken.latency_ms };
    } catch { /* Text remains the exact verified model output. */ }
  }
  return { conversationId, reply: understanding.reply, userMessageId, assistantId: assistant.id,
    understanding, tts, stage: { startedAt, totalLatencyMs: Date.now() - startedAt, llm: true,
      understandLatencyMs, review: false, retrieveEngine: "none", tts: Boolean(tts), ttsLatencyMs: tts?.latencyMs } };
}
