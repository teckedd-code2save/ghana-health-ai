import type { Prisma } from "@prisma/client";
import type { LanguageCode, ConversationChannel } from "@prisma/client";
import { prisma } from "@/db/prisma";
import { isModalTtsConfigured, modalSpeak } from "@/lib/modal-tts";
import { understandUtterance, type UnderstandResult } from "@/lib/understand";
import { writeAudit } from "@/lib/audit";
import { getAgentMemory, memoryPrompt, rememberFromTurn } from "@/lib/agent-memory";
import { executeCommercePlan, type CommerceExecutionResult } from "@/lib/commerce-execute";

type SessionUser = {
  id: string;
  preferredLang: LanguageCode;
} | null;

export type TurnTranscriptMeta = {
  mode?: string;
  model?: string;
  latencyMs?: number;
  speaker?: string;
  language?: string;
  languageProbability?: number;
  route?: string;
  duration?: number;
  rms?: number;
};

export type ConversationTurnInput = {
  user: SessionUser;
  ip?: string;
  text: string;
  language?: LanguageCode;
  focus?: "health" | "commerce";
  instruction?: string;
  conversationId?: string;
  channel: ConversationChannel;
  speak?: boolean;
  transcript?: TurnTranscriptMeta;
  onStage?: (stage: {
    name:
      | "conversation"
      | "user_message"
      | "understanding"
      | "assistant_message"
      | "reply_delta"
      | "tts"
      | "audit";
    detail?: string;
    chunk?: string;
    at: number;
  }) => void | Promise<void>;
};

export type ConversationTurnResult = {
  conversationId: string;
  reply: string;
  userMessageId: string;
  assistantId: string;
  understanding: UnderstandResult;
  commerceExecution?: CommerceExecutionResult;
  tts: {
    audioBase64: string;
    sampleRate?: number;
    format?: string;
    model?: string;
    provider?: string;
    latencyMs?: number;
  } | null;
  stage: {
    startedAt: number;
    totalLatencyMs: number;
    llm: boolean;
    understandLatencyMs: number;
    review: boolean;
    retrieveEngine: string;
    tts: boolean;
    ttsLatencyMs?: number;
  };
};

export async function runConversationTurn(
  input: ConversationTurnInput,
): Promise<ConversationTurnResult> {
  const startedAt = Date.now();
  const language = input.language ?? input.user?.preferredLang ?? "tw";
  const text = input.text.trim();
  if (!text) throw new Error("Text is required.");

  let conversation = input.conversationId
    ? await prisma.conversation.findUnique({ where: { id: input.conversationId } })
    : null;

  if (!conversation) {
    conversation = await prisma.conversation.create({
      data: {
        userId: input.user?.id,
        language,
        channel: input.channel,
        title: text.slice(0, 60),
        intent: "UNKNOWN",
      },
    });
  }
  await input.onStage?.({
    name: "conversation",
    detail: conversation.id,
    at: Date.now(),
  });

  const prior = await prisma.message.findMany({
    where: { conversationId: conversation.id },
    orderBy: { createdAt: "asc" },
    take: 12,
  });

  const history = prior.map((m) => ({
    role: (m.role === "USER" ? "user" : "assistant") as "user" | "assistant",
    content: m.content,
  }));
  const owner = { userId: input.user?.id, sessionId: conversation.id };
  const focusScope =
    input.focus === "commerce" ? "COMMERCE" : input.focus === "health" ? "HEALTH" : undefined;
  const memories = await getAgentMemory(owner, focusScope);
  const profileMemories = await getAgentMemory(owner, "PROFILE");
  const memory = memoryPrompt([...profileMemories, ...memories]);

  const userMessage = await prisma.message.create({
    data: {
      conversationId: conversation.id,
      role: "USER",
      content: text,
      language,
      speakerLabel: input.transcript?.speaker,
      latencyMs: input.transcript?.latencyMs,
      metadata: {
        mode: input.transcript?.mode ?? (input.channel === "VOICE" ? "voice" : "text"),
        asrModel: input.transcript?.model,
        asrLanguage: input.transcript?.language,
        asrLanguageProbability: input.transcript?.languageProbability,
        asrRoute: input.transcript?.route,
        duration: input.transcript?.duration,
        rms: input.transcript?.rms,
        focus: input.focus,
      },
    },
  });
  await input.onStage?.({ name: "user_message", at: Date.now() });

  const understandStartedAt = Date.now();
  await input.onStage?.({ name: "understanding", detail: "started", at: Date.now() });
  const understanding = await understandUtterance({
    text,
    language,
    history,
    memory,
    transcript: input.transcript,
    focus: input.focus,
    instruction: input.instruction,
  });
  const commerceExecution =
    understanding.intent === "ECOMMERCE"
      ? await executeCommercePlan(understanding.commerce)
      : undefined;
  const understandLatencyMs = Date.now() - understandStartedAt;
  await input.onStage?.({
    name: "understanding",
    detail: understanding.retrieve?.engine ?? understanding.engine,
    at: Date.now(),
  });

  await prisma.conversation.update({
    where: { id: conversation.id },
    data: { intent: understanding.intent, language },
  });

  if (understanding.intent === "HEALTH") {
    await prisma.symptomCheck.create({
      data: {
        userId: input.user?.id,
        symptoms: text.split(/[\s,]+/).slice(0, 12),
        freeText: text,
        severity: understanding.severity,
        advice: understanding.reply,
        escalate: understanding.escalate,
        language,
        disclaimer: "Model-generated health information; not a diagnosis.",
      },
    });
  }

  const meta: Prisma.InputJsonValue = {
    intent: understanding.intent,
    severity: understanding.severity,
    escalate: understanding.escalate,
    engine: understanding.engine,
    model: understanding.model,
    replyLanguage: understanding.replyLanguage,
    retrieve: understanding.retrieve ?? null,
    review: understanding.review ?? null,
    health: understanding.health ?? null,
    commerce: understanding.commerce ?? null,
    commerceExecution: commerceExecution ?? null,
    transcript: input.transcript ?? null,
    memory: memory ?? null,
    focus: input.focus,
  };

  const assistant = await prisma.message.create({
    data: {
      conversationId: conversation.id,
      role: "ASSISTANT",
      content: understanding.reply,
      language,
      intent: understanding.intent,
      disclaimer: understanding.intent === "HEALTH",
      latencyMs: Date.now() - startedAt,
      metadata: meta,
    },
  });
  await input.onStage?.({ name: "assistant_message", at: Date.now() });

  await rememberFromTurn({
    owner,
    text,
    reply: understanding.reply,
    language,
    focus: input.focus,
    intent: understanding.intent,
    severity: understanding.severity,
  });

  if (input.onStage) {
    for (const chunk of chunkReplyForLiveStream(understanding.reply)) {
      await input.onStage({
        name: "reply_delta",
        chunk,
        at: Date.now(),
      });
      await sleep(38);
    }
  }

  let tts: ConversationTurnResult["tts"] = null;
  if (input.speak && isModalTtsConfigured()) {
    await input.onStage?.({ name: "tts", detail: "started", at: Date.now() });
    const ttsLang = understanding.replyLanguage === "en" ? "en" : "tw";
    const spoken = await modalSpeak(understanding.reply, ttsLang);
    if (spoken.audio_base64) {
      tts = {
        audioBase64: spoken.audio_base64,
        sampleRate: spoken.sample_rate,
        format: spoken.format || "wav",
        model: spoken.model,
        provider: spoken.provider,
        latencyMs: spoken.latency_ms,
      };
    }
    await input.onStage?.({
      name: "tts",
      detail: tts?.model ?? "unavailable",
      at: Date.now(),
    });
  }

  await writeAudit({
    action: input.channel === "VOICE" ? "voice.turn" : "chat.turn",
    actorId: input.user?.id,
    entityType: "conversation",
    entityId: conversation.id,
    ip: input.ip,
    meta: {
      intent: understanding.intent,
      engine: understanding.engine,
      severity: understanding.severity,
      escalate: understanding.escalate,
      retrieve: understanding.retrieve ?? null,
      review: understanding.review ?? null,
      health: understanding.health ?? null,
      asrModel: input.transcript?.model,
      asrRoute: input.transcript?.route,
      commerce: understanding.commerce ?? null,
      commerceExecution: commerceExecution ?? null,
      tts: Boolean(tts?.audioBase64),
    },
  });
  await input.onStage?.({ name: "audit", at: Date.now() });

  return {
    conversationId: conversation.id,
    reply: understanding.reply,
    userMessageId: userMessage.id,
    assistantId: assistant.id,
    understanding,
    commerceExecution,
    tts,
    stage: {
      startedAt,
      totalLatencyMs: Date.now() - startedAt,
      llm: understanding.engine !== "fallback",
      understandLatencyMs,
      review: understanding.review?.engine === "llm",
      retrieveEngine: understanding.retrieve?.engine ?? "none",
      tts: Boolean(tts?.audioBase64),
      ttsLatencyMs: tts?.latencyMs,
    },
  };
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function chunkReplyForLiveStream(reply: string): string[] {
  const parts = reply.match(/\S+\s*/g) ?? [];
  const chunks: string[] = [];
  let current = "";

  for (const part of parts) {
    current += part;
    if (current.length >= 28 || /[.!?]\s*$/.test(current)) {
      chunks.push(current);
      current = "";
    }
  }

  if (current) chunks.push(current);
  return chunks;
}
