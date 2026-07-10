import { z } from "zod";
import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { understandUtterance } from "@/lib/understand";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";
import { isModalTtsConfigured, modalSpeak } from "@/lib/modal-tts";

const schema = z.object({
  message: z.string().min(1).max(4000),
  conversationId: z.string().uuid().optional(),
  language: z.enum(["tw", "en", "ga", "ee", "dag"]).optional(),
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
    const started = Date.now();

    let conversation = body.conversationId
      ? await prisma.conversation.findUnique({ where: { id: body.conversationId } })
      : null;

    if (!conversation) {
      conversation = await prisma.conversation.create({
        data: {
          userId: user?.id,
          language,
          channel: "WEB",
          title: body.message.slice(0, 60),
          intent: "UNKNOWN",
        },
      });
    }

    const prior = await prisma.message.findMany({
      where: { conversationId: conversation.id },
      orderBy: { createdAt: "asc" },
      take: 12,
    });

    await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "USER",
        content: body.message,
        language,
      },
    });

    const understanding = await understandUtterance({
      text: body.message,
      language,
      history: prior.map((m) => ({
        role: m.role === "USER" ? "user" : "assistant",
        content: m.content,
      })),
    });

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: { intent: understanding.intent, language },
    });

    if (understanding.intent === "HEALTH") {
      await prisma.symptomCheck.create({
        data: {
          userId: user?.id,
          symptoms: body.message.split(/[\s,]+/).slice(0, 12),
          freeText: body.message,
          severity: understanding.severity,
          advice: understanding.reply,
          escalate: understanding.escalate,
          language,
          disclaimer: "LLM companion — not medical advice",
        },
      });
    }

    // Lightweight product assist when intent is ecommerce — still from real DB, not hardcoded
    let reply = understanding.reply;
    if (understanding.intent === "ECOMMERCE") {
      const terms = body.message.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
      const products = await prisma.product.findMany({
        where: {
          isActive: true,
          OR: terms.flatMap((t) => [
            { nameEn: { contains: t, mode: "insensitive" as const } },
            { nameTw: { contains: t, mode: "insensitive" as const } },
            { tags: { has: t } },
          ]),
        },
        take: 4,
      });
      if (products.length) {
        const lines = products
          .map((p) => `• ${p.nameTw} / ${p.nameEn} — GH₵ ${Number(p.priceGhs).toFixed(2)}`)
          .join("\n");
        reply = `${understanding.reply}\n\n${lines}`;
      }
    }

    const meta: Prisma.InputJsonValue = {
      intent: understanding.intent,
      severity: understanding.severity,
      escalate: understanding.escalate,
      engine: understanding.engine,
    };

    const assistant = await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "ASSISTANT",
        content: reply,
        language,
        intent: understanding.intent,
        disclaimer: understanding.intent === "HEALTH",
        latencyMs: Date.now() - started,
        metadata: meta,
      },
    });

    let tts: { audioBase64: string; sampleRate?: number; model?: string } | null = null;
    if (body.speak && isModalTtsConfigured()) {
      try {
        const spoken = await modalSpeak(reply, language === "en" ? "en" : "tw");
        if (spoken.audio_base64) {
          tts = {
            audioBase64: spoken.audio_base64,
            sampleRate: spoken.sample_rate,
            model: spoken.model,
          };
        }
      } catch (e) {
        console.error("[chat tts]", e);
      }
    }

    await writeAudit({
      action: "chat.understand",
      actorId: user?.id,
      entityType: "conversation",
      entityId: conversation.id,
      ip,
      meta: {
        intent: understanding.intent,
        engine: understanding.engine,
        severity: understanding.severity,
      },
    });

    return jsonOk({
      conversationId: conversation.id,
      message: {
        id: assistant.id,
        role: assistant.role,
        content: assistant.content,
        intent: understanding.intent,
        latencyMs: assistant.latencyMs,
        metadata: meta,
      },
      tts,
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
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
