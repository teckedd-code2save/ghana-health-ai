import { z } from "zod";
import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { detectIntent, generateHealthReply } from "@/lib/health-rag";
import { jsonError, jsonOk } from "@/lib/api";

const schema = z.object({
  message: z.string().min(1).max(4000),
  conversationId: z.string().uuid().optional(),
  language: z.enum(["tw", "en", "ga", "ee", "dag"]).optional(),
});

export async function POST(req: Request) {
  try {
    const body = schema.parse(await req.json());
    const user = await getSessionUser();
    const language = body.language ?? user?.preferredLang ?? "tw";
    const intent = detectIntent(body.message);
    const started = Date.now();

    let conversation = body.conversationId
      ? await prisma.conversation.findUnique({ where: { id: body.conversationId } })
      : null;

    if (!conversation) {
      conversation = await prisma.conversation.create({
        data: {
          userId: user?.id,
          language,
          intent,
          channel: "WEB",
          title: body.message.slice(0, 60),
        },
      });
    } else {
      await prisma.conversation.update({
        where: { id: conversation.id },
        data: { intent, language },
      });
    }

    await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "USER",
        content: body.message,
        language,
        intent,
      },
    });

    let replyText: string;
    let meta: Prisma.InputJsonValue = { intent };
    let disclaimer = false;

    if (intent === "HEALTH") {
      const health = await generateHealthReply(body.message, language);
      replyText = health.reply;
      disclaimer = true;
      meta = {
        intent,
        severity: health.severity,
        escalate: health.escalate,
        sources: health.sources,
      };

      await prisma.symptomCheck.create({
        data: {
          userId: user?.id,
          symptoms: body.message.split(/[\s,]+/).slice(0, 12),
          freeText: body.message,
          severity: health.severity,
          advice: health.reply,
          escalate: health.escalate,
          language,
          disclaimer: health.disclaimer,
        },
      });
    } else if (intent === "ECOMMERCE") {
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
        take: 5,
      });

      if (products.length === 0) {
        replyText =
          language === "tw"
            ? "Menhui nneɛma no wɔ gua so. Sɔ hwɛ 'rice', 'paracetamol', anaa 'soap'."
            : "I couldn't find matching market items. Try rice, paracetamol, or soap.";
      } else {
        const lines = products.map(
          (p) =>
            `• ${language === "tw" ? p.nameTw : p.nameEn} — GH₵ ${Number(p.priceGhs).toFixed(2)} (${p.stock} left)`,
        );
        replyText =
          (language === "tw"
            ? "Gua so nneɛma a mehui:\n"
            : "Here's what I found in the market:\n") +
          lines.join("\n") +
          (language === "tw"
            ? "\n\nKɔ Market tab so de fa ka cart."
            : "\n\nOpen the Market tab to add to cart.");
        meta = {
          intent,
          products: products.map((p) => ({
            id: p.id,
            sku: p.sku,
            nameEn: p.nameEn,
            priceGhs: Number(p.priceGhs),
          })),
        };
      }
    } else {
      replyText =
        language === "tw"
          ? "Mema wo akwaaba ɔ Ghana Health AI. Bisa health asɛm (nyinsen, afe) anaa market (tɔ nneɛma). Wobetumi nso de voice reka."
          : "Welcome to Ghana Health AI. Ask a health question (pregnancy, fever) or shop the market by voice/text.";
    }

    const assistant = await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "ASSISTANT",
        content: replyText,
        language,
        intent,
        disclaimer,
        latencyMs: Date.now() - started,
        metadata: meta,
      },
    });

    return jsonOk({
      conversationId: conversation.id,
      message: {
        id: assistant.id,
        role: assistant.role,
        content: assistant.content,
        intent,
        latencyMs: assistant.latencyMs,
        metadata: meta,
      },
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
