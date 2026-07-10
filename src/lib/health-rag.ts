import { prisma } from "@/db/prisma";
import type { LanguageCode, SeverityFlag } from "@prisma/client";
import { chatComplete, isLlmConfigured } from "@/lib/llm";

export const HEALTH_DISCLAIMER_TW =
  "Yɛi nyɛ oduruyɛfoɔ adwuma — kɔ oduruyɛfoɔ anaa community health worker hɔ sɛ ɛhia. This is not medical advice.";

export const HEALTH_DISCLAIMER_EN =
  "This is not a substitute for professional medical care. If symptoms are severe, contact a clinic, community health worker, or emergency services immediately.";

export const EMERGENCY_HOTLINE =
  process.env.HEALTH_ESCALATION_HOTLINE ||
  "112 or your nearest clinic / community health worker";

const DANGER_KEYWORDS = [
  "bleeding",
  "mogya",
  "convulsion",
  "seizure",
  "unconscious",
  "difficulty breathing",
  "shortness of breath",
  "chest pain",
  "severe headache",
  "swollen",
  "water broke",
  "no fetal movement",
  "fever high",
  "emergency",
  "suicide",
  "self harm",
];

const MEDIUM_KEYWORDS = [
  "fever",
  "headache",
  "vomiting",
  "nausea",
  "pain",
  "dizzy",
  "fatigue",
  "swelling",
  "pregnancy",
  "nyinsen",
  "afe",
  "malaria",
  "diarrhea",
];

export function assessSeverity(text: string): SeverityFlag {
  const lower = text.toLowerCase();
  if (DANGER_KEYWORDS.some((k) => lower.includes(k))) return "EMERGENCY";
  if (MEDIUM_KEYWORDS.some((k) => lower.includes(k))) return "MEDIUM";
  return "LOW";
}

export function detectIntent(text: string): "HEALTH" | "ECOMMERCE" | "GENERAL" {
  const lower = text.toLowerCase();
  const shop = [
    "buy", "price", "market", "tɔ", "boɔ", "order", "cart", "momo", "rice",
    "paracetamol", "soap", "shop", "pay",
  ];
  const health = [
    "pain", "fever", "pregnant", "pregnancy", "symptom", "doctor", "hospital",
    "nyinsen", "yare", "afe", "headache", "maternal", "anc", "malaria", "clinic",
  ];
  if (shop.some((k) => lower.includes(k))) return "ECOMMERCE";
  if (health.some((k) => lower.includes(k))) return "HEALTH";
  return "GENERAL";
}

export async function retrieveHealthKnowledge(query: string, limit = 4) {
  const terms = query
    .toLowerCase()
    .split(/[\s,.;!?]+/)
    .filter((t) => t.length > 2)
    .slice(0, 10);

  if (terms.length === 0) {
    return prisma.knowledgeArticle.findMany({
      where: { isActive: true },
      take: limit,
      orderBy: { updatedAt: "desc" },
    });
  }

  const articles = await prisma.knowledgeArticle.findMany({
    where: {
      isActive: true,
      OR: terms.flatMap((t) => [
        { titleEn: { contains: t, mode: "insensitive" as const } },
        { titleTw: { contains: t, mode: "insensitive" as const } },
        { bodyEn: { contains: t, mode: "insensitive" as const } },
        { bodyTw: { contains: t, mode: "insensitive" as const } },
        { tags: { has: t } },
        { category: { contains: t, mode: "insensitive" as const } },
      ]),
    },
    take: limit * 3,
  });

  const ranked = articles
    .map((a) => {
      const blob =
        `${a.titleEn} ${a.titleTw} ${a.bodyEn} ${a.bodyTw} ${a.tags.join(" ")} ${a.category}`.toLowerCase();
      const score = terms.reduce((s, t) => s + (blob.includes(t) ? 1 : 0), 0);
      return { article: a, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((r) => r.article);

  if (ranked.length === 0) {
    return prisma.knowledgeArticle.findMany({
      where: { isActive: true, category: "maternal" },
      take: limit,
    });
  }
  return ranked;
}

function templateReply(
  userText: string,
  language: LanguageCode,
  articles: Awaited<ReturnType<typeof retrieveHealthKnowledge>>,
  severity: SeverityFlag,
): string {
  const disclaimer = language === "tw" ? HEALTH_DISCLAIMER_TW : HEALTH_DISCLAIMER_EN;
  let body: string;
  if (articles.length > 0) {
    body = language === "tw" ? articles[0].bodyTw : articles[0].bodyEn;
  } else if (language === "tw") {
    body =
      "Mede asɛm no ate. Sɛ wo yare anaa wo wɔ nyinsen ho nsɛm a, ka kyerɛ me sɛnea ɛte. Mesɛ sɛ wo kɔ oduruyɛfoɔ hɔ sɛ ɛyɛ den.";
  } else {
    body =
      "I heard you. Share more about your symptoms or pregnancy concerns and I will guide you with general information. Please see a clinician if this feels urgent.";
  }

  let urgency = "";
  if (severity === "EMERGENCY") {
    urgency =
      language === "tw"
        ? `\n\n⚠️ EYI BETUMI AYƐ ƆHAW: Kɔ hospital anaa frɛ ${EMERGENCY_HOTLINE} ntɛm ara.`
        : `\n\n⚠️ THIS MAY BE URGENT: Go to a hospital or call ${EMERGENCY_HOTLINE} immediately.`;
  } else if (severity === "MEDIUM") {
    urgency =
      language === "tw"
        ? "\n\nSɛ ɛkɔ so anaa ɛyɛ den a, kɔ clinic."
        : "\n\nIf this continues or worsens, visit a clinic.";
  }

  return `${body}${urgency}\n\n—\n${disclaimer}`;
}

export async function generateHealthReply(
  userText: string,
  language: LanguageCode = "tw",
): Promise<{
  reply: string;
  severity: SeverityFlag;
  escalate: boolean;
  sources: { slug: string; title: string }[];
  disclaimer: string;
  engine: "llm" | "template";
}> {
  const severity = assessSeverity(userText);
  const escalate = severity === "EMERGENCY" || severity === "HIGH";
  const articles = await retrieveHealthKnowledge(userText, 4);
  const disclaimer = language === "tw" ? HEALTH_DISCLAIMER_TW : HEALTH_DISCLAIMER_EN;
  const sources = articles.map((a) => ({
    slug: a.slug,
    title: language === "tw" ? a.titleTw : a.titleEn,
  }));

  const kbBlock = articles
    .map(
      (a, i) =>
        `[${i + 1}] ${a.titleEn} / ${a.titleTw}\n${language === "tw" ? a.bodyTw : a.bodyEn}`,
    )
    .join("\n\n");

  if (isLlmConfigured()) {
    const system = `You are Ghana Health AI, a careful community health companion for Ghana.
Rules:
- Respond primarily in ${language === "tw" ? "Twi (Akan) with light English mix where natural" : "clear simple English"}.
- You are NOT a doctor. Never diagnose definitively or prescribe specific drug doses.
- Use ONLY the knowledge base excerpts when making health claims; if unsure, say so and urge clinic or community health worker care.
- Always end with a short disclaimer.
- If severity is EMERGENCY, lead with urgent action and hotline ${EMERGENCY_HOTLINE}.
- Cultural humility: respect Ghanaian family and community health worker care pathways.
- Never use bare acronyms like CHW — say "community health worker".
Severity assessed: ${severity}
Knowledge base:
${kbBlock || "(no articles matched — speak generally and urge professional care)"}`;

    const llm = await chatComplete(
      [
        { role: "system", content: system },
        { role: "user", content: userText },
      ],
      { temperature: 0.25, maxTokens: 500 },
    );

    if (llm) {
      const withDisclaimer = llm.includes("not medical") || llm.includes("nyɛ oduruyɛfoɔ")
        ? llm
        : `${llm}\n\n—\n${disclaimer}`;
      return {
        reply: withDisclaimer,
        severity,
        escalate,
        sources,
        disclaimer,
        engine: "llm",
      };
    }
  }

  return {
    reply: templateReply(userText, language, articles, severity),
    severity,
    escalate,
    sources,
    disclaimer,
    engine: "template",
  };
}
