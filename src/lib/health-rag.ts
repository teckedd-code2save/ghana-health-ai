import { prisma } from "@/db/prisma";
import type { LanguageCode, SeverityFlag } from "@prisma/client";

export const HEALTH_DISCLAIMER_TW =
  "Yɛi nyɛ oduruyɛfoɔ adwuma — kɔ oduruyɛfoɔ anaa CHW hɔ sɛ ɛhia. This is not medical advice.";

export const HEALTH_DISCLAIMER_EN =
  "This is not a substitute for professional medical care. If symptoms are severe, contact a health worker, CHW, or emergency services immediately.";

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
    "buy",
    "price",
    "market",
    "tɔ",
    "boɔ",
    "order",
    "cart",
    "momo",
    "rice",
    "paracetamol",
    "soap",
    "shop",
  ];
  const health = [
    "pain",
    "fever",
    "pregnant",
    "pregnancy",
    "symptom",
    "doctor",
    "hospital",
    "nyinsen",
    "yare",
    "afe",
    "headache",
    "maternal",
    "anc",
  ];
  if (shop.some((k) => lower.includes(k))) return "ECOMMERCE";
  if (health.some((k) => lower.includes(k))) return "HEALTH";
  return "GENERAL";
}

export async function retrieveHealthKnowledge(query: string, limit = 3) {
  const terms = query
    .toLowerCase()
    .split(/[\s,.;!?]+/)
    .filter((t) => t.length > 2)
    .slice(0, 8);

  if (terms.length === 0) {
    return prisma.knowledgeArticle.findMany({
      where: { isActive: true },
      take: limit,
      orderBy: { updatedAt: "desc" },
    });
  }

  // Simple keyword match over title/body (MVP). Phase 2: vector hybrid search.
  const articles = await prisma.knowledgeArticle.findMany({
    where: {
      isActive: true,
      OR: terms.flatMap((t) => [
        { titleEn: { contains: t, mode: "insensitive" as const } },
        { titleTw: { contains: t, mode: "insensitive" as const } },
        { bodyEn: { contains: t, mode: "insensitive" as const } },
        { bodyTw: { contains: t, mode: "insensitive" as const } },
        { tags: { has: t } },
      ]),
    },
    take: limit * 2,
  });

  // Rank by number of term hits
  const ranked = articles
    .map((a) => {
      const blob = `${a.titleEn} ${a.titleTw} ${a.bodyEn} ${a.bodyTw} ${a.tags.join(" ")}`.toLowerCase();
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

export async function generateHealthReply(
  userText: string,
  language: LanguageCode = "tw",
): Promise<{
  reply: string;
  severity: SeverityFlag;
  escalate: boolean;
  sources: { slug: string; title: string }[];
  disclaimer: string;
}> {
  const severity = assessSeverity(userText);
  const escalate = severity === "EMERGENCY" || severity === "HIGH";
  const articles = await retrieveHealthKnowledge(userText, 3);
  const disclaimer = language === "tw" ? HEALTH_DISCLAIMER_TW : HEALTH_DISCLAIMER_EN;

  const sources = articles.map((a) => ({
    slug: a.slug,
    title: language === "tw" ? a.titleTw : a.titleEn,
  }));

  let body: string;
  if (articles.length > 0) {
    const primary = articles[0];
    body = language === "tw" ? primary.bodyTw : primary.bodyEn;
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
        ? "\n\n⚠️ EYI BETUMI AYƐ ƆHAW: Kɔ hospital anaa frɛ emergency / CHW ntɛm ara."
        : "\n\n⚠️ THIS MAY BE URGENT: Go to a hospital or call emergency / your CHW immediately.";
  } else if (severity === "MEDIUM") {
    urgency =
      language === "tw"
        ? "\n\nSɛ ɛkɔ so anaa ɛyɛ den a, kɔ clinic."
        : "\n\nIf this continues or worsens, visit a clinic.";
  }

  const reply = `${body}${urgency}\n\n—\n${disclaimer}`;
  return { reply, severity, escalate, sources, disclaimer };
}
