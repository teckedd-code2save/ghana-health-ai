/**
 * Understanding — research path.
 *
 * 1. English ONLY if preferredLang === "en"
 * 2. ABENA retrieval over Twi knowledge/products (Ghana-NLP encoder)
 * 3. Twi-first generation from transcript + retrieved Twi context
 * 4. No hand-written semantic bank; no English KB dump as primary brain
 *
 * See docs/research-stack.md
 */

import { chatComplete, isLlmConfigured } from "@/lib/llm";
import { retrieveTwiContext } from "@/lib/twi-retrieve";
import type { LanguageCode } from "@prisma/client";

export type UnderstandResult = {
  reply: string;
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  engine: "abena+llm" | "llm" | "fallback";
  model?: string;
  replyLanguage: "tw" | "en";
  retrieve?: {
    engine: string;
    model?: string;
    articles: { slug: string; score: number }[];
    products: { id: string; score: number }[];
  };
};

const HOTLINE =
  process.env.HEALTH_ESCALATION_HOTLINE || "112 or your nearest clinic / community health worker";

export function resolveReplyLanguage(preferred?: LanguageCode): "tw" | "en" {
  return preferred === "en" ? "en" : "tw";
}

export function detectReplyLanguage(
  _text: string,
  preferred?: LanguageCode,
): "tw" | "en" {
  return resolveReplyLanguage(preferred);
}

const SYSTEM_TWI = `Wo yɛ Ghana Health AI — voice companion ma Ghana.
Ka **Twi** a ɛte sɛ kasa a nnipa ka. Light English mix nko sɛ ɛhia (din, aduro din).

Wobenya:
- asɛm a user no kae (ASR betumi ayɛ noise)
- nkyerɛkyerɛmu a wɔtwee fii Twi knowledge (sɛ ɛwɔ hɔ a)

Rules:
1. Gye user asɛm no so. Sɛ ɛyɛ noise / menteee a, bisa wɔn nka bio prɛko.
2. Fa Twi knowledge no di dwuma sɛ ɛfa asɛm no ho — nnyɛ copy-paste.
3. Nnyɛ oduruyɛfoɔ. Mma drug dosage. Mma definitive diagnosis.
4. Danger (mogya a ɛsen, home yɛ den, seizure, ba a ɔnte yafunu mu, suicide): di kan ka hospital / frɛ ${HOTLINE}.
5. No canned closers ("Me ho yɛ hɔ", "Feel free to ask").
6. "community health worker", "antenatal care" — no bare CHW/ANC.
7. 2–5 nkyerɛaseɛ a wɔbɛtumi aka.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

const SYSTEM_EN = `You are Ghana Health AI. The user prefers **English** — reply in clear simple English only.

Use the transcript and any retrieved knowledge. Not a doctor. No dosages/diagnoses.
Emergencies: lead with hospital / ${HOTLINE}.
No canned closers. 2–5 short spoken sentences.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

const CANNED: RegExp[] = [
  /\s*Sɛ wopɛ nsɛm pii a,?\s*me ho yɛ hɔ!?\s*$/iu,
  /\s*Me ho yɛ hɔ!?\s*$/iu,
  /\s*I'm here if you (need|want).*$/iu,
  /\s*Feel free to (ask|reach).*$/iu,
  /\s*If you (need|want|have) (any )?(more )?(questions|help).*$/iu,
];

export function stripCannedClosers(text: string): string {
  let out = text.trim();
  let prev = "";
  while (out !== prev) {
    prev = out;
    for (const re of CANNED) out = out.replace(re, "").trim();
  }
  return out
    .replace(/\bCHWs\b/g, "community health workers")
    .replace(/\bCHW\b/g, "community health worker")
    .replace(/\bANC\b/g, "antenatal care");
}

function dangerHeuristic(text: string): boolean {
  const t = text.toLowerCase();
  return [
    "bleeding",
    "mogya",
    "unconscious",
    "can't breathe",
    "cannot breathe",
    "seizure",
    "convulsion",
    "suicide",
    "no fetal",
    "water broke",
    "home yɛ den",
    "nte home",
  ].some((k) => t.includes(k));
}

function parseJson(raw: string): Partial<UnderstandResult> | null {
  try {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    return JSON.parse(raw.slice(start, end + 1)) as Partial<UnderstandResult>;
  } catch {
    return null;
  }
}

function guessIntent(
  text: string,
  hasProducts: boolean,
  hasHealth: boolean,
): UnderstandResult["intent"] {
  const t = text.toLowerCase();
  const shop = ["tɔ", "boɔ", "cart", "price", "buy", "market", "ɛmo", "paracetamol", "sapo"];
  if (hasProducts && shop.some((k) => t.includes(k))) return "ECOMMERCE";
  if (hasHealth || dangerHeuristic(text)) return "HEALTH";
  if (shop.some((k) => t.includes(k))) return "ECOMMERCE";
  return "GENERAL";
}

export async function understandUtterance(input: {
  text: string;
  language?: LanguageCode;
  history?: { role: "user" | "assistant"; content: string }[];
}): Promise<UnderstandResult> {
  const language = input.language ?? "tw";
  const replyLang = resolveReplyLanguage(language);
  const urgent = dangerHeuristic(input.text);

  // Research path: ABENA (or fallback) retrieval on Twi content
  const ctx = await retrieveTwiContext(input.text);
  const retrieveMeta = {
    engine: ctx.engine,
    model: ctx.model,
    articles: ctx.articles.map((a) => ({ slug: a.slug, score: Number(a.score.toFixed(4)) })),
    products: ctx.products.map((p) => ({ id: p.id, score: Number(p.score.toFixed(4)) })),
  };

  const intentGuess = guessIntent(
    input.text,
    ctx.products.length > 0,
    ctx.articles.length > 0,
  );

  // Build Twi-primary context block for the generator
  const twiKb = ctx.articles
    .map((a, i) => `[${i + 1}] ${a.titleTw}\n${a.bodyTw}`)
    .join("\n\n");
  const productBlock = ctx.products
    .map((p) => `• ${p.nameTw} / ${p.nameEn} — GH₵ ${p.priceGhs.toFixed(2)}`)
    .join("\n");

  if (!isLlmConfigured()) {
    const seed =
      replyLang === "tw"
        ? urgent
          ? `⚠️ EYI BETUMI AYƐ ƆHAW. Kɔ hospital anaa frɛ ${HOTLINE} ntɛm.`
          : ctx.articles[0]?.bodyTw ||
            "LLM key nni hɔ. Fa GROQ_API_KEY anaa OPENAI_API_KEY hyɛ mu."
        : urgent
          ? `⚠️ This may be urgent. Go to hospital or call ${HOTLINE}.`
          : "No LLM key configured.";
    return {
      reply: stripCannedClosers(seed),
      intent: urgent ? "HEALTH" : intentGuess,
      severity: urgent ? "EMERGENCY" : "LOW",
      escalate: urgent,
      engine: "fallback",
      replyLanguage: replyLang,
      retrieve: retrieveMeta,
    };
  }

  const history = (input.history ?? []).slice(-6).map((m) => ({
    role: m.role === "user" ? ("user" as const) : ("assistant" as const),
    content: m.content,
  }));

  const userPayload = {
    asr_transcript: input.text,
    speak_language: replyLang === "en" ? "english" : "twi",
    english_allowed: replyLang === "en",
    retrieve_engine: ctx.engine,
    twi_knowledge: twiKb || null,
    market_hits: productBlock || null,
  };

  const raw = await chatComplete(
    [
      { role: "system", content: replyLang === "en" ? SYSTEM_EN : SYSTEM_TWI },
      ...history,
      { role: "user", content: JSON.stringify(userPayload) },
    ],
    { temperature: 0.25, maxTokens: 400 },
  );

  if (!raw) {
    return {
      reply:
        replyLang === "tw"
          ? "Menteee wo yie. Ka bio kakra, anaa twerɛ asɛm no."
          : "I could not form a reply. Please say that again.",
      intent: "UNKNOWN",
      severity: urgent ? "EMERGENCY" : "LOW",
      escalate: urgent,
      engine: ctx.engine === "abena" ? "abena+llm" : "llm",
      model: ctx.model,
      replyLanguage: replyLang,
      retrieve: retrieveMeta,
    };
  }

  const parsed = parseJson(raw);
  let reply = parsed?.reply ? String(parsed.reply) : raw.trim();

  // Attach market lines if ecommerce and we have priced hits
  if (
    (parsed?.intent === "ECOMMERCE" || intentGuess === "ECOMMERCE") &&
    productBlock &&
    !reply.includes("GH₵")
  ) {
    reply = `${stripCannedClosers(reply)}\n\n${productBlock}`;
  } else {
    reply = stripCannedClosers(reply);
  }

  const severity =
    (parsed?.severity as UnderstandResult["severity"]) || (urgent ? "EMERGENCY" : "LOW");

  return {
    reply,
    intent: (parsed?.intent as UnderstandResult["intent"]) || intentGuess,
    severity,
    escalate: Boolean(parsed?.escalate) || severity === "EMERGENCY" || severity === "HIGH" || urgent,
    engine: ctx.engine === "abena" || ctx.engine === "openai" ? "abena+llm" : "llm",
    model: ctx.model,
    replyLanguage: replyLang,
    retrieve: retrieveMeta,
  };
}
