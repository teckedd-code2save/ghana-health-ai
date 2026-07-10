/**
 * Conversational understanding — LLM is the brain.
 * Grounded on the user's actual words; no canned closers.
 */

import { chatComplete, isLlmConfigured } from "@/lib/llm";
import type { LanguageCode } from "@prisma/client";

export type UnderstandResult = {
  reply: string;
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  engine: "llm" | "fallback";
  model?: string;
};

const HOTLINE = process.env.HEALTH_ESCALATION_HOTLINE || "112 / nearest CHW / facility";

const SYSTEM = `You are Ghana Health AI — a careful voice companion for people in Ghana.

Languages: Twi (Akan), English, and code-mixed Twi-English as Ghanaians actually speak.
Match the user's language mix. Prefer clear short spoken answers (2–5 sentences).

What you do well:
- Maternal health, symptoms, wellbeing — general guidance only, never as a doctor
- Market / shopping intent in natural conversation
- Everyday questions with cultural awareness (family, CHW, clinic pathways)

Hard rules:
1. Answer ONLY what the user actually said. Do not invent a different topic.
2. If the user message looks like ASR noise, garbage, or unrelated filler (nonsense loops, random "mmea" boilerplate, empty meaning), reply briefly that you did not catch them and ask them to repeat once — do NOT give a long lecture on an unrelated subject.
3. NOT a medical professional. Never invent drug dosages or definitive diagnoses.
4. Danger signs (heavy bleeding/mogya, unconscious, can't breathe, seizures, no fetal movement, suicidal thoughts): lead with URGENT action + hotline ${HOTLINE}.
5. No canned closers or catchphrases. Never end with stock lines like:
   "Sɛ wopɛ nsɛm pii a, me ho yɛ hɔ", "Me ho yɛ hɔ", "Feel free to ask", "I'm here if you need me".
   Stop when the answer is complete.
6. Prefer concrete next steps (rest, hydrate, see CHW/clinic, price/cart) over vague encouragement.

Respond with ONLY a JSON object (no markdown fences):
{
  "reply": "spoken answer to the user",
  "intent": "HEALTH" | "ECOMMERCE" | "GENERAL",
  "severity": "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY",
  "escalate": boolean
}`;

const CANNED_CLOSER_PATTERNS: RegExp[] = [
  /\s*Sɛ wopɛ nsɛm pii a,?\s*me ho yɛ hɔ!?\s*$/iu,
  /\s*Sɛ wopɛ (nsɛm|biribi) (pii|bio).*?(hɔ|ho)!?\s*$/iu,
  /\s*Me ho yɛ hɔ!?\s*$/iu,
  /\s*I'm here if you (need|want).*$/iu,
  /\s*Feel free to (ask|reach).*$/iu,
  /\s*If you (need|want|have) (any )?(more )?(questions|info|information|help).*$/iu,
  /\s*Let me know if you (need|want).*$/iu,
  /\s*Don't hesitate to (ask|reach).*$/iu,
];

export function stripCannedClosers(text: string): string {
  let out = text.trim();
  let prev = "";
  while (out !== prev) {
    prev = out;
    for (const re of CANNED_CLOSER_PATTERNS) {
      out = out.replace(re, "").trim();
    }
  }
  return out;
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
    "kill myself",
    "no fetal",
    "water broke",
  ].some((k) => t.includes(k));
}

function parseJsonReply(raw: string): Partial<UnderstandResult> | null {
  try {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    return JSON.parse(raw.slice(start, end + 1)) as Partial<UnderstandResult>;
  } catch {
    return null;
  }
}

export async function understandUtterance(input: {
  text: string;
  language?: LanguageCode;
  history?: { role: "user" | "assistant"; content: string }[];
}): Promise<UnderstandResult> {
  const language = input.language ?? "tw";
  const urgent = dangerHeuristic(input.text);

  if (!isLlmConfigured()) {
    const reply =
      language === "tw"
        ? urgent
          ? `⚠️ EYI BETUMI AYƐ ƆHAW. Kɔ hospital anaa frɛ ${HOTLINE} ntɛm. (LLM key missing.)`
          : "LLM key nni hɔ. Fa GROQ_API_KEY anaa OPENAI_API_KEY hyɛ configuration mu."
        : urgent
          ? `⚠️ This may be urgent. Go to a hospital or call ${HOTLINE} now. (LLM key missing.)`
          : "No LLM API key configured. Set GROQ_API_KEY or OPENAI_API_KEY.";
    return {
      reply,
      intent: urgent ? "HEALTH" : "GENERAL",
      severity: urgent ? "EMERGENCY" : "LOW",
      escalate: urgent,
      engine: "fallback",
    };
  }

  const history = (input.history ?? []).slice(-8).map((m) => ({
    role: m.role === "user" ? ("user" as const) : ("assistant" as const),
    content: m.content,
  }));

  const raw = await chatComplete(
    [
      { role: "system", content: SYSTEM },
      ...history,
      {
        role: "user",
        content: `Preferred language code: ${language}\nTranscript (what the user said — reply to THIS only):\n"""${input.text}"""`,
      },
    ],
    { temperature: 0.25, maxTokens: 420 },
  );

  if (!raw) {
    return {
      reply:
        language === "tw"
          ? "Menteee wo yie. Ka bio kakra, anaa twerɛ asɛm no."
          : "I couldn't form a reply. Please say that again or type it.",
      intent: "UNKNOWN",
      severity: urgent ? "EMERGENCY" : "LOW",
      escalate: urgent,
      engine: "fallback",
    };
  }

  const parsed = parseJsonReply(raw);
  if (parsed?.reply) {
    const severity =
      (parsed.severity as UnderstandResult["severity"]) || (urgent ? "EMERGENCY" : "LOW");
    return {
      reply: stripCannedClosers(String(parsed.reply)),
      intent: (parsed.intent as UnderstandResult["intent"]) || "GENERAL",
      severity,
      escalate: Boolean(parsed.escalate) || severity === "EMERGENCY" || urgent,
      engine: "llm",
    };
  }

  return {
    reply: stripCannedClosers(raw.trim()),
    intent: urgent ? "HEALTH" : "GENERAL",
    severity: urgent ? "EMERGENCY" : "LOW",
    escalate: urgent,
    engine: "llm",
  };
}
