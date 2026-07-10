/**
 * Conversational understanding — LLM is the brain.
 * No canned closers; answers end when the content is done.
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

const SYSTEM = `You are Ghana Health AI — a voice companion for people in Ghana.

You understand spoken Twi (Akan), English, and code-mixed Twi-English.
Reply in the same language mix the user used when possible.
If user language is "tw", prefer natural Twi with light English as Ghanaians speak.

You can:
- Discuss health, pregnancy, symptoms, wellbeing — carefully, never as a doctor
- Help with market/shopping intent (products, prices, cart) conversationally
- Chat generally with cultural awareness

Hard rules:
- You are NOT a medical professional. Never invent drug dosages or diagnoses.
- For danger signs (heavy bleeding, unconscious, can't breathe, seizures, no fetal movement, suicidal thoughts): lead with URGENT action + hotline ${HOTLINE}.
- Keep replies short enough to speak aloud (2–6 sentences) unless user asks for detail.
- Be warm, clear, practical for low-literacy and rural users.
- NEVER append canned closers, catchphrases, or filler invitations such as:
  "Sɛ wopɛ nsɛm pii a, me ho yɛ hɔ!", "If you need anything else…", "Feel free to ask…",
  "Me ho yɛ hɔ", "I'm here if you need me", or similar stock endings.
  End the reply when the answer is complete — no signature line, no invitation to continue unless the user asked a follow-up question that requires it.
- Do not repeat the same closing sentence across turns.

Respond with ONLY a JSON object (no markdown fences):
{
  "reply": "spoken answer to the user",
  "intent": "HEALTH" | "ECOMMERCE" | "GENERAL",
  "severity": "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY",
  "escalate": boolean
}`;

/** Known LLM stock endings to strip if the model ignores the system prompt. */
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
  // Multiple passes — models sometimes stack closers
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
    // Config error only — not a content template for normal turns
    const reply =
      language === "tw"
        ? urgent
          ? `⚠️ EYI BETUMI AYƐ ƆHAW. Kɔ hospital anaa frɛ ${HOTLINE} ntɛm. (LLM key missing — set GROQ_API_KEY or OPENAI_API_KEY.)`
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
        content: `Preferred language code: ${language}\nUser said: ${input.text}`,
      },
    ],
    { temperature: 0.35, maxTokens: 500 },
  );

  if (!raw) {
    return {
      reply:
        language === "tw"
          ? "Menni mmuae mprempren. Sɔ bio anaa kɔ clinic sɛ ɛyɛ den."
          : "I couldn't form a reply just now. Try again, or visit a clinic if this is serious.",
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
