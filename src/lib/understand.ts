/**
 * Conversational understanding — LLM is the brain.
 * Not keyword RAG. Optional brief safety layer only.
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

Respond with ONLY a JSON object (no markdown fences):
{
  "reply": "spoken answer to the user",
  "intent": "HEALTH" | "ECOMMERCE" | "GENERAL",
  "severity": "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY",
  "escalate": boolean
}`;

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
          ? `⚠️ EYI BETUMI AYƐ ƆHAW. Kɔ hospital anaa frɛ ${HOTLINE} ntɛm. (LLM not configured — set GROQ_API_KEY or OPENAI_API_KEY.)`
          : "Mede wo asɛm ate, nanso LLM key nni hɔ. Fa GROQ_API_KEY anaa OPENAI_API_KEY hyɛ Infisical mu na me nte ase yie."
        : urgent
          ? `⚠️ This may be urgent. Go to a hospital or call ${HOTLINE} now. (LLM not configured.)`
          : "I heard you, but no LLM API key is configured. Add GROQ_API_KEY or OPENAI_API_KEY in Infisical for real understanding.";
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
    const severity = (parsed.severity as UnderstandResult["severity"]) || (urgent ? "EMERGENCY" : "LOW");
    return {
      reply: String(parsed.reply),
      intent: (parsed.intent as UnderstandResult["intent"]) || "GENERAL",
      severity,
      escalate: Boolean(parsed.escalate) || severity === "EMERGENCY" || urgent,
      engine: "llm",
    };
  }

  // Model returned prose — still usable
  return {
    reply: raw.trim(),
    intent: urgent ? "HEALTH" : "GENERAL",
    severity: urgent ? "EMERGENCY" : "LOW",
    escalate: urgent,
    engine: "llm",
  };
}
