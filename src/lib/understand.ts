/**
 * Understanding — research path.
 *
 * 1. Speech transcript + ASR quality first.
 * 2. LLM understanding decides intent, safety, and clarification.
 * 3. Retrieval is optional grounding, not the product brain.
 *
 * See docs/research-stack.md
 */

import { chatComplete, isLlmConfigured, llmProviderInfo } from "@/lib/llm";
import { buildCommerceActionPlan, type CommerceActionPlan } from "@/lib/commerce-plan";
import { buildHealthUnderstanding, type HealthUnderstanding } from "@/lib/health-plan";
import {
  formatUnderstandingModelHint,
  recoverUnderstandingWithModel,
  understandingModelMode,
  type UnderstandingModelMode,
  type UnderstandingModelPrediction,
} from "@/lib/understanding-model";
import type { LanguageCode } from "@prisma/client";
import { z } from "zod";

export type UnderstandResult = {
  reply: string;
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  health?: HealthUnderstanding;
  commerce?: CommerceUnderstanding;
  engine: "abena+llm" | "openai+llm" | "lexical+llm" | "llm" | "fallback";
  model?: string;
  replyLanguage: "tw" | "en";
  comprehension?: {
    understood: boolean;
    meaning?: string | null;
    uncertaintyReason?: string | null;
    model?: UnderstandingModelPrediction | null;
  };
  retrieve?: {
    engine: string;
    model?: string;
    articles: { slug: string; score: number }[];
    products: { id: string; score: number }[];
    cacheMisses?: number;
  };
  review?: {
    engine: "llm" | "fallback";
    revised: boolean;
  };
  synthesis?: {
    mode: "live_model" | "degraded_fallback";
    provider?: string;
    model?: string;
    usedHistory: boolean;
    usedMemory: boolean;
    safetyEnforced: boolean;
    understandingModel?: {
      mode: UnderstandingModelMode;
      used: boolean;
      model?: string;
      latencyMs?: number;
    };
  };
};

export type CommerceUnderstanding = {
  action: "buy" | "order" | "find" | "price" | "availability" | "unknown";
  item?: string;
  quantity?: string;
  location?: string;
  fulfillment?: "delivery" | "pickup" | "unknown";
  confidence: number;
  source: "deterministic";
  plan?: CommerceActionPlan;
};

type TranscriptQualityInput = {
  mode?: string;
  model?: string;
  latencyMs?: number;
  speaker?: string;
  language?: string;
  duration?: number;
  rms?: number;
  languageProbability?: number;
  route?: string;
};

const directTurnSchema = z.object({
  understood: z.boolean(),
  understoodMeaning: z.string().min(1).nullable(),
  uncertaintyReason: z.string().min(1).nullable(),
  reply: z.string().min(1).nullable(),
  intent: z.enum(["HEALTH", "ECOMMERCE", "GENERAL", "UNKNOWN"]),
  severity: z.enum(["LOW", "MEDIUM", "HIGH", "EMERGENCY"]),
  escalate: z.boolean(),
});

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

const SYSTEM_DIRECT = `Understand and respond directly to the latest message in its conversation. Handle natural Twi/Akan, English, and code-switching without translating aloud or discussing the wording. Treat ordinary spelling mistakes, omitted Twi diacritics, and informal keyboard substitutions such as 3 for ɛ as normal when the meaning remains clear. Reply simply and naturally in the requested language.

Start immediately with the useful answer or the one necessary follow-up question. The first clause must add new information or ask a question; it must never be a standalone acknowledgement. Do not acknowledge, repeat, paraphrase, summarize, or translate what the user just said before answering. Never restate the user's request in first person as though it were your own request. Use conversation history silently: a short message such as "MacBook" adds detail to the preceding request and should not cause you to repeat the combined request. If one detail is missing, ask only for that detail instead of first describing the item or request. Restate user information only when resolving a genuine ambiguity that cannot be handled with a direct question. Do not open with "Okay", "I understand", "You said", "Ɛyɛ", "Aane", "Aane, me tee ase", or "Mepa wo kyɛw". Bad: "Ɛyɛ. MacBook charger no..." Good: "MacBook Air anaa Pro, na afe bɛn?"

Set understood=false only when the core meaning or intent cannot be recovered. Missing details do not make a message unclear: when the request is understood but needs a brand, model, symptom detail, location, or other slot, set understood=true, preserve the recovered meaning, and put the direct follow-up question in reply. If the message is genuinely unclear, set understood=false and reply=null rather than guessing. For health, do not diagnose or prescribe dosage and treat explicit emergencies urgently. For commerce, do not invent prices, stores, or availability. Return JSON only:
Known Twi health meanings that must be preserved: "m'ani kum", "mani kum", and "ani kum" mean eye pain/eye ache in this product's health context, not sleepiness, sadness, or indifference. "abɔ waw" and "abo waw" mean cough/coughing. "mehome yɛ den" and "ahome yɛ den" mean difficulty breathing. "me koko mu yɛ me yaw" means chest pain, not heartburn.
{"understood":boolean,"understoodMeaning":"brief faithful English meaning for internal records or null","uncertaintyReason":"brief reason or null","reply":"direct response or null","intent":"HEALTH|ECOMMERCE|GENERAL|UNKNOWN","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

function dangerOverride(text: string) {
  const lower = normalizeHealthText(text);
  const pregnancy =
    /\b(pregnan|nyinsɛn|nyinsen|yafunu)\b/.test(lower);
  const severeHeadache =
    /\b(severe headache|bad headache|ti yɛ me yaw paa|ti yaw.*den|ti yare.*den)\b/.test(lower);
  const swelling =
    /\b(swelling|swollen|ahonhon|anim.*hon|nsa.*hon)\b/.test(lower);
  const bleeding = /\b(bleeding|mogya|blood)\b/.test(lower);
  const chestPain = /\b(chest pain|crushing chest|kokom.*yaw)\b/.test(lower);
  const breathing = /\b(can'?t breathe|breathing trouble|home.*den|ahome.*den)\b/.test(lower);

  if ((pregnancy && (severeHeadache || swelling || bleeding)) || chestPain || breathing) {
    return { severity: "EMERGENCY" as const, escalate: true };
  }
  return null;
}

function normalizeHealthText(text: string) {
  return text
    .toLowerCase()
    .replace(/\b3y3\b/g, "ɛyɛ")
    .replace(/\by3\b/g, "yɛ")
    .replace(/\bserius\b/g, "serious")
    .replace(/\s+/g, " ")
    .trim();
}

function hasEyePainPhrase(text: string) {
  return /\b(?:m'?ani|mani|ani)\s+kum\b/.test(normalizeHealthText(text));
}

function asksWhichHospital(text: string) {
  const lower = normalizeHealthText(text);
  return /\bhospital\b/.test(lower) && /\bb[eɛ]n\b|\bwhich\b|\bwhere\b|\bhe\b/.test(lower);
}

function continuesSeriousMigraine(
  text: string,
  history?: { role: "user" | "assistant"; content: string }[],
) {
  const lower = normalizeHealthText(text);
  const serious = /\bserious\b|\bden\b|\bpaa\b/.test(lower);
  const priorMigraine = (history ?? []).some((message) => /migraine/i.test(message.content));
  return serious && priorMigraine;
}

function hasCommerceIntent(text: string) {
  const lower = text.toLowerCase();
  return /\b(buy|order|shop|shopping|market|cart|checkout|price|cost|delivery|pickup|store|stores|available|availability|find|get)\b/.test(
    lower,
  ) || /\b(mepɛ|me pɛ|metɔ|me tɔ|mɛtɔ|mebɛtɔ|atɔ|tɔ|boɔ|bo|dwa|market|hwehwɛ|hwehwɛɛ|kra|fa brɛ|brɛ me)\b/.test(
    lower,
  );
}

function extractCommerceUnderstanding(text: string, focus: "health" | "commerce"): CommerceUnderstanding | undefined {
  if (focus !== "commerce" && !hasCommerceIntent(text)) return undefined;
  const lower = text.toLowerCase();
  const action = inferCommerceAction(lower);
  const item = extractCommerceItem(text);
  const quantity = extractCommerceQuantity(text);
  const location = extractCommerceLocation(text);
  const fulfillment = /\b(deliver|delivery|fa brɛ|brɛ me|bring|send)\b/i.test(text)
    ? "delivery"
    : /\b(pickup|pick up|come for|meba|mɛba)\b/i.test(text)
      ? "pickup"
      : "unknown";
  const confidence =
    0.35 +
    (action !== "unknown" ? 0.18 : 0) +
    (item ? 0.28 : 0) +
    (quantity ? 0.08 : 0) +
    (location ? 0.08 : 0) +
    (fulfillment !== "unknown" ? 0.03 : 0);

  return {
    action,
    ...(item ? { item } : {}),
    ...(quantity ? { quantity } : {}),
    ...(location ? { location } : {}),
    fulfillment,
    confidence: Math.min(0.92, Math.round(confidence * 100) / 100),
    source: "deterministic",
  };
}

function inferCommerceAction(lower: string): CommerceUnderstanding["action"] {
  if (/\b(price|cost|how much|boɔ|bo)\b/.test(lower)) return "price";
  if (/\b(available|availability|in stock|do you have)\b/.test(lower)) return "availability";
  if (/\b(find|search|hwehwɛ|hwehwɛɛ|near|bɛn)\b/.test(lower)) return "find";
  if (/\b(order|checkout|kra)\b/.test(lower)) return "order";
  if (/\b(buy|get|shop|metɔ|me tɔ|mɛtɔ|mebɛtɔ|atɔ|tɔ)\b/.test(lower)) return "buy";
  if (/(?:me|mɛ|mebɛ|mepɛ sɛ)\s*tɔ/.test(lower)) return "buy";
  return "unknown";
}

function extractCommerceItem(text: string) {
  const cleaned = text
    .replace(/[?.!,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const patterns = [
    /\bhow much is\s+(.+?)(?:\s+(?:near|around|in|at|for)\b|$)/i,
    /\bdo you have\s+(.+?)(?:\s+(?:near|around|in|at|for|in the market|available)\b|$)/i,
    /\bis\s+(.+?)\s+available\b/i,
    /\b(?:price|cost|boɔ|bo)\s+(?:of|for)?\s*(.+?)(?:\s+(?:near|around|in|at|for)\b|$)/i,
    /\b(?:buy|order|get|find|search for)\s+(.+?)(?:\s+(?:for me|online|near|around|in|at|to|for|ma me|ma|wɔ|wo|a ɛbɛn|bɛn)\b|$)/i,
    /\b(?:metɔ|me tɔ|mɛtɔ|mebɛtɔ|mepɛ sɛ metɔ|mepɛ sɛ me tɔ|mepɛ|me pɛ|hwehwɛ)\s+(.+?)(?:\s+(?:ma me|ma|wɔ|wo|a ɛbɛn|bɛn|online|near|around|for)\b|$)/i,
  ];
  for (const pattern of patterns) {
    const match = cleaned.match(pattern);
    const item = cleanSlot(match?.[1]);
    if (item) return item;
  }
  return undefined;
}

function extractCommerceQuantity(text: string) {
  const patterns = [
    /\b(\d+\s*(?:kg|kilo|kilos|crate|crates|bag|bags|pack|packs|pieces?|bunch(?:es)?|dozen|litres?|l|sachet(?:s)?))\b/i,
    /\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:kg|kilo|kilos|crate|crates|bag|bags|pack|packs|pieces?|bunch(?:es)?|dozen|litres?|l|sachet(?:s)?)\b/i,
    /\b(?:baako|mmienu|mmiɛnsa|anan|anum|nsia|nson|nwɔtwe|nkron|du)\s+(?:kilo|bag|pack|pieces?|bunch|crate)\b/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    const quantity = cleanSlot(match?.[0]);
    if (quantity) return quantity;
  }
  return undefined;
}

function extractCommerceLocation(text: string) {
  const patterns = [
    /\b(?:near|around|in|at)\s+([a-zA-ZɛɔƐƆŋŊ][\wɛɔƐƆŋŊ -]{2,40})/i,
    /\b(?:wɔ|wo|bɛn|a ɛbɛn)\s+([a-zA-ZɛɔƐƆŋŊ][\wɛɔƐƆŋŊ -]{2,40})/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    const location = cleanSlot(match?.[1]);
    if (location) return location;
  }
  return undefined;
}

function cleanSlot(value?: string) {
  const cleaned = value
    ?.replace(/\b(please|pls|ma me|for me|online|today|tomorrow|nnɛ|kyena)\b/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned || cleaned.length < 2) return undefined;
  return cleaned.slice(0, 80);
}

function inferFocus(text: string, explicit?: "health" | "commerce") {
  if (explicit) return explicit;
  return hasCommerceIntent(text) ? "commerce" : "health";
}

function applyIntentOverride(input: {
  text: string;
  focus: "health" | "commerce";
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  reply: string;
}) {
  if (input.focus === "commerce" && hasCommerceIntent(input.text)) {
    return {
      ...input,
      intent: "ECOMMERCE" as const,
      severity: input.severity === "EMERGENCY" ? input.severity : ("LOW" as const),
      escalate: input.severity === "EMERGENCY" ? input.escalate : false,
    };
  }
  return input;
}

function applySafetyOverride(input: {
  text: string;
  reply: string;
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  replyLang: "tw" | "en";
}) {
  const override = input.intent === "HEALTH" ? dangerOverride(input.text) : null;
  if (!override) return input;
  const prefix =
    input.replyLang === "en"
      ? `This may be urgent. Please go to a hospital now or call ${HOTLINE}. `
      : `Eyi betumi ayɛ emergency. Kɔ hospital ntɛm anaa frɛ ${HOTLINE}. `;
  return {
    ...input,
    severity: override.severity,
    escalate: override.escalate,
    reply: input.reply.startsWith(prefix) ? input.reply : `${prefix}${input.reply}`,
  };
}

function includesAnyTerm(text: string, terms: string[]) {
  const lower = text.toLowerCase();
  return terms.some((term) => lower.includes(term.toLowerCase()));
}

function applyHealthMeaningGuard(input: {
  text: string;
  reply: string;
  intent: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  replyLang: "tw" | "en";
  focus: "health" | "commerce";
  history?: { role: "user" | "assistant"; content: string }[];
}) {
  if (input.focus !== "health") return input;

  if (hasEyePainPhrase(input.text)) {
    const clinicLine =
      input.replyLang === "en"
        ? "If the eye pain is strong, worsening, or affects your vision, please go to a clinic today."
        : "Sɛ ani yaw no yɛ den, ɛrekɔ so, anaa ɛka w'ani so hwɛ a, kɔ clinic nnɛ.";
    return {
      ...input,
      intent: "HEALTH" as const,
      severity: input.severity === "EMERGENCY" ? input.severity : ("MEDIUM" as const),
      escalate: input.severity === "EMERGENCY" ? input.escalate : false,
      reply: includesAnyTerm(input.reply, ["ani", "eye"]) && includesAnyTerm(input.reply, ["clinic"])
        ? input.reply
        : `${input.reply} ${clinicLine}`,
    };
  }

  if (asksWhichHospital(input.text)) {
    const detailLine =
      input.replyLang === "en"
        ? "Tell me your town or area and the main symptom, then I can help you choose the right hospital level."
        : "Ka wo town anaa area ne symptom titiriw no, na mɛboa wo apaw hospital level a ɛfata.";
    return {
      ...input,
      intent: "HEALTH" as const,
      severity: input.severity === "EMERGENCY" ? input.severity : ("MEDIUM" as const),
      escalate: input.severity === "EMERGENCY" ? input.escalate : false,
      reply: includesAnyTerm(input.reply, ["town", "area", "symptom"]) ? input.reply : `${input.reply} ${detailLine}`,
    };
  }

  if (continuesSeriousMigraine(input.text, input.history)) {
    const clinicLine =
      input.replyLang === "en"
        ? "Because the migraine is serious, arrange clinic or hospital care today, especially if it is new, severe, or comes with weakness, fever, vision changes, or vomiting."
        : "Sɛ migraine no yɛ serious saa a, kɔ clinic anaa hospital nnɛ, titiriw sɛ ɛyɛ foforo, ɛyɛ den paa, anaa ɛka ahoɔden so tew, fever, ani so haw, anaa vomiting ho.";
    return {
      ...input,
      intent: "HEALTH" as const,
      severity: "HIGH" as const,
      escalate: true,
      reply: includesAnyTerm(input.reply, ["migraine"]) && includesAnyTerm(input.reply, ["clinic", "hospital"])
        ? input.reply
        : `${input.reply} ${clinicLine}`,
    };
  }

  return input;
}

function fallbackReply(input: {
  text: string;
  replyLang: "tw" | "en";
  focus: "health" | "commerce";
  commerce?: CommerceUnderstanding;
  transcript?: TranscriptQualityInput;
  history?: { role: "user" | "assistant"; content: string }[];
}) {
  const baseIntent = input.focus === "commerce" ? "ECOMMERCE" : "HEALTH";
  const safety = dangerOverride(input.text);
  if (safety) {
    return {
      reply:
        input.replyLang === "en"
          ? `The understanding model is unavailable, but the transcript contains an explicit danger sign. Please seek urgent care now or call ${HOTLINE}.`
          : `Understanding model no nni hɔ seesei, nanso transcript no mu wɔ danger sign pefee. Kɔ hospital ntɛm anaa frɛ ${HOTLINE}.`,
      intent: "HEALTH" as const,
      severity: "EMERGENCY" as const,
      escalate: true,
    };
  }

  return {
    reply:
      input.replyLang === "en"
        ? "The language understanding model is unavailable right now, so I will not guess what you meant. Please say it again in different words shortly."
        : "Language understanding model no nni hɔ seesei, enti merensusu nea wokyerɛe. Mesrɛ wo, san ka nea wohia no wɔ ɔkwan foforo so akyiri yi.",
    intent: baseIntent as "HEALTH" | "ECOMMERCE",
    severity: "LOW" as const,
    escalate: false,
  };
}

function fallbackUnderstanding(input: {
  text: string;
  replyLang: "tw" | "en";
  focus: "health" | "commerce";
  commerce?: CommerceUnderstanding;
  transcript?: TranscriptQualityInput;
  history?: { role: "user" | "assistant"; content: string }[];
  retrieveMeta: UnderstandResult["retrieve"];
}): UnderstandResult {
  const fallback = fallbackReply(input);
  const health =
    fallback.intent === "HEALTH"
      ? buildHealthUnderstanding({
          text: input.text,
          severity: fallback.severity,
          escalate: fallback.escalate,
          transcript: input.transcript,
          hotline: HOTLINE,
        })
      : undefined;

  return {
    reply: fallback.reply,
    intent: fallback.intent,
    severity: fallback.severity,
    escalate: fallback.escalate,
    health,
    commerce: fallback.intent === "ECOMMERCE" ? input.commerce : undefined,
    engine: "fallback",
    replyLanguage: input.replyLang,
    comprehension: {
      understood: false,
      meaning: null,
      uncertaintyReason: "language_understanding_unavailable",
    },
    retrieve: input.retrieveMeta,
    review: {
      engine: "fallback",
      revised: false,
    },
    synthesis: {
      mode: "degraded_fallback",
      usedHistory: Boolean(input.history?.length),
      usedMemory: false,
      safetyEnforced: true,
    },
  };
}

function parseJson(raw: string): unknown {
  try {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    return JSON.parse(raw.slice(start, end + 1)) as unknown;
  } catch {
    return null;
  }
}

function honestNotUnderstood(replyLang: "tw" | "en") {
  return replyLang === "en"
    ? "I did not understand that clearly enough to answer. Please say it again in different words."
    : "Mante nea wokae no ase yie sɛ memmua. Mesrɛ wo, san ka no wɔ ɔkwan foforo so.";
}

function isFailureReply(content: string) {
  const normalized = content.toLowerCase().replace(/\s+/g, " ").trim();
  return (
    normalized.includes("mante nea wokae no ase") ||
    normalized.includes("i did not understand that clearly enough") ||
    normalized.includes("service is temporarily unavailable") ||
    normalized.includes("ntumi mmua wo seesei")
  );
}

function engineFromRetrieve(
  retrieveEngine: string,
  hasLlm: boolean,
): UnderstandResult["engine"] {
  if (!hasLlm) return "fallback";
  if (retrieveEngine === "none") return "llm";
  if (retrieveEngine === "abena") return "abena+llm";
  if (retrieveEngine === "openai") return "openai+llm";
  if (retrieveEngine === "lexical") return "lexical+llm";
  return "llm";
}

export async function understandUtterance(input: {
  text: string;
  language?: LanguageCode;
  history?: { role: "user" | "assistant"; content: string }[];
  memory?: unknown;
  transcript?: TranscriptQualityInput;
  focus?: "health" | "commerce";
  instruction?: string;
  understandingModelMode?: UnderstandingModelMode;
}): Promise<UnderstandResult> {
  const language = input.language ?? "tw";
  const provider = llmProviderInfo();
  const replyLang = resolveReplyLanguage(language);
  const focus = inferFocus(input.text, input.focus);
  const modelMode = input.understandingModelMode ?? understandingModelMode();
  const commerce = withCommercePlan(extractCommerceUnderstanding(input.text, focus));
  const retrieveMeta = {
    engine: "none",
    model: undefined,
    articles: [],
    products: [],
    cacheMisses: 0,
  };
  const history = (input.history ?? [])
    .filter((message) => message.role === "user" || !isFailureReply(message.content))
    .slice(-6)
    .map((m) => ({
      role: m.role === "user" ? ("user" as const) : ("assistant" as const),
      content: m.content,
    }));
  const modelPrediction =
    modelMode === "assist"
      ? await recoverUnderstandingWithModel({
          text: input.text,
          language,
          focus,
          history,
          memory: input.memory,
          transcript: input.transcript
            ? {
                language: input.transcript.language,
                languageProbability: input.transcript.languageProbability,
                model: input.transcript.model,
                route: input.transcript.route,
              }
            : undefined,
        })
      : modelMode === "assist_v1"
        ? await recoverUnderstandingWithModel({
            text: input.text,
            language,
            focus,
            history,
            memory: input.memory,
            mode: modelMode,
            transcript: input.transcript
              ? {
                  language: input.transcript.language,
                  languageProbability: input.transcript.languageProbability,
                  model: input.transcript.model,
                  route: input.transcript.route,
                }
              : undefined,
          })
      : null;

  if (!isLlmConfigured()) {
    return fallbackUnderstanding({
      text: input.text,
      replyLang,
      focus,
      commerce,
      transcript: input.transcript,
      retrieveMeta,
      history: input.history,
    });
  }
  const modelHint =
    modelMode === "assist" || modelMode === "assist_v1"
      ? formatUnderstandingModelHint(modelPrediction)
      : "";

  const directRaw = await chatComplete(
    [
      {
        role: "system",
        content: [
          SYSTEM_DIRECT,
          `Requested reply language: ${replyLang === "en" ? "English" : "Twi/Akan"}.`,
          modelHint,
        ]
          .filter(Boolean)
          .join("\n\n"),
      },
      ...history,
      { role: "user", content: input.text },
    ],
    { temperature: 0.3, maxTokens: 650 },
  );

  if (!directRaw) {
    return fallbackUnderstanding({
      text: input.text,
      replyLang,
      focus,
      commerce,
      transcript: input.transcript,
      retrieveMeta,
      history: input.history,
    });
  }

  const direct = directTurnSchema.safeParse(parseJson(directRaw));
  if (
    !direct.success ||
    !direct.data.understood ||
    !direct.data.reply
  ) {
    const uncertaintyReason = direct.success
      ? direct.data.uncertaintyReason ?? "meaning_not_recovered"
      : "invalid_understanding_output";
    const intent = direct.success ? direct.data.intent : "UNKNOWN";
    const health =
      focus === "health"
        ? buildHealthUnderstanding({
            text: input.text,
            severity: "LOW",
            escalate: false,
            transcript: input.transcript,
            hotline: HOTLINE,
          })
        : undefined;
    return {
      reply: honestNotUnderstood(replyLang),
      intent,
      severity: "LOW",
      escalate: false,
      health,
      engine: "llm",
      model: provider?.model,
      replyLanguage: replyLang,
      comprehension: {
        understood: false,
        meaning: null,
        uncertaintyReason,
        model: modelPrediction,
      },
      retrieve: retrieveMeta,
      review: { engine: "fallback", revised: false },
      synthesis: {
        mode: "live_model",
        provider: provider?.provider,
        model: provider?.model,
        usedHistory: history.length > 0,
        usedMemory: Boolean(input.memory),
        safetyEnforced: true,
        understandingModel: {
          mode: modelMode,
          used: Boolean(modelPrediction),
          model: modelPrediction?.model,
          latencyMs: modelPrediction?.latencyMs,
        },
      },
    };
  }

  const intentChecked = applyIntentOverride({
    text: input.text,
    focus,
    reply: direct.data.reply.trim(),
    intent: direct.data.intent,
    severity: direct.data.severity,
    escalate: direct.data.escalate,
  });
  const guarded = applyHealthMeaningGuard({
    ...intentChecked,
    text: input.text,
    replyLang,
    focus,
    history: input.history,
  });
  const final = applySafetyOverride({
    text: input.text,
    reply: guarded.reply.trim(),
    intent: guarded.intent,
    severity: guarded.severity,
    escalate: guarded.escalate,
    replyLang,
  });
  const health =
    final.intent === "HEALTH"
      ? buildHealthUnderstanding({
          text: input.text,
          severity: final.severity,
          escalate: final.escalate,
          transcript: input.transcript,
          hotline: HOTLINE,
        })
      : undefined;

  return {
    reply: final.reply.trim(),
    intent: final.intent,
    severity: final.severity,
    escalate: final.escalate,
    health,
    commerce: final.intent === "ECOMMERCE" ? commerce : undefined,
    engine: engineFromRetrieve("none", true),
    model: provider?.model,
    replyLanguage: replyLang,
    comprehension: {
      understood: true,
      meaning: direct.data.understoodMeaning ?? modelPrediction?.naturalEnglish ?? null,
      uncertaintyReason: null,
      model: modelPrediction,
    },
    retrieve: retrieveMeta,
    review: {
      engine: "fallback",
      revised: false,
    },
    synthesis: {
      mode: "live_model",
      provider: provider?.provider,
      model: provider?.model,
      usedHistory: history.length > 0,
      usedMemory: Boolean(input.memory),
      safetyEnforced: true,
      understandingModel: {
        mode: modelMode,
        used: Boolean(modelPrediction),
        model: modelPrediction?.model,
        latencyMs: modelPrediction?.latencyMs,
      },
    },
  };
}

function withCommercePlan(commerce: CommerceUnderstanding | undefined) {
  if (!commerce) return undefined;
  return {
    ...commerce,
    plan: buildCommerceActionPlan(commerce),
  };
}
