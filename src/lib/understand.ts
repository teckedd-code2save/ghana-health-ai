/**
 * Understanding — research path.
 *
 * 1. Speech transcript + ASR quality first.
 * 2. LLM understanding decides intent, safety, and clarification.
 * 3. Retrieval is optional grounding, not the product brain.
 *
 * See docs/research-stack.md
 */

import { chatComplete, isLlmConfigured } from "@/lib/llm";
import { buildCommerceActionPlan, type CommerceActionPlan } from "@/lib/commerce-plan";
import { buildHealthUnderstanding, type HealthUnderstanding } from "@/lib/health-plan";
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

const understandSchema = z.object({
  reply: z.string().min(1),
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

const SYSTEM_TWI = `Wo yɛ Ghana Health AI — voice companion ma Ghana.
Ka **Twi** a ɛte sɛ kasa a nnipa ka. Light English mix nko sɛ ɛhia (din, aduro din).

Wobenya:
- asɛm a user no kae (ASR betumi ayɛ noise)
- transcript_quality: ASR confidence, audio length, loudness, route/model
- selected_focus: "health" anaa "commerce"; di focus_instruction so sɛ ɛne safety rules no hyia
- commerce mode yɛ shopping/action understanding; no connected store/search source yet
- memory: nneɛma a user aka dada; fa boa continuity nko ara, mma ɛnsesa asɛm foforo a ɔreka

Rules:
1. Gye user asɛm no so. Sɛ ɛyɛ noise / menteee a, bisa wɔn nka bio prɛko.
2. Health mode: fa transcript no ankasa na kyerɛ nea wote ase. Sɛ wunte ase yie a, bisa asɛmmisa baako; nsusuw sɛnea ɛte.
3. Nnyɛ oduruyɛfoɔ. Mma drug dosage. Mma definitive diagnosis.
4. Sɛ transcript_quality kyerɛ sɛ audio no yɛ tiaa, ɛyɛ quiet, anaa language confidence yɛ low a, na asɛm no mu nna hɔ a, bisa clarifying question baako; nnyɛ medical advice fi guess so.
5. Sɛ asɛm no yɛ emergency anaa danger sign a, ma hospital / ${HOTLINE} nyɛ asɛm a edi kan.
6. No canned closers ("Me ho yɛ hɔ", "Feel free to ask").
7. "community health worker", "antenatal care" — no bare CHW/ANC.
8. 2–5 nkyerɛaseɛ a wɔbɛtumi aka.
9. Sɛ selected_focus yɛ commerce a, te nea ɔpɛ sɛ ɔtɔ no ase. Yi item, quantity, location, delivery/pickup needs. Nkyerɛ price/store sɛ source nni hɔ; bisa next question baako.
10. Memory yɛ context nko ara; sɛ current transcript ne memory nhyia a, di current transcript so.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

const SYSTEM_EN = `You are Ghana Health AI. The user prefers **English** — reply in clear simple English only.

Use the transcript and ASR quality first. Not a doctor. No dosages/diagnoses.
The payload includes selected_focus and focus_instruction. Follow them when they do not conflict with safety.
The payload may include memory from earlier turns. Use it for continuity only; never let memory override the current transcript or become a medical fact.
If transcript_quality shows short, quiet, or low-confidence speech and the transcript is unclear, ask one brief clarifying question instead of guessing.
Emergencies: lead with hospital / ${HOTLINE}.
Commerce focus: understand the shopping request, item, quantity, location, delivery/pickup needs. Do not invent stores, prices, or availability; ask one useful next question when ordering/search is not connected.
No canned closers. 2–5 short spoken sentences.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

const SYSTEM_REVIEW = `You are the final safety and product-quality reviewer for Ghana Health AI.
Review the draft against the transcript, retrieved context, and language target.

Rules:
- Return the final user-facing answer, not commentary about the review.
- If this is health guidance, keep it general and do not diagnose or give drug dosage.
- If the transcript suggests urgent danger, the final answer must lead with seeking urgent care.
- If the transcript is unclear or ASR quality is weak, the final answer should ask a brief clarifying question.
- If transcript_quality shows weak ASR evidence, do not invent health facts beyond what the transcript supports.
- If this is ecommerce, do not invent stores, prices, or availability. Preserve shopping intent and ask the next useful question.
- Preserve the target language: English only for english, otherwise Twi.
- Avoid canned closers and acronyms.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL|UNKNOWN","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

const SYSTEM_REPAIR = `Convert the supplied model output into this exact JSON schema.
Do not add new medical or product claims. Preserve the reply meaning.
Choose exactly one intent and exactly one severity. Never copy enum options as the value.

JSON only:
{"reply":"...","intent":"HEALTH|ECOMMERCE|GENERAL|UNKNOWN","severity":"LOW|MEDIUM|HIGH|EMERGENCY","escalate":boolean}`;

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

function hasHistoryTopic(history: { role: "user" | "assistant"; content: string }[] | undefined, pattern: RegExp) {
  return (history ?? []).some((turn) => pattern.test(normalizeHealthText(turn.content)));
}

function healthReplyOverride(input: {
  text: string;
  replyLang: "tw" | "en";
  history?: { role: "user" | "assistant"; content: string }[];
}) {
  const lower = normalizeHealthText(input.text);
  const danger = dangerOverride(input.text);
  if (danger) return null;

  const eyePain =
    /\b(?:m'?ani|mani|ani)\s+(?:kum|yɛ\s+me\s+yaw|yare|red|hye)\b/.test(lower) ||
    /\beye\s+(?:pain|hurts?|red)\b/.test(lower);
  if (eyePain) {
    return {
      reply:
        input.replyLang === "en"
          ? "I understand you have strong eye pain. Please avoid rubbing the eye and do not put medicine in it unless a clinician gave it to you. If your vision is blurred, there is injury, swelling, or the pain is severe, go to a clinic or eye unit today."
          : "Mate ase sɛ w'ani yɛ wo yaw paa. Mfa wo nsa nnya ani no mu, na mfa aduro ngu mu gye sɛ clinician na ɔmaa wo. Sɛ w'ani so yɛ kusuu, biribi apira no, ahonhon wɔ hɔ, anaa yaw no yɛ den paa a, kɔ clinic anaa eye unit nnɛ.",
      severity: "MEDIUM" as const,
      escalate: false,
    };
  }

  const hospitalChoice =
    /\bhospital\b.*\b(?:bɛn|ben|where|which|menkɔ|nkɔ|go)\b/.test(lower) ||
    /\b(?:yɛsɛ|ɛsɛ|should)\b.*\bhospital\b/.test(lower);
  if (hospitalChoice) {
    return {
      reply:
        input.replyLang === "en"
          ? "Tell me your town or area and the main symptom, then I can guide the next step. If breathing is difficult, bleeding is heavy, there is chest pain, pregnancy danger signs, or a child is very weak, go to the nearest emergency unit now or call the emergency line."
          : "Ka wo town anaa area ne symptom titiriw no, na mɛkyerɛ wo step a edi hɔ. Sɛ ahome yɛ den, mogya retu bebree, koko mu yaw wɔ hɔ, pregnancy danger signs wɔ hɔ, anaa abofra ayɛ mmerɛ paa a, kɔ emergency unit a ɛbɛn wo seesei anaa frɛ emergency line.",
      severity: "MEDIUM" as const,
      escalate: false,
    };
  }

  const seriousFollowUp =
    /\b(?:ɛyɛ|eye|it'?s|it is)\s+(?:serious|den|bad)\b/.test(lower) ||
    /\bserious\s+paa\b/.test(lower);
  const migraineOrHeadache = /\b(?:migraine|headache|ti\s+(?:yaw|yare)|ti\s+yɛ\s+me\s+yaw)\b/;
  if (seriousFollowUp && hasHistoryTopic(input.history, migraineOrHeadache)) {
    return {
      reply:
        input.replyLang === "en"
          ? "If the migraine is that severe, please get checked at a clinic or hospital today. Go urgently if it started suddenly, your vision changes, you feel weak on one side, you are confused, fever comes with neck stiffness, or you are pregnant."
          : "Sɛ migraine no yɛ den saa a, kɔ clinic anaa hospital nnɛ ma wɔnhwɛ wo. Kɔ ntɛm paa sɛ ɛfirii ase prɛko pɛ, w'ani so resesa, wo fã baako ayɛ mmerɛ, wo tirim ayɛ wo basaa, fever ne kɔn mu den ka ho, anaa woyɛ nyinsɛn.",
      severity: "HIGH" as const,
      escalate: true,
    };
  }

  return null;
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
  const severity = safety?.severity ?? ("LOW" as const);
  const escalate = safety?.escalate ?? false;

  if (baseIntent === "ECOMMERCE") {
    const plan = input.commerce?.plan;
    const item = input.commerce?.item;
    const missing = plan?.missing?.[0];
    const reply =
      input.replyLang === "en"
        ? missing === "item"
          ? "What item do you want me to help you find or buy?"
          : missing === "quantity"
            ? `How much ${item ?? "of it"} do you want?`
            : missing === "location"
              ? `Where should I search or deliver ${item ?? "it"}?`
              : item
                ? `I understand you want help with ${item}. I can check the connected catalog and ask you to confirm before anything is added to cart.`
                : "Tell me the item, quantity, and location so I can help with the shopping request."
        : missing === "item"
          ? "Dɛn na wopɛ sɛ meboa wo hwehwɛ anaa tɔ?"
          : missing === "quantity"
            ? `${item ?? "Adeɛ no"} dodow sɛn na wopɛ?`
            : missing === "location"
              ? `Ɛhe na memfa ${item ?? "adeɛ no"} nhwehwɛ anaa menfa mmrɛ wo?`
              : item
                ? `Mate ase sɛ wopɛ mmoa wɔ ${item} ho. Mɛhwehwɛ connected catalog no mu, na mɛma wo confirm ansa na biribi akɔ cart mu.`
                : "Ka adeɛ no, dodow, ne beaeɛ no na memmoa wo wɔ shopping no ho.";

    return {
      reply,
      intent: "ECOMMERCE" as const,
      severity: "LOW" as const,
      escalate: false,
    };
  }

  const override = healthReplyOverride({
    text: input.text,
    replyLang: input.replyLang,
    history: input.history,
  });
  if (override) {
    return {
      reply: override.reply,
      intent: "HEALTH" as const,
      severity: override.severity,
      escalate: override.escalate,
    };
  }

  const health = buildHealthUnderstanding({
    text: input.text,
    severity,
    escalate,
    transcript: input.transcript,
    hotline: HOTLINE,
  });

  if (health.plan.status === "needs_clarification") {
    return {
      reply:
        input.replyLang === "en"
          ? "I did not hear that clearly. Please repeat the main symptom or question in one short sentence."
          : "Mante no yie. Mesrɛ wo, san ka symptom anaa asɛmmisa titiriw no wɔ sentence tiawa baako mu.",
      intent: "HEALTH" as const,
      severity: "LOW" as const,
      escalate: false,
    };
  }

  if (health.plan.status === "urgent_referral") {
    return {
      reply:
        input.replyLang === "en"
          ? `This may be urgent. Please go to a hospital now or call ${HOTLINE}.`
          : `Eyi betumi ayɛ emergency. Kɔ hospital ntɛm anaa frɛ ${HOTLINE}.`,
      intent: "HEALTH" as const,
      severity: "EMERGENCY" as const,
      escalate: true,
    };
  }

  if (health.plan.status === "clinic_recommended") {
    return {
      reply:
        input.replyLang === "en"
          ? "This may need a clinician to check. Please visit a clinic soon, especially if it is getting worse or the person is weak."
          : "Ɛbɛhia sɛ nurse anaa doctor hwɛ no. Kɔ clinic ntɛm, titiriw sɛ ɛreyɛ den anaa onipa no ayɛ mmerɛ.",
      intent: "HEALTH" as const,
      severity: health.plan.urgency === "urgent" ? ("HIGH" as const) : ("MEDIUM" as const),
      escalate: health.plan.urgency === "urgent",
    };
  }

  return {
    reply:
      input.replyLang === "en"
        ? "I can give only general guidance from what I heard. Rest, drink fluids, and watch the symptoms; if it worsens or you are worried, visit a clinic."
        : "Nea mate no nti, metumi ama general advice nko ara. Gye w'ahome, nom nsuo, na hwɛ sɛ symptom no resesa; sɛ ɛyɛ den a, kɔ clinic.",
    intent: "HEALTH" as const,
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
    retrieve: input.retrieveMeta,
    review: {
      engine: "fallback",
      revised: false,
    },
  };
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

function normalizeIntent(value: unknown) {
  if (typeof value !== "string") return value;
  const upper = value.toUpperCase().replace(/[^A-Z]/g, "");
  if (upper.includes("ECOMMERCE")) return "ECOMMERCE";
  if (upper.includes("HEALTH")) return "HEALTH";
  if (upper.includes("GENERAL")) return "GENERAL";
  if (upper.includes("UNKNOWN")) return "UNKNOWN";
  if (["SHOPPING", "MARKET", "PRODUCT", "ORDER", "COMMERCE"].includes(upper)) {
    return "ECOMMERCE";
  }
  if (["MEDICAL", "SYMPTOM", "SAFETY"].includes(upper)) return "HEALTH";
  if (["OTHER", "CHAT"].includes(upper)) return "GENERAL";
  return upper;
}

function normalizeSeverity(value: unknown) {
  if (typeof value !== "string") return value;
  const upper = value.toUpperCase().replace(/[^A-Z]/g, "");
  if (["NONE", "NONMEDICAL", "NORMAL"].includes(upper)) return "LOW";
  if (["URGENT", "CRITICAL"].includes(upper)) return "EMERGENCY";
  return upper;
}

function normalizeEscalate(value: unknown) {
  if (typeof value !== "string") return value;
  const lower = value.trim().toLowerCase();
  if (["true", "yes", "y"].includes(lower)) return true;
  if (["false", "no", "n"].includes(lower)) return false;
  return value;
}

function normalizeCandidate(parsed: Partial<UnderstandResult> | null) {
  if (!parsed) return null;
  const record = parsed as Partial<UnderstandResult> & {
    answer?: unknown;
    response?: unknown;
    message?: unknown;
  };
  const reply =
    typeof record.reply === "string"
      ? record.reply
      : typeof record.answer === "string"
        ? record.answer
        : typeof record.response === "string"
          ? record.response
          : typeof record.message === "string"
            ? record.message
            : record.reply;
  const intent = normalizeIntent(parsed.intent);
  const neutralSafetyIntent =
    intent === "ECOMMERCE" || intent === "GENERAL" || intent === "UNKNOWN";
  const severity = normalizeSeverity(parsed.severity);
  const escalate = normalizeEscalate(parsed.escalate);
  return {
    ...parsed,
    reply,
    intent,
    severity: neutralSafetyIntent
      ? ["LOW", "MEDIUM", "HIGH", "EMERGENCY"].includes(String(severity))
        ? severity
        : "LOW"
      : severity,
    escalate: neutralSafetyIntent
      ? typeof escalate === "boolean"
        ? escalate
        : false
      : escalate,
  };
}

async function parseOrRepairModelJson(input: {
  raw: string;
  label: "draft" | "reviewed";
  context: unknown;
}) {
  const checked = understandSchema.safeParse(normalizeCandidate(parseJson(input.raw)));
  if (checked.success) return checked.data;

  const repairedRaw = await chatComplete(
    [
      { role: "system", content: SYSTEM_REPAIR },
      {
        role: "user",
        content: JSON.stringify({
          label: input.label,
          context: input.context,
          model_output: input.raw,
          validation_error: checked.error.flatten(),
        }),
      },
    ],
    { temperature: 0, maxTokens: 400 },
  );

  if (!repairedRaw) {
    throw new Error(`LLM did not return a repaired ${input.label} response.`);
  }

  const repaired = understandSchema.safeParse(
    normalizeCandidate(parseJson(repairedRaw)),
  );
  if (!repaired.success) {
    throw new Error(`LLM returned an invalid ${input.label} response schema.`);
  }

  return repaired.data;
}

export async function understandUtterance(input: {
  text: string;
  language?: LanguageCode;
  history?: { role: "user" | "assistant"; content: string }[];
  memory?: unknown;
  transcript?: TranscriptQualityInput;
  focus?: "health" | "commerce";
  instruction?: string;
}): Promise<UnderstandResult> {
  const language = input.language ?? "tw";
  const replyLang = resolveReplyLanguage(language);
  const focus = inferFocus(input.text, input.focus);
  const commerce = withCommercePlan(extractCommerceUnderstanding(input.text, focus));
  const retrieveMeta = {
    engine: "none",
    model: undefined,
    articles: [],
    products: [],
    cacheMisses: 0,
  };

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

  const history = (input.history ?? []).slice(-6).map((m) => ({
    role: m.role === "user" ? ("user" as const) : ("assistant" as const),
    content: m.content,
  }));

  const userPayload = {
    asr_transcript: input.text,
    transcript_quality: input.transcript
      ? {
          mode: input.transcript.mode,
          asr_model: input.transcript.model,
          asr_language: input.transcript.language,
          language_probability: input.transcript.languageProbability,
          asr_route: input.transcript.route,
          duration_seconds: input.transcript.duration,
          rms: input.transcript.rms,
          latency_ms: input.transcript.latencyMs,
        }
      : null,
    speak_language: replyLang === "en" ? "english" : "twi",
    english_allowed: replyLang === "en",
    selected_focus: focus,
    focus_instruction: input.instruction ?? null,
    commerce_understanding: commerce ?? null,
    memory: input.memory ?? null,
    retrieve_engine: "none",
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

  const checked = await parseOrRepairModelJson({
    raw,
    label: "draft",
    context: userPayload,
  });

  const reviewPayload = {
    transcript: input.text,
    transcript_quality: userPayload.transcript_quality,
    target_language: replyLang === "en" ? "english" : "twi",
    selected_focus: focus,
    focus_instruction: input.instruction ?? null,
    commerce_understanding: commerce ?? null,
    memory: input.memory ?? null,
    draft: checked,
    retrieve_engine: "none",
  };

  const reviewedRaw = await chatComplete(
    [
      { role: "system", content: SYSTEM_REVIEW },
      { role: "user", content: JSON.stringify(reviewPayload) },
    ],
    { temperature: 0.1, maxTokens: 400 },
  );

  if (!reviewedRaw) {
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

  const reviewed = await parseOrRepairModelJson({
    raw: reviewedRaw,
    label: "reviewed",
    context: reviewPayload,
  });
  const intentChecked = applyIntentOverride({
    text: input.text,
    focus,
    reply: reviewed.reply.trim(),
    intent: reviewed.intent,
    severity: reviewed.severity,
    escalate: reviewed.escalate,
  });
  const healthOverride =
    intentChecked.intent === "HEALTH"
      ? healthReplyOverride({ text: input.text, replyLang, history: input.history })
      : null;
  const responseChecked = healthOverride
    ? {
        ...intentChecked,
        reply: healthOverride.reply,
        severity: healthOverride.severity,
        escalate: healthOverride.escalate,
      }
    : intentChecked;
  const final = applySafetyOverride({
    text: input.text,
    reply: responseChecked.reply.trim(),
    intent: responseChecked.intent,
    severity: responseChecked.severity,
    escalate: responseChecked.escalate,
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
    replyLanguage: replyLang,
    retrieve: retrieveMeta,
    review: {
      engine: "llm",
      revised: final.reply.trim() !== checked.reply.trim(),
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
