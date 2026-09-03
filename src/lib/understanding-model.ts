import type { LanguageCode } from "@prisma/client";
import { z } from "zod";

export type UnderstandingModelPrediction = {
  normalizedTwi?: string;
  naturalEnglish?: string;
  literalEnglish?: string;
  intent?: string;
  entities?: Record<string, string | number | boolean | null>;
  ambiguities?: string;
  requiresClarification?: boolean;
  model?: string;
  latencyMs?: number;
};

const entityValueSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);

const predictionSchema = z.object({
  normalized_twi: z.string().optional().nullable(),
  normalizedTwi: z.string().optional().nullable(),
  natural_english: z.string().optional().nullable(),
  naturalEnglish: z.string().optional().nullable(),
  literal_english: z.string().optional().nullable(),
  literalEnglish: z.string().optional().nullable(),
  intent: z.string().optional().nullable(),
  entities: z.record(z.string(), entityValueSchema).optional().nullable(),
  ambiguities: z.string().optional().nullable(),
  requires_clarification: z.boolean().optional().nullable(),
  requiresClarification: z.boolean().optional().nullable(),
  model: z.string().optional().nullable(),
  latency_ms: z.number().optional().nullable(),
  latencyMs: z.number().optional().nullable(),
});

function cleanUrl(value: string | undefined) {
  return value?.replace(/\/$/, "") || "";
}

export function isUnderstandingModelConfigured() {
  return Boolean(cleanUrl(process.env.UNDERSTANDING_MODEL_URL));
}

export function understandingModelMode(): "shadow" | "assist" {
  return process.env.UNDERSTANDING_MODEL_MODE === "assist" ? "assist" : "shadow";
}

export async function recoverUnderstandingWithModel(input: {
  text: string;
  language: LanguageCode;
  focus: "health" | "commerce";
  history?: { role: "user" | "assistant"; content: string }[];
  memory?: unknown;
  transcript?: {
    language?: string;
    languageProbability?: number;
    model?: string;
    route?: string;
  };
}): Promise<UnderstandingModelPrediction | null> {
  const baseUrl = cleanUrl(process.env.UNDERSTANDING_MODEL_URL);
  if (!baseUrl) return null;

  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    Number(process.env.UNDERSTANDING_MODEL_TIMEOUT_MS || 12_000),
  );

  try {
    const res = await fetch(`${baseUrl}/understand`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(process.env.UNDERSTANDING_MODEL_TOKEN
          ? { Authorization: `Bearer ${process.env.UNDERSTANDING_MODEL_TOKEN}` }
          : {}),
      },
      body: JSON.stringify({
        text: input.text,
        language: input.language,
        focus: input.focus,
        history: input.history?.slice(-6) ?? [],
        memory: input.memory ?? null,
        transcript: input.transcript ?? null,
      }),
    });
    const raw = (await res.json().catch(() => null)) as unknown;
    if (!res.ok || !raw) {
      console.error("[understanding-model]", res.status, raw);
      return null;
    }

    const parsed = predictionSchema.safeParse(raw);
    if (!parsed.success) {
      console.error("[understanding-model] invalid response", parsed.error.issues);
      return null;
    }
    const data = parsed.data;
    return {
      normalizedTwi: data.normalized_twi ?? data.normalizedTwi ?? undefined,
      naturalEnglish: data.natural_english ?? data.naturalEnglish ?? undefined,
      literalEnglish: data.literal_english ?? data.literalEnglish ?? undefined,
      intent: data.intent ?? undefined,
      entities: data.entities ?? undefined,
      ambiguities: data.ambiguities ?? undefined,
      requiresClarification:
        data.requires_clarification ?? data.requiresClarification ?? undefined,
      model: data.model ?? undefined,
      latencyMs: data.latency_ms ?? data.latencyMs ?? undefined,
    };
  } catch (error) {
    console.error("[understanding-model]", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export function formatUnderstandingModelHint(
  prediction: UnderstandingModelPrediction | null,
) {
  if (!prediction) return "";
  return [
    "Candidate semantic recovery from the project-trained Twi understanding model.",
    "Use it silently as a hint only; do not mention it to the user.",
    prediction.normalizedTwi ? `normalized_twi: ${prediction.normalizedTwi}` : "",
    prediction.naturalEnglish ? `natural_english: ${prediction.naturalEnglish}` : "",
    prediction.literalEnglish ? `literal_english: ${prediction.literalEnglish}` : "",
    prediction.intent ? `intent: ${prediction.intent}` : "",
    prediction.entities ? `entities: ${JSON.stringify(prediction.entities)}` : "",
    prediction.ambiguities ? `ambiguities: ${prediction.ambiguities}` : "",
    typeof prediction.requiresClarification === "boolean"
      ? `requires_clarification: ${prediction.requiresClarification}`
      : "",
  ]
    .filter(Boolean)
    .join("\n");
}
