import { isLlmConfigured } from "@/lib/llm";
import { isModalAsrConfigured, resolveModalAsrRoute } from "@/lib/modal-asr";
import { isModalTtsConfigured, resolveTtsRoute } from "@/lib/modal-tts";
import { isAbenaConfigured, isEmbedConfigured } from "@/lib/embed";
import { prisma } from "@/db/prisma";

export type ReadinessLevel = "pass" | "degraded" | "fail";

export type ReadinessCheck = {
  key: string;
  label: string;
  level: ReadinessLevel;
  required: boolean;
  detail: string;
};

export type ProductReadiness = {
  ok: boolean;
  status: "ready" | "degraded" | "blocked";
  service: "ghana-health-ai";
  mode: "voice-first";
  checks: ReadinessCheck[];
  summary: {
    requiredPassed: number;
    requiredTotal: number;
    degraded: number;
    failed: number;
  };
  ts: string;
};

function check(input: ReadinessCheck): ReadinessCheck {
  return input;
}

function summarize(checks: ReadinessCheck[]): ProductReadiness["summary"] {
  const required = checks.filter((item) => item.required);
  return {
    requiredPassed: required.filter((item) => item.level === "pass").length,
    requiredTotal: required.length,
    degraded: checks.filter((item) => item.level === "degraded").length,
    failed: checks.filter((item) => item.level === "fail").length,
  };
}

export async function getProductReadiness(): Promise<ProductReadiness> {
  let dbOk = false;
  try {
    await prisma.$queryRaw`SELECT 1`;
    dbOk = true;
  } catch {
    dbOk = false;
  }

  const twiRoute = resolveModalAsrRoute("tw");
  const englishRoute = resolveModalAsrRoute("en");
  const twiTtsRoute = resolveTtsRoute("tw");
  const llm = isLlmConfigured();

  const checks: ReadinessCheck[] = [
    check({
      key: "db",
      label: "Database",
      level: dbOk ? "pass" : "fail",
      required: true,
      detail: dbOk ? "Database query succeeded." : "Database query failed.",
    }),
    check({
      key: "asr_twi",
      label: "Twi ASR",
      level: isModalAsrConfigured() && twiRoute?.name === "twi-default" ? "pass" : "fail",
      required: true,
      detail: twiRoute
        ? `Route ${twiRoute.name} is configured.`
        : "No Twi ASR route is configured.",
    }),
    check({
      key: "asr_english",
      label: "English ASR",
      level: englishRoute?.name === "english" ? "pass" : "fail",
      required: true,
      detail:
        englishRoute?.name === "english"
          ? "Separate English ASR route is configured."
          : "English would not use the separate ASR route.",
    }),
    check({
      key: "tts",
      label: "TTS",
      level: isModalTtsConfigured() ? "pass" : "fail",
      required: true,
      detail: isModalTtsConfigured()
        ? `TTS route is configured: ${twiTtsRoute?.provider ?? "unknown"} (${twiTtsRoute?.modelLabel ?? "unknown"}).`
        : "No TTS route is configured.",
    }),
    check({
      key: "llm",
      label: "LLM Response Model",
      level: llm ? "pass" : "degraded",
      required: true,
      detail: llm
        ? "LLM provider is configured."
        : "LLM is missing; deterministic fallback will answer with reduced quality.",
    }),
    check({
      key: "response_fallback",
      label: "Response Fallback",
      level: "pass",
      required: true,
      detail: "Health and commerce fallback paths are implemented for LLM outage/no-key cases.",
    }),
    check({
      key: "abena_embed",
      label: "ABENA Embed",
      level: isAbenaConfigured() ? "pass" : isEmbedConfigured() ? "degraded" : "fail",
      required: false,
      detail: isAbenaConfigured()
        ? "ABENA Modal embed route is configured."
        : isEmbedConfigured()
          ? "OpenAI embedding fallback is configured; ABENA is not active."
          : "No embedding path is configured.",
    }),
    check({
      key: "asr_model_promotion",
      label: "ASR Model Promotion",
      level: "degraded",
      required: false,
      detail:
        "No ASR checkpoint currently passes all promotion gates; v6 remains hold-and-validate for Twi.",
    }),
    check({
      key: "product_eval_data",
      label: "Product ASR Eval Data",
      level: "degraded",
      required: false,
      detail:
        "Product ASR buckets are not full yet; collect consented health, commerce, code-switch, English, and phone/noise speech.",
    }),
  ];

  const summary = summarize(checks);
  const requiredFailed = checks.some((item) => item.required && item.level === "fail");
  const requiredDegraded = checks.some((item) => item.required && item.level === "degraded");
  const status = requiredFailed ? "blocked" : requiredDegraded ? "degraded" : "ready";

  return {
    ok: status !== "blocked",
    status,
    service: "ghana-health-ai",
    mode: "voice-first",
    checks,
    summary,
    ts: new Date().toISOString(),
  };
}
