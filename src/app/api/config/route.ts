import { jsonOk } from "@/lib/api";
import { isPaystackConfigured, getPaystackPublicKey } from "@/lib/paystack";
import { isLlmConfigured } from "@/lib/llm";
import { isModalAsrConfigured, resolveModalAsrRoute } from "@/lib/modal-asr";
import { isModalTtsConfigured, resolveTtsRoute } from "@/lib/modal-tts";
import { isAbenaConfigured, isEmbedConfigured } from "@/lib/embed";

/** Public runtime flags (no secrets). */
export async function GET(request: Request) {
  const englishRoute = resolveModalAsrRoute("en");
  const twiTtsRoute = resolveTtsRoute("tw");
  const englishTtsRoute = resolveTtsRoute("en");
  const configuredOrigin = process.env.NEXT_PUBLIC_APP_URL;
  const forwardedHost = request.headers.get("x-forwarded-host");
  const forwardedProto = request.headers.get("x-forwarded-proto") || "https";
  const requestOrigin = forwardedHost
    ? `${forwardedProto}://${forwardedHost}`
    : new URL(request.url).origin;
  return jsonOk({
    voiceMode: process.env.VOICE_MODE || "modal",
    modalAsr: isModalAsrConfigured(),
    modalAsrEnglish: englishRoute?.name === "english",
    asrRoutes: {
      tw: resolveModalAsrRoute("tw")?.name ?? "unconfigured",
      en: englishRoute?.name ?? "unconfigured",
    },
    modalTts: isModalTtsConfigured(),
    ttsRoutes: {
      tw: twiTtsRoute
        ? { provider: twiTtsRoute.provider, model: twiTtsRoute.modelLabel }
        : { provider: "unconfigured", model: "unconfigured" },
      en: englishTtsRoute
        ? { provider: englishTtsRoute.provider, model: englishTtsRoute.modelLabel }
        : { provider: "unconfigured", model: "unconfigured" },
    },
    /** True only when Modal ABENA URL is set (research path). */
    abenaEmbed: isAbenaConfigured(),
    /** Any embed path (ABENA or OpenAI emergency fallback). */
    embedAny: isEmbedConfigured(),
    llm: isLlmConfigured(),
    paystack: isPaystackConfigured(),
    paystackPublicKey: getPaystackPublicKey() || process.env.NEXT_PUBLIC_PAYSTACK_PUBLIC_KEY || "",
    stack: {
      asr: "Twi: teckedd/gha-whisper-small-twi-v6; English: separate openai/whisper-small route; DONDO research checkpoint",
      embed: "Ghana-NLP/abena-base-asante-twi-uncased",
      tts: twiTtsRoute?.modelLabel ?? "unconfigured",
      language: "Master Twi + English first; Ga next",
      docs: "docs/research-stack.md",
    },
    languages: ["tw", "en", "ga"],
    hotline: process.env.HEALTH_ESCALATION_HOTLINE || "112 / community health worker",
    appUrl: configuredOrigin || requestOrigin,
  });
}
