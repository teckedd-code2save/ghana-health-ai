export type ModalTtsResult = {
  audio_base64: string;
  sample_rate: number;
  format: string;
  duration?: number;
  latency_ms: number;
  model: string;
  provider?: TtsProvider;
  text?: string;
  language?: string;
  error?: string;
};

export type TtsProvider = "mms" | "stable-twi" | "nano-twi" | "qwen";

export type TtsRoute = {
  provider: TtsProvider;
  url: string;
  configured: boolean;
  modelLabel: string;
};

/**
 * MODAL_TTS_URL can be either:
 * - full speak endpoint: https://...--ghana-health-tts-speak.modal.run
 * - base with /speak path (legacy ASGI)
 */
export const DEFAULT_MODAL_TTS_URL =
  "https://createdliving1000--ghana-health-tts-speak.modal.run";
export const DEFAULT_STABLE_TWI_TTS_URL =
  "https://createdliving1000--ghana-health-tts-stable-twi-speak.modal.run";

function normalizeSpeakUrl(rawValue: string | undefined, fallback?: string): string | null {
  const raw = (rawValue || fallback || "").replace(/\/$/, "");
  if (!raw) return null;
  if (raw.endsWith("/speak") || raw.includes("-speak")) return raw;
  return `${raw}/speak`;
}

function providerFromEnv(language: string): TtsProvider {
  const configured =
    language === "en"
      ? process.env.TTS_EN_PROVIDER || process.env.TTS_PROVIDER
      : process.env.TTS_TWI_PROVIDER || process.env.TTS_PROVIDER;
  if (configured === "mms" || configured === "stable-twi" || configured === "nano-twi" || configured === "qwen") {
    return configured;
  }
  return language === "en" ? "mms" : "stable-twi";
}

function urlForProvider(provider: TtsProvider) {
  if (provider === "stable-twi") {
    return normalizeSpeakUrl(
      process.env.STABLE_TWI_TTS_URL || process.env.TTS_STABLE_TWI_URL,
      DEFAULT_STABLE_TWI_TTS_URL,
    );
  }
  if (provider === "nano-twi") {
    return normalizeSpeakUrl(process.env.NANO_TWI_TTS_URL || process.env.TTS_NANO_TWI_URL);
  }
  if (provider === "qwen") {
    return normalizeSpeakUrl(process.env.QWEN_TTS_URL || process.env.TTS_QWEN_URL);
  }
  return normalizeSpeakUrl(process.env.MODAL_TTS_URL, DEFAULT_MODAL_TTS_URL);
}

export function resolveTtsRoute(language: string = "tw", requestedProvider?: TtsProvider): TtsRoute | null {
  const lang = language === "en" ? "en" : "tw";
  const provider = lang === "en" ? "mms" : requestedProvider ?? providerFromEnv(lang);
  const url = urlForProvider(provider);
  if (!url) return null;
  const modelLabel =
    provider === "mms"
      ? lang === "en"
        ? process.env.TTS_ENG_MODEL_ID || "facebook/mms-tts-eng"
        : process.env.TTS_MODEL_ID || "facebook/mms-tts-aka"
      : provider === "stable-twi"
        ? "ghananlpcommunity/stable-twi-tts"
        : provider === "nano-twi"
          ? "ghananlpcommunity/nano-twi"
          : "qwen3-tts-12hz";
  return {
    provider,
    url,
    configured: Boolean(url),
    modelLabel,
  };
}

export function isModalTtsConfigured(): boolean {
  return Boolean(resolveTtsRoute("tw") || resolveTtsRoute("en"));
}

/** Expand jargon + strip symbols so speech models don't say "C H W". */
export function speakableText(text: string, language: string = "tw"): string {
  let clean = text;
  const pairs: [RegExp, string][] = [
    [/\bCHWs?\b/gi, language === "en" ? "community health workers" : "community health worker"],
    [/\bANC\b/g, "antenatal care"],
    [/\bOTC\b/g, "over the counter"],
    [/\bMoMo\b/gi, "mobile money"],
    [/\bGHS\b/g, "Ghana Health Service"],
    [/\bWHO\b/g, "World Health Organization"],
  ];
  for (const [re, repl] of pairs) clean = clean.replace(re, repl);
  return clean
    .replace(/[*_#`>~\[\]()]/g, " ")
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 500);
}

export async function modalSpeak(
  text: string,
  language: string = "tw",
  opts?: { provider?: TtsProvider },
): Promise<ModalTtsResult> {
  const lang = language === "en" ? "en" : "tw";
  const route = resolveTtsRoute(lang, opts?.provider);
  if (!route) throw new Error(`TTS route is not configured for ${lang}`);

  const clean = speakableText(text, lang);
  if (!clean) {
    return {
      audio_base64: "",
      sample_rate: 16000,
      format: "wav",
      latency_ms: 0,
      model: "skipped",
      provider: route.provider,
      error: "empty_text",
    };
  }

  return speakViaRoute(clean, lang, route);
}

async function speakViaRoute(
  clean: string,
  lang: "tw" | "en",
  route: TtsRoute,
  allowFallback = true,
): Promise<ModalTtsResult> {
  const res = await fetch(route.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_TTS_TOKEN
        ? { Authorization: `Bearer ${process.env.MODAL_TTS_TOKEN}` }
        : {}),
    },
    body: JSON.stringify({
      text: clean,
      language: lang,
      provider: route.provider,
      model: route.modelLabel,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    if (allowFallback && lang === "tw" && route.provider !== "mms") {
      const fallback = resolveTtsRoute("tw", "mms");
      if (fallback) return speakViaRoute(clean, lang, fallback, false);
    }
    throw new Error(`Modal TTS ${res.status}: ${body.slice(0, 200)}`);
  }
  const result = (await res.json()) as ModalTtsResult;
  if (result.error && allowFallback && lang === "tw" && route.provider !== "mms") {
    const fallback = resolveTtsRoute("tw", "mms");
    if (fallback) return speakViaRoute(clean, lang, fallback, false);
  }
  return {
    ...result,
    provider: result.provider ?? route.provider,
    model: result.model || route.modelLabel,
    language: result.language ?? lang,
  };
}
