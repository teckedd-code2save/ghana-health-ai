/**
 * Modal ASR client — real Twi Whisper fine-tune on GPU.
 */

export type ModalAsrResult = {
  text: string;
  language: string;
  language_probability?: number;
  duration?: number;
  rms?: number;
  segments?: { start: number; end: number; text: string }[];
  latency_ms: number;
  model: string;
  speaker?: string;
  verified?: boolean | null;
  error?: string;
  rejected_text?: string;
  route?: ModalAsrRoute;
};

export type ModalAsrRoute = {
  name: "twi-default" | "english" | "english-fallback" | "dondo";
  url: string;
  requestedLanguage?: string;
  routedLanguage: "tw" | "en";
};

export const DEFAULT_MODAL_ASR_EN_URL =
  "https://createdliving1000--ghana-health-asr-en-api.modal.run";

export const DEFAULT_MODAL_ASR_DONDO_URL =
  "https://createdliving1000--ghana-health-asr-dondo-api.modal.run";

function cleanUrl(value: string | undefined): string | null {
  return value?.replace(/\/$/, "") || null;
}

export function resolveModalAsrRoute(
  language?: string,
  asrModel?: string,
): ModalAsrRoute | null {
  const defaultUrl = cleanUrl(process.env.MODAL_ASR_URL);
  const englishUrl = cleanUrl(process.env.MODAL_ASR_EN_URL || DEFAULT_MODAL_ASR_EN_URL);
  const dondoUrl = cleanUrl(
    process.env.MODAL_ASR_DONDO_URL || DEFAULT_MODAL_ASR_DONDO_URL,
  );
  const wantsEnglish = language === "en";

  if (wantsEnglish && englishUrl) {
    return {
      name: "english",
      url: englishUrl,
      requestedLanguage: language,
      routedLanguage: "en",
    };
  }

  // Explicit opt-in A/B route: DONDO CTC serves Twi turns only.
  if (asrModel === "dondo" && dondoUrl) {
    return {
      name: "dondo",
      url: dondoUrl,
      requestedLanguage: language,
      routedLanguage: "tw",
    };
  }

  if (!defaultUrl) return null;

  return {
    name: wantsEnglish ? "english-fallback" : "twi-default",
    url: defaultUrl,
    requestedLanguage: language,
    routedLanguage: wantsEnglish ? "en" : "tw",
  };
}

export function isModalAsrConfigured(): boolean {
  return Boolean(resolveModalAsrRoute());
}

export async function modalTranscribe(
  audio: ArrayBuffer | Buffer | Uint8Array,
  opts?: { language?: string; contentType?: string; filename?: string; asrModel?: string },
): Promise<ModalAsrResult> {
  const route = resolveModalAsrRoute(opts?.language, opts?.asrModel);
  if (!route) throw new Error("MODAL_ASR_URL is not set");

  const bytes =
    audio instanceof ArrayBuffer
      ? Buffer.from(audio)
      : Buffer.isBuffer(audio)
        ? audio
        : Buffer.from(audio);

  const form = new FormData();
  form.append(
    "audio",
    new Blob([new Uint8Array(bytes)], { type: opts?.contentType ?? "audio/webm" }),
    opts?.filename ?? "utterance.webm",
  );

  const qs = opts?.language ? `?language=${encodeURIComponent(opts.language)}` : "";
  const token =
    route.name === "english" && process.env.MODAL_ASR_EN_TOKEN
      ? process.env.MODAL_ASR_EN_TOKEN
      : process.env.MODAL_ASR_TOKEN;
  const res = await fetch(`${route.url}/transcribe${qs}`, {
    method: "POST",
    body: form,
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });

  const data = (await res.json().catch(() => ({}))) as ModalAsrResult & {
    detail?: string;
  };

  if (!res.ok) {
    const msg =
      data.error ||
      data.detail ||
      `Modal ASR ${res.status}`;
    // Surface structured error so UI can show "speak louder" etc.
    return {
      text: data.text || "",
      language: data.language || "tw",
      latency_ms: data.latency_ms || 0,
      model: data.model || "unknown",
      error: msg,
      rejected_text: data.rejected_text,
      duration: data.duration,
      rms: data.rms,
      route,
    };
  }
  return { ...data, route };
}
