export type ModalTtsResult = {
  audio_base64: string;
  sample_rate: number;
  format: string;
  duration?: number;
  latency_ms: number;
  model: string;
  text?: string;
  language?: string;
  error?: string;
};

/**
 * MODAL_TTS_URL can be either:
 * - full speak endpoint: https://...--ghana-health-tts-speak.modal.run
 * - base with /speak path (legacy ASGI)
 */
export const DEFAULT_MODAL_TTS_URL =
  "https://createdliving1000--ghana-health-tts-speak.modal.run";

function ttsSpeakUrl(): string | null {
  const raw = (process.env.MODAL_TTS_URL || DEFAULT_MODAL_TTS_URL).replace(/\/$/, "");
  if (!raw) return null;
  if (raw.endsWith("/speak") || raw.includes("-speak")) return raw;
  return `${raw}/speak`;
}

export function isModalTtsConfigured(): boolean {
  return Boolean(ttsSpeakUrl());
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
): Promise<ModalTtsResult> {
  const url = ttsSpeakUrl();
  if (!url) throw new Error("MODAL_TTS_URL is not set");

  const lang = language === "en" ? "en" : "tw";
  const clean = speakableText(text, lang);
  if (!clean) {
    return {
      audio_base64: "",
      sample_rate: 16000,
      format: "wav",
      latency_ms: 0,
      model: "skipped",
      error: "empty_text",
    };
  }

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_TTS_TOKEN
        ? { Authorization: `Bearer ${process.env.MODAL_TTS_TOKEN}` }
        : {}),
    },
    body: JSON.stringify({ text: clean, language: lang }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Modal TTS ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ModalTtsResult;
}
