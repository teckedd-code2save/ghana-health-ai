export type ModalTtsResult = {
  audio_base64: string;
  sample_rate: number;
  format: string;
  duration?: number;
  latency_ms: number;
  model: string;
  text?: string;
  error?: string;
};

/**
 * MODAL_TTS_URL can be either:
 * - full speak endpoint: https://...--ghana-health-tts-speak.modal.run
 * - base with /speak path (legacy ASGI)
 */
function ttsSpeakUrl(): string | null {
  const raw = process.env.MODAL_TTS_URL?.replace(/\/$/, "");
  if (!raw) return null;
  if (raw.endsWith("/speak") || raw.includes("-speak")) return raw;
  return `${raw}/speak`;
}

export function isModalTtsConfigured(): boolean {
  return Boolean(process.env.MODAL_TTS_URL);
}

export async function modalSpeak(
  text: string,
  language: string = "tw",
): Promise<ModalTtsResult> {
  const url = ttsSpeakUrl();
  if (!url) throw new Error("MODAL_TTS_URL is not set");

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_TTS_TOKEN
        ? { Authorization: `Bearer ${process.env.MODAL_TTS_TOKEN}` }
        : {}),
    },
    body: JSON.stringify({ text, language }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Modal TTS ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ModalTtsResult;
}
