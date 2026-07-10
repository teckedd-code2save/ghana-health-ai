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

function ttsBaseUrl(): string | null {
  return process.env.MODAL_TTS_URL?.replace(/\/$/, "") || null;
}

export function isModalTtsConfigured(): boolean {
  return Boolean(ttsBaseUrl());
}

export async function modalSpeak(
  text: string,
  language: string = "tw",
): Promise<ModalTtsResult> {
  const base = ttsBaseUrl();
  if (!base) throw new Error("MODAL_TTS_URL is not set");

  const res = await fetch(`${base}/speak`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_TTS_TOKEN
        ? { Authorization: `Bearer ${process.env.MODAL_TTS_TOKEN}` }
        : {}),
    },
    body: JSON.stringify({ text, language, return_bytes: false }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Modal TTS ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ModalTtsResult;
}
