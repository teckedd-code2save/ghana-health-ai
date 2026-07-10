/**
 * Modal ASR client — real Twi Whisper fine-tune on GPU.
 */

export type ModalAsrResult = {
  text: string;
  language: string;
  language_probability?: number;
  duration?: number;
  segments?: { start: number; end: number; text: string }[];
  latency_ms: number;
  model: string;
  speaker?: string;
  verified?: boolean | null;
  error?: string;
};

function asrBaseUrl(): string | null {
  return process.env.MODAL_ASR_URL?.replace(/\/$/, "") || null;
}

export function isModalAsrConfigured(): boolean {
  // Prefer real ASR whenever URL is set (VOICE_MODE optional)
  return Boolean(asrBaseUrl());
}

export async function modalTranscribe(
  audio: ArrayBuffer | Buffer | Uint8Array,
  opts?: { language?: string; contentType?: string; filename?: string },
): Promise<ModalAsrResult> {
  const base = asrBaseUrl();
  if (!base) throw new Error("MODAL_ASR_URL is not set");

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
  const res = await fetch(`${base}/transcribe${qs}`, {
    method: "POST",
    body: form,
    headers: process.env.MODAL_ASR_TOKEN
      ? { Authorization: `Bearer ${process.env.MODAL_ASR_TOKEN}` }
      : undefined,
  });

  if (!res.ok) {
    throw new Error(`Modal ASR ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  return (await res.json()) as ModalAsrResult;
}
