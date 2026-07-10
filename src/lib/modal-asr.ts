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
};

function asrBaseUrl(): string | null {
  return process.env.MODAL_ASR_URL?.replace(/\/$/, "") || null;
}

export function isModalAsrConfigured(): boolean {
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
    };
  }
  return data;
}
