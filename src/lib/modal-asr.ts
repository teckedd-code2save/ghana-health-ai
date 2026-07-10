/**
 * Client for Modal ASR service (faster-whisper on GPU).
 * Falls back handled by caller when VOICE_MODE=stub or MODAL_ASR_URL unset.
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
  const url = process.env.MODAL_ASR_URL?.replace(/\/$/, "");
  return url || null;
}

export function isModalAsrConfigured(): boolean {
  return process.env.VOICE_MODE === "modal" && Boolean(asrBaseUrl());
}

export async function modalTranscribe(
  audio: ArrayBuffer | Buffer | Uint8Array,
  opts?: { language?: string; contentType?: string; filename?: string },
): Promise<ModalAsrResult> {
  const base = asrBaseUrl();
  if (!base) {
    throw new Error("MODAL_ASR_URL is not set");
  }

  const bytes =
    audio instanceof ArrayBuffer
      ? Buffer.from(audio)
      : Buffer.isBuffer(audio)
        ? audio
        : Buffer.from(audio);

  const form = new FormData();
  const blob = new Blob([new Uint8Array(bytes)], {
    type: opts?.contentType ?? "audio/webm",
  });
  form.append("audio", blob, opts?.filename ?? "utterance.webm");

  const qs = opts?.language ? `?language=${encodeURIComponent(opts.language)}` : "";
  const res = await fetch(`${base}/transcribe${qs}`, {
    method: "POST",
    body: form,
    // Modal public endpoints don't need auth by default; add header if you lock them down
    headers: process.env.MODAL_ASR_TOKEN
      ? { Authorization: `Bearer ${process.env.MODAL_ASR_TOKEN}` }
      : undefined,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Modal ASR ${res.status}: ${body.slice(0, 200)}`);
  }

  return (await res.json()) as ModalAsrResult;
}

export async function modalAsrHealth(): Promise<{ ok: boolean; service?: string }> {
  const base = asrBaseUrl();
  if (!base) return { ok: false };
  const res = await fetch(`${base}/health`, { next: { revalidate: 0 } });
  if (!res.ok) return { ok: false };
  return res.json();
}
