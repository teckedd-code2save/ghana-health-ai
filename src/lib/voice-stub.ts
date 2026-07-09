/**
 * Stub voice pipeline for local MVP.
 * Production path: Modal Parakeet + Sortformer streaming (see /modal).
 */

export type VoiceChunkResult = {
  text: string;
  speaker: string;
  language: "tw" | "en";
  verified: boolean | null;
  latencyMs: number;
  mode: "stub";
};

const STUB_PHRASES = [
  { text: "Me ti yɛ me ya", lang: "tw" as const },
  { text: "Mepɛ paracetamol boɔ", lang: "tw" as const },
  { text: "Me yare afe", lang: "tw" as const },
  { text: "How much is rice?", lang: "en" as const },
  { text: "Me wɔ nyinsen, me yare", lang: "tw" as const },
];

let phraseIndex = 0;

export async function stubTranscribeChunk(
  _audio?: ArrayBuffer | null,
  speakerId?: string | null,
): Promise<VoiceChunkResult> {
  const started = Date.now();
  // Simulate partial ASR latency
  await new Promise((r) => setTimeout(r, 80 + Math.random() * 120));
  const phrase = STUB_PHRASES[phraseIndex % STUB_PHRASES.length];
  phraseIndex += 1;
  return {
    text: phrase.text,
    speaker: speakerId ? `Speaker (${speakerId.slice(0, 6)})` : "Speaker 1 (User)",
    language: phrase.lang,
    verified: speakerId ? Math.random() > 0.2 : null,
    latencyMs: Date.now() - started,
    mode: "stub",
  };
}

/** Deterministic fake embedding from passphrase for Voice ID MVP */
export function stubEmbeddingFromPassphrase(passphrase: string): number[] {
  const dim = 64;
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < passphrase.length; i++) {
    out[i % dim] += passphrase.charCodeAt(i) / 255;
  }
  const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0)) || 1;
  return out.map((v) => v / norm);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

export function embeddingToB64(embedding: number[]): string {
  return Buffer.from(Float32Array.from(embedding).buffer).toString("base64");
}

export function embeddingFromB64(b64: string): number[] {
  const buf = Buffer.from(b64, "base64");
  const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return Array.from(arr);
}
