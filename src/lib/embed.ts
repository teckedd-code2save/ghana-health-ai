/**
 * Twi text embeddings — research path.
 *
 * Prefer Modal ABENA (Ghana-NLP/abena-base-asante-twi-uncased).
 * Optional OpenAI embeddings only as emergency fallback (not Twi-native).
 */

import { cosineSimilarity as cos } from "@/lib/voice-embed";

export type EmbedResult = {
  vectors: number[][];
  dim: number;
  model: string;
  engine: "abena" | "openai" | "none";
  latencyMs?: number;
};

function embedUrl(): string | null {
  const raw = process.env.MODAL_EMBED_URL?.replace(/\/$/, "");
  return raw || null;
}

export function isEmbedConfigured(): boolean {
  return Boolean(process.env.MODAL_EMBED_URL || process.env.OPENAI_API_KEY);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  return cos(a, b);
}

async function embedViaModal(
  texts: string[],
  mode: "query" | "passage",
): Promise<EmbedResult | null> {
  const url = embedUrl();
  if (!url) return null;

  const started = Date.now();
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_EMBED_TOKEN
        ? { Authorization: `Bearer ${process.env.MODAL_EMBED_TOKEN}` }
        : {}),
    },
    body: JSON.stringify({ texts, mode }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Modal embed ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = (await res.json()) as {
    embeddings?: number[][];
    dim?: number;
    model?: string;
    engine?: string;
    latency_ms?: number;
    error?: string;
  };
  if (data.error || !data.embeddings?.length) {
    throw new Error(data.error || "empty embeddings");
  }
  return {
    vectors: data.embeddings,
    dim: data.dim ?? data.embeddings[0]!.length,
    model: data.model || "abena",
    engine: "abena",
    latencyMs: data.latency_ms ?? Date.now() - started,
  };
}

async function embedViaOpenAI(texts: string[]): Promise<EmbedResult | null> {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return null;
  const base = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(
    /\/$/,
    "",
  );
  const model = process.env.EMBED_MODEL || "text-embedding-3-small";
  const started = Date.now();
  const res = await fetch(`${base}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input: texts }),
  });
  if (!res.ok) {
    throw new Error(`OpenAI embed ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data = (await res.json()) as {
    data?: { embedding: number[]; index: number }[];
    model?: string;
  };
  const sorted = [...(data.data ?? [])].sort((a, b) => a.index - b.index);
  const vectors = sorted.map((d) => {
    const v = d.embedding;
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / n);
  });
  if (!vectors.length) throw new Error("empty openai embeddings");
  return {
    vectors,
    dim: vectors[0]!.length,
    model: data.model || model,
    engine: "openai",
    latencyMs: Date.now() - started,
  };
}

export async function embedTexts(
  texts: string[],
  opts?: { mode?: "query" | "passage" },
): Promise<EmbedResult> {
  const mode = opts?.mode ?? "query";
  if (!texts.length) {
    return { vectors: [], dim: 0, model: "none", engine: "none" };
  }

  if (process.env.MODAL_EMBED_URL) {
    try {
      const r = await embedViaModal(texts, mode);
      if (r) return r;
    } catch (e) {
      console.error("[embed abena]", e);
    }
  }

  try {
    const r = await embedViaOpenAI(texts);
    if (r) return r;
  } catch (e) {
    console.error("[embed openai]", e);
  }

  return { vectors: [], dim: 0, model: "none", engine: "none" };
}

export async function embedOne(
  text: string,
  mode: "query" | "passage" = "query",
): Promise<{ vector: number[]; model: string; engine: EmbedResult["engine"] } | null> {
  const r = await embedTexts([text], { mode });
  if (!r.vectors[0]?.length) return null;
  return { vector: r.vectors[0], model: r.model, engine: r.engine };
}
