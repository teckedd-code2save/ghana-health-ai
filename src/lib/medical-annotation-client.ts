import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { ChatMessage } from "./llm";

type Json = Record<string, unknown>;

export class MedicalAnnotationProviderError extends Error {
  constructor(public readonly status: number, public readonly code: string) {
    super(`Annotation provider HTTP ${status} (${code})`);
  }

  get fatal() {
    return [400, 401, 403, 404].includes(this.status) ||
      ["insufficient_quota", "credit_balance_exhausted"].includes(this.code);
  }
}

function parseObject(content: string): Json | null {
  const fenced = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const cleaned = (fenced?.[1] ?? content).trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const parsed = JSON.parse(cleaned.slice(start, end + 1));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    if (!Array.isArray(parsed.rows) && Array.isArray(parsed.response_shape?.rows)) {
      return parsed.response_shape;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function createMedicalAnnotationClient(cacheDirectory: string) {
  return async (messages: ChatMessage[], model: string, maxTokens: number): Promise<Json> => {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new MedicalAnnotationProviderError(401, "missing_api_key");
    const baseUrl = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
    const request = { model, messages, max_completion_tokens: maxTokens };
    const requestHash = crypto.createHash("sha256").update(JSON.stringify({ baseUrl, request })).digest("hex");
    const cachePath = path.join(cacheDirectory, `${requestHash}.json`);
    try {
      const saved = JSON.parse(await fs.readFile(cachePath, "utf8"));
      if (saved.request_hash !== requestHash || !saved.payload?.rows) {
        throw new Error(`Invalid annotation checkpoint: ${requestHash}`);
      }
      return saved.payload as Json;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }

    await fs.mkdir(cacheDirectory, { recursive: true });
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const started = Date.now();
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(240_000),
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        // Provider messages can echo credentials; persist only status and error code.
        const code = String(errorBody.error?.code || "request_failed").replace(/[^a-zA-Z0-9_]/g, "").slice(0, 80);
        const error = new MedicalAnnotationProviderError(response.status, code);
        if (error.fatal || code === "insufficient_quota" || attempt === 3) throw error;
        await new Promise((resolve) => setTimeout(resolve, 1500 * attempt));
        continue;
      }
      const data = await response.json();
      const choice = data.choices?.[0];
      const payload = choice?.finish_reason === "stop" ? parseObject(choice.message?.content || "") : null;
      const checkpoint = {
        schema_version: 1,
        request_hash: requestHash,
        requested_model: model,
        output_token_budget: request.max_completion_tokens,
        returned_model: data.model,
        generated_at: new Date().toISOString(),
        elapsed_ms: Date.now() - started,
        request_id: response.headers.get("x-request-id"),
        usage: data.usage,
        finish_reason: choice?.finish_reason,
        raw_text: choice?.message?.content || "",
        payload,
      };
      const attemptPath = path.join(cacheDirectory, `${requestHash}.attempt-${Date.now()}-${attempt}.json`);
      await fs.writeFile(attemptPath, `${JSON.stringify(checkpoint)}\n`, { mode: 0o600 });
      if (payload && Array.isArray(payload.rows)) {
        const temporary = `${cachePath}.${process.pid}.tmp`;
        await fs.writeFile(temporary, `${JSON.stringify(checkpoint)}\n`, { mode: 0o600 });
        await fs.rename(temporary, cachePath);
        return payload;
      }
      if (choice?.finish_reason === "length") request.max_completion_tokens *= 2;
    }
    throw new Error(`${model} did not return a complete rows object after three attempts.`);
  };
}
