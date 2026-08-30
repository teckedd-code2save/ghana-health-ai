/**
 * OpenAI-compatible chat completion (OpenAI, Groq, or any base URL).
 * Falls back to null if no key configured so callers can use RAG-only path.
 */

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

function resolveProvider(): {
  apiKey: string;
  baseUrl: string;
  model: string;
  fallbackModel?: string;
} | null {
  const groq = process.env.GROQ_API_KEY;
  if (groq) {
    return {
      apiKey: groq,
      baseUrl: "https://api.groq.com/openai/v1",
      model: process.env.LLM_MODEL || "llama-3.3-70b-versatile",
    };
  }
  const openai = process.env.OPENAI_API_KEY;
  if (openai) {
    const legacyModel = process.env.LLM_MODEL?.trim();
    return {
      apiKey: openai,
      baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      // Keep the language model independent from the historical provider-wide
      // override so an old gpt-4o-mini setting cannot silently pin this path.
      model: process.env.OPENAI_LANGUAGE_MODEL?.trim() || "gpt-5.6-sol",
      fallbackModel:
        legacyModel && !["gpt-5.6-sol", "gpt-5.4-mini"].includes(legacyModel)
          ? legacyModel
          : "gpt-5.4-mini",
    };
  }
  return null;
}

export function llmProviderInfo(): { provider: "groq" | "openai"; model: string } | null {
  const resolved = resolveProvider();
  if (!resolved) return null;
  return {
    provider: process.env.GROQ_API_KEY ? "groq" : "openai",
    model: resolved.model,
  };
}

export function isLlmConfigured(): boolean {
  return Boolean(resolveProvider());
}

export async function chatComplete(
  messages: ChatMessage[],
  opts?: { temperature?: number; maxTokens?: number },
): Promise<string | null> {
  const provider = resolveProvider();
  if (!provider) return null;

  const models = [provider.model, provider.fallbackModel].filter(
    (model, index, all): model is string =>
      Boolean(model) && all.indexOf(model) === index,
  );
  for (const model of models) {
    try {
      const isGpt5 = model.startsWith("gpt-5");
      const res = await fetch(`${provider.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages,
          ...(isGpt5
            ? { max_completion_tokens: opts?.maxTokens ?? 600 }
            : {
                temperature: opts?.temperature ?? 0.3,
                max_tokens: opts?.maxTokens ?? 600,
              }),
        }),
      });
      if (!res.ok) {
        console.error("[llm]", model, res.status, await res.text());
        continue;
      }
      const data = (await res.json()) as {
        choices?: { message?: { content?: string } }[];
      };
      const content = data.choices?.[0]?.message?.content?.trim();
      if (content) return content;
    } catch (e) {
      console.error("[llm]", model, e);
    }
  }
  return null;
}
