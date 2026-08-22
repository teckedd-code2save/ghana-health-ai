/**
 * OpenAI-compatible chat completion (OpenAI, Groq, or any base URL).
 * Falls back to null if no key configured so callers can use RAG-only path.
 */

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

function resolveProvider(): {
  apiKey: string;
  baseUrl: string;
  model: string;
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
    return {
      apiKey: openai,
      baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      model: process.env.LLM_MODEL || "gpt-4o-mini",
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

  try {
    const res = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${provider.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: provider.model,
        messages,
        temperature: opts?.temperature ?? 0.3,
        max_tokens: opts?.maxTokens ?? 600,
      }),
    });
    if (!res.ok) {
      console.error("[llm]", res.status, await res.text());
      return null;
    }
    const data = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    return data.choices?.[0]?.message?.content?.trim() || null;
  } catch (e) {
    console.error("[llm]", e);
    return null;
  }
}
