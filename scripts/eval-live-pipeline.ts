import fs from "node:fs/promises";
import path from "node:path";

type Fixture = {
  id: string;
  language: "tw" | "en" | "ga";
  message: string;
  expectedIntent?: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  expectedEscalate?: boolean;
  expectedSeverity?: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  expectedClarifying?: boolean;
  expectedCommerceExecution?: {
    mode?: "none" | "local_catalog_search" | "order_draft";
    status?: "not_applicable" | "needs_clarification" | "ready" | "no_matches";
  };
  forbiddenTerms?: string[];
};

type ChatResponse = {
  conversationId?: string;
  message?: {
    content?: string;
    metadata?: {
      engine?: string;
      review?: { revised?: boolean } | null;
      retrieve?: { engine?: string } | null;
    };
  };
  understanding?: {
    intent?: string;
    engine?: string;
    severity?: string;
    escalate?: boolean;
    commerceExecution?: {
      mode?: string;
      status?: string;
      products?: unknown[];
      draft?: unknown;
    } | null;
  };
  stage?: {
    llm?: boolean;
    review?: boolean;
    retrieveEngine?: string;
    totalLatencyMs?: number;
  };
  error?: string;
};

type HealthResponse = {
  ok?: boolean;
  dependencies?: {
    db?: boolean;
  };
};

const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");
const fixturePath =
  process.env.EVAL_FIXTURES ||
  path.join(process.cwd(), "scripts", "live-pipeline-fixtures.json");
const limit = Number(process.env.EVAL_LIMIT || process.argv.find((a) => a.startsWith("--limit="))?.split("=")[1] || 0);
const useStream = process.env.EVAL_STREAM === "1" || process.argv.includes("--stream");

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function looksLikeOldTemplate(text: string): boolean {
  return [
    "This is not a substitute for professional medical care",
    "Yɛi nyɛ oduruyɛfoɔ adwuma",
    "I heard you. Share more about your symptoms",
    "Mede asɛm no ate.",
    "LLM key nni hɔ",
  ].some((needle) => text.includes(needle));
}

async function readSse(
  res: Response,
): Promise<{
  data: ChatResponse;
  stages: string[];
  replyDeltaChars: number;
  events: string[];
  stageEvents: { name: string; eventIndex: number }[];
}> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("missing response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData: ChatResponse | null = null;
  const stages: string[] = [];
  const stageEvents: { name: string; eventIndex: number }[] = [];
  let replyDeltaChars = 0;
  const events: string[] = [];

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const eventLine = chunk.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = chunk.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.slice("event:".length).trim();
      events.push(event);
      const parsed = JSON.parse(dataLine.slice("data:".length).trim()) as ChatResponse & {
        name?: string;
        error?: string;
      };
      if (event === "stage" && parsed.name) {
        stages.push(parsed.name);
        stageEvents.push({ name: parsed.name, eventIndex: events.length - 1 });
      }
      if (event === "reply_delta") {
        const delta = parsed as { chunk?: string };
        replyDeltaChars += delta.chunk?.length ?? 0;
      }
      if (event === "error") throw new Error(parsed.error || "stream error");
      if (event === "final") finalData = parsed;
    }
  }

  if (!finalData) throw new Error("stream ended without final event");
  return { data: finalData, stages, replyDeltaChars, events, stageEvents };
}

async function callChat(
  fx: Fixture,
): Promise<{
  data: ChatResponse;
  stages: string[];
  replyDeltaChars: number;
  events: string[];
  stageEvents: { name: string; eventIndex: number }[];
}> {
  const endpoint = useStream ? "/api/chat/stream" : "/api/chat";
  const res = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: fx.message,
      language: fx.language,
      speak: false,
    }),
  });

  if (useStream) {
    assertOk(res.ok, `${fx.id}: stream HTTP ${res.status}`);
    return readSse(res);
  }

  const data = (await res.json()) as ChatResponse;
  assertOk(res.ok, `${fx.id}: HTTP ${res.status} ${data.error || ""}`);
  return { data, stages: [], replyDeltaChars: 0, events: [], stageEvents: [] };
}

function assertStreamOrder(
  fixtureId: string,
  events: string[],
  stages: string[],
  stageEvents: { name: string; eventIndex: number }[],
) {
  const firstUnderstandingStage = stages.indexOf("understanding");
  const firstAssistantStage = stages.indexOf("assistant_message");
  const understandingEvent = stageEvents.find((event) => event.name === "understanding");
  const assistantEvent = stageEvents.find((event) => event.name === "assistant_message");
  const firstReplyDelta = events.indexOf("reply_delta");
  const firstFinal = events.indexOf("final");

  assertOk(firstUnderstandingStage >= 0, `${fixtureId}: stream missing understanding stage`);
  assertOk(firstAssistantStage >= 0, `${fixtureId}: stream missing assistant stage`);
  assertOk(understandingEvent, `${fixtureId}: stream missing understanding event index`);
  assertOk(assistantEvent, `${fixtureId}: stream missing assistant event index`);
  assertOk(firstReplyDelta >= 0, `${fixtureId}: stream missing reply delta event`);
  assertOk(firstFinal >= 0, `${fixtureId}: stream missing final event`);
  assertOk(
    firstUnderstandingStage < firstAssistantStage,
    `${fixtureId}: assistant stage appeared before understanding stage`,
  );
  assertOk(
    events.indexOf("stage") >= 0 && events.indexOf("stage") < firstReplyDelta,
    `${fixtureId}: reply started before any progress stage`,
  );
  assertOk(
    understandingEvent.eventIndex < assistantEvent.eventIndex,
    `${fixtureId}: assistant event arrived before understanding event`,
  );
  assertOk(
    assistantEvent.eventIndex < firstReplyDelta,
    `${fixtureId}: reply chunk arrived before assistant stage`,
  );
  assertOk(
    firstReplyDelta < firstFinal,
    `${fixtureId}: final event arrived before streamed reply chunks`,
  );
}

async function main() {
  const healthRes = await fetch(`${baseUrl}/api/health`);
  const health = (await healthRes.json().catch(() => ({}))) as HealthResponse;
  assertOk(healthRes.ok, `health preflight failed: HTTP ${healthRes.status}`);
  assertOk(health.ok === true, "health preflight failed: app is not ready");
  assertOk(health.dependencies?.db === true, "health preflight failed: database is not reachable");

  const raw = await fs.readFile(fixturePath, "utf8");
  const fixtures = JSON.parse(raw) as Fixture[];
  const selected = limit > 0 ? fixtures.slice(0, limit) : fixtures;
  const results: {
    id: string;
    ok: boolean;
    intent?: string;
    engine?: string;
    retrieve?: string;
    latencyMs?: number;
    error?: string;
  }[] = [];

  for (const fx of selected) {
    try {
      const { data, stages, replyDeltaChars, events, stageEvents } = await callChat(fx);
      if (useStream) {
        assertStreamOrder(fx.id, events, stages, stageEvents);
        assertOk(replyDeltaChars > 0, `${fx.id}: stream missing reply deltas`);
      }

      const content = data.message?.content?.trim() || "";
      assertOk(content.length >= 8, `${fx.id}: empty/short model response`);
      assertOk(data.stage?.llm === true, `${fx.id}: did not use live LLM stage`);
      assertOk(data.stage?.review === true, `${fx.id}: did not use model review stage`);
      assertOk(data.understanding?.engine !== "fallback", `${fx.id}: fallback engine used`);
      assertOk(!looksLikeOldTemplate(content), `${fx.id}: old template response detected`);
      if (fx.expectedIntent) {
        assertOk(
          data.understanding?.intent === fx.expectedIntent,
          `${fx.id}: expected ${fx.expectedIntent}, got ${data.understanding?.intent}`,
        );
      }
      if (typeof fx.expectedEscalate === "boolean") {
        assertOk(
          data.understanding?.escalate === fx.expectedEscalate,
          `${fx.id}: expected escalate=${fx.expectedEscalate}, got ${data.understanding?.escalate}`,
        );
      }
      if (fx.expectedSeverity) {
        assertOk(
          data.understanding?.severity === fx.expectedSeverity,
          `${fx.id}: expected severity ${fx.expectedSeverity}, got ${data.understanding?.severity}`,
        );
      }
      if (fx.expectedClarifying) {
        const lower = content.toLowerCase();
        assertOk(
          content.includes("?") ||
            lower.includes("clarify") ||
            lower.includes("repeat") ||
            lower.includes("bio") ||
            lower.includes("nte") ||
            lower.includes("nkyerɛ") ||
            lower.includes("kyerɛ me") ||
            lower.includes("understand"),
          `${fx.id}: expected a model-generated clarification, got: ${content}`,
        );
      }
      if (fx.expectedCommerceExecution) {
        const execution = data.understanding?.commerceExecution;
        assertOk(execution, `${fx.id}: missing commerce execution`);
        if (fx.expectedCommerceExecution.mode) {
          assertOk(
            execution.mode === fx.expectedCommerceExecution.mode,
            `${fx.id}: expected commerce execution mode ${fx.expectedCommerceExecution.mode}, got ${execution.mode}`,
          );
        }
        if (fx.expectedCommerceExecution.status) {
          assertOk(
            execution.status === fx.expectedCommerceExecution.status,
            `${fx.id}: expected commerce execution status ${fx.expectedCommerceExecution.status}, got ${execution.status}`,
          );
        }
      }
      for (const term of fx.forbiddenTerms ?? []) {
        assertOk(
          !content.toLowerCase().includes(term.toLowerCase()),
          `${fx.id}: forbidden term detected: ${term}`,
        );
      }

      results.push({
        id: fx.id,
        ok: true,
        intent: data.understanding?.intent,
        engine: data.understanding?.engine,
        retrieve: data.stage?.retrieveEngine,
        latencyMs: data.stage?.totalLatencyMs,
      });
    } catch (e) {
      results.push({
        id: fx.id,
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  for (const r of results) {
    if (r.ok) {
      console.log(
        `ok ${r.id} mode=${useStream ? "stream" : "json"} intent=${r.intent} engine=${r.engine} retrieve=${r.retrieve} latency=${r.latencyMs}ms`,
      );
    } else {
      console.error(`fail ${r.id}: ${r.error}`);
    }
  }

  const failed = results.filter((r) => !r.ok);
  if (failed.length) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
