import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { isModelMode, isResearchR1Mode, ResearchR1Error } from "../src/lib/model-modes";
import { R1_EXECUTION, R1_MODEL, R1_REVISION, researchR1Reply, validateResearchHistory, verifyResearchResult } from "../src/lib/research-r1";

const messages = [{ role: "user" as const, content: "Hello" }];
function result() {
  return { variant: "meaning-adapter", output: "Hello.", generated: true, complete: true,
    load_verified: true, finish_reason: "eos", prompt_tokens: 20, context_tokens: 4096, request_seconds: 1,
    identity: { variant: "meaning-adapter", model: R1_MODEL, revision: R1_REVISION,
      execution_manifest_sha256: R1_EXECUTION,
      adapter_weights_sha256: "73ee9854c4975ce2d79ae7dbabb2fd5d021e9579a7f81ac26ec14b4a5a7fbb2b",
      base_weights_sha256: "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6" } };
}

test("explicit research modes do not replace Stable", () => {
  assert(isModelMode("shadow")); assert(!isResearchR1Mode("shadow"));
  for (const mode of ["r1_base", "r1_meaning", "r1_response"]) assert(isResearchR1Mode(mode));
  assert(!isModelMode("r1_unknown"));
});
test("history preserves complete alternating turns and never truncates", () => {
  validateResearchHistory(messages);
  validateResearchHistory([...messages, { role: "assistant", content: "Hello." }, ...messages]);
  assert.throws(() => validateResearchHistory([...messages, ...messages]), ResearchR1Error);
  assert.throws(() => validateResearchHistory([{ role: "assistant", content: "Hello" }]), ResearchR1Error);
  assert.throws(() => validateResearchHistory(Array.from({ length: 81 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: "x" }))), ResearchR1Error);
});
test("result binds model, revision, checkpoint and completion", () => {
  assert.equal(verifyResearchResult(result(), "r1_meaning").research.complete, true);
  assert.throws(() => verifyResearchResult(result(), "r1_response"), ResearchR1Error);
  assert.throws(() => verifyResearchResult({ ...result(), identity: { ...result().identity, model: "another-model" } }, "r1_meaning"), ResearchR1Error);
  assert.throws(() => verifyResearchResult({ ...result(), complete: false }, "r1_meaning"), ResearchR1Error);
  assert.equal(verifyResearchResult({ ...result(), complete: false, finish_reason: "repetition_guard" }, "r1_meaning").research.complete, false);
});
test("fragmented SSE arrives as real deltas before final, with no fallback", async () => {
  const originalFetch = globalThis.fetch;
  const previous = [process.env.RESEARCH_R1_URL, process.env.RESEARCH_R1_TOKEN];
  process.env.RESEARCH_R1_URL = "https://example.modal.run";
  process.env.RESEARCH_R1_TOKEN = "x".repeat(48);
  let calls = 0;
  let observed = "";
  const encoder = new TextEncoder();
  globalThis.fetch = async (url, init) => {
    calls++;
    assert.equal(String(url), "https://example.modal.run/stream");
    assert.deepEqual(JSON.parse(init!.body as string), { messages, variant: "meaning-adapter" });
    return new Response(new ReadableStream({ start(controller) {
      controller.enqueue(encoder.encode('data: {"type":"delta","chunk":"Hel'));
      controller.enqueue(encoder.encode('lo."}\n\n'));
      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "final", result: result() })}\n\n`));
      controller.close();
    } }));
  };
  try {
    const reply = await researchR1Reply(messages, "r1_meaning", (chunk) => { observed += chunk; });
    assert.equal(observed, "Hello."); assert.equal(reply.result.output, observed); assert.equal(calls, 1);
    globalThis.fetch = async () => { calls++; return new Response("unavailable", { status: 503 }); };
    await assert.rejects(researchR1Reply(messages, "r1_meaning"), /No other model/);
    assert.equal(calls, 2);
    globalThis.fetch = async () => new Response('data: {"type":"delta","chunk":"Hello."}\n\n');
    await assert.rejects(researchR1Reply(messages, "r1_meaning"), /incomplete/);
    globalThis.fetch = async () => new Response(`data: ${JSON.stringify({ type: "final", result: result() })}\n\n`);
    await assert.rejects(researchR1Reply(messages, "r1_meaning"), /inconsistent/);
  } finally {
    globalThis.fetch = originalFetch;
    for (const [i, key] of ["RESEARCH_R1_URL", "RESEARCH_R1_TOKEN"].entries()) {
      if (previous[i] === undefined) delete process.env[key]; else process.env[key] = previous[i];
    }
  }
});
test("direct research branch cannot invoke legacy understanding, memory or commerce", () => {
  const turn = readFileSync("src/lib/research-r1-turn.ts", "utf8");
  assert(!/understandUtterance|rememberFromTurn|executeCommercePlan|chatComplete/.test(turn));
  assert(turn.includes("if (!input.user)"));
  assert(turn.includes("conversation.userId !== userId"));
  assert(turn.includes('clinicalAssessment: "not_performed"'));
  const feedback = readFileSync("src/app/api/research/reply-feedback/route.ts", "utf8");
  assert(feedback.includes("conversation: { userId: user.id }"));
  assert(feedback.includes("trainingConsent: false"));
  assert(feedback.includes("prisma.auditLog.create"));
});
