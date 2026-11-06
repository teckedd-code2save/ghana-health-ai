// Run from the repo root: node --import tsx --test scripts/test-research-r1-turn.ts
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { test } from "node:test";
import { compileFunction } from "node:vm";
import ts from "typescript";
import { z } from "zod";
import * as modelModes from "../src/lib/model-modes";
import { R1_EXECUTION, R1_MODEL, R1_REVISION, R1_VARIANTS, isCompletedResearchReply, isContinuableResearchReply, validateResearchHistory, verifyResearchResult } from "../src/lib/research-r1";
import type { ConversationTurnInput, ConversationTurnResult } from "../src/lib/conversation-turn";
import type { ResearchR1Mode } from "../src/lib/model-modes";
import type { R1Message } from "../src/lib/research-r1";

const OWNER = "10000000-0000-4000-8000-000000000001";
const OTHER = "10000000-0000-4000-8000-000000000002";
const CONVERSATION = "20000000-0000-4000-8000-000000000001";
const MESSAGE = "30000000-0000-4000-8000-000000000001";
const hash = (text: string) => createHash("sha256").update(text).digest("hex");
type Data = Record<string, unknown>;
type Message = { id: string; conversationId: string; role: string; content: string; metadata: Data | null };
type Call = { name: string; args: unknown };
type Generation = ReturnType<typeof verifyResearchResult> & { outputSha256: string };

// Execute the actual modules with an allowlist of injected dependencies. A new
// module instance per test also isolates the turn handler's in-flight user lock.
function loadModule<T>(relativePath: string, dependencies: Record<string, unknown>): T {
  const filename = resolve(relativePath);
  const { outputText, diagnostics } = ts.transpileModule(readFileSync(filename, "utf8"), {
    fileName: filename, reportDiagnostics: true,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  });
  assert.equal(diagnostics?.length ?? 0, 0, `Transpile failed: ${relativePath}`);
  const loaded = { exports: {} };
  const noNetwork = () => { throw new Error("Unexpected external API call in isolated runtime test"); };
  compileFunction(outputText, ["exports", "require", "module", "fetch", "process", "globalThis"], { filename })(
    loaded.exports,
    (name: string) => {
      assert(Object.hasOwn(dependencies, name), `Unexpected dependency (possible fallback): ${name}`);
      return dependencies[name];
    },
    loaded, noNetwork, Object.freeze({ env: Object.freeze({}) }), Object.freeze({ fetch: noNetwork }),
  );
  return loaded.exports as T;
}

function generated(mode: ResearchR1Mode = "r1_meaning", output = "A response\nwith exact spacing.", complete = true): Generation {
  const variant = R1_VARIANTS[mode];
  const adapterSha = mode === "r1_meaning" ? "73ee9854c4975ce2d79ae7dbabb2fd5d021e9579a7f81ac26ec14b4a5a7fbb2b"
    : mode === "r1_response" ? "0be5dcd4dbda0c37460bfe4e8ac19833326edb04a32fb22252b089dd868a9bc1" : null;
  const verified = verifyResearchResult({ variant, output, generated: true, load_verified: true, complete,
    finish_reason: complete ? "eos" : "repetition_guard", prompt_tokens: 40, context_tokens: 4096, request_seconds: 1,
    identity: { variant, model: R1_MODEL, revision: R1_REVISION, execution_manifest_sha256: R1_EXECUTION,
      adapter_weights_sha256: adapterSha, base_weights_sha256: "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6" },
  }, mode);
  return { ...verified, outputSha256: hash(output) };
}

function prior(role: string, content: string, research: Data = { ...generated().research }): Message {
  return { id: role === "ASSISTANT" ? MESSAGE : "30000000-0000-4000-8000-000000000002",
    conversationId: CONVERSATION, role, content, metadata: { research } };
}

function fixture() {
  const calls: Call[] = [];
  const conversations = new Map([[CONVERSATION, { id: CONVERSATION, userId: OWNER }]]);
  const messages: Message[] = [];
  const auditRows: Data[] = [];
  const state = {
    user: { id: OWNER, preferredLang: "tw", role: "USER" } as { id: string; preferredLang: string; role: string } | null,
    allowed: true, ttsConfigured: false, ttsError: false, auditError: false, assistantWriteError: false,
    generation: generated(),
    reply: undefined as undefined | ((messages: R1Message[], mode: ResearchR1Mode, onDelta?: (chunk: string) => Promise<void>) => Promise<Generation>),
  };
  const log = (name: string, args: unknown) => calls.push({ name, args });
  const prisma = {
    conversation: {
      async findUnique(args: { where: { id: string } }) { log("conversation.findUnique", args); return conversations.get(args.where.id) ?? null; },
      async create(args: { data: Data }) {
        log("conversation.create", args);
        const row = { id: `20000000-0000-4000-8000-${String(conversations.size + 1).padStart(12, "0")}`, userId: String(args.data.userId) };
        conversations.set(row.id, row); return row;
      },
      async update(args: unknown) { log("conversation.update", args); return {}; },
    },
    message: {
      async findMany(args: { where: { conversationId: string }; orderBy: { createdAt: string }; take: number }) {
        log("message.findMany", args);
        assert.equal(args.orderBy.createdAt, "asc");
        return messages.filter(m => m.conversationId === args.where.conversationId).slice(0, args.take);
      },
      async create(args: { data: Data }) {
        log("message.create", args);
        if (state.assistantWriteError && args.data.role === "ASSISTANT") throw new Error("Mock database unavailable");
        const row: Message = { id: `30000000-0000-4000-8000-${String(messages.length + 10).padStart(12, "0")}`,
          conversationId: String(args.data.conversationId), role: String(args.data.role), content: String(args.data.content), metadata: args.data.metadata as Data };
        messages.push(row); return row;
      },
      async findFirst(args: { where: { id: string; conversationId: string; role: string; conversation: { userId: string } } }) {
        log("message.findFirst", args);
        const where = args.where;
        assert.equal(where.role, "ASSISTANT");
        assert.equal(where.conversation.userId, state.user?.id);
        return messages.find(m => m.id === where.id && m.conversationId === where.conversationId && m.role === where.role
          && conversations.get(m.conversationId)?.userId === where.conversation.userId) ?? null;
      },
    },
    auditLog: { async create(args: { data: Data }) {
      log("auditLog.create", args);
      if (state.auditError) throw new Error("Mock audit write unavailable");
      auditRows.push(args.data); return { id: "saved-review" };
    } },
  };
  const dependencies = {
    "@/db/prisma": { prisma }, "@/lib/model-modes": modelModes,
    "@/lib/rate-limit": { rateLimit: (...args: unknown[]) => { log("rateLimit", args); return { allowed: state.allowed }; } },
    "@/lib/research-r1": { R1_VARIANTS, isCompletedResearchReply, isContinuableResearchReply, validateResearchHistory,
      researchR1Reply: async (messages: R1Message[], mode: ResearchR1Mode, onDelta?: (chunk: string) => Promise<void>) => {
        log("researchR1Reply", { messages, mode });
        if (state.reply) return state.reply(messages, mode, onDelta);
        await onDelta?.(state.generation.result.output);
        return state.generation;
      },
    },
    "@/lib/modal-tts": { isModalTtsConfigured: () => state.ttsConfigured,
      modalSpeak: async (...args: unknown[]) => { log("modalSpeak", args); if (state.ttsError) throw new Error("Mock TTS unavailable"); return { audio_base64: "mock-audio", sample_rate: 24000 }; } },
    "@/lib/auth": { getSessionUser: async () => { log("getSessionUser", null); return state.user; } },
    "@/lib/api": { jsonOk: (body: unknown) => Response.json(body), jsonError: (error: string, status = 400) => Response.json({ error }, { status }) },
    "node:crypto": { createHash }, zod: { z },
  };
  const turn = loadModule<{ runResearchR1Turn: (input: ConversationTurnInput, mode: ResearchR1Mode) => Promise<ConversationTurnResult> }>("src/lib/research-r1-turn.ts", dependencies);
  const feedback = loadModule<{ POST: (request: Request) => Promise<Response> }>("src/app/api/research/reply-feedback/route.ts", dependencies);
  const input = (overrides: Partial<ConversationTurnInput> = {}): ConversationTurnInput => ({ user: { id: OWNER, preferredLang: "tw" }, text: "Hello", channel: "WEB", ...overrides });
  return { ...turn, ...feedback, state, calls, conversations, messages, auditRows, input,
    named: (name: string) => calls.filter(call => call.name === name) };
}

async function expectStatus(action: Promise<unknown>, status: number) {
  await assert.rejects(action, error => error instanceof modelModes.ResearchR1Error && error.status === status);
}
function request(body: unknown = { conversationId: CONVERSATION, messageId: MESSAGE, assessment: "useful", note: "" }) {
  return new Request("http://localhost/api/research/reply-feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

const modes = ["r1_base", "r1_meaning", "r1_response"] as const;
for (const mode of modes) test(`completion guard: accepts only the exact completed ${mode} identity`, () => {
  const research = generated(mode).research;
  for (const selected of modes) assert.equal(isCompletedResearchReply(research, selected), mode === selected);
});
test("completion guard: every identity and completion field is required, including execution", () => {
  for (const mode of modes) {
    const research = generated(mode).research;
    for (const field of Object.keys(research)) {
      const incomplete: Data = { ...research };
      delete incomplete[field];
      assert.equal(isCompletedResearchReply(incomplete, mode), false, `${mode} without ${field}`);
    }
  }
});
test("completion guard: complete=true cannot override a missing or non-EOS finish reason", () => {
  for (const mode of modes) {
    for (const finishReason of [undefined, null, "", "EOS", "max_new_tokens", "time_limit", "repetition_guard"]) {
      assert.equal(isCompletedResearchReply({ ...generated(mode).research, finishReason }, mode), false, `${mode}: ${finishReason}`);
    }
  }
});
test("completion guard: EOS cannot override false, missing or truthy non-boolean completion", () => {
  for (const complete of [false, undefined, null, "true", 1]) {
    assert.equal(isCompletedResearchReply({ ...generated().research, complete }, "r1_meaning"), false);
  }
});
test("completion guard: malformed values fail closed without throwing", () => {
  for (const value of [null, undefined, false, true, 0, "meaning-adapter", [], {}]) {
    assert.equal(isCompletedResearchReply(value, "r1_meaning"), false);
  }
});
for (const finishReason of ["max_new_tokens", "time_limit"]) test(`turn: verified bounded ${finishReason} history continues without relabelling completion`, async () => {
  const f = fixture();
  const research = { ...generated().research, complete: false, finishReason };
  assert.equal(isContinuableResearchReply(research, "r1_meaning"), true);
  assert.equal(isContinuableResearchReply({ ...research, adapterSha: "other" }, "r1_meaning"), false);
  f.messages.push(prior("USER", "Original question"), prior("ASSISTANT", "Bounded answer", research));
  await f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning");
  assert.deepEqual(f.named("researchR1Reply")[0].args, { mode: "r1_meaning", messages: [
    { role: "user", content: "Original question" }, { role: "assistant", content: "Bounded answer" },
    { role: "user", content: "Hello" },
  ] });
  assert.equal((f.messages[1].metadata?.research as Data).complete, false);
});
test("turn: empty bounded history remains blocked", async () => {
  const f = fixture();
  f.messages.push(prior("USER", "Question"), prior("ASSISTANT", "  ", { ...generated().research, complete: false, finishReason: "time_limit" }));
  await expectStatus(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning"), 409);
  assert.equal(f.named("researchR1Reply").length, 0);
});
test("completion guard: base requires explicit null weights and adapters require pinned weights", () => {
  assert.equal(isCompletedResearchReply({ ...generated("r1_base").research, adapterSha: "" }, "r1_base"), false);
  assert.equal(isCompletedResearchReply({ ...generated("r1_base").research, adapterSha: undefined }, "r1_base"), false);
  for (const mode of ["r1_meaning", "r1_response"] as const) {
    assert.equal(isCompletedResearchReply({ ...generated(mode).research, adapterSha: null }, mode), false);
  }
});

test("turn: sign-in is required before quota, database or generation", async () => {
  const f = fixture();
  await expectStatus(f.runResearchR1Turn(f.input({ user: null }), "r1_meaning"), 401);
  assert.deepEqual(f.calls, []);
});
test("turn: research quota is 30 per hour per account and rejects before work", async () => {
  const f = fixture(); f.state.allowed = false;
  await expectStatus(f.runResearchR1Turn(f.input(), "r1_meaning"), 429);
  assert.deepEqual(f.calls, [{ name: "rateLimit", args: [`r1-user:${OWNER}`, 30, 3600] }]);
});
for (const owner of [null, OTHER]) test(`turn: missing or foreign conversation (${owner}) cannot be replaced or read`, async () => {
  const f = fixture();
  if (owner) f.conversations.get(CONVERSATION)!.userId = owner; else f.conversations.delete(CONVERSATION);
  await expectStatus(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning"), 404);
  assert.deepEqual(f.calls.map(c => c.name), ["rateLimit", "conversation.findUnique"]);
});
test("turn: blank input creates no conversation or model call", async () => {
  const f = fixture();
  await expectStatus(f.runResearchR1Turn(f.input({ text: " \n " }), "r1_meaning"), 400);
  assert.deepEqual(f.calls.map(c => c.name), ["rateLimit"]);
});

for (const mode of ["r1_base", "r1_meaning", "r1_response"] as const) test(`turn: ${mode} saves exact output and evidence, with no clinical or training approval`, async () => {
  const f = fixture(); const stages: Array<{ name: string; chunk?: string }> = [];
  const output = "Me ho y\u025b.\n\n  \u0190y\u025b with  two spaces.\n\"Quoted\" text.";
  f.state.generation = generated(mode, output);
  const result = await f.runResearchR1Turn(f.input({ text: "  Hello  ", onStage: stage => { stages.push(stage); } }), mode);
  const assistant = f.messages.find(m => m.role === "ASSISTANT")!;
  assert.equal(assistant.content, output); assert.equal(result.reply, output); assert.equal(result.understanding.reply, output);
  assert.equal(result.assistantId, assistant.id); assert.equal(result.userMessageId, f.messages[0].id);
  assert.equal(result.conversationId, assistant.conversationId);
  assert.deepEqual(assistant.metadata?.research, f.state.generation.research);
  assert.deepEqual(assistant.metadata?.evidence, f.state.generation.result);
  assert.equal(assistant.metadata?.outputSha256, hash(output));
  assert.equal(assistant.metadata?.trainingConsent, false); assert.equal(assistant.metadata?.researchOnly, true);
  assert.equal(assistant.metadata?.clinicalAssessment, "not_performed");
  assert.equal(result.understanding.comprehension, undefined); assert.equal(result.understanding.synthesis?.safetyEnforced, false);
  assert.equal(result.stage.review, false); assert.equal(result.stage.retrieveEngine, "none");
  assert.deepEqual(f.named("researchR1Reply")[0].args, { messages: [{ role: "user", content: "Hello" }], mode });
  assert.equal(stages.filter(stage => stage.name === "reply_delta").map(stage => stage.chunk).join(""), output);
  assert.equal(stages.at(-1)?.name, "assistant_message"); assert.equal(f.named("modalSpeak").length, 0);
});
test("turn: complete same-checkpoint history retains every turn, role and spelling", async () => {
  const f = fixture();
  for (let i = 0; i < 20; i++) f.messages.push(prior(i % 2 ? "ASSISTANT" : "USER", `Original ${i}\n  spacing`));
  const expected = f.messages.map(m => ({ role: m.role === "USER" ? "user" : "assistant", content: m.content }));
  const result = await f.runResearchR1Turn(f.input({ conversationId: CONVERSATION, text: "Follow-up" }), "r1_meaning");
  assert.deepEqual(f.named("researchR1Reply")[0].args, { messages: [...expected, { role: "user", content: "Follow-up" }], mode: "r1_meaning" });
  assert.equal(result.understanding.synthesis?.usedHistory, true);
  assert.equal(f.named("conversation.create").length, 0);
});

const rejectedHistories: Array<[string, () => Message[]]> = [
  ["legacy reply", () => [prior("USER", "Hi"), { ...prior("ASSISTANT", "Legacy"), metadata: { synthesis: { provider: "legacy" } } }]],
  ["different variant", () => [prior("USER", "Hi"), prior("ASSISTANT", "Other", { ...generated("r1_response").research })]],
  ["incomplete reply", () => [prior("USER", "Hi"), prior("ASSISTANT", "Partial", { ...generated("r1_meaning", "Partial", false).research })]],
  ["unknown completion", () => [prior("USER", "Hi"), prior("ASSISTANT", "Unknown", { variant: "meaning-adapter" })]],
  ["missing execution identity", () => {
    const research: Data = { ...generated().research }; delete research.executionSha;
    return [prior("USER", "Hi"), prior("ASSISTANT", "Unbound", research)];
  }],
  ["complete flag with non-EOS finish", () => [prior("USER", "Hi"), prior("ASSISTANT", "Partial", { ...generated().research, finishReason: "max_new_tokens" })]],
  ["dangling user after failure", () => [prior("USER", "Unanswered")]],
  ["non-alternating roles", () => [prior("USER", "One"), prior("USER", "Two")]],
  ["oversized full history", () => Array.from({ length: 82 }, (_, i) => prior(i % 2 ? "ASSISTANT" : "USER", `Turn ${i}`))],
  ["total context character overflow", () => Array.from({ length: 8 }, (_, i) => prior(i % 2 ? "ASSISTANT" : "USER", "x".repeat(8000)))],
];
for (const [label, history] of rejectedHistories) test(`turn: rejects ${label} without truncation, generation or writes`, async () => {
  const f = fixture(); f.messages.push(...history()); const before = structuredClone(f.messages);
  await expectStatus(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning"), 409);
  assert.deepEqual(f.messages, before);
  assert.equal(f.named("researchR1Reply").length, 0); assert.equal(f.named("message.create").length, 0);
});
test("turn: incomplete generated output is saved honestly, never spoken or reused", async () => {
  const f = fixture(); f.state.ttsConfigured = true; f.state.generation = generated("r1_meaning", "Partial answer", false);
  const result = await f.runResearchR1Turn(f.input({ conversationId: CONVERSATION, speak: true }), "r1_meaning");
  assert.equal(result.understanding.research?.complete, false); assert.equal(result.understanding.research?.finishReason, "repetition_guard");
  assert.equal(result.reply, "Partial answer"); assert.equal(result.tts, null); assert.equal(f.named("modalSpeak").length, 0);
  await expectStatus(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning"), 409);
  assert.equal(f.named("researchR1Reply").length, 1);
});
test("turn: failed generation after a delta cannot save a fake assistant or silently fall back", async () => {
  const f = fixture(); const failure = new modelModes.ResearchR1Error("No other model was used.");
  const stages: string[] = []; f.state.ttsConfigured = true;
  f.state.reply = async (_messages, _mode, onDelta) => { await onDelta?.("Unverified partial"); throw failure; };
  await assert.rejects(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION, speak: true, onStage: stage => { stages.push(stage.name); } }), "r1_meaning"), error => error === failure);
  assert.deepEqual(f.messages.map(m => m.role), ["USER"]); assert(stages.includes("reply_delta"));
  assert(!stages.includes("assistant_message")); assert.equal(f.named("modalSpeak").length, 0);
  assert.equal(f.named("conversation.update").length, 0); assert.equal(f.named("researchR1Reply").length, 1);
  f.state.reply = undefined;
  await f.runResearchR1Turn(f.input(), "r1_meaning"); // Failure releases the account lock.
  assert.equal(f.named("researchR1Reply").length, 2);
});
test("turn: account lock rejects concurrent generation, releases after completion", async () => {
  const f = fixture(); let release!: () => void; let started!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const entered = new Promise<void>(resolve => { started = resolve; });
  f.state.reply = async () => { started(); await gate; return f.state.generation; };
  const first = f.runResearchR1Turn(f.input(), "r1_meaning"); await entered;
  try {
    await expectStatus(f.runResearchR1Turn(f.input(), "r1_response"), 409);
    assert.equal(f.named("rateLimit").length, 1); assert.equal(f.named("researchR1Reply").length, 1);
  } finally { release(); await first; }
  f.state.reply = undefined; await f.runResearchR1Turn(f.input(), "r1_meaning");
});
test("turn: TTS failure keeps the exact completed text, without another model call", async () => {
  const f = fixture(); f.state.ttsConfigured = true; f.state.ttsError = true;
  const result = await f.runResearchR1Turn(f.input({ speak: true }), "r1_meaning");
  assert.equal(result.reply, f.state.generation.result.output); assert.equal(result.tts, null);
  assert.equal(f.named("modalSpeak").length, 1); assert.equal(f.named("researchR1Reply").length, 1);
});
test("turn: failed assistant persistence propagates instead of claiming completion", async () => {
  const f = fixture(); f.state.assistantWriteError = true; const stages: string[] = [];
  await assert.rejects(f.runResearchR1Turn(f.input({ onStage: stage => { stages.push(stage.name); } }), "r1_meaning"), /Mock database/);
  assert.deepEqual(f.messages.map(m => m.role), ["USER"]); assert(!stages.includes("assistant_message"));
  assert.equal(f.named("conversation.update").length, 0);
});

test("feedback: unauthenticated requests stop before parsing, quota or lookup", async () => {
  const f = fixture(); f.state.user = null;
  const response = await f.POST(new Request("http://localhost", { method: "POST", body: "invalid json" }));
  assert.equal(response.status, 401); assert.deepEqual(f.calls.map(c => c.name), ["getSessionUser"]);
});
test("feedback: rate limit rejects before lookup or writes", async () => {
  const f = fixture(); f.state.allowed = false;
  assert.equal((await f.POST(request())).status, 429);
  assert.deepEqual(f.named("rateLimit")[0].args, [`r1-feedback:${OWNER}`, 40, 60]);
  assert.equal(f.named("message.findFirst").length, 0); assert.equal(f.auditRows.length, 0);
});
for (const assessment of ["useful", "incorrect", "unsure"] as const) test(`feedback: ${assessment} is an explicit review, never training consent`, async () => {
  const f = fixture(); f.messages.push(prior("ASSISTANT", "Exact output\n  with spaces", { ...generated("r1_meaning", "partial", false).research }));
  const before = structuredClone(f.messages);
  const response = await f.POST(request({ conversationId: CONVERSATION, messageId: MESSAGE, assessment, note: "  A correction.  " }));
  assert.equal(response.status, 200); assert.deepEqual(await response.json(), { saved: true, reviewId: "saved-review" });
  assert.deepEqual(f.named("message.findFirst")[0].args, { where: { id: MESSAGE, conversationId: CONVERSATION, role: "ASSISTANT", conversation: { userId: OWNER } } });
  assert.deepEqual(f.auditRows, [{ action: "research.r1.reply_feedback", actorId: OWNER, entityType: "message", entityId: MESSAGE,
    meta: { conversationId: CONVERSATION, assessment, note: "A correction.", research: before[0].metadata?.research,
      outputSha256: hash(before[0].content), trainingConsent: false, source: "app-r1-review" } }]);
  assert.deepEqual(f.messages, before); assert.equal(f.named("message.create").length, 0);
});
for (const scenario of ["other owner", "other conversation", "missing message", "user message", "legacy message"]) test(`feedback: ${scenario} cannot be reviewed`, async () => {
  const f = fixture(); f.messages.push(prior("ASSISTANT", "Reply"));
  if (scenario === "other owner") f.conversations.get(CONVERSATION)!.userId = OTHER;
  if (scenario === "other conversation") f.messages[0].conversationId = OTHER;
  if (scenario === "missing message") f.messages.length = 0;
  if (scenario === "user message") f.messages[0].role = "USER";
  if (scenario === "legacy message") f.messages[0].metadata = { synthesis: {} };
  const response = await f.POST(request());
  assert.equal(response.status, 404); assert.deepEqual(await response.json(), { error: "Research reply not found." });
  assert.equal(f.auditRows.length, 0);
});
for (const change of [{ assessment: undefined }, { assessment: "approved" }, { note: "x".repeat(4001) }, { trainingConsent: true }, { conversationId: "invalid" }, { messageId: "invalid" }]) test(`feedback: invalid input ${Object.keys(change)[0]} is rejected before lookup`, async () => {
  const f = fixture(); const response = await f.POST(request({ conversationId: CONVERSATION, messageId: MESSAGE, assessment: "useful", ...change }));
  assert.equal(response.status, 400); assert.equal(f.named("message.findFirst").length, 0); assert.equal(f.auditRows.length, 0);
});
test("feedback: optional empty note and 4,000-character boundary are accepted", async () => {
  const f = fixture(); f.messages.push(prior("ASSISTANT", "Reply"));
  for (const note of [undefined, "x".repeat(4000)]) {
    assert.equal((await f.POST(request({ conversationId: CONVERSATION, messageId: MESSAGE, assessment: "unsure", note }))).status, 200);
  }
  assert.equal((f.auditRows[0].meta as Data).note, "");
  assert.equal(((f.auditRows[1].meta as Data).note as string).length, 4000);
});
test("feedback: storage failure never returns saved or a review ID", async () => {
  const f = fixture(); f.messages.push(prior("ASSISTANT", "Reply")); f.state.auditError = true;
  const response = await f.POST(request()); const body = await response.json();
  assert.equal(response.status, 500); assert.equal(body.saved, undefined); assert.equal(body.reviewId, undefined);
  assert.equal(f.auditRows.length, 0);
});

// These are regression checks, not permissions to normalize invalid history.
for (const scenario of ["wrong execution", "wrong revision", "wrong model", "wrong adapter", "SYSTEM role"]) test(`turn: rejects mixed history with ${scenario}`, async () => {
  const f = fixture(); const research = { ...generated().research };
  if (scenario === "wrong execution") research.executionSha = "unrelated-execution";
  if (scenario === "wrong revision") research.revision = "unrelated-revision";
  if (scenario === "wrong model") research.model = "unrelated-model";
  if (scenario === "wrong adapter") research.adapterSha = "unrelated-adapter";
  f.messages.push(prior("USER", "Earlier request"), prior(scenario === "SYSTEM role" ? "SYSTEM" : "ASSISTANT", "Earlier answer", research));
  await expectStatus(f.runResearchR1Turn(f.input({ conversationId: CONVERSATION }), "r1_meaning"), 409);
  assert.equal(f.named("researchR1Reply").length, 0); assert.equal(f.named("message.create").length, 0);
});
