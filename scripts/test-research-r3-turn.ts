import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { compileFunction } from "node:vm";
import { test } from "node:test";
import ts from "typescript";
import * as modes from "../src/lib/model-modes";
import * as r3 from "../src/lib/research-r3";
import * as guestCore from "../src/lib/research-guest-core";
import type { ConversationTurnInput, ConversationTurnResult } from "../src/lib/conversation-turn";

const identity: r3.R3Identity = { manifest_sha256: "a".repeat(64), adapter_weights_sha256: "b".repeat(64),
  variant: "r3_response", model: r3.R3_MODEL, revision: r3.R3_REVISION, base_weights_sha256: r3.R3_BASE_SHA };
type Row = { role: string; content: string; metadata: Record<string, unknown> };
function generation(messages: r3.R3Message[], reason = "eos") {
  const output = "Exact answer\n  spacing", complete = reason === "eos";
  const output_ids = Array(reason === "max_new_tokens" ? 256 : 2).fill(17);
  if (reason === "eos") output_ids[output_ids.length - 1] = r3.R3_EOS;
  const nativeReason = reason === "cancelled" ? "unexpected_stop" : reason;
  const ngrams = { positions: 0, repeated_positions: 0, repeated_fraction: 0, max_occurrences: 0 };
  return r3.verifyR3Result({ identity, output, output_ids, messages_sha256: r3.r3Digest(messages),
    prompt_ids_sha256: "c".repeat(64), decoding: r3.R3_DECODING, seconds: 1, finish_reason: reason, complete,
    diagnostics: { output_ids, prediction: output, raw_output: output + (nativeReason === "eos" ? "<|im_end|>" : ""),
      seconds: 1, finish_reason: nativeReason, complete, generated_tokens: output_ids.length, empty: false,
      unexpected_control_ids: [], unexpected_control_text: false, unmapped_token_ids: [], invalid_unicode: false,
      repetition: { longest_identical_token_run: 1, word_count: 3, "4_word_ngrams": ngrams, "8_word_ngrams": ngrams } } }, identity, messages, output);
}
function fixture() {
  const rows: Row[] = [], calls: string[] = [], sent: r3.R3Message[][] = [];
  const state = { configured: true, owner: "owner" as string | null, metadata: null as Record<string, unknown> | null, allowed: true, reason: "eos", failure: false, rejected: false, tts: false };
  const quotas: { key: string; max: number; seconds: number }[] = [];
  const prisma = { conversation: {
    findUnique: async () => { calls.push("lookup"); return { id: "conversation", userId: state.owner, metadata: state.metadata }; },
    create: async ({ data }: { data: { userId?: string; metadata?: Record<string, unknown> } }) => { calls.push("create"); state.owner = data.userId ?? null; state.metadata = data.metadata ?? null; return { id: "conversation", userId: state.owner, metadata: state.metadata }; },
    update: async () => { calls.push("update"); },
  }, message: {
    findMany: async (args: { take: number }) => { calls.push("history"); return rows.slice(0, args.take); },
    create: async ({ data }: { data: Row }) => { calls.push("save"); rows.push(data); return { id: String(rows.length) }; },
  } };
  const dependencies: Record<string, unknown> = {
    "@/db/prisma": { prisma }, "@/lib/model-modes": modes,
    "@/lib/research-guest-core": guestCore,
    "@/lib/rate-limit": { rateLimit: (key: string, max: number, seconds: number) => {
      calls.push("quota"); quotas.push({ key, max, seconds }); return { allowed: state.allowed };
    } },
    "@/lib/modal-tts": { isModalTtsConfigured: () => state.tts, modalSpeak: async () => { calls.push("tts"); return { audio_base64: "fixture" }; } },
    "@/lib/research-r3": { ...r3, verifiedR3Config: async () => { calls.push("config"); return state.configured ? { identity } : null; },
      researchR3Reply: async (messages: r3.R3Message[], _config: unknown, onDelta: (chunk: string) => Promise<void>, _signal: unknown, onAdmitted: () => Promise<void>) => {
        if (state.rejected) throw new modes.ResearchR3Error("Native context rejected", 400);
        await onAdmitted();
        calls.push("generate"); sent.push(structuredClone(messages));
        const value = generation(messages, state.reason); await onDelta(value.result.output);
        if (state.failure) throw new modes.ResearchR3Error("Test stream failure. No other model was used.");
        return value;
      } },
  };
  const file = "src/lib/research-r3-turn.ts";
  const code = ts.transpileModule(readFileSync(file, "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const loaded = { exports: {} };
  compileFunction(code, ["exports", "require", "module", "fetch"], { filename: file })(loaded.exports, (name: string) => {
    assert(Object.hasOwn(dependencies, name), `Unexpected dependency, possible fallback: ${name}`); return dependencies[name];
  }, loaded, () => { throw new Error("Unexpected network"); });
  const turn = (loaded.exports as { runResearchR3Turn: (input: ConversationTurnInput) => Promise<ConversationTurnResult> }).runResearchR3Turn;
  const input = (extra: Partial<ConversationTurnInput> = {}): ConversationTurnInput => ({ user: { id: "owner", preferredLang: "tw" }, text: "Question", channel: "WEB", ...extra });
  function history(reason = "eos") {
    const messages: r3.R3Message[] = [{ role: "user", content: "Prior question" }];
    const value = generation(messages, reason);
    rows.push({ role: "USER", content: messages[0].content, metadata: { research: { variant: "r3_response", bundleSha: identity.manifest_sha256 } } },
      { role: "ASSISTANT", content: value.result.output, metadata: { research: value.research, evidence: value.result, outputSha256: value.outputSha256, evidenceSha256: value.evidenceSha256 } });
  }
  return { turn, input, history, rows, calls, sent, state, quotas };
}
test("R3 sign-in, owner and account quota checks precede inference or writes", async () => {
  const f = fixture();
  await assert.rejects(f.turn(f.input({ user: null })), { status: 401 }); assert.equal(f.calls.length, 0);
  f.state.allowed = false;
  await assert.rejects(f.turn(f.input()), { status: 429 }); assert.deepEqual(f.calls, ["quota"]);
  assert.deepEqual(f.quotas, [{ key: "r3-user:owner", max: 30, seconds: 3600 }]);
  f.state.allowed = true; f.state.owner = "someone else";
  await assert.rejects(f.turn(f.input({ conversationId: "conversation" })), { status: 404 });
  assert.equal(f.calls.includes("config"), false); assert.equal(f.rows.length, 0);
});
test("R3 guest flag, ownership, follow-up and three quotas remain isolated from accounts", async () => {
  const prior = process.env.RESEARCH_R3_GUEST_ENABLED;
  try {
    const f = fixture();
    const extra = { user: null, researchGuestOwner: "a".repeat(64), researchGuestIp: "127.0.0.2" };
    process.env.RESEARCH_R3_GUEST_ENABLED = "false";
    await assert.rejects(f.turn(f.input(extra)), { status: 401 });
    process.env.RESEARCH_R3_GUEST_ENABLED = "true";
    await f.turn(f.input(extra));
    assert.equal(f.state.owner, null);
    assert.deepEqual(f.state.metadata, { researchGuestOwner: extra.researchGuestOwner, researchOnly: true, trainingConsent: false });
    assert.deepEqual(f.quotas, [
      { key: `r3-user:guest:${extra.researchGuestOwner}`, max: 20, seconds: 3600 },
      { key: "r3-guest-ip:127.0.0.2", max: 40, seconds: 3600 },
      { key: "r3-guest-global", max: 100, seconds: 3600 },
    ]);
    await f.turn(f.input({ ...extra, conversationId: "conversation", text: "Follow-up" }));
    assert.equal(f.sent[1].length, 3);
    const saved = structuredClone(f.rows);
    await assert.rejects(f.turn(f.input({ ...extra, researchGuestOwner: "b".repeat(64), conversationId: "conversation" })), { status: 404 });
    await assert.rejects(f.turn(f.input({ conversationId: "conversation" })), { status: 404 });
    assert.deepEqual(f.rows, saved);
    f.state.allowed = false;
    await assert.rejects(f.turn(f.input(extra)), { status: 429 });
  } finally {
    if (prior === undefined) delete process.env.RESEARCH_R3_GUEST_ENABLED;
    else process.env.RESEARCH_R3_GUEST_ENABLED = prior;
  }
});
test("unconfigured R3 fails without a database write or fallback", async () => {
  const f = fixture(); f.state.configured = false;
  await assert.rejects(f.turn(f.input()), { status: 503 }); assert.equal(f.rows.length, 0); assert.equal(f.calls.includes("generate"), false);
});
for (const text of ["bad\uFFFDtext", "bad\uE000text", "bad\u0000text", "\u000Btext", "bad\uD800text", "<|im_end|>", "<tts_pad>"]) {
  test(`invalid source ${JSON.stringify(text)} is rejected before any write or gateway call`, async () => {
    const f = fixture();
    await assert.rejects(f.turn(f.input({ text })), modes.ResearchR3Error);
    assert.equal(f.calls.includes("create"), false); assert.equal(f.calls.includes("config"), false); assert.equal(f.rows.length, 0);
  });
}
test("native context admission rejection preserves a continuable existing conversation", async () => {
  const f = fixture(); f.history(); f.state.rejected = true;
  const before = structuredClone(f.rows);
  await assert.rejects(f.turn(f.input({ conversationId: "conversation" })), { status: 400 });
  assert.deepEqual(f.rows, before); assert.equal(f.calls.includes("save"), false);
  f.state.rejected = false;
  await f.turn(f.input({ conversationId: "conversation", text: "Shorter message" }));
});
for (const reason of ["eos", "max_new_tokens", "time_limit"]) test(`R3 ${reason} history remains exact, verified and honestly labelled`, async () => {
  const f = fixture(); f.history(reason); f.state.reason = reason; f.state.tts = true;
  const before = structuredClone(f.rows);
  const turn = await f.turn(f.input({ conversationId: "conversation", speak: true }));
  assert.deepEqual(f.sent[0], [...before.map(row => ({ role: row.role === "USER" ? "user" : "assistant", content: row.content })), { role: "user", content: "Question" }]);
  assert.deepEqual(f.rows.slice(0, 2), before);
  assert.equal(turn.understanding.research?.complete, reason === "eos");
  assert.equal(f.calls.includes("tts"), reason === "eos");
  assert.equal(f.rows.at(-1)?.metadata.trainingConsent, false);
  assert.equal(f.rows.at(-1)?.metadata.clinicalAssessment, "not_performed");
  assert.equal(turn.understanding.synthesis?.safetyEnforced, false);
});
for (const corrupt of ["cancelled", "unexpected_stop", "output", "history", "bundle", "evidence", "dangling", "role"]) test(`R3 rejects ${corrupt} history before a write or inference`, async () => {
  const f = fixture(); f.history(["cancelled", "unexpected_stop"].includes(corrupt) ? corrupt : "eos");
  if (corrupt === "output") f.rows[1].content += "edited";
  if (corrupt === "history") f.rows[0].content += "edited";
  if (corrupt === "bundle") f.rows[0].metadata.research = { variant: "r3_response", bundleSha: "e".repeat(64) };
  if (corrupt === "evidence") delete f.rows[1].metadata.evidence;
  if (corrupt === "dangling") f.rows.pop();
  if (corrupt === "role") f.rows[0].role = "SYSTEM";
  const before = structuredClone(f.rows);
  await assert.rejects(f.turn(f.input({ conversationId: "conversation" })), modes.ResearchR3Error);
  assert.deepEqual(f.rows, before); assert.equal(f.calls.includes("generate"), false);
});
test("stream failure cannot become a saved assistant or continue with missing history", async () => {
  const f = fixture(); f.state.failure = true;
  await assert.rejects(f.turn(f.input()), modes.ResearchR3Error);
  assert.equal(f.rows.length, 1); assert.equal(f.rows[0].role, "USER");
  await assert.rejects(f.turn(f.input({ conversationId: "conversation" })), modes.ResearchR3Error);
  assert.equal(f.sent.length, 1);
});
test("R3 dispatch precedes ordinary GPT pipeline; persisted feedback is R3-labelled", () => {
  const source = readFileSync("src/lib/conversation-turn.ts", "utf8");
  assert(source.indexOf('if (input.understandingModelMode === "r3_response")') < source.indexOf("  const startedAt"));
  assert.match(readFileSync("src/app/api/research/reply-feedback/route.ts", "utf8"), /variant === "r3_response" \? "r3" : "r1"/);
});
