import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { repeatedPassage, LOOP_POLICY } from "../src/lib/response-loop-guard";
import { MODEL_MODES, ResearchR3Error, isResearchModelMode } from "../src/lib/model-modes";
import { R3_MODEL, R3_REVISION, R3_BASE_SHA, R3_EOS, R3_DECODING, r3Config, verifiedR3Config, r3Digest,
  researchR3Reply, validateR3History, verifyR3Result, verifyR3StoredReply, type R3Identity, type R3Message } from "../src/lib/research-r3";

// Synthetic pins only. These are not R3 training receipts or a model evaluation.
const identity: R3Identity = { manifest_sha256: "a".repeat(64), variant: "r3_response", model: R3_MODEL,
  revision: R3_REVISION, base_weights_sha256: R3_BASE_SHA, adapter_weights_sha256: "b".repeat(64) };
const messages: R3Message[] = [{ role: "user", content: "Keep exact\nspacing and Twi: \u025b" }];
const requestId = "c".repeat(32);
const config = { url: new URL("https://fixture.modal.run"), token: "test-only-credential-not-real-000000", identity };
function result(output = "Exact\n  answer.", finish_reason = "eos", complete = true) {
  const output_ids = Array(finish_reason === "max_new_tokens" ? 256 : 3).fill(12);
  if (finish_reason === "eos") output_ids[output_ids.length - 1] = R3_EOS;
  const nativeReason = finish_reason === "cancelled" ? "unexpected_stop" : finish_reason;
  const ngrams = { positions: 0, repeated_positions: 0, repeated_fraction: 0, max_occurrences: 0 };
  return { identity, messages_sha256: r3Digest(messages), prompt_ids_sha256: "d".repeat(64), decoding: R3_DECODING,
    output_ids, output, seconds: 1, finish_reason, complete,
    diagnostics: { output_ids, prediction: output, raw_output: output + (nativeReason === "eos" ? "<|im_end|>" : ""),
      seconds: 1, finish_reason: nativeReason, complete, generated_tokens: output_ids.length, empty: !output.trim(),
      unexpected_control_ids: [], unexpected_control_text: false, unmapped_token_ids: [], invalid_unicode: false,
      repetition: { longest_identical_token_run: 2, word_count: 2, "4_word_ngrams": ngrams, "8_word_ngrams": ngrams } } };
}
function events(value = result()): Array<Record<string, unknown>> {
  return [{ type: "start", protocol: "r2-demo-stream-v1", request_id: requestId, call_id: "fc-test123", identity, messages_sha256: r3Digest(messages) },
    { type: "delta", request_id: requestId, index: 0, text: value.output },
    { type: "final", request_id: requestId, result: value }];
}
function response(items: unknown[], tail = "") {
  const bytes = new TextEncoder().encode(": keep-alive\n\n" + items.map(e => `data: ${JSON.stringify(e)}\n\n`).join("") + tail);
  let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset >= bytes.length) controller.close();
    else { controller.enqueue(bytes.slice(offset, offset + 7)); offset += 7; }
  } }), { headers: { "content-type": "text/event-stream" } });
}
const health = { service: "research-r3-gateway-v1", configured: true, upstream_checked: false,
  protocol: "r2-demo-stream-v1", actor: { app_name: "fixture-r3", class_name: "UnderstandingR3Demo" }, identity,
  variants: ["r3_response"], decoding: R3_DECODING, limits: { context_tokens: 4096, body_bytes: 65536 } };

test("R3 explicit mode preserves Stable default and R1 modes", () => {
  assert.equal(MODEL_MODES[0], "shadow");
  assert.equal(isResearchModelMode("r3_response"), true);
  assert.equal(isResearchModelMode("shadow"), false);
  assert.match(readFileSync("src/lib/understanding-model-store.ts", "utf8"), /return "shadow"/);
});
test("repetition detector allows varied lists and short emphasis but detects Twi loops", () => {
  assert.equal(repeatedPassage("thank you ".repeat(5)), null);
  assert.equal(repeatedPassage("Please do not take another person's medicine. ".repeat(2)), null);
  assert.equal(repeatedPassage(["food", "water", "comfort", "sleep", "attention", "clothes", "help"].map(x => "It could be that the child wants " + x).join(" ")), null);
  assert.deepEqual(repeatedPassage("Ebia ɛyɛ aduru bi a ɛhaw no. ".repeat(8)), { block_words: 7, repetitions: 3, end_word: 32 });
  const long = Array.from({ length: 27 }, (_, i) => `word${i}`).join(" ") + " ";
  assert.deepEqual(repeatedPassage(long.repeat(2)), { block_words: 27, repetitions: 2, end_word: 54 });
});
test("paused reply needs reproducible guard evidence and remains usable as exact follow-up context", () => {
  const output = "Ebia ɛyɛ aduru bi a ɛhaw no. ".repeat(8);
  const raw = { ...result(output, "cancelled", false), loop_guard: { version: LOOP_POLICY, trigger: repeatedPassage(output) } };
  const verified = verifyR3Result(raw, identity, messages, output, true);
  assert.equal(verified.research.finishReason, "loop_guard");
  assert.equal(verified.research.complete, false);
  const meta = { research: verified.research, evidence: raw, outputSha256: verified.outputSha256, evidenceSha256: verified.evidenceSha256 };
  verifyR3StoredReply(meta, output, identity, messages);
  assert.throws(() => verifyR3Result(raw, identity, messages, output), ResearchR3Error);
  assert.throws(() => verifyR3Result({ ...raw, loop_guard: { version: LOOP_POLICY, trigger: { block_words: 8, repetitions: 3, end_word: 32 } } }, identity, messages, output, true), ResearchR3Error);
  assert.throws(() => verifyR3Result({ ...raw, complete: true }, identity, messages, output, true), ResearchR3Error);
  assert.throws(() => verifyR3Result(result(), identity, messages, "Exact\n  answer.", true), ResearchR3Error);
});
test("guarded transport opts in explicitly while raw transport contract stays unchanged", async t => {
  const output = "Ebia ɛyɛ aduru bi a ɛhaw no. ".repeat(8);
  const raw = { ...result(output, "cancelled", false), loop_guard: { version: LOOP_POLICY, trigger: repeatedPassage(output) } };
  const sequence = events(raw);
  sequence[0].loop_guard = LOOP_POLICY;
  t.mock.method(globalThis, "fetch", async (_url: unknown, init?: RequestInit) => {
    assert.deepEqual(JSON.parse(String(init?.body)), { messages, variant: "r3_response", loop_guard: true });
    return response(sequence);
  });
  const actual = await researchR3Reply(messages, { ...config, loopGuard: true });
  assert.equal(actual.result.output, output);
  assert.equal(actual.research.finishReason, "loop_guard");
});
test("config allows 20s for cold health and fails closed without approval, identity or authentication", async t => {
  const keys = ["ENABLED", "URL", "TOKEN", "MANIFEST_SHA256", "ADAPTER_SHA256"].map(k => `RESEARCH_R3_${k}`);
  const before = keys.map(k => process.env[k]);
  t.after(() => keys.forEach((k, i) => before[i] === undefined ? delete process.env[k] : process.env[k] = before[i]));
  for (const k of keys) delete process.env[k];
  let calls = 0, status = 200, timedOut = false;
  const deadlineSignal = new AbortController().signal;
  t.mock.method(AbortSignal, "timeout", (milliseconds: number) => {
    assert.equal(milliseconds, 20000);
    return deadlineSignal;
  });
  t.mock.method(globalThis, "fetch", async (url: RequestInfo | URL, init?: RequestInit) => {
    calls++; assert.equal(String(url), "https://fixture.modal.run/health");
    assert.equal((init?.headers as Record<string, string>).Authorization, `Bearer ${config.token}`);
    assert.equal(init?.redirect, "error");
    assert.equal(init?.signal, deadlineSignal);
    if (timedOut) throw new DOMException("Synthetic health timeout", "TimeoutError");
    return Response.json(health, { status });
  });
  assert.equal(await verifiedR3Config(), null); assert.equal(calls, 0);
  Object.assign(process.env, { RESEARCH_R3_ENABLED: "true", RESEARCH_R3_URL: config.url.href, RESEARCH_R3_TOKEN: config.token,
    RESEARCH_R3_MANIFEST_SHA256: identity.manifest_sha256, RESEARCH_R3_ADAPTER_SHA256: identity.adapter_weights_sha256 });
  assert.deepEqual((await verifiedR3Config())?.identity, identity);
  timedOut = true;
  const beforeTimeout = calls;
  assert.equal(await verifiedR3Config(), null);
  assert.equal(calls, beforeTimeout + 1, "No automatic retry on health timeout");
  timedOut = false;
  status = 401; assert.equal(await verifiedR3Config(), null); status = 200;
  process.env.RESEARCH_R3_ADAPTER_SHA256 = "f".repeat(64);
  assert.equal(await verifiedR3Config(), null);
  for (const url of ["http://fixture.modal.run", "https://fixture.modal.run.evil.test", "https://user:pass@fixture.modal.run", "https://fixture.modal.run?key=secret"]) {
    process.env.RESEARCH_R3_URL = url; assert.equal(r3Config(), null);
  }
});
test("history is complete, alternating and body bounded, never silently truncated", () => {
  validateR3History(messages);
  for (const history of [[], [...messages, ...messages], [{ role: "user", content: " " }],
    [{ role: "user", content: "x".repeat(16001) }], Array.from({ length: 131 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: "x" })),
    Array.from({ length: 5 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: "\u025b".repeat(10000) }))]) {
    assert.throws(() => validateR3History(history as R3Message[]), ResearchR3Error);
  }
  assert.equal(r3Digest([{ role: "user", content: "hello" }]), "d98167dd28f22e330824942ba4d4ce217c2411a0d1141d60b40fe4cb8dc0d232");
  assert.equal(r3Digest([{ content: "hello", role: "user" }]), r3Digest([{ role: "user", content: "hello" }]));
});
test("final binds checkpoint, ordered full history, prompt hash, decoding and exact output", () => {
  const raw = result();
  const verified = verifyR3Result(raw, identity, messages, raw.output);
  assert.equal(verified.research.bundleSha, identity.manifest_sha256);
  const bad = [{ ...raw, identity: { ...identity, adapter_weights_sha256: "e".repeat(64) } },
    { ...raw, messages_sha256: "e".repeat(64) }, { ...raw, prompt_ids_sha256: "" },
    { ...raw, decoding: { ...R3_DECODING, truncate: true } }, { ...raw, extra: true },
    { ...raw, output: raw.output.trim() + " " }, { ...raw, diagnostics: null },
    { ...raw, diagnostics: { ...raw.diagnostics, prediction: "other" } }, { ...raw, seconds: Infinity },
    { ...raw, finish_reason: "time_limit", complete: true }, { ...raw, output_ids: [] }];
  for (const value of bad) assert.throws(() => verifyR3Result(value, identity, messages, raw.output), ResearchR3Error);
});
for (const reason of ["eos", "max_new_tokens", "time_limit", "unexpected_stop", "cancelled"]) test(`stored ${reason} result is rechecked before continuation`, () => {
  const raw = result("Original answer", reason, reason === "eos");
  const verified = verifyR3Result(raw, identity, messages, raw.output);
  const meta = { research: verified.research, evidence: raw, outputSha256: verified.outputSha256, evidenceSha256: verified.evidenceSha256 };
  const check = () => verifyR3StoredReply(meta, raw.output, identity, messages);
  if (["eos", "max_new_tokens", "time_limit"].includes(reason)) check(); else assert.throws(check, ResearchR3Error);
  assert.throws(() => verifyR3StoredReply(meta, "changed answer", identity, messages), ResearchR3Error);
  assert.throws(() => verifyR3StoredReply(meta, raw.output, identity, [{ role: "user", content: "Changed history" }]), ResearchR3Error);
});
test("saved evidence integrity binds every diagnostic field; impossible EOS/vocabulary are rejected", () => {
  const raw = result(); const verified = verifyR3Result(raw, identity, messages, raw.output);
  for (const ids of [[248077], [R3_EOS, 12], [12]]) {
    const corrupt = { ...raw, output_ids: ids, diagnostics: { ...raw.diagnostics, output_ids: ids } };
    assert.throws(() => verifyR3Result(corrupt, identity, messages, raw.output), ResearchR3Error);
  }
  const changed = structuredClone(raw); changed.diagnostics.repetition.word_count++;
  assert.throws(() => verifyR3StoredReply({ ...verified, evidence: changed }, raw.output, identity, messages), ResearchR3Error);
});
test("gateway context rejection is not transient service failure and never admits a write", async t => {
  t.mock.method(globalThis, "fetch", async () => new Response("Invalid context", { status: 400 }));
  let admitted = false;
  await assert.rejects(researchR3Reply(messages, config, undefined, undefined, async () => { admitted = true; }), { status: 400 });
  assert.equal(admitted, false);
});
test("valid non-BMP Unicode and gateway-sized output are not rejected by UTF-16 length", async t => {
  validateR3History([{ role: "user", content: "\u{1F642}".repeat(8000) }]);
  const raw = result("\u{1F642}".repeat(40000));
  t.mock.method(globalThis, "fetch", async () => Response.json({}));
  assert.equal(verifyR3Result(raw, identity, messages, raw.output).result.output, raw.output);
  assert.throws(() => verifyR3Result(result("x".repeat(65537)), identity, messages, "x".repeat(65537)), ResearchR3Error);
});
test("fragmented SSE verifies start before deltas and final before success", async t => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async (url: RequestInfo | URL, init?: RequestInit) => {
    calls++; assert.equal(String(url), "https://fixture.modal.run/stream");
    assert.equal(init?.redirect, "error");
    assert.deepEqual(JSON.parse(String(init?.body)), { messages, variant: "r3_response" });
    assert.equal((init?.headers as Record<string, string>).Authorization, `Bearer ${config.token}`);
    return response(events());
  });
  const deltas: string[] = [];
  const verified = await researchR3Reply(messages, config, chunk => { deltas.push(chunk); });
  assert.equal(deltas.join(""), result().output); assert.equal(verified.result.output, result().output); assert.equal(calls, 1);
});
for (const kind of ["no-start", "wrong-start", "wrong-request", "wrong-index", "no-final", "duplicate-final", "after-final", "wrong-output", "error", "truncated"]) {
  test(`SSE ${kind} fails without fallback`, async t => {
    const stream = events(); let tail = "";
    if (kind === "no-start") stream.shift();
    if (kind === "wrong-start") stream[0] = { ...stream[0], identity: { ...identity, manifest_sha256: "e".repeat(64) } };
    if (kind === "wrong-request") stream[1].request_id = "f".repeat(32);
    if (kind === "wrong-index") stream[1].index = 1;
    if (kind === "no-final") stream.pop();
    if (kind === "duplicate-final") stream.push(stream[2]);
    if (kind === "after-final") stream.push(stream[1]);
    if (kind === "wrong-output") stream[2].result = result("Not streamed");
    if (kind === "error") stream[2] = { type: "error", request_id: requestId };
    if (kind === "truncated") tail = "data: {";
    let calls = 0;
    t.mock.method(globalThis, "fetch", async () => { calls++; return response(stream, tail); });
    await assert.rejects(researchR3Reply(messages, config), ResearchR3Error); assert.equal(calls, 1);
  });
}
test("downstream abort cancels fetch and never returns partial text as final", async t => {
  const controller = new AbortController(); let cancelled = false;
  t.mock.method(globalThis, "fetch", async (_url: RequestInfo | URL, init?: RequestInit) => {
    init?.signal?.addEventListener("abort", () => { cancelled = true; });
    controller.abort();
    return response(events());
  });
  await assert.rejects(researchR3Reply(messages, config, undefined, controller.signal), ResearchR3Error);
  assert.equal(cancelled, true);
});
