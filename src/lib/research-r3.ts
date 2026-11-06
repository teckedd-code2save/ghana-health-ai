import { createHash } from "node:crypto";
import { z } from "zod";
import { ResearchR3Error, type ResearchR3Metadata } from "@/lib/model-modes";
import { LOOP_POLICY, repeatedPassage } from "@/lib/response-loop-guard";

export const R3_MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1";
export const R3_REVISION = "693f12e444a30971ca53889128d7bade9491ecb3";
export const R3_BASE_SHA = "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6";
export const R3_EOS = 248046;
const R3_VOCAB = 248077;
const MAX_OUTPUT_CHARS = 65536;
const codepoints = (text: string) => Array.from(text).length;
const RESERVED = ["endoftext", "im_start", "im_end", "object_ref_start", "object_ref_end", "box_start", "box_end",
  "quad_start", "quad_end", "vision_start", "vision_end", "vision_pad", "image_pad", "video_pad", "audio_start", "audio_end", "audio_pad"]
  .map(name => `<|${name}|>`).concat(["<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"]);
function validSource(text: string) {
  return !RESERVED.some(token => text.includes(token)) && !Array.from(text).some(c =>
    /[\uFFFD\p{Co}\p{Cs}]/u.test(c) || (/\p{Cc}/u.test(c) && !"\n\r\t".includes(c)));
}
export const R3_DECODING = { do_sample: false, enable_thinking: false, max_new_tokens: 256,
  num_beams: 1, repetition_penalty: 1, seed: 17, truncate: false } as const;
const sha = z.string().regex(/^[a-f0-9]{64}$/);
const identitySchema = z.object({ manifest_sha256: sha, variant: z.literal("r3_response"),
  model: z.literal(R3_MODEL), revision: z.literal(R3_REVISION),
  base_weights_sha256: z.literal(R3_BASE_SHA), adapter_weights_sha256: sha }).strict();
export type R3Identity = z.infer<typeof identitySchema>;
export type R3Message = { role: "user" | "assistant"; content: string };
type R3Config = { url: URL; token: string; identity: R3Identity; loopGuard?: boolean };
const fail = (status = 503): never => { throw new ResearchR3Error("R3 evidence could not be verified. No other model was used.", status); };

// Matches Python's sorted, UTF-8 JSON digest used by the sealed actor bundle.
function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, canonical(v)]));
  return value;
}
export const r3Digest = (value: unknown) => createHash("sha256").update(JSON.stringify(canonical(value))).digest("hex");
export const r3OutputHash = (value: string) => createHash("sha256").update(value).digest("hex");
const equal = (a: unknown, b: unknown) => a !== undefined && b !== undefined && r3Digest(a) === r3Digest(b);

export function r3Config(): R3Config | null {
  if (process.env.RESEARCH_R3_ENABLED !== "true" || (process.env.RESEARCH_R3_TOKEN?.length ?? 0) < 32) return null;
  try {
    const url = new URL(process.env.RESEARCH_R3_URL!);
    if (url.protocol !== "https:" || !url.hostname.endsWith(".modal.run") || url.username || url.password || url.search || url.hash || url.port) return null;
    const identity = identitySchema.parse({ manifest_sha256: process.env.RESEARCH_R3_MANIFEST_SHA256,
      variant: "r3_response", model: R3_MODEL, revision: R3_REVISION, base_weights_sha256: R3_BASE_SHA,
      adapter_weights_sha256: process.env.RESEARCH_R3_ADAPTER_SHA256 });
    return { url, token: process.env.RESEARCH_R3_TOKEN!, identity, loopGuard: process.env.RESEARCH_R3_LOOP_GUARD === "true" };
  } catch { return null; }
}

function endpoint(config: R3Config, path: string) {
  const url = new URL(config.url);
  url.pathname = url.pathname.replace(/\/$/, "") + path;
  return url;
}

export async function verifiedR3Config(): Promise<R3Config | null> {
  const config = r3Config();
  if (!config) return null;
  try {
    const response = await fetch(endpoint(config, "/health"), { cache: "no-store", redirect: "error", signal: AbortSignal.timeout(20000),
      headers: { Authorization: `Bearer ${config.token}` } });
    if (!response.ok) return null;
    const health = await response.json();
    if (health.service !== "research-r3-gateway-v1" || health.configured !== true || health.upstream_checked !== false ||
        health.protocol !== "r2-demo-stream-v1" || !equal(health.identity, config.identity) ||
        health.actor?.class_name !== "UnderstandingR3Demo" || typeof health.actor?.app_name !== "string" ||
        !equal(health.variants, ["r3_response"]) || !equal(health.decoding, R3_DECODING) ||
        health.limits?.context_tokens !== 4096 || health.limits?.body_bytes !== 65536) return null;
    if (config.loopGuard && health.loop_guard !== LOOP_POLICY) return null;
    return config;
  } catch { return null; }
}

export function validateR3History(messages: R3Message[]) {
  const parsed = z.array(z.object({ role: z.enum(["user", "assistant"]), content: z.string().min(1).refine(text => codepoints(text) <= 16000 && validSource(text)) }).strict()).min(1).max(129).safeParse(messages);
  if (!parsed.success || messages.length % 2 !== 1 || messages.some((m, i) => !m.content.trim() || m.role !== (i % 2 ? "assistant" : "user")) ||
      Buffer.byteLength(JSON.stringify({ messages, variant: "r3_response" })) > 65536) {
    throw new ResearchR3Error("Start a new R3 conversation. No history will be silently removed.", 409);
  }
}

const tokenIds = z.array(z.number().int().nonnegative().max(R3_VOCAB - 1)).max(256);
const ngrams = z.object({ positions: z.number().int().nonnegative(), repeated_positions: z.number().int().nonnegative(),
  repeated_fraction: z.number().min(0).max(1), max_occurrences: z.number().int().nonnegative() }).strict();
const diagnosticsSchema = z.object({ output_ids: tokenIds, raw_output: z.string(), prediction: z.string(), generated_tokens: z.number().int().min(1).max(256),
  finish_reason: z.enum(["eos", "max_new_tokens", "time_limit", "unexpected_stop"]), complete: z.boolean(),
  unexpected_control_ids: tokenIds, empty: z.boolean(), seconds: z.number().nonnegative().max(300),
  unexpected_control_text: z.boolean(), unmapped_token_ids: tokenIds, invalid_unicode: z.boolean(),
  repetition: z.object({ longest_identical_token_run: z.number().int().nonnegative(), word_count: z.number().int().nonnegative(),
    "4_word_ngrams": ngrams, "8_word_ngrams": ngrams }).strict() }).strict();
const resultSchema = z.object({ identity: identitySchema, messages_sha256: sha, prompt_ids_sha256: sha,
  decoding: z.record(z.string(), z.unknown()), output: z.string().refine(text => codepoints(text) <= MAX_OUTPUT_CHARS),
  output_ids: tokenIds.min(1), seconds: z.number().nonnegative().max(300),
  finish_reason: z.enum(["eos", "max_new_tokens", "time_limit", "unexpected_stop", "cancelled"]),
  complete: z.boolean(), diagnostics: diagnosticsSchema,
  loop_guard: z.object({ version: z.literal(LOOP_POLICY), trigger: z.object({ block_words: z.number().int().min(6).max(80),
    repetitions: z.number().int().min(2).max(3), end_word: z.number().int().min(32) }).strict().nullable() }).strict().optional(),
}).strict();

export function verifyR3Result(value: unknown, identity: R3Identity, messages: R3Message[], streamed: string, loopGuard = false) {
  const parsed = resultSchema.safeParse(value);
  if (!parsed.success) return fail();
  const result = parsed.data;
  if (loopGuard !== (result.loop_guard !== undefined)) return fail();
  const paused = Boolean(result.loop_guard?.trigger);
  if (paused && (!equal(result.loop_guard!.trigger, repeatedPassage(result.output)) || result.finish_reason !== "cancelled" || result.complete)) return fail();
  if (!equal(result.identity, identity) || result.messages_sha256 !== r3Digest(messages) || !equal(result.decoding, R3_DECODING) ||
      result.output !== streamed || (result.complete && (result.finish_reason !== "eos" || !result.output.trim())) ||
      (result.finish_reason === "max_new_tokens" && result.output_ids.length !== 256)) return fail();
  const d = result.diagnostics;
  if (!equal(d.output_ids, result.output_ids) || d.prediction !== result.output || d.seconds !== result.seconds ||
      d.generated_tokens !== result.output_ids.length || d.empty !== !result.output.trim() ||
      result.output_ids.slice(0, -1).includes(R3_EOS) || (d.finish_reason === "eos") !== (result.output_ids.at(-1) === R3_EOS) ||
      d.raw_output !== result.output + (d.finish_reason === "eos" ? "<|im_end|>" : "") ||
      (d.complete && (d.empty || d.unexpected_control_ids.length > 0 || d.unexpected_control_text || d.unmapped_token_ids.length > 0 || d.invalid_unicode)) ||
      (result.finish_reason !== "cancelled" && d.finish_reason !== result.finish_reason) ||
      result.complete !== (d.complete === true && result.finish_reason !== "cancelled")) return fail();
  const research: ResearchR3Metadata = { variant: "r3_response", model: identity.model, revision: identity.revision,
    executionSha: identity.manifest_sha256, bundleSha: identity.manifest_sha256, adapterSha: identity.adapter_weights_sha256,
    finishReason: paused ? "loop_guard" : result.finish_reason, complete: result.complete };
  return { result, research, outputSha256: r3OutputHash(result.output), evidenceSha256: r3Digest(result) };
}

export function verifyR3StoredReply(metadata: unknown, content: string, identity: R3Identity, messages: R3Message[]) {
  if (!metadata || typeof metadata !== "object" || !content.trim()) return fail(409);
  const meta = metadata as Record<string, unknown>;
  const storedPolicy = !!(meta.evidence && typeof meta.evidence === "object" && "loop_guard" in meta.evidence);
  const verified = verifyR3Result(meta.evidence, identity, messages, content, storedPolicy);
  if (!equal(meta.research, verified.research) || meta.outputSha256 !== verified.outputSha256 || meta.evidenceSha256 !== verified.evidenceSha256 ||
      !(verified.research.complete || ["max_new_tokens", "time_limit", "loop_guard"].includes(verified.research.finishReason))) return fail(409);
}

export async function researchR3Reply(messages: R3Message[], config: R3Config,
  onDelta?: (chunk: string) => Promise<void> | void, signal?: AbortSignal, onAdmitted?: () => Promise<void>) {
  validateR3History(messages);
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) controller.abort();
  const timeout = setTimeout(abort, 910000);
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let streamed = "", requestId = "", index = 0;
  let final: ReturnType<typeof verifyR3Result> | undefined;
  try {
    const response = await fetch(endpoint(config, "/stream"), { method: "POST", signal: controller.signal, cache: "no-store", redirect: "error",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.token}` }, body: JSON.stringify({ messages, variant: "r3_response", ...(config.loopGuard ? { loop_guard: true } : {}) }) });
    if ([400, 413].includes(response.status)) throw new ResearchR3Error("R3 rejected this message or its full conversation context. No history was truncated. Shorten the message or start a new conversation.", 400);
    if (!response.ok || !response.body || !response.headers.get("content-type")?.startsWith("text/event-stream")) return fail(response.status === 429 ? 429 : 503);
    // A successful SSE response follows native tokenizer/context admission at the gateway.
    // Persist only now, so rejected prompts cannot poison the stored conversation.
    await onAdmitted?.();
    reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      if (buffer.length > 1048576) return fail();
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const data = block.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (!data) continue;
        if (final) return fail();
        const event = JSON.parse(data);
        if (event.type === "start") {
          if (requestId || event.protocol !== "r2-demo-stream-v1" || !/^[a-f0-9]{32}$/.test(event.request_id) ||
              typeof event.call_id !== "string" || !/^fc-[A-Za-z0-9]+$/.test(event.call_id) || !equal(event.identity, config.identity) || event.messages_sha256 !== r3Digest(messages)) return fail();
          requestId = event.request_id;
          if (event.loop_guard !== (config.loopGuard ? LOOP_POLICY : undefined)) return fail();
        } else {
          if (!requestId || event.request_id !== requestId) return fail();
          if (event.type === "delta") {
            if (event.index !== index++ || index > 1024 || typeof event.text !== "string" || !event.text || codepoints(streamed + event.text) > MAX_OUTPUT_CHARS) return fail();
            streamed += event.text;
            await onDelta?.(event.text);
          } else if (event.type === "final") final = verifyR3Result(event.result, config.identity, messages, streamed, config.loopGuard);
          else return fail();
        }
      }
      if (done) break;
    }
    if (!final || buffer.trim() || controller.signal.aborted) return fail();
    return final;
  } catch (error) {
    if (error instanceof ResearchR3Error) throw error;
    throw new ResearchR3Error("R3 could not finish. No other model was used.", controller.signal.aborted ? 504 : 503);
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
    controller.abort();
    await reader?.cancel().catch(() => {});
  }
}
