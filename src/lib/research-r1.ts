import { createHash } from "node:crypto";
import { z } from "zod";
import { ResearchR1Error, type ResearchR1Mode, type ResearchR1Metadata } from "@/lib/model-modes";

export const R1_EXECUTION = "a9a09a5ba056983f08977308beb6ae1e78667258deda7287f3ce45a7e8d2ff70";
export const R1_MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1";
export const R1_REVISION = "693f12e444a30971ca53889128d7bade9491ecb3";
export const R1_VARIANTS = { r1_base: "base", r1_meaning: "meaning-adapter", r1_response: "response-adapter" } as const;
const WEIGHTS = {
  base: null,
  "meaning-adapter": "73ee9854c4975ce2d79ae7dbabb2fd5d021e9579a7f81ac26ec14b4a5a7fbb2b",
  "response-adapter": "0be5dcd4dbda0c37460bfe4e8ac19833326edb04a32fb22252b089dd868a9bc1",
};
export type R1Message = { role: "user" | "assistant"; content: string };
const messageSchema = z.object({ role: z.enum(["user", "assistant"]), content: z.string().min(1).max(16000) }).strict();
const resultSchema = z.object({
  variant: z.enum(["base", "meaning-adapter", "response-adapter"]),
  output: z.string().min(1).max(32000), generated: z.literal(true), load_verified: z.literal(true),
  complete: z.boolean(), finish_reason: z.enum(["eos", "max_new_tokens", "time_limit", "repetition_guard"]),
  identity: z.object({ model: z.literal(R1_MODEL), revision: z.literal(R1_REVISION),
    execution_manifest_sha256: z.literal(R1_EXECUTION), variant: z.string(),
    adapter_weights_sha256: z.string().nullable(),
    base_weights_sha256: z.literal("6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6") }),
  prompt_tokens: z.number().int().positive().max(3840), context_tokens: z.literal(4096),
  request_seconds: z.number().nonnegative(),
}).passthrough();

export function validateResearchHistory(messages: R1Message[]) {
  const parsed = z.array(messageSchema).min(1).max(79).safeParse(messages);
  if (!parsed.success || messages.length % 2 !== 1 || messages.some((m, i) => m.role !== (i % 2 ? "assistant" : "user")) ||
      messages.reduce((n, m) => n + m.content.length, 0) > 60000) {
    throw new ResearchR1Error("This research conversation cannot be continued. Start a new conversation; no history will be silently removed.", 409);
  }
}

export function isCompletedResearchReply(value: unknown, mode: ResearchR1Mode) {
  if (!value || typeof value !== "object") return false;
  const research = value as Partial<ResearchR1Metadata>;
  const variant = R1_VARIANTS[mode];
  return research.variant === variant && research.model === R1_MODEL && research.revision === R1_REVISION &&
    research.executionSha === R1_EXECUTION && research.adapterSha === WEIGHTS[variant] &&
    research.complete === true && research.finishReason === "eos";
}

export function isContinuableResearchReply(value: unknown, mode: ResearchR1Mode) {
  if (isCompletedResearchReply(value, mode)) return true;
  if (!value || typeof value !== "object") return false;
  const research = value as Partial<ResearchR1Metadata>;
  if (research.complete !== false || !["max_new_tokens", "time_limit"].includes(research.finishReason ?? "")) return false;
  return isCompletedResearchReply({ ...research, complete: true, finishReason: "eos" }, mode);
}

export function verifyResearchResult(value: unknown, mode: ResearchR1Mode) {
  const parsed = resultSchema.safeParse(value);
  if (!parsed.success) throw new ResearchR1Error("The research model returned unverifiable output. No other model was used.");
  const result = parsed.data;
  const variant = R1_VARIANTS[mode];
  if (result.variant !== variant || result.identity.variant !== variant || result.identity.adapter_weights_sha256 !== WEIGHTS[variant] ||
      result.complete !== (result.finish_reason === "eos")) {
    throw new ResearchR1Error("The research checkpoint or completion did not match. No other model was used.");
  }
  const research: ResearchR1Metadata = { variant, model: R1_MODEL, revision: R1_REVISION,
    executionSha: R1_EXECUTION, adapterSha: WEIGHTS[variant], finishReason: result.finish_reason, complete: result.complete };
  return { result, research };
}

export function researchR1Configured() {
  return Boolean(process.env.RESEARCH_R1_URL && (process.env.RESEARCH_R1_TOKEN?.length ?? 0) >= 32);
}

export async function researchR1Reply(messages: R1Message[], mode: ResearchR1Mode, onDelta?: (chunk: string) => Promise<void> | void) {
  validateResearchHistory(messages);
  if (!researchR1Configured()) throw new ResearchR1Error("Research R1 is not available yet. No other model was used.");
  const url = new URL(process.env.RESEARCH_R1_URL!);
  if (url.protocol !== "https:" || !url.hostname.endsWith(".modal.run")) throw new ResearchR1Error("Research endpoint configuration is invalid.");
  url.pathname = url.pathname.replace(/\/$/, "") + "/stream";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 250000);
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let streamed = "";
  let final: ReturnType<typeof verifyResearchResult> | undefined;
  try {
    const response = await fetch(url, { method: "POST", signal: controller.signal, cache: "no-store",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RESEARCH_R1_TOKEN}` },
      body: JSON.stringify({ messages, variant: R1_VARIANTS[mode] }) });
    if (!response.ok || !response.body) throw new ResearchR1Error("The research service is unavailable. No other model was used.");
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      if (buffer.length > 256000) throw new ResearchR1Error("Invalid research stream.");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
        if (!data) continue;
        const event = JSON.parse(data);
        if (event.type === "error") throw new ResearchR1Error("The research model could not finish. Start a new conversation; no other model was used.");
        if (event.type === "delta") {
          if (final || typeof event.chunk !== "string" || streamed.length + event.chunk.length > 32000) throw new ResearchR1Error("Invalid research stream.");
          streamed += event.chunk;
          await onDelta?.(event.chunk);
        } else if (event.type === "final") {
          if (final) throw new ResearchR1Error("Duplicate research result.");
          final = verifyResearchResult(event.result, mode);
        }
      }
      if (done) break;
    }
    if (!final || streamed.trim() !== final.result.output) throw new ResearchR1Error("The research stream was incomplete or inconsistent. No other model was used.");
    return { ...final, outputSha256: createHash("sha256").update(final.result.output).digest("hex") };
  } catch (error) {
    if (error instanceof ResearchR1Error) throw error;
    throw new ResearchR1Error(controller.signal.aborted ? "Research R1 timed out. No other model was used." : "Research R1 failed. No other model was used.", controller.signal.aborted ? 504 : 503);
  } finally {
    clearTimeout(timer);
    controller.abort();
    await reader?.cancel().catch(() => {});
  }
}
