import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { appendFile, mkdir, readFile, rmdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { z } from "zod";

export const runtime = "nodejs";
const execute = promisify(execFile);
const releasePattern = /^[a-zA-Z0-9_-]{1,100}$/;
const query = z.object({ release: z.string().regex(releasePattern), collection: z.enum(["language", "meaning", "conversation", "health", "teacher"]), offset: z.coerce.number().int().min(0).max(2000000) });
const decisionSchema = query.extend({ id: z.string().min(1).max(500), source_hash: z.string().regex(/^[a-f0-9]{64}$/),
  reference_analysis_id: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  decision: z.enum(["reviewed", "needs_second_review", "exclude"]), selected: z.number().int().min(-1).max(1),
  correction: z.string().max(12000), notes: z.string().max(4000), request_id: z.string().uuid(),
  correction_field: z.enum(["meaning_english", "suggested_normalization", "reference_answer", "conversation", "structured"]),
  issue: z.enum(["none", "meaning", "negation", "experiencer", "quantity_time", "naturalness", "unsupported_claim", "other"]) });

function localOnly(request: Request) {
  return (process.env.NODE_ENV !== "production" || process.env.CORPUS_STANDALONE_REVIEW === "1") &&
    ["localhost", "127.0.0.1", "[::1]"].includes(new URL(request.url).hostname);
}

async function rowFor(input: z.infer<typeof query>) {
  if (input.collection === "teacher") {
    const candidateFile = path.join(process.cwd(), "tmp/corpus-reference-review", input.release, "qualification-review.json");
    const candidates = await readFile(candidateFile, "utf8").catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return null; throw error; });
    if (candidates) {
      const rows = JSON.parse(candidates) as Array<Record<string, unknown>>;
      const row = rows[input.offset];
      return { ready: true, total: rows.length, rows: row ? [{ ...row, original: row.original_tw, reference_english: row.reference_en }] : [] };
    }
    const file = path.join(process.cwd(), "tmp/corpus-releases", input.release, "teacher/review.json");
    const contents = await readFile(file, "utf8").catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return null; throw error; });
    if (!contents) return { ready: false, total: 0, rows: [], error: "Teacher qualification has no translation results. The source datasets remain available." };
    const rows = JSON.parse(contents) as Array<Record<string, unknown>>;
    const original = rows[input.offset];
    return { ready: true, total: rows.length, rows: original ? [{ ...original,
      original: original.original_tw ?? original.original_en, source: "Qwen teacher qualification", language: original.original_tw ? "tw" : "en",
      source_hash: createHash("sha256").update(JSON.stringify(original)).digest("hex"), reference_english: original.reference_en ?? null }] : [] };
  }
  const result = await execute(process.env.CORPUS_PYTHON ?? "python3", ["scripts/corpus_review_row.py", "--release", input.release,
    "--collection", input.collection, "--offset", String(input.offset)], { cwd: process.cwd(), timeout: 15000, maxBuffer: 1024 * 1024 });
  return JSON.parse(result.stdout) as { ready: boolean; total: number; rows: Array<Record<string, unknown>> };
}

export async function GET(request: Request) {
  if (!localOnly(request)) return new Response(null, { status: 404 });
  const parsed = query.safeParse(Object.fromEntries(new URL(request.url).searchParams));
  if (!parsed.success) return Response.json({ error: "Invalid dataset selection." }, { status: 400 });
  try {
    const data = await rowFor(parsed.data);
    const decisionsFile = path.join(process.cwd(), "tmp/corpus-review", parsed.data.release, "decisions.jsonl");
    const text = await readFile(decisionsFile, "utf8").catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return ""; throw error; });
    const decisions = text.split("\n").filter(Boolean).map((line) => JSON.parse(line));
    return Response.json({ ...data, rows: data.rows.map((row) => ({ ...row, review: decisions.findLast((r) => r.id === row.id && r.source_hash === row.source_hash) ?? null })) });
  } catch {
    return Response.json({ error: "This release is still being prepared. No source data was changed." }, { status: 409 });
  }
}

export async function POST(request: Request) {
  if (!localOnly(request)) return new Response(null, { status: 404 });
  if (request.headers.get("origin") !== new URL(request.url).origin) return new Response(null, { status: 403 });
  const parsed = decisionSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Check the review fields." }, { status: 400 });
  const input = parsed.data;
  const data = await rowFor(input).catch(() => null);
  if (!data) return Response.json({ error: "The source is unavailable. Your correction has not been saved." }, { status: 409 });
  const row = data.rows[0];
  if (!row || row.id !== input.id || row.source_hash !== input.source_hash) return Response.json({ error: "This source changed. Reload it before saving." }, { status: 409 });
  if (input.reference_analysis_id && input.reference_analysis_id !== (row.reference_analysis as {id?: string} | undefined)?.id)
    return Response.json({ error: "This annotation changed. Reload it before saving." }, { status: 409 });
  if (input.selected >= 0 && row.reference_analysis && !input.reference_analysis_id)
    return Response.json({ error: "Reload this annotation before selecting a proposal." }, { status: 409 });
  if (input.selected >= ((row.alternatives as unknown[] | undefined)?.length ?? 0)) return Response.json({ error: "That proposal does not exist." }, { status: 400 });
  const folder = path.join(process.cwd(), "tmp/corpus-review", input.release);
  await mkdir(folder, { recursive: true });
  const file = path.join(folder, "decisions.jsonl");
  const lock = path.join(folder, ".review-lock");
  try { await mkdir(lock); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") return Response.json({ error: "Another save is finishing. Please retry." }, { status: 409 });
    throw error;
  }
  try {
    const previous = await readFile(file, "utf8").catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return ""; throw error; });
    const existing = previous.split("\n").filter(Boolean).map((line) => JSON.parse(line)).find((r) => r.request_id === input.request_id);
    if (existing) {
      if (Object.keys(input).some((key) => existing[key] !== input[key as keyof typeof input])) return Response.json({ error: "This save ID belongs to a different correction." }, { status: 409 });
      return Response.json({ saved: true, storage: "local" });
    }
    const record = { ...input, reviewer: "local_reviewer", created_at: new Date().toISOString() };
    await appendFile(file, JSON.stringify(record) + "\n", "utf8");
  } finally { await rmdir(lock); }
  let storage = "local";
  if (process.env.DATABASE_URL) {
    try {
      const { prisma } = await import("@/db/prisma");
      const rowId = "corpus-release:" + createHash("sha256").update(input.release + input.id + input.source_hash).digest("hex");
      const fields = { rowKind: "corpus_release", decision: input.decision, normalizedTwi: "", naturalEnglish: "",
        notes: JSON.stringify({ release: input.release, source_id: input.id, source_hash: input.source_hash, reference_analysis_id: input.reference_analysis_id,
          correction: input.correction, correction_field: input.correction_field, issue: input.issue, notes: input.notes }),
        selectedProposalId: String(input.selected), synthesisVersion: input.source_hash, reviewer: "local_reviewer" };
      await prisma.researchUnderstandingReview.upsert({ where: { rowId }, create: { rowId, ...fields }, update: fields });
      storage = "local_and_postgres";
    } catch { /* The durable local correction remains available for later sync. */ }
  }
  return Response.json({ saved: true, storage });
}
