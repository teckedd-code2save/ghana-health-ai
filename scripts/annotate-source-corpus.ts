import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import "../src/config/load-env";
import { createMedicalAnnotationClient, MedicalAnnotationProviderError } from "../src/lib/medical-annotation-client";
import { annotationWorkPlan } from "../src/lib/annotation-work-plan";
import {
  corpusDigest, sourceAnnotationRubric, sourceAnnotationSchema, sourceCorpusHash, sourceCorpusIntents,
  sourceCorpusPromptVersion, sourceCorpusSchema, sourceProposalProblems, sourceProposalSchema,
  type SourceAnnotation, type SourceCorpusRow, type SourceProposal,
} from "../src/lib/source-corpus-annotations";

function argument(name: string, fallback: string) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1] ?? fallback;
}
function integer(name: string, fallback: number, minimum = 0) {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isInteger(value) || value < minimum) throw new Error(`Invalid ${name}`);
  return value;
}
async function jsonl<T>(file: string, parse: (value: unknown) => T, optional = false): Promise<T[]> {
  try { return (await fs.readFile(file, "utf8")).split("\n").filter((line) => line.trim()).map((line) => parse(JSON.parse(line))); }
  catch (error) { if (optional && (error as NodeJS.ErrnoException).code === "ENOENT") return []; throw error; }
}

const shape = {
  normalized_text: "same-language spelling normalization only",
  natural_english: "faithful English meaning of utterance alone, with uncertainty preserved",
  intent: "one allowed intent",
  entities: { symptoms: [], conditions: [], medicines: [], body_parts: [], durations: [], quantities: [], persons: [], locations: [], other: [] },
  uncertainty_notes: "missing context or genuine ambiguity; empty when none",
  requires_clarification: false,
  safety_level: "routine|same_day|urgent|emergency|needs_review",
  source_answer_assessment: "not_applicable|supportable|needs_minor_edit|needs_expert_review|unsafe_or_incorrect",
  source_answer_issues: [], reply: "same-language concise reply if reference_answer exists, otherwise null",
  evidence_spans: ["exact source text"], confidence: 0.0,
};

async function main() {
  if (argument("--provider", "modal") !== "mock") {
    const release = argument("--release", "");
    if (!release) throw new Error("Modal-only annotation: supply --release <versioned corpus directory>. No default 12-row or paid-API run.");
    const action = process.argv.includes("--execute") ? "run" : "plan";
    const args = ["scripts/corpus_annotation.py", action, "--release", release];
    if (action === "run") args.push("--gate", argument("--gate", ""), "--max-batches", argument("--max-batches", "0"));
    const result = spawnSync("python3", args, { stdio: "inherit" });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error("Modal corpus annotation did not complete; retained checkpoints were not replaced.");
    return;
  }
  const mockUrl = new URL(process.env.OPENAI_BASE_URL ?? "https://disabled.invalid");
  if (mockUrl.protocol !== "http:" || !["localhost", "127.0.0.1", "[::1]"].includes(mockUrl.hostname)) {
    throw new Error("The legacy annotation client is available only for loopback mock tests.");
  }
  const input = argument("--input", "data/annotated-source-corpus/sources.v1.jsonl");
  const output = argument("--out", "data/annotated-source-corpus/annotations.v1.jsonl");
  const summaryFile = `${output}.summary.json`;
  const limit = integer("--limit", 0);
  const offset = integer("--offset", 0);
  const chunkSize = integer("--chunk-size", 4, 1);
  const concurrency = integer("--concurrency", 8, 1);
  const models = {
    teacher_a: "mock:teacher-a",
    teacher_b: "mock:teacher-b",
    judge: "mock:judge",
  };
  const pipelineId = corpusDigest(JSON.stringify({ prompt: sourceCorpusPromptVersion, models }));
  const sourceFilter = new Set(argument("--source", "").split(",").filter(Boolean));
  const sources = await jsonl(input, (row) => sourceCorpusSchema.parse(row));
  if (new Set(sources.map((row) => row.id)).size !== sources.length) throw new Error("Duplicate source IDs");
  for (const row of sources) if (sourceCorpusHash(row) !== row.source_hash) throw new Error(`Changed source ${row.id}`);
  const pool = sources.filter((row) => (!sourceFilter.size || sourceFilter.has(row.source)) &&
    (!process.argv.includes("--train-only") || row.split === "train"))
    .sort((a, b) => a.source_hash.localeCompare(b.source_hash));
  if (!pool.length) throw new Error("Empty source selection");
  const selection = limit ? pool.slice(offset, offset + limit) : pool.slice(offset);
  const previous = await jsonl(output, (row) => sourceAnnotationSchema.parse(row), true);
  const sourceById = new Map(sources.map((row) => [row.id, row]));
  for (const row of previous) {
    if (row.pipeline_id !== pipelineId || sourceById.get(row.row_id)?.source_hash !== row.source_hash) throw new Error("Stale output: choose a new output file");
  }
  if (new Set(previous.map((row) => row.row_id)).size !== previous.length) throw new Error("Duplicate output IDs");
  const records = new Map(previous.map((row) => [row.row_id, row]));
  const pending = selection.filter((row) => !records.has(row.id));
  await fs.mkdir(path.dirname(output), { recursive: true });
  const planned = await annotationWorkPlan({
    file: `${output}.plan.json`, cacheDirectory: `${output}.requests`,
    signature: corpusDigest(JSON.stringify({ pipelineId, sources: selection.map((row) => [row.id, row.source_hash]) })),
    selectedIds: selection.map((row) => row.id), pendingIds: pending.map((row) => row.id), chunkSize,
  });
  if (process.argv.includes("--dry-run")) {
    console.log(JSON.stringify({ source_pool: pool.length, selected: selection.length, pending: pending.length, paid_requests_made: 0 }));
    return;
  }
  const complete = createMedicalAnnotationClient(`${output}.requests`);
  const sourcePayload = (rows: SourceCorpusRow[]) => rows.map(({ id, language, task, utterance, reference_answer }) => ({ row_id: id, language, task, utterance, reference_answer }));

  async function teacher(rows: SourceCorpusRow[], role: "teacher_a" | "teacher_b") {
    const payload = await complete([
      { role: "system", content: `${sourceAnnotationRubric}\nYou are the independent ${role === "teacher_a" ? "meaning-first annotator" : "semantic and safety critic"}. Return JSON only.` },
      { role: "user", content: JSON.stringify({ prompt_version: sourceCorpusPromptVersion, allowed_intents: sourceCorpusIntents, rows: sourcePayload(rows), response_shape: { rows: [{ row_id: "unchanged row id", ...shape }] } }) },
    ], models[role], Math.max(4000, rows.length * 1800));
    const values = payload.rows as Array<Record<string, unknown>>;
    const byId = new Map(values.map((row) => [row.row_id, row]));
    if (byId.size !== rows.length || values.length !== rows.length) throw new Error(`${role} returned mismatched row IDs`);
    return rows.map((row) => sourceProposalSchema.parse(byId.get(row.id)));
  }

  async function annotate(rows: SourceCorpusRow[]) {
    const a = await teacher(rows, "teacher_a");
    if (fatal) throw fatal;
    const b = await teacher(rows, "teacher_b");
    if (fatal) throw fatal;
    const payload = await complete([
      { role: "system", content: `${sourceAnnotationRubric}\nYou are a separate adjudicator. Compare both alternatives with original evidence. Select a fully faithful proposal or synthesize a correction using only that evidence. If meaning, medical correctness or evidence remains doubtful, choose needs_review. Do not equate model agreement with truth. Return JSON only.` },
      { role: "user", content: JSON.stringify({
        prompt_version: sourceCorpusPromptVersion, allowed_intents: sourceCorpusIntents,
        rows: sourcePayload(rows).map((source, i) => ({ ...source, teacher_a: a[i], teacher_b: b[i] })),
        response_shape: { rows: [{ row_id: "unchanged row id", choice: "teacher_a|teacher_b|synthesized|needs_review", confidence: 0.0, rationale: "evidence-based reason", review_reasons: [], synthesized: "null unless synthesizing; then a complete annotation with all fields from the proposals" }] },
      }) },
    ], models.judge, Math.max(4000, rows.length * 2000));
    const values = payload.rows as Array<Record<string, unknown>>;
    const byId = new Map(values.map((row) => [row.row_id, row]));
    if (byId.size !== rows.length || values.length !== rows.length) throw new Error("Judge returned mismatched row IDs");
    return rows.map((source, i): SourceAnnotation => {
      const decision = byId.get(source.id);
      if (!decision || !["teacher_a", "teacher_b", "synthesized", "needs_review"].includes(String(decision.choice))) throw new Error("Invalid judge decision");
      const selected: SourceProposal = decision.choice === "teacher_b" ? b[i]
        : decision.choice === "synthesized" ? sourceProposalSchema.parse(decision.synthesized) : a[i];
      const reasons = new Set(sourceProposalProblems(source, selected));
      if (decision.choice === "needs_review") reasons.add("adjudicator_requested_review");
      if (typeof decision.confidence !== "number" || decision.confidence < 0.95) reasons.add("adjudicator_uncertainty");
      if (!Array.isArray(decision.review_reasons) || decision.review_reasons.some((reason) => typeof reason !== "string")) throw new Error("Missing judge review reasons");
      for (const reason of decision.review_reasons as string[]) if (reason.trim()) reasons.add(reason);
      if (a[i].safety_level !== b[i].safety_level || [a[i].source_answer_assessment, b[i].source_answer_assessment].includes("unsafe_or_incorrect")) reasons.add("safety_disagreement");
      return sourceAnnotationSchema.parse({
        row_id: source.id, source_hash: source.source_hash, prompt_version: sourceCorpusPromptVersion, pipeline_id: pipelineId,
        generated_at: new Date().toISOString(),
        proposals: [{ role: "teacher_a", model: models.teacher_a, fields: a[i] }, { role: "teacher_b", model: models.teacher_b, fields: b[i] }],
        selected, adjudication: { model: models.judge, choice: decision.choice, confidence: decision.confidence, rationale: decision.rationale },
        status: reasons.size ? "needs_review" : "model_consensus", review_reasons: [...reasons].sort(),
        eligible_for_production_training: false, eligible_for_final_evaluation: false,
      });
    });
  }

  await fs.mkdir(path.dirname(output), { recursive: true });
  const failures: Array<{ row_ids: string[]; error: string }> = [];
  let fatal: Error | null = null;
  let completed = 0;
  let saveQueue = Promise.resolve();
  async function save(final: boolean) {
    const all = [...records.values()].sort((a, b) => a.row_id.localeCompare(b.row_id));
    const temporary = `${output}.${process.pid}.tmp`;
    await fs.writeFile(temporary, all.map((row) => JSON.stringify(row)).join("\n") + (all.length ? "\n" : ""));
    await fs.rename(temporary, output);
    await fs.writeFile(summaryFile, JSON.stringify({
      prompt_version: sourceCorpusPromptVersion, pipeline_id: pipelineId, models, source_pool: pool.length,
      selected: selection.length, annotations_saved: records.size, completed_this_run: completed,
      remaining: selection.filter((row) => !records.has(row.id)).length,
      consensus: all.filter((row) => row.status === "model_consensus").length,
      review: all.filter((row) => row.status === "needs_review").length,
      failures, interrupted_by_provider: Boolean(fatal), finished: final,
      ready_for_training: false, note: "Populated annotations, not an automatically accepted training export.",
    }, null, 2) + "\n");
  }
  async function processChunk(rows: SourceCorpusRow[]): Promise<void> {
    if (fatal || rows.every((row) => records.has(row.id))) return;
    try {
      const annotations = await annotate(rows);
      for (const row of annotations) if (!records.has(row.row_id)) {
        records.set(row.row_id, row);
        completed += 1;
      }
      saveQueue = saveQueue.then(() => save(false));
      await saveQueue;
      console.log(JSON.stringify({ completed, requested: pending.length, saved: records.size }));
    } catch (error) {
      if (error instanceof MedicalAnnotationProviderError) fatal = error;
      else if (rows.length > 1) {
        const middle = Math.ceil(rows.length / 2);
        await processChunk(rows.slice(0, middle));
        await processChunk(rows.slice(middle));
        return;
      }
      failures.push({ row_ids: rows.map((row) => row.id), error: error instanceof Error ? error.message : String(error) });
      console.error(JSON.stringify(failures.at(-1)));
    }
  }
  const pendingIds = new Set(pending.map((row) => row.id));
  const chunks = planned.filter((ids) => ids.some((id) => pendingIds.has(id))).map((ids) => ids.map((id) => sourceById.get(id)!));
  let next = 0;
  if (chunks.length) await processChunk(chunks[next++]);
  async function worker() { while (!fatal && next < chunks.length) await processChunk(chunks[next++]); }
  await Promise.all(Array.from({ length: Math.min(concurrency, Math.max(0, chunks.length - next)) }, worker));
  await saveQueue;
  await save(true);
  console.log(await fs.readFile(summaryFile, "utf8"));
  if (failures.length || fatal) process.exitCode = 1;
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
