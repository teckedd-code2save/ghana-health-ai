import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";

type Json = Record<string, unknown>;

type ModelProposal = {
  normalized_twi: string;
  natural_english: string;
  literal_english: string;
  intent: string;
  entities: string;
  ambiguities: string;
  requires_clarification: boolean;
  model: string;
  status: "not_requested" | "draft";
};

type Candidate = {
  id: string;
  source: string;
  domain: string;
  text: string;
  normalized_text: string;
  model_proposal: ModelProposal;
};

type AnnotatedRow = {
  id: string;
  normalized_twi?: unknown;
  natural_english?: unknown;
  literal_english?: unknown;
  intent?: unknown;
  entities?: unknown;
  ambiguities?: unknown;
  requires_clarification?: unknown;
};

const root = process.cwd();
const defaultInput = path.join(root, "data", "understanding-corpus", "candidates.v0.jsonl");
const defaultOut = path.join(root, "tmp", "understanding-corpus", "candidates.annotated.v0.jsonl");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function numericArgValue(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function argSet(name: string) {
  return new Set(
    argValue(name)
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function compactJson(value: unknown) {
  if (typeof value === "string") return value.trim();
  if (value === undefined || value === null) return "";
  return JSON.stringify(value);
}

function parseJsonPayload(content: string): Json | null {
  const cleaned = content
    .trim()
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  try {
    return JSON.parse(cleaned) as Json;
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    try {
      return JSON.parse(cleaned.slice(start, end + 1)) as Json;
    } catch {
      return null;
    }
  }
}

function parseJsonl(raw: string): Candidate[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Candidate);
}

function serializeJsonl(rows: Candidate[]) {
  return `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`;
}

async function writeJsonl(filePath: string, rows: Candidate[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tmpPath, serializeJsonl(rows), "utf8");
  await fs.rename(tmpPath, filePath);
}

function selectTargets(rows: Candidate[]) {
  const sources = argSet("--source");
  const domains = argSet("--domain");
  const force = process.argv.includes("--force");
  const offset = numericArgValue("--offset", 0);
  const maxNew = numericArgValue("--max-new", 100);
  const selected: { row: Candidate; index: number }[] = [];

  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    if (sources.size > 0 && !sources.has(row.source)) continue;
    if (domains.size > 0 && !domains.has(row.domain)) continue;
    if (!force && row.model_proposal?.status === "draft") continue;
    selected.push({ row, index });
  }

  return selected.slice(offset, offset + maxNew);
}

async function annotateChunk(targets: { row: Candidate; index: number }[]) {
  const provider = llmProviderInfo();
  if (!provider) throw new Error("No LLM provider configured. Run through `sec --` or set OPENAI_API_KEY/GROQ_API_KEY.");

  const content = await chatComplete(
    [
      {
        role: "system",
        content:
          "You are preparing draft semantic labels for Ghanaian-language understanding research. Return only valid JSON. Do not answer the speaker. Preserve uncertainty and mark unclear text instead of inventing meaning.",
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction:
            "For each Twi/Akan/code-switched transcript, produce draft labels for human review. Use faithful natural English meaning, a short literal English gloss, a snake_case intent, compact entities, ambiguity notes, and requires_clarification. For general non-medical speech, use honest intents like general_statement, narrative_fragment, question, commerce_request, health_symptom_report, or unclear_fragment. Do not create diagnoses or emergency advice.",
          rows: targets.map(({ row }) => ({
            id: row.id,
            source: row.source,
            domain: row.domain,
            transcript: row.text,
            normalized_hint: row.normalized_text,
          })),
          response_shape: {
            rows: [
              {
                id: "same id",
                normalized_twi: "cleaned Twi/Akan transcript",
                natural_english: "faithful English meaning",
                literal_english: "closer literal gloss",
                intent: "snake_case",
                entities: { symptom: [], products: [], quantities: [], locations: [], time: [] },
                ambiguities: "short notes for reviewer",
                requires_clarification: false,
              },
            ],
          },
        }),
      },
    ],
    { temperature: 0, maxTokens: Math.max(900, targets.length * 420) },
  );
  if (!content) throw new Error("LLM returned no annotation content.");

  const parsed = parseJsonPayload(content);
  const rows = Array.isArray(parsed?.rows) ? (parsed.rows as AnnotatedRow[]) : [];
  const byId = new Map(rows.map((row) => [asString(row.id), row]));
  const model = `${provider.provider}:${provider.model}`;

  return targets.map(({ row, index }) => {
    const draft = byId.get(row.id);
    if (!draft) return { index, row, updated: false };
    return {
      index,
      row: {
        ...row,
        model_proposal: {
          normalized_twi: asString(draft.normalized_twi) || row.model_proposal.normalized_twi,
          natural_english: asString(draft.natural_english) || row.model_proposal.natural_english,
          literal_english: asString(draft.literal_english) || row.model_proposal.literal_english,
          intent: asString(draft.intent) || row.model_proposal.intent,
          entities: compactJson(draft.entities) || row.model_proposal.entities,
          ambiguities: compactJson(draft.ambiguities) || row.model_proposal.ambiguities,
          requires_clarification: draft.requires_clarification === true,
          model,
          status: "draft" as const,
        },
      },
      updated: true,
    };
  });
}

async function main() {
  const input = argValue("--input", defaultInput);
  const out = process.argv.includes("--in-place") ? input : argValue("--out", defaultOut);
  const chunkSize = Math.max(1, numericArgValue("--chunk-size", 8));
  const rows = parseJsonl(await fs.readFile(input, "utf8"));
  const targets = selectTargets(rows);
  let newlyAnnotated = 0;
  let failedChunks = 0;

  for (let start = 0; start < targets.length; start += chunkSize) {
    const chunk = targets.slice(start, start + chunkSize);
    try {
      const updates = await annotateChunk(chunk);
      for (const update of updates) {
        if (!update.updated) continue;
        rows[update.index] = update.row;
        newlyAnnotated += 1;
      }
      await writeJsonl(out, rows);
      console.log(
        JSON.stringify({
          chunk: `${start + 1}-${Math.min(start + chunkSize, targets.length)}`,
          newlyAnnotated,
          out,
        }),
      );
    } catch (error) {
      failedChunks += 1;
      console.error(
        JSON.stringify({
          chunk: `${start + 1}-${Math.min(start + chunkSize, targets.length)}`,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  }

  const annotatedAfter = rows.filter((row) => row.model_proposal?.status === "draft").length;
  const summary = {
    input,
    out,
    selected: targets.length,
    newlyAnnotated,
    failedChunks,
    annotatedAfter,
    bySource: rows.reduce<Record<string, number>>((acc, row) => {
      if (row.model_proposal?.status === "draft") acc[row.source] = (acc[row.source] ?? 0) + 1;
      return acc;
    }, {}),
  };
  console.log(JSON.stringify(summary, null, 2));
  if (targets.length > 0 && newlyAnnotated === 0) process.exit(1);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
