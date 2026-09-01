import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";

type SourceRow = {
  id: string;
  source_dataset: string;
  source_config: string;
  source_url: string;
  source_row_index: number;
  license: string;
  english_user: string;
  faithful_english_meaning: string;
  english_answer: string;
  intent: string;
  entities: unknown;
};

type DraftRow = SourceRow & {
  twi_user: string;
  twi_answer: string;
  safety_level: string;
  translation_status: "draft";
  model: string;
};

type Translation = {
  id?: unknown;
  twi_user?: unknown;
  twi_answer?: unknown;
  intent?: unknown;
  entities?: unknown;
  safety_level?: unknown;
};

type Json = Record<string, unknown>;

const root = process.cwd();
const defaultInput = path.join(root, "data", "medical-response-corpus", "english-medical-qa.v0.jsonl");
const defaultOut = path.join(root, "data", "medical-response-corpus", "medical-qa-twi-drafts.v0.jsonl");

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

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
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

function parseJsonl<T>(raw: string) {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

async function writeJsonl(filePath: string, rows: DraftRow[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tmpPath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.rename(tmpPath, filePath);
}

async function translateChunk(rows: SourceRow[]) {
  const provider = llmProviderInfo();
  if (!provider) throw new Error("No LLM provider configured. Run through `sec --` or set OPENAI_API_KEY/GROQ_API_KEY.");
  const content = await chatComplete(
    [
      {
        role: "system",
        content:
          "You translate patient-facing English medical QA into natural Ghanaian Twi for dataset drafting. Return only valid JSON. Do not add new medical claims.",
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction:
            "Translate each English user question and answer into natural, clear Twi suitable for a Ghanaian voice health assistant. Keep emergency instructions direct. Preserve the exact medical meaning. Also refine intent/entities if obvious. These are drafts for human review, not gold labels.",
          rows: rows.map((row) => ({
            id: row.id,
            english_user: row.english_user,
            english_answer: row.english_answer,
            intent: row.intent,
            entities: row.entities,
          })),
          response_shape: {
            rows: [
              {
                id: "same id",
                twi_user: "Twi question",
                twi_answer: "Twi answer",
                intent: "snake_case",
                entities: {},
                safety_level: "routine|urgent|emergency|needs_review",
              },
            ],
          },
        }),
      },
    ],
    { temperature: 0, maxTokens: Math.max(1200, rows.length * 850) },
  );
  if (!content) throw new Error("LLM returned no translation content.");
  const parsed = parseJsonPayload(content);
  const translated = Array.isArray(parsed?.rows) ? (parsed.rows as Translation[]) : [];
  const byId = new Map(translated.map((row) => [asString(row.id), row]));
  const model = `${provider.provider}:${provider.model}`;

  return rows.flatMap((row): DraftRow[] => {
    const draft = byId.get(row.id);
    const twiUser = asString(draft?.twi_user);
    const twiAnswer = asString(draft?.twi_answer);
    if (!twiUser || !twiAnswer) return [];
    return [
      {
        ...row,
        twi_user: twiUser,
        twi_answer: twiAnswer,
        intent: asString(draft?.intent) || row.intent,
        entities: draft?.entities ?? row.entities,
        safety_level: asString(draft?.safety_level) || "needs_review",
        translation_status: "draft",
        model,
      },
    ];
  });
}

async function main() {
  const input = argValue("--input", defaultInput);
  const out = argValue("--out", defaultOut);
  const limit = numericArgValue("--limit", 24);
  const offset = numericArgValue("--offset", 0);
  const chunkSize = Math.max(1, numericArgValue("--chunk-size", 4));
  const force = process.argv.includes("--force");
  const existing = await fs
    .readFile(out, "utf8")
    .then((raw) => parseJsonl<DraftRow>(raw))
    .catch(() => []);
  const existingIds = new Set(existing.map((row) => row.id));
  const sourceRows = parseJsonl<SourceRow>(await fs.readFile(input, "utf8"))
    .slice(offset, offset + limit)
    .filter((row) => force || !existingIds.has(row.id));
  const translated: DraftRow[] = force ? [] : [...existing];
  let failedChunks = 0;

  for (let start = 0; start < sourceRows.length; start += chunkSize) {
    const chunk = sourceRows.slice(start, start + chunkSize);
    try {
      const chunkTranslations = await translateChunk(chunk);
      if (chunkTranslations.length === 0) throw new Error("Chunk produced no usable Twi translations.");
      translated.push(...chunkTranslations);
      await writeJsonl(out, translated);
      console.log(JSON.stringify({ chunk: `${start + 1}-${Math.min(start + chunkSize, sourceRows.length)}`, translated: translated.length, out }));
    } catch (error) {
      failedChunks += 1;
      console.error(JSON.stringify({ chunk: `${start + 1}-${Math.min(start + chunkSize, sourceRows.length)}`, error: error instanceof Error ? error.message : String(error) }));
    }
  }

  await writeJsonl(out, translated);
  console.log(JSON.stringify({ input, out, sourceRows: sourceRows.length, translated: translated.length, failedChunks }, null, 2));
  if (sourceRows.length > 0 && translated.length === 0) process.exit(1);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
