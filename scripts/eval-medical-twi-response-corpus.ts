import fs from "node:fs/promises";
import path from "node:path";

type Row = {
  id?: string;
  english_user?: string;
  faithful_english_meaning?: string;
  intent?: string;
  answer?: string;
  safety_level?: string;
  source_urls?: string[];
  twi_user?: string;
  twi_answer?: string;
  translation_status?: string;
};

const root = process.cwd();
const defaultPath = path.join(root, "data", "medical-response-corpus", "twi-drafts.v0.jsonl");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function parseJsonl(raw: string): Row[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Row);
}

function hasText(value: unknown) {
  return typeof value === "string" && value.trim().length > 0;
}

async function main() {
  const input = argValue("--input", defaultPath);
  const rows = parseJsonl(await fs.readFile(input, "utf8"));
  const errors: string[] = [];
  const ids = new Set<string>();

  for (const [index, row] of rows.entries()) {
    const label = row.id || `row_${index + 1}`;
    if (!hasText(row.id)) errors.push(`${label}: missing id`);
    if (row.id && ids.has(row.id)) errors.push(`${label}: duplicate id`);
    if (row.id) ids.add(row.id);
    if (!hasText(row.english_user)) errors.push(`${label}: missing english_user`);
    if (!hasText(row.faithful_english_meaning)) errors.push(`${label}: missing faithful_english_meaning`);
    if (!hasText(row.intent)) errors.push(`${label}: missing intent`);
    if (!hasText(row.answer)) errors.push(`${label}: missing answer`);
    if (!hasText(row.twi_user)) errors.push(`${label}: missing twi_user`);
    if (!hasText(row.twi_answer)) errors.push(`${label}: missing twi_answer`);
    if (!["routine", "same_day", "urgent", "emergency"].includes(row.safety_level ?? "")) {
      errors.push(`${label}: invalid safety_level`);
    }
    if (!Array.isArray(row.source_urls) || row.source_urls.length === 0) {
      errors.push(`${label}: missing source_urls`);
    }
    if (row.translation_status !== "draft") errors.push(`${label}: translation_status is not draft`);
  }

  const summary = {
    input,
    rows: rows.length,
    errors,
    ready_for_review: errors.length === 0,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (errors.length > 0) throw new Error("Medical Twi response corpus failed validation.");
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
