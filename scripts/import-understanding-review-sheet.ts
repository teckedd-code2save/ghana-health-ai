import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  buildUnderstandingTrainingExportFromReviews,
  readCorpusCandidates,
  readFileUnderstandingReviews,
  type UnderstandingReview,
} from "../src/lib/research-understanding-store";

const root = process.cwd();
const defaultIn = path.join(root, "tmp", "understanding-corpus", "review-sheet.v0.csv");
const defaultOut = path.join(root, "tmp", "understanding-review", "reviews.v0.jsonl");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function parseCsv(raw: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < raw.length; i += 1) {
    const char = raw[i];
    const next = raw[i + 1];
    if (quoted) {
      if (char === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((line) => line.some((value) => value.trim().length > 0));
}

function rowValue(row: Record<string, string>, key: string) {
  return row[key]?.trim() ?? "";
}

async function main() {
  const input = argValue("--input", defaultIn);
  const output = argValue("--out", defaultOut);
  const reviewer = argValue("--reviewer", "bulk_reviewer");
  const raw = await fs.readFile(input, "utf8");
  const [header, ...lines] = parseCsv(raw);
  if (!header) throw new Error("Review sheet is empty.");

  const imported = lines.map((line) =>
    Object.fromEntries(header.map((key, index) => [key, line[index] ?? ""])),
  );
  const existing = await readFileUnderstandingReviews();
  const existingById = new Map(existing.map((review) => [review.id, review]));
  const now = new Date().toISOString();
  let saved = 0;

  for (const row of imported) {
    const id = rowValue(row, "id");
    if (!id) continue;
    const decision = rowValue(row, "decision") || "unreviewed";
    if (!["unreviewed", "reviewed", "needs_second_review", "exclude"].includes(decision)) {
      throw new Error(`Invalid decision for ${id}: ${decision}`);
    }

    const review: UnderstandingReview = {
      id,
      normalizedTwi: rowValue(row, "review_normalized_twi"),
      naturalEnglish: rowValue(row, "review_natural_english"),
      literalEnglish: rowValue(row, "review_literal_english"),
      intent: rowValue(row, "review_intent"),
      entities: rowValue(row, "review_entities"),
      ambiguities: rowValue(row, "review_ambiguities"),
      decision: decision as UnderstandingReview["decision"],
      notes: rowValue(row, "review_notes"),
      reviewer: rowValue(row, "reviewer") || reviewer,
      createdAt: existingById.get(id)?.createdAt ?? now,
      updatedAt: now,
    };
    existingById.set(id, review);
    saved += 1;
  }

  const reviews = Array.from(existingById.values()).sort((a, b) => a.id.localeCompare(b.id));
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, `${reviews.map((review) => JSON.stringify(review)).join("\n")}\n`, "utf8");

  const candidates = await readCorpusCandidates();
  const exportPayload = buildUnderstandingTrainingExportFromReviews(candidates, reviews);
  console.log(
    JSON.stringify(
      {
        input,
        output,
        saved,
        reviews: reviews.length,
        accepted: exportPayload.accepted,
        splits: exportPayload.splits,
        readiness: exportPayload.readiness,
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
