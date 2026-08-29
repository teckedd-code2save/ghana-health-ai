import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  buildUnderstandingTrainingExportFromReviews,
  parseUnderstandingReviewSheetCsv,
  readCorpusCandidates,
  readFileUnderstandingReviews,
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

async function main() {
  const input = argValue("--input", defaultIn);
  const output = argValue("--out", defaultOut);
  const reviewer = argValue("--reviewer", "bulk_reviewer");
  const raw = await fs.readFile(input, "utf8");
  const imported = parseUnderstandingReviewSheetCsv(raw, reviewer);
  const existing = await readFileUnderstandingReviews();
  const existingById = new Map(existing.map((review) => [review.id, review]));

  for (const review of imported.reviews) {
    existingById.set(review.id, {
      ...review,
      createdAt: existingById.get(review.id)?.createdAt ?? review.createdAt,
    });
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
        saved: imported.reviews.length,
        skipped: imported.skipped,
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
