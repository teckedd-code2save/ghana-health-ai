import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  buildUnderstandingTrainingExport,
  buildUnderstandingTrainingExportFromReviews,
  readCorpusCandidates,
} from "../src/lib/research-understanding-store";

type Review = {
  id: string;
  normalizedTwi: string;
  naturalEnglish: string;
  literalEnglish: string;
  intent: string;
  entities: string;
  ambiguities: string;
  decision: "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
  notes: string;
  reviewer: string;
  updatedAt?: string;
};

const root = process.cwd();
const defaultReviews = path.join(root, "tmp", "understanding-review", "reviews.v0.jsonl");
const defaultOutDir = path.join(root, "tmp", "understanding-corpus", "exports", "v0");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function parseJsonl<T>(raw: string): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

async function main() {
  const reviewPath = argValue("--reviews", defaultReviews);
  const outDir = argValue("--out-dir", defaultOutDir);
  const reviewSource = argValue("--review-source", "auto");
  const strict = process.argv.includes("--strict");

  let exportPayload;
  if (reviewSource === "file") {
    const [candidates, reviews] = await Promise.all([
      readCorpusCandidates(),
      fs
        .readFile(reviewPath, "utf8")
        .then((raw) => parseJsonl<Review>(raw))
        .catch((error) => {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
          throw error;
        }),
    ]);
    exportPayload = buildUnderstandingTrainingExportFromReviews(candidates, reviews);
  } else {
    exportPayload = await buildUnderstandingTrainingExport();
  }
  const bySplit = {
    train: exportPayload.rows.filter((row) => row.split === "train"),
    dev: exportPayload.rows.filter((row) => row.split === "dev"),
    test: exportPayload.rows.filter((row) => row.split === "test"),
  };
  const summary = {
    schema_version: exportPayload.schema_version,
    created_at: exportPayload.created_at,
    candidates: exportPayload.candidates,
    reviews: exportPayload.reviews,
    accepted: exportPayload.accepted,
    rejected: exportPayload.rejected,
    splits: exportPayload.splits,
    rejected_reasons: exportPayload.rejected_reasons,
  };

  await fs.mkdir(outDir, { recursive: true });
  await Promise.all(
    Object.entries(bySplit).map(([split, rows]) =>
      fs.writeFile(
        path.join(outDir, `${split}.jsonl`),
        rows.length ? `${rows.map((row) => JSON.stringify(row)).join("\n")}\n` : "",
        "utf8",
      ),
    ),
  );
  await fs.writeFile(path.join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify({ outDir, ...summary }, null, 2));

  if (strict && exportPayload.accepted === 0) {
    throw new Error("No reviewed rows are eligible for training export yet.");
  }
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
