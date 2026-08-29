import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  buildUnderstandingReviewSheetCsv,
  buildUnderstandingReviewSheetCsvFromReviews,
  readCorpusCandidates,
  readFileUnderstandingReviews,
  selectMinimumTrainingReviewCandidates,
} from "../src/lib/research-understanding-store";

const root = process.cwd();
const defaultOut = path.join(root, "tmp", "understanding-corpus", "review-sheet.v0.csv");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

async function main() {
  const out = argValue("--out", defaultOut);
  const reviewSource = argValue("--review-source", "file");
  const scope = argValue("--scope", "all");
  const csv = await (async () => {
    if (reviewSource !== "file") {
      return buildUnderstandingReviewSheetCsv(scope === "minimum-training" ? "minimum-training" : "all");
    }
    const [candidates, reviews] = await Promise.all([
      readCorpusCandidates(),
      readFileUnderstandingReviews(),
    ]);
    const scopedCandidates =
      scope === "minimum-training"
        ? selectMinimumTrainingReviewCandidates(candidates, reviews)
        : candidates;
    return buildUnderstandingReviewSheetCsvFromReviews(scopedCandidates, reviews);
  })();
  await fs.mkdir(path.dirname(out), { recursive: true });
  await fs.writeFile(out, csv, "utf8");
  console.log(JSON.stringify({ out, bytes: Buffer.byteLength(csv, "utf8") }, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
