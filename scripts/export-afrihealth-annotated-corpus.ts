import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { medicalResponseAnnotationSchema } from "../src/lib/medical-response-annotations";
import {
  annotationCalibrationSchema,
  buildMedicalResponseExport,
  licensedMedicalSourceSchema,
  medicalExportReviewSchema,
} from "../src/lib/medical-response-export";

const root = process.cwd();
const corpusDirectory = path.join(root, "data", "medical-response-corpus");

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

async function readRows<T>(file: string, parse: (value: unknown) => T, optional = false): Promise<T[]> {
  try {
    return (await fs.readFile(file, "utf8")).split("\n").filter((line) => line.trim()).map((line) => parse(JSON.parse(line)));
  } catch (error) {
    if (optional && (error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function main() {
  const sourcePath = argValue("--source", path.join(corpusDirectory, "afrihealth-akan-source.v1.jsonl"));
  const annotationPath = argValue("--annotations", path.join(corpusDirectory, "afrihealth-akan-annotations.v1.jsonl"));
  const reviewPath = argValue("--reviews", path.join(root, "tmp", "understanding-review", "reviews.v0.jsonl"));
  const calibrationPath = argValue("--calibration", "");
  const output = argValue("--out", path.join(root, "tmp", "afrihealth-annotated-corpus", "v1"));
  const minimumSourceRows = Number(argValue("--minimum-source-rows", "7000"));
  if (!Number.isInteger(minimumSourceRows) || minimumSourceRows < 1) throw new Error("Invalid minimum source count");
  const sources = await readRows(sourcePath, (value) => licensedMedicalSourceSchema.parse(value));
  const annotations = await readRows(annotationPath, (value) => medicalResponseAnnotationSchema.parse(value));
  const fromDatabase = process.argv.includes("--reviews-db");
  const reviews = fromDatabase ? await (async () => {
    await import("../src/config/load-env");
    const { prisma } = await import("../src/db/prisma");
    try {
      const saved = await prisma.researchUnderstandingReview.findMany({ where: { rowId: { startsWith: "afrihealth_akan_" } } });
      return saved.map((row) => medicalExportReviewSchema.parse({
        ...row, id: row.rowId, updatedAt: row.updatedAt.toISOString(),
      }));
    } finally {
      await prisma.$disconnect();
    }
  })() : await readRows(reviewPath, (value) => medicalExportReviewSchema.parse(value), true);
  const parsePool = (value: unknown) => value as { source_record_id: string; question: string };
  const trainPool = await readRows(path.join(corpusDirectory, "afrihealth-ghana-response-train.v1.jsonl"), parsePool);
  const evalPool = await readRows(path.join(corpusDirectory, "afrihealth-ghana-response-eval.v1.jsonl"), parsePool);
  const calibration = calibrationPath
    ? annotationCalibrationSchema.parse(JSON.parse(await fs.readFile(calibrationPath, "utf8"))) : null;
  const result = buildMedicalResponseExport({
    sources, annotations, reviews, calibration,
    trainRecordIds: new Set(trainPool.map((row) => row.source_record_id)),
    evalRecordIds: new Set(evalPool.map((row) => row.source_record_id)),
    heldOutQuestions: evalPool.map((row) => row.question),
  });
  const train = result.rows.filter((row) => row.split === "train");
  const evaluation = result.rows.filter((row) => row.split === "eval");
  await fs.mkdir(output, { recursive: true });
  const artifacts = [];
  for (const [file, rows] of [["train.jsonl", train], ["eval.jsonl", evaluation], ["excluded.jsonl", result.rejected]] as const) {
    const content = rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length ? "\n" : "");
    const target = path.join(output, file);
    const temporary = `${target}.${process.pid}.tmp`;
    await fs.writeFile(temporary, content);
    await fs.rename(temporary, target);
    artifacts.push({ file, rows: rows.length, sha256: crypto.createHash("sha256").update(content).digest("hex") });
  }
  const exclusions = result.rejected.reduce<Record<string, number>>((counts, row) => {
    counts[row.reason] = (counts[row.reason] || 0) + 1;
    return counts;
  }, {});
  const uniqueTrainSources = new Set(train.map((row) => row.provenance.row_id)).size;
  const trainLanguages = { tw: train.filter((row) => row.language === "tw").length, en: train.filter((row) => row.language === "en").length };
  const ready = uniqueTrainSources >= minimumSourceRows && trainLanguages.tw === trainLanguages.en && evaluation.length > 0;
  const manifest = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    ready_for_research_training: ready,
    annotations_available: annotations.length,
    unique_training_source_rows: uniqueTrainSources,
    training_examples: train.length,
    train_languages: trainLanguages,
    evaluation_examples: evaluation.length,
    reviews_loaded: reviews.length,
    review_source: fromDatabase ? "postgres" : path.relative(root, reviewPath),
    source_sha256: crypto.createHash("sha256").update(await fs.readFile(sourcePath)).digest("hex"),
    annotation_sha256: crypto.createHash("sha256").update(await fs.readFile(annotationPath)).digest("hex"),
    calibration_sha256: calibrationPath ? crypto.createHash("sha256").update(await fs.readFile(calibrationPath)).digest("hex") : null,
    artifacts,
    exclusions,
    minimum_unique_training_sources: minimumSourceRows,
    policy: "Twi and English are paired views of the same source, with the same split. Counts of examples are not counts of unique source records. Model silver is separately identified from human review. Source-validation rows remain evaluation-only. This export does not establish final medical or product readiness.",
  };
  await fs.writeFile(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(JSON.stringify(manifest, null, 2));
  if (process.argv.includes("--strict") && !ready) process.exitCode = 1;
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
