import fs from "node:fs/promises";
import path from "node:path";
import { buildMedicalPilot, digest } from "../src/lib/medical-response-pilot";
import { medicalResponseAnnotationSchema } from "../src/lib/medical-response-annotations";
import { licensedMedicalSourceSchema } from "../src/lib/medical-response-export";

const dir = "data/medical-response-corpus";
const output = "tmp/medical-response-pilot/v1";
const read = async (file: string) => (await fs.readFile(file, "utf8")).split("\n").filter(Boolean).map((line) => JSON.parse(line));

async function main() {
  if (!process.argv.includes("--acknowledge-uncalibrated-experiment")) throw new Error("This is not the accepted corpus. Explicit experimental acknowledgement required.");
  const annotationFile = `${dir}/afrihealth-teacher-v3-train.jsonl`;
  const sourceFile = `${dir}/afrihealth-akan-source.v1.jsonl`;
  const sources = (await read(sourceFile)).map((row) => licensedMedicalSourceSchema.parse(row));
  const annotations = (await read(annotationFile)).map((row) => medicalResponseAnnotationSchema.parse(row));
  const trainPool = await read(`${dir}/afrihealth-ghana-response-train.v1.jsonl`);
  const locked = await read(`${dir}/afrihealth-ghana-response-eval.v1.jsonl`);
  const product = await read(`${dir}/response-product-eval.v1.jsonl`);
  const reviewsArg = process.argv.indexOf("--reviews");
  if (reviewsArg < 0 || !process.argv[reviewsArg + 1]) throw new Error("Supply a current review export via --reviews; do not silently ignore human decisions.");
  const reviews = await read(process.argv[reviewsArg + 1]);
  const result = buildMedicalPilot({ sources, annotations,
    trainRecordIds: new Set(trainPool.map((row) => row.source_record_id)),
    heldOutQuestions: [...locked.map((row) => row.question), ...product.flatMap((row) => row.messages.filter((m: { role: string }) => m.role === "user").map((m: { content: string }) => m.content))],
    humanBlockedIds: new Set(reviews.filter((row) => row.decision !== "unreviewed").map((row) => row.rowId ?? row.id)),
  });
  const train = result.rows.filter((row) => row.split === "pilot_train");
  const holdout = result.rows.filter((row) => row.split === "pilot_holdout");
  if (train.length < 60 || holdout.length < 12) throw new Error("Insufficient non-overlapping pilot rows");
  await fs.mkdir(output, { recursive: true });
  const artifacts = [];
  for (const [file, rows] of [["train.jsonl", train], ["holdout.jsonl", holdout], ["excluded.jsonl", result.excluded]] as const) {
    const content = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
    await fs.writeFile(path.join(output, file), content);
    artifacts.push({ file, rows: rows.length, sha256: digest(content) });
  }
  const unique = (rows: typeof train) => new Set(rows.map((row) => row.provenance.row_id)).size;
  const manifest = {
    schema_version: 1, experiment: "afrihealth_teacher_v3_pilot_v1", created_at: new Date().toISOString(),
    purpose: "User-requested bounded learning experiment on saved annotations, not a calibrated corpus export",
    accepted_corpus_gate_unchanged: true, ready_for_production: false, human_reviewed_sources: 0,
    annotation_rows: annotations.length, selected_sources: unique(result.rows), source_groups: result.sourceGroups,
    training_sources: unique(train), holdout_sources: unique(holdout), training_examples: train.length, holdout_examples: holdout.length,
    artifacts, annotation_sha256: digest(await fs.readFile(annotationFile, "utf8")),
    source_sha256: digest(await fs.readFile(sourceFile, "utf8")),
    reviews_sha256: digest(await fs.readFile(process.argv[reviewsArg + 1], "utf8")), reviews_loaded: reviews.length,
    base_model: "ghananlpcommunity/MiniCPM5-1B-Twi", base_revision: "d807ca1a3323972afafabff8f9affe2639e37b5c",
    limitations: ["Model labels are uncalibrated, not human gold", "All selected safety labels are routine; safety and clarification classifiers are NOT trained",
      "Mostly adolescent and reproductive health education, not acute clinical dialogue", "English replies are translated views, not additional independent sources",
      "Source-group holdout measures agreement with model annotations, not clinical accuracy", "Original source validation and product cases remain excluded from all training",
      "No production deployment or public Hub push is authorized by this artifact"],
  };
  await fs.writeFile(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(JSON.stringify(manifest, null, 2));
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
