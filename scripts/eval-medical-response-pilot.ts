import assert from "node:assert/strict";
import fs from "node:fs";
import { buildMedicalPilot, digest } from "../src/lib/medical-response-pilot";
import { medicalResponseAnnotationSchema } from "../src/lib/medical-response-annotations";
import { buildMedicalResponseExport, licensedMedicalSourceSchema } from "../src/lib/medical-response-export";

const dir = "data/medical-response-corpus/";
const read = (file: string) => fs.readFileSync(dir + file, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const sources = read("afrihealth-akan-source.v1.jsonl").map((row) => licensedMedicalSourceSchema.parse(row));
const annotations = read("afrihealth-teacher-v3-train.jsonl").map((row) => medicalResponseAnnotationSchema.parse(row));
const trainRecordIds = new Set<string>(read("afrihealth-ghana-response-train.v1.jsonl").map((row) => row.source_record_id));
const input = { sources, annotations, trainRecordIds, heldOutQuestions: read("afrihealth-ghana-response-eval.v1.jsonl").map((row) => row.question as string) };
const result = buildMedicalPilot(input);
assert.ok(result.rows.length > 60);
const trains = result.rows.filter((row) => row.split === "pilot_train");
const heldout = result.rows.filter((row) => row.split === "pilot_holdout");
for (const field of ["row_id", "source_record_hash", "group_id"] as const) {
  const seen = new Set(trains.map((row) => row.provenance[field]));
  assert.ok(heldout.every((row) => !seen.has(row.provenance[field])));
}
assert.ok(result.rows.every((row) => !row.eligible_for_research_training && !row.eligible_for_final_evaluation && !row.provenance.human_reviewed));
for (const row of result.rows.filter((row) => row.task === "interpret_and_reply")) {
  const target = JSON.parse(row.messages.at(-1)!.content);
  assert.ok(!("safety_level" in target), "Do not teach a constant routine safety classifier");
}
const id = trains[0].provenance.row_id;
const source = sources.find((row) => row.id === id)!;
const annotation = annotations.find((row) => row.row_id === id)!;
const one = { ...input, annotations: [annotation] };
assert.equal(buildMedicalPilot({ ...one, humanBlockedIds: new Set([id]) }).rows.length, 0);
assert.equal(buildMedicalPilot({ ...one, heldOutQuestions: [source.question_twi_source] }).rows.length, 0);
assert.equal(buildMedicalPilot({ ...one, trainRecordIds: new Set() }).rows.length, 0);
assert.equal(buildMedicalPilot({ ...one, sources: [{ ...source, question_twi_source: "tampered" }] }).rows.length, 0);
assert.equal(buildMedicalPilot({ ...one, annotations: [{ ...annotation, eligible_for_training: false }] }).rows.length, 0);
assert.equal(buildMedicalPilot({ ...one, annotations: [{ ...annotation, adjudication: { ...annotation.adjudication, status: "needs_human_review" } }] }).rows.length, 0);
assert.throws(() => buildMedicalPilot({ ...one, annotations: [annotation, annotation] }), /Duplicate annotations/);
assert.equal(buildMedicalResponseExport({ ...input, reviews: [], calibration: null, evalRecordIds: new Set() }).rows.length, 0, "Normal training gate must remain closed");
assert.equal(digest(JSON.stringify(result)), digest(JSON.stringify(buildMedicalPilot({ ...input, annotations: [...annotations].reverse() }))));
console.log(`Pilot contracts passed: ${trains.length} training views, ${heldout.length} held-out views; normal corpus gate remains closed.`);
