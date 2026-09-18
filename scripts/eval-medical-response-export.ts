import assert from "node:assert/strict";
import fs from "node:fs";
import {
  medicalAnnotationPipelineId,
  medicalResponseAnnotationSchema,
  medicalQuestionEntityViolations,
} from "../src/lib/medical-response-annotations";
import {
  buildMedicalResponseExport,
  licensedMedicalSourceSchema,
  medicalExportReviewSchema,
} from "../src/lib/medical-response-export";

const directory = "data/medical-response-corpus/";
const readRows = (file: string) => fs.readFileSync(directory + file, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const sources = readRows("afrihealth-akan-source.v1.jsonl").map((row) => licensedMedicalSourceSchema.parse(row));
const annotations = readRows("afrihealth-akan-annotations.v1.jsonl").map((row) => medicalResponseAnnotationSchema.parse(row));
const trainIds = new Set<string>(readRows("afrihealth-ghana-response-train.v1.jsonl").map((row) => row.source_record_id));
const original = annotations.find((row) => row.eligible_for_training && sources.some((source) => source.id === row.row_id && trainIds.has(source.source_record_id)))!;
const annotation = structuredClone(original);
assert.ok(annotation, "Require a real annotated source row for the export contract");
const source = sources.find((row) => row.id === annotation.row_id)!;
const selected = [...annotation.proposals, ...(annotation.synthesized_proposal ? [annotation.synthesized_proposal] : [])]
  .find((row) => row.proposal_id === annotation.recommended_proposal_id)!;
// Old proposals used entities from both QA fields; ground this export fixture in the question.
const ungrounded = medicalQuestionEntityViolations(source.question_twi_source, selected.entities);
for (const category of Object.keys(selected.entities) as Array<keyof typeof selected.entities>) {
  selected.entities[category] = selected.entities[category]
    .filter((value) => !ungrounded.some((issue) => issue.category === category && issue.value === value));
}
const input: Parameters<typeof buildMedicalResponseExport>[0] = {
  sources: [source], annotations: [annotation], reviews: [], trainRecordIds: trainIds,
  evalRecordIds: new Set(), heldOutQuestions: [], calibration: null,
};
assert.equal(buildMedicalResponseExport(input).rows.length, 0, "Model confidence alone cannot admit training rows");
const passingCalibration = {
  gate: "pass" as const, failures: [], pipeline_ids: [medicalAnnotationPipelineId(annotation)],
  candidate_sha256: "0".repeat(64), reference_sha256: "1".repeat(64),
};
const calibrated = { ...input, calibration: passingCalibration };
const pair = buildMedicalResponseExport(calibrated).rows;
assert.equal(pair.length, 2);
assert.deepEqual(pair.map((row) => row.language).sort(), ["en", "tw"]);
assert.equal(pair[0].provenance.source_record_hash, pair[1].provenance.source_record_hash);
assert.ok(pair.every((row) => row.split === "train" && row.provenance.review_status === "model_silver"));
assert.equal(buildMedicalResponseExport({ ...input, calibration: { ...passingCalibration, gate: "fail" } }).rows.length, 0);
assert.equal(buildMedicalResponseExport({ ...input, calibration: { ...passingCalibration, pipeline_ids: ["wrong"] } }).rows.length, 0);
const contaminated = structuredClone(annotation);
const contaminatedSelection = [...contaminated.proposals, ...(contaminated.synthesized_proposal ? [contaminated.synthesized_proposal] : [])]
  .find((row) => row.proposal_id === contaminated.recommended_proposal_id)!;
contaminatedSelection.entities.medicines.push("answer-only-medicine");
const blocked = buildMedicalResponseExport({ ...calibrated, annotations: [contaminated] });
assert.equal(blocked.rows.length, 0, "Consensus must not promote entities that the user did not mention");
assert.equal(blocked.rejected[0].reason, "entities_not_grounded_in_question");
const emptyEntities = { symptoms: [], conditions: [], medicines: [], body_parts: [], durations: [], quantities: [], persons: [], locations: [], other: [] };
assert.equal(medicalQuestionEntityViolations("My child has pain.", { ...emptyEntities, symptoms: ["pain"], persons: ["child"] }).length, 0);
assert.equal(medicalQuestionEntityViolations("My child has pain.", { ...emptyEntities, symptoms: ["fever"], persons: ["I"] }).length, 2);
assert.equal(medicalQuestionEntityViolations("My child's pain", { ...emptyEntities, persons: ["child\u2019s"] }).length, 0);

const review = medicalExportReviewSchema.parse({
  id: source.id, decision: "reviewed", selectedProposalId: selected.proposal_id,
  synthesisVersion: annotation.prompt_version, reviewer: "contract-reviewer", updatedAt: "2026-09-06T00:00:00Z",
  normalizedTwi: selected.question_twi_normalized, naturalEnglish: selected.question_english,
  intent: selected.intent, entities: JSON.stringify(selected.entities),
  replyTwi: selected.reply_twi, safetyLevel: selected.safety_level,
  ambiguities: JSON.stringify({ topics: selected.topics }), requiresClarification: false,
});
const humanRows = buildMedicalResponseExport({ ...input, reviews: [review] }).rows;
assert.equal(humanRows.length, 2, "Human review can admit a row independently of a model calibration");
assert.equal(JSON.parse(humanRows[0].messages[2].content).requires_clarification, false, "Nonempty review metadata is not a clarification decision");
assert.equal(humanRows[0].provenance.review_status, "human_reviewed");
assert.equal(buildMedicalResponseExport({ ...calibrated, reviews: [{ ...review, decision: "exclude" }] }).rows.length, 0);
assert.equal(buildMedicalResponseExport({ ...input, reviews: [{ ...review, synthesisVersion: "stale" }] }).rows.length, 0);
assert.equal(buildMedicalResponseExport({ ...input, reviews: [{ ...review, selectedProposalId: "missing" }] }).rows.length, 0);
const edited = buildMedicalResponseExport({ ...input, reviews: [{ ...review, replyTwi: "Human corrected response" }] });
assert.equal(edited.rows.length, 1, "Do not attach a stale English translation to an edited Twi reply");
assert.equal(edited.rows[0].language, "tw");
assert.ok(edited.rejected.some((row) => row.reason === "english_view_needs_translation_of_human_edit"));
assert.equal(buildMedicalResponseExport({ ...calibrated, heldOutQuestions: [selected.question_english] }).rows.length, 0);
assert.equal(buildMedicalResponseExport({ ...calibrated, sources: [{ ...source, question_twi_source: "tampered source" }] }).rows.length, 0);
assert.equal(buildMedicalResponseExport({ ...calibrated, trainRecordIds: new Set() }).rows.length, 0);
assert.throws(() => buildMedicalResponseExport({ ...input, annotations: [annotation, annotation] }), /Duplicate annotation/);
const validation = buildMedicalResponseExport({
  ...calibrated, sources: [{ ...source, source_split: "validation" }], evalRecordIds: new Set([source.source_record_id]),
});
assert.equal(validation.rows.length, 2);
assert.ok(validation.rows.every((row) => row.split === "eval" && !row.eligible_for_research_training && !row.eligible_for_final_evaluation));
console.log("Medical export: calibration enforcement, reviewed selections, corrected replies, bilingual lineage, source hashes, and held-out isolation passed.");
