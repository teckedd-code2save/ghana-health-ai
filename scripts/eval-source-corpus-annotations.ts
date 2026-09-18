import assert from "node:assert/strict";
import fs from "node:fs";
import { corpusTextKey, sourceCorpusHash, sourceCorpusSchema, sourceProposalProblems, sourceProposalSchema } from "../src/lib/source-corpus-annotations";

const source = sourceCorpusSchema.parse({
  id: "contract", source: "contract", source_record_id: "contract", source_revision: "contract",
  source_url: "https://example.com/source", license: "cc0-1.0", license_evidence_url: "https://example.com/license", attribution: "Contract test",
  split: "train", language: "en", task: "understanding_and_response",
  utterance: "Where can I find a clinic?", reference_answer: "Ask your local health service.", speaker_id: null, source_hash: "0".repeat(64),
});
source.source_hash = sourceCorpusHash(source);
const fields = sourceProposalSchema.parse({
  normalized_text: source.utterance, natural_english: source.utterance, intent: "health_service_question",
  entities: { locations: ["clinic"] }, uncertainty_notes: "", requires_clarification: false,
  safety_level: "routine", source_answer_assessment: "supportable", source_answer_issues: [],
  reply: source.reference_answer, evidence_spans: [source.reference_answer], confidence: 0.99,
});
assert.deepEqual(sourceProposalProblems(source, fields), []);
assert.ok(sourceProposalProblems(source, { ...fields, entities: { ...fields.entities, medicines: ["invented medicine"] } }).includes("entities_not_grounded_in_question"));
assert.ok(sourceProposalProblems(source, { ...fields, reply: "Take 2 tablets." }).includes("unsupported_numeric_claim"));
assert.ok(sourceProposalProblems(source, { ...fields, evidence_spans: ["unrelated evidence"] }).includes("ungrounded_evidence"));
assert.ok(sourceProposalProblems({ ...source, utterance: "Changed" }, fields).includes("source_hash_mismatch"));
const narrative = { ...source, task: "understanding" as const, reference_answer: null };
narrative.source_hash = sourceCorpusHash(narrative);
assert.ok(sourceProposalProblems(narrative, fields).includes("invented_reference_response"));
assert.deepEqual(sourceProposalProblems(narrative, { ...fields, reply: null, source_answer_assessment: "not_applicable", evidence_spans: [narrative.utterance] }), []);
assert.equal(sourceProposalSchema.safeParse({ ...fields, requires_clarification: undefined }).success, false);
assert.equal(sourceProposalSchema.safeParse({ ...fields, intent: "arbitrary_label" }).success, false);

const rows = fs.readFileSync("data/annotated-source-corpus/sources.v1.jsonl", "utf8").split("\n").filter(Boolean).map((line) => sourceCorpusSchema.parse(JSON.parse(line)));
assert.equal(rows.length, new Set(rows.map((row) => row.id)).size);
assert.ok(rows.every((row) => sourceCorpusHash(row) === row.source_hash));
assert.ok(rows.every((row) => ["ImhotepSystems/AfriHealth-QA", "waxal", "ghana_nlp_speech"].includes(row.source)));
const heldOut = new Set(rows.filter((row) => row.split !== "train").map((row) => corpusTextKey(row.utterance)));
assert.ok(rows.filter((row) => row.split === "train").every((row) => !heldOut.has(corpusTextKey(row.utterance))));
assert.ok(rows.filter((row) => row.source === "ghana_nlp_speech").every((row) => row.license === "cc-by-nc-4.0"));
console.log("Source corpus: hashes, source allowlist, held-out deduplication, mandatory fields, entity/evidence grounding, and response separation passed.");
