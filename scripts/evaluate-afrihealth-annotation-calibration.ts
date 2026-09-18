import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  medicalAnnotationPipelineId,
  medicalResponseAnnotationSchema,
  type MedicalResponseAnnotation,
  type MedicalResponseProposal,
  type MedicalResponseSynthesis,
} from "../src/lib/medical-response-annotations";

type FinalProposal = MedicalResponseProposal | MedicalResponseSynthesis;

const root = process.cwd();
const defaultReference = path.join(
  root,
  "data",
  "medical-response-corpus",
  "afrihealth-akan-annotations.v1.jsonl",
);
const defaultCandidate = path.join(
  root,
  "tmp",
  "understanding-corpus",
  "afrihealth-pivot-reference-62.imported.jsonl",
);
const defaultSummary = path.join(
  root,
  "tmp",
  "understanding-corpus",
  "afrihealth-pivot-reference-62.calibration.json",
);

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

function hasFlag(name: string) {
  return process.argv.includes(name);
}

async function readAnnotations(filePath: string) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => (
    medicalResponseAnnotationSchema.parse(JSON.parse(line))
  ));
}

function finalProposal(row: MedicalResponseAnnotation): FinalProposal | undefined {
  if (row.synthesized_proposal?.proposal_id === row.recommended_proposal_id) {
    return row.synthesized_proposal;
  }
  return row.proposals.find((proposal) => proposal.proposal_id === row.recommended_proposal_id);
}

function normalizedTokens(value: string) {
  return new Set(value.toLocaleLowerCase("en").match(/[\p{L}\p{N}]+/gu) ?? []);
}

function jaccard(left: Iterable<string>, right: Iterable<string>) {
  const a = new Set(left);
  const b = new Set(right);
  const union = new Set([...a, ...b]);
  if (!union.size) return 1;
  let intersection = 0;
  for (const value of a) if (b.has(value)) intersection += 1;
  return intersection / union.size;
}

function entityValues(proposal: FinalProposal) {
  return Object.values(proposal.entities).flat().map((value) => value.toLocaleLowerCase("en"));
}

function mean(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function ratio(numerator: number, denominator: number) {
  return denominator ? numerator / denominator : 0;
}

async function main() {
  const referencePath = argValue("--reference", defaultReference);
  const candidatePath = argValue("--candidate", defaultCandidate);
  const summaryPath = argValue("--summary", defaultSummary);
  const references = await readAnnotations(referencePath);
  const candidates = await readAnnotations(candidatePath);
  for (const [name, rows] of [["reference", references], ["candidate", candidates]] as const) {
    if (new Set(rows.map((row) => row.row_id)).size !== rows.length) {
      throw new Error(`Duplicate row IDs in ${name} calibration artifact`);
    }
  }
  const referenceById = new Map(references.map((row) => [row.row_id, row]));
  const comparisons = [];
  const sourceMismatches: string[] = [];

  for (const candidate of candidates) {
    const reference = referenceById.get(candidate.row_id);
    const expected = reference ? finalProposal(reference) : undefined;
    const actual = finalProposal(candidate);
    if (!reference || !expected || !actual) continue;
    if (reference.source_record_hash !== candidate.source_record_hash) {
      sourceMismatches.push(candidate.row_id);
      continue;
    }

    const intent = actual.intent === expected.intent;
    const safety = actual.safety_level === expected.safety_level;
    const clarification = actual.requires_clarification === expected.requires_clarification;
    const assessment = actual.source_answer_assessment === expected.source_answer_assessment;
    const topic = jaccard(actual.topics, expected.topics);
    const entities = jaccard(entityValues(actual), entityValues(expected));
    const question = jaccard(normalizedTokens(actual.question_english), normalizedTokens(expected.question_english));
    const answer = jaccard(normalizedTokens(actual.answer_english), normalizedTokens(expected.answer_english));
    const reply = jaccard(normalizedTokens(actual.reply_english), normalizedTokens(expected.reply_english));
    const proxyMismatch = !intent
      || !safety
      || !clarification
      || !assessment
      || topic < 0.5;
    const routedToReview = candidate.adjudication.status === "needs_human_review";
    const silverProxyPass = !proxyMismatch && candidate.adjudication.status === "silver_consensus";
    comparisons.push({
      row_id: candidate.row_id,
      intent,
      safety,
      clarification,
      assessment,
      topic,
      entities,
      question,
      answer,
      reply,
      proxy_mismatch: proxyMismatch,
      routed_to_review: routedToReview,
      silver_proxy_pass: silverProxyPass,
      proposed_silver: candidate.adjudication.status === "silver_consensus",
    });
  }

  const mismatches = comparisons.filter((row) => row.proxy_mismatch);
  const silver = comparisons.filter((row) => row.proposed_silver);
  const silverPasses = silver.filter((row) => row.silver_proxy_pass);
  const metrics = {
    compared_rows: comparisons.length,
    reference_rows: references.length,
    candidate_rows: candidates.length,
    categorical_agreement: {
      intent: mean(comparisons.map((row) => Number(row.intent))),
      safety_level: mean(comparisons.map((row) => Number(row.safety))),
      requires_clarification: mean(comparisons.map((row) => Number(row.clarification))),
      source_answer_assessment: mean(comparisons.map((row) => Number(row.assessment))),
    },
    semantic_similarity: {
      topics_jaccard: mean(comparisons.map((row) => row.topic)),
      entities_jaccard: mean(comparisons.map((row) => row.entities)),
      question_token_jaccard: mean(comparisons.map((row) => row.question)),
      answer_token_jaccard: mean(comparisons.map((row) => row.answer)),
      reply_token_jaccard: mean(comparisons.map((row) => row.reply)),
    },
    proxy_mismatches: mismatches.length,
    review_capture_rate: mismatches.length ? ratio(mismatches.filter((row) => row.routed_to_review).length, mismatches.length) : 1,
    proposed_silver_rows: silver.length,
    proposed_silver_proxy_pass_rate: ratio(silverPasses.length, silver.length),
  };
  const thresholds = {
    compared_rows: 50,
    intent_agreement: 0.9,
    safety_agreement: 0.9,
    source_answer_assessment_agreement: 0.85,
    review_capture_rate: 0.85,
    proposed_silver_proxy_pass_rate: 0.9,
  };
  const failures = [
    sourceMismatches.length ? "source_hash_mismatch" : "",
    comparisons.length / Math.max(1, references.length) < 0.95 ? "incomplete_reference_coverage" : "",
    metrics.compared_rows < thresholds.compared_rows ? "insufficient_overlap" : "",
    metrics.categorical_agreement.intent < thresholds.intent_agreement ? "intent_agreement" : "",
    metrics.categorical_agreement.safety_level < thresholds.safety_agreement ? "safety_agreement" : "",
    metrics.categorical_agreement.source_answer_assessment < thresholds.source_answer_assessment_agreement
      ? "source_answer_assessment_agreement"
      : "",
    metrics.review_capture_rate < thresholds.review_capture_rate ? "review_capture_rate" : "",
    metrics.proposed_silver_proxy_pass_rate < thresholds.proposed_silver_proxy_pass_rate
      ? "proposed_silver_proxy_pass_rate"
      : "",
  ].filter(Boolean);
  const report = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    reference_path: path.relative(root, referencePath),
    candidate_path: path.relative(root, candidatePath),
    candidate_sha256: crypto.createHash("sha256").update(await fs.readFile(candidatePath)).digest("hex"),
    reference_sha256: crypto.createHash("sha256").update(await fs.readFile(referencePath)).digest("hex"),
    pipeline_ids: [...new Set(candidates.map(medicalAnnotationPipelineId))].sort(),
    source_hash_mismatches: sourceMismatches,
    reference_note: "The reference is a stronger model-assisted calibration set, not human gold.",
    metrics,
    thresholds,
    gate: failures.length ? "fail" : "pass",
    failures,
    mismatched_row_ids: mismatches.map((row) => row.row_id),
  };
  await fs.mkdir(path.dirname(summaryPath), { recursive: true });
  await fs.writeFile(summaryPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(report, null, 2));
  if (hasFlag("--strict") && failures.length) process.exitCode = 1;
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
