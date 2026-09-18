import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  medicalQuestionEntityViolations,
  medicalResponseAnnotationSchema,
  medicalResponseSourceSchema,
} from "../src/lib/medical-response-annotations";

function argument(name: string, fallback: string) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1] ?? fallback;
}

async function main() {
  const directory = path.join(process.cwd(), "data", "medical-response-corpus");
  const sourcePath = argument("--source", path.join(directory, "afrihealth-akan-source.v1.jsonl"));
  const annotationPath = argument("--annotations", path.join(directory, "afrihealth-teacher-v2-calibration.jsonl"));
  const summaryPath = argument("--summary", `${annotationPath}.grounding.summary.json`);
  const sourceRaw = await fs.readFile(sourcePath, "utf8");
  const annotationRaw = await fs.readFile(annotationPath, "utf8");
  const sources = sourceRaw.split("\n").filter((line) => line.trim())
    .map((line) => medicalResponseSourceSchema.parse(JSON.parse(line)));
  const annotations = annotationRaw.split("\n").filter((line) => line.trim())
    .map((line) => medicalResponseAnnotationSchema.parse(JSON.parse(line)));
  if (new Set(annotations.map((row) => row.row_id)).size !== annotations.length) throw new Error("Duplicate annotations");
  const sourceById = new Map(sources.map((source) => [source.id, source]));
  const rows = annotations.map((annotation) => {
    const source = sourceById.get(annotation.row_id);
    if (!source || source.record_source_hash !== annotation.source_record_hash) throw new Error("Missing or mismatched source");
    const proposals = [...annotation.proposals, ...(annotation.synthesized_proposal ? [annotation.synthesized_proposal] : [])];
    const selected = proposals.find((proposal) => proposal.proposal_id === annotation.recommended_proposal_id);
    if (!selected) throw new Error("Missing selected proposal");
    return {
      row_id: annotation.row_id,
      status: annotation.adjudication.status,
      question: source.question_twi_source,
      question_english: selected.question_english,
      selected_violations: medicalQuestionEntityViolations(source.question_twi_source, selected.entities),
      proposal_violations: proposals.map((proposal) => ({
        proposal_id: proposal.proposal_id,
        role: proposal.agent_role,
        violations: medicalQuestionEntityViolations(source.question_twi_source, proposal.entities),
      })),
    };
  });
  const failing = rows.filter((row) => row.selected_violations.length);
  const summary = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    annotation_sha256: crypto.createHash("sha256").update(annotationRaw).digest("hex"),
    source_sha256: crypto.createHash("sha256").update(sourceRaw).digest("hex"),
    prompt_versions: [...new Set(annotations.map((row) => row.prompt_version))],
    annotations: annotations.length,
    selected_rows_with_ungrounded_entities: failing.length,
    silver_rows_with_ungrounded_entities: failing.filter((row) => row.status === "silver_consensus").length,
    selected_entity_violations: failing.reduce((sum, row) => sum + row.selected_violations.length, 0),
    note: "Question-only lexical span check, not semantic or medical validation. No source or annotation is modified. A passing span may still have an incorrect category, negation or experiencer.",
    rows,
  };
  await fs.mkdir(path.dirname(summaryPath), { recursive: true });
  await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify({ ...summary, rows: undefined }, null, 2));
  if (process.argv.includes("--strict") && failing.length) process.exitCode = 1;
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
