import fs from "node:fs/promises";
import path from "node:path";
import {
  buildMedicalResponseAnnotation,
  medicalResponseAnnotationSchema,
  medicalResponseOpenPromptVersion,
  medicalResponsePivotPromptVersion,
  medicalResponseProposalSchema,
  medicalResponseSourceSchema,
  medicalResponseSynthesisSchema,
  normalizeMedicalModelFields,
  stableMedicalProposalId,
  type MedicalResponseAnnotation,
  type MedicalResponseProposal,
  type MedicalResponseSource,
  type MedicalResponseSynthesis,
} from "../src/lib/medical-response-annotations";

type Json = Record<string, unknown>;
type RawBundle = { model?: unknown; parsed?: unknown } | null;
type RawRow = {
  prompt_version?: unknown;
  generated_at?: unknown;
  row_id?: unknown;
  source_record_hash?: unknown;
  teacher_a?: RawBundle;
  teacher_b?: RawBundle;
  adjudicator?: RawBundle;
};

const supportedPromptVersions = new Set([
  medicalResponseOpenPromptVersion,
  medicalResponsePivotPromptVersion,
]);

const root = process.cwd();
const defaultInput = path.join(root, "tmp", "understanding-corpus", "afrihealth-modal-v1.raw.jsonl");
const defaultSource = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-source.v1.jsonl");
const defaultOutput = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-open-annotations.v1.jsonl");
const defaultSummary = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-open-annotations.v1.summary.json");

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown) {
  return Array.isArray(value) ? value.map(asString).filter(Boolean) : [];
}

function parsedBundle(bundle: RawBundle | undefined) {
  if (typeof bundle?.parsed !== "object" || bundle.parsed === null) return undefined;
  const parsed = bundle.parsed as Json;
  const annotationKeys = [
    "question_english",
    "answer_english",
    "intent",
    "topics",
    "entities",
    "safety_level",
    "requires_clarification",
    "reply_twi",
    "selected_proposal",
    "material_disagreement_fields",
  ];
  const nested = typeof parsed.response_shape === "object" && parsed.response_shape !== null
    ? parsed.response_shape as Json
    : undefined;
  const outerHasPayload = annotationKeys.some((key) => key in parsed);
  const nestedHasPayload = nested && annotationKeys.some((key) => key in nested);
  return !outerHasPayload && nestedHasPayload ? nested : parsed;
}

function parseJsonl<T>(raw: string, parse: (value: unknown) => T) {
  return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => parse(JSON.parse(line)));
}

async function readExisting(filePath: string) {
  try {
    return parseJsonl(await fs.readFile(filePath, "utf8"), (value) => medicalResponseAnnotationSchema.parse(value));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function writeJsonl(filePath: string, rows: MedicalResponseAnnotation[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(temporary, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.rename(temporary, filePath);
}

function proposal(
  source: MedicalResponseSource,
  bundle: RawBundle | undefined,
  role: "teacher_a" | "teacher_b",
  generatedAt: string,
  promptVersion: string,
): MedicalResponseProposal {
  const model = `modal:${asString(bundle?.model) || role}`;
  return medicalResponseProposalSchema.parse({
    proposal_id: stableMedicalProposalId(source.id, role, model, promptVersion),
    row_id: source.id,
    agent_role: role,
    model,
    prompt_version: promptVersion,
    generated_at: generatedAt,
    ...normalizeMedicalModelFields(parsedBundle(bundle)),
  });
}

async function main() {
  const input = argValue("--input", defaultInput);
  const sourcePath = argValue("--source", defaultSource);
  const output = argValue("--out", defaultOutput);
  const summaryPath = argValue("--summary", defaultSummary);
  const rawRows = parseJsonl<RawRow>(await fs.readFile(input, "utf8"), (value) => value as RawRow);
  const sources = parseJsonl(await fs.readFile(sourcePath, "utf8"), (value) => medicalResponseSourceSchema.parse(value));
  const sourceById = new Map(sources.map((row) => [row.id, row]));
  const records = new Map((await readExisting(output)).map((row) => [row.row_id, row]));
  const rejected: Array<{ row_id: string; reason: string }> = [];

  for (const raw of rawRows) {
    const rowId = asString(raw.row_id);
    const promptVersion = asString(raw.prompt_version);
    const source = sourceById.get(rowId);
    if (!source) {
      rejected.push({ row_id: rowId, reason: "missing_source" });
      continue;
    }
    if (!supportedPromptVersions.has(promptVersion) || raw.source_record_hash !== source.record_source_hash) {
      rejected.push({ row_id: rowId, reason: "version_or_source_hash_mismatch" });
      continue;
    }
    const teacherAFields = parsedBundle(raw.teacher_a ?? null);
    const teacherBFields = parsedBundle(raw.teacher_b ?? null);
    const judge = parsedBundle(raw.adjudicator ?? null);
    if (!teacherAFields || !teacherBFields || !judge) {
      rejected.push({ row_id: rowId, reason: "unparsed_model_output" });
      continue;
    }
    try {
      const generatedAt = asString(raw.generated_at) || new Date().toISOString();
      const teacherA = proposal(source, raw.teacher_a, "teacher_a", generatedAt, promptVersion);
      const teacherB = proposal(source, raw.teacher_b, "teacher_b", generatedAt, promptVersion);
      const judgeModel = `modal:${asString(raw.adjudicator?.model) || "adjudicator"}`;
      const selection = asString(judge.selected_proposal);
      const reviewReasons = stringArray(judge.review_reasons);
      const synthesisId = stableMedicalProposalId(
        source.id,
        "adjudicator",
        judgeModel,
        promptVersion,
      );
      let selectedProposalId = selection === "teacher_a"
        ? teacherA.proposal_id
        : selection === "teacher_b"
          ? teacherB.proposal_id
          : selection === "synthesized" ? synthesisId : "needs_human_review";
      let synthesis: MedicalResponseSynthesis | null = null;
      if (selection === "synthesized") {
        const parsedSynthesis = medicalResponseSynthesisSchema.safeParse({
          proposal_id: synthesisId,
          row_id: source.id,
          agent_role: "adjudicator",
          model: judgeModel,
          prompt_version: promptVersion,
          generated_at: generatedAt,
          ...normalizeMedicalModelFields(judge.synthesized),
        });
        if (parsedSynthesis.success) {
          synthesis = parsedSynthesis.data;
        } else {
          selectedProposalId = "needs_human_review";
          reviewReasons.push("invalid_synthesis");
        }
      }
      if (selection === "needs_human_review") reviewReasons.push("adjudicator_requested_review");
      records.set(source.id, buildMedicalResponseAnnotation({
        source,
        proposals: [teacherA, teacherB],
        synthesis,
        selectedProposalId,
        adjudicatorModel: judgeModel,
        adjudicatorConfidence: typeof judge.confidence === "number" ? judge.confidence : Number(judge.confidence) || 0,
        adjudicatorRationale: asString(judge.rationale),
        adjudicatorDisagreements: stringArray(judge.material_disagreement_fields),
        adjudicatorReviewReasons: reviewReasons,
        promptVersion,
      }));
    } catch (error) {
      rejected.push({
        row_id: rowId,
        reason: error instanceof Error ? `invalid_annotation:${error.message}` : "invalid_annotation",
      });
    }
  }

  const annotations = [...records.values()].sort((left, right) => left.row_id.localeCompare(right.row_id));
  await writeJsonl(output, annotations);
  const statuses = annotations.reduce<Record<string, number>>((counts, row) => {
    counts[row.adjudication.status] = (counts[row.adjudication.status] ?? 0) + 1;
    return counts;
  }, {});
  const summary = {
    schema_version: 1,
    input_prompt_versions: [...new Set(rawRows.map((row) => asString(row.prompt_version)).filter(Boolean))].sort(),
    corpus_prompt_versions: [...new Set(annotations.map((row) => row.prompt_version))].sort(),
    raw_rows: rawRows.length,
    imported_annotations: annotations.length,
    statuses,
    eligible_for_training: annotations.filter((row) => row.eligible_for_training).length,
    rejected: rejected.length,
    rejected_rows: rejected,
  };
  await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(summary, null, 2));
  if (rejected.length) process.exitCode = 1;
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
