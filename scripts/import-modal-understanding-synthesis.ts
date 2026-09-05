import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  annotationProposalSchema,
  buildSynthesisRecord,
  corpusSynthesisPromptVersion,
  corpusSynthesisSchema,
  hasAuthoritativeSourceMeaning,
  semanticAmbiguities,
  type AnnotationProposal,
  type CorpusSynthesis,
} from "../src/lib/research-synthesis";
import { readCorpusCandidates, type CorpusCandidate } from "../src/lib/research-understanding-store";

type Json = Record<string, unknown>;
type RawAgent = { model?: string; parsed?: Json | null };
type RawRow = {
  prompt_version?: string;
  row_id?: string;
  source_hash?: string;
  generated_at?: string;
  translator?: RawAgent;
  semantic?: RawAgent;
  adjudicator?: RawAgent;
};

const root = process.cwd();
const defaultInput = path.join(root, "tmp", "understanding-corpus", "modal-synthesis-v9.raw.jsonl");
const defaultOutput = path.join(root, "tmp", "understanding-corpus", "synthesis.v1.jsonl");

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown) {
  return Array.isArray(value)
    ? value.map((item) => asString(item) || (
        typeof item === "object" && item !== null ? asString((item as Json).field) : ""
      )).filter(Boolean)
    : [];
}

function completeAnnotation(value: Json | null | undefined): value is Json {
  return Boolean(
    value &&
    asString(value.normalized_twi) &&
    asString(value.natural_english) &&
    asString(value.intent) &&
    typeof value.entities === "object" &&
    value.entities !== null,
  );
}

function compactJson(value: unknown) {
  if (typeof value === "string") return value.trim();
  return value === undefined || value === null ? "{}" : JSON.stringify(value);
}

function safetyLevel(value: unknown): AnnotationProposal["safety_level"] {
  return value === "routine" || value === "same_day" || value === "urgent" || value === "emergency"
    ? value
    : "";
}

function proposalId(rowId: string, role: string, model: string) {
  return crypto
    .createHash("sha256")
    .update(`${corpusSynthesisPromptVersion}:${rowId}:${role}:${model}`)
    .digest("hex")
    .slice(0, 20);
}

function proposalFromFields(input: {
  row: CorpusCandidate;
  fields: Json;
  role: AnnotationProposal["agent_role"];
  model: string;
  generatedAt: string;
}) {
  return annotationProposalSchema.parse({
    proposal_id: proposalId(input.row.id, input.role, input.model),
    row_id: input.row.id,
    agent_role: input.role,
    model: input.model,
    prompt_version: corpusSynthesisPromptVersion,
    generated_at: input.generatedAt,
    normalized_twi: asString(input.fields.normalized_twi) || input.row.text,
    natural_english: asString(input.fields.natural_english),
    literal_english: asString(input.fields.literal_english),
    intent: asString(input.fields.intent),
    entities: compactJson(input.fields.entities),
    ambiguities: compactJson(input.fields.ambiguities),
    reply_twi: asString(input.fields.reply_twi),
    safety_level: safetyLevel(input.fields.safety_level),
    requires_clarification: input.fields.requires_clarification === true,
  });
}

function sourceProposal(row: CorpusCandidate, generatedAt: string) {
  const source = row.model_proposal;
  return proposalFromFields({
    row,
    role: "source",
    model: source.model || `source:${row.source}`,
    generatedAt,
    fields: {
      normalized_twi: source.normalized_twi || row.text,
      natural_english: source.natural_english,
      literal_english: source.literal_english,
      intent: source.intent,
      entities: source.entities || "{}",
      ambiguities: semanticAmbiguities(source.ambiguities),
      reply_twi: source.reply_twi,
      safety_level: source.safety_level,
      requires_clarification: source.requires_clarification,
    },
  });
}

async function readExisting(output: string) {
  try {
    const raw = await fs.readFile(output, "utf8");
    return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => corpusSynthesisSchema.parse(JSON.parse(line)));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function main() {
  const input = argValue("--input", defaultInput);
  const output = argValue("--out", defaultOutput);
  const candidates = await readCorpusCandidates();
  const candidateById = new Map(candidates.map((row) => [row.id, row]));
  const existing = await readExisting(output);
  const recordsById = new Map(existing.map((row) => [row.row_id, row]));
  const rawRows = (await fs.readFile(input, "utf8"))
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RawRow);
  const rejected: Array<{ row_id: string; reason: string }> = [];
  let imported = 0;

  for (const raw of rawRows) {
    const rowId = asString(raw.row_id);
    const row = candidateById.get(rowId);
    if (!row) {
      rejected.push({ row_id: rowId, reason: "missing_candidate" });
      continue;
    }
    if (raw.prompt_version !== corpusSynthesisPromptVersion || raw.source_hash !== row.source_hash) {
      rejected.push({ row_id: rowId, reason: "version_or_source_hash_mismatch" });
      continue;
    }
    const translatorFields = raw.translator?.parsed;
    const semanticFields = raw.semantic?.parsed;
    const judge = raw.adjudicator?.parsed;
    if (
      !completeAnnotation(translatorFields) ||
      !completeAnnotation(semanticFields) ||
      !completeAnnotation(judge) ||
      !asString(judge.selected_proposal_id) ||
      typeof judge.confidence !== "number"
    ) {
      rejected.push({ row_id: rowId, reason: "unparsed_agent_output" });
      continue;
    }
    const generatedAt = asString(raw.generated_at) || new Date().toISOString();
    const source = sourceProposal(row, generatedAt);
    const translator = proposalFromFields({
      row,
      fields: translatorFields,
      role: "translator",
      model: asString(raw.translator?.model) || "modal:translator",
      generatedAt,
    });
    const semantic = proposalFromFields({
      row,
      fields: semanticFields,
      role: "semantic_annotator",
      model: asString(raw.semantic?.model) || "modal:semantic",
      generatedAt,
    });
    const judgeModel = asString(raw.adjudicator?.model) || "modal:adjudicator";
    const namedSelections = new Map([
      ["source", source.proposal_id],
      ["translator", translator.proposal_id],
      ["semantic", semantic.proposal_id],
    ]);
    const selectedName = asString(judge.selected_proposal_id);
    const selectedProposalId = namedSelections.get(selectedName) ?? selectedName;
    const authoritativeSource = hasAuthoritativeSourceMeaning(row.source, row.model_proposal.model);
    const synthesizedProposal = authoritativeSource
      ? proposalFromFields({
          row,
          fields: {
            normalized_twi: row.text,
            natural_english: row.model_proposal.natural_english,
            literal_english: asString(judge.literal_english) || row.model_proposal.literal_english,
            intent: row.model_proposal.intent,
            entities: judge.entities || row.model_proposal.entities || "{}",
            ambiguities: semanticAmbiguities(row.model_proposal.ambiguities),
            reply_twi: row.model_proposal.reply_twi,
            safety_level: row.model_proposal.safety_level,
            requires_clarification: row.model_proposal.requires_clarification,
          },
          role: "adjudicator",
          model: `${judgeModel}:source-grounded-synthesis`,
          generatedAt,
        })
      : selectedName === "synthesized"
      ? proposalFromFields({ row, fields: judge, role: "adjudicator", model: judgeModel, generatedAt })
      : null;
    const record = buildSynthesisRecord({
      rowId: row.id,
      sourceHash: row.source_hash,
      promptVersion: corpusSynthesisPromptVersion,
      proposals: [source, translator, semantic],
      sourceEnglish: row.model_proposal.natural_english,
      sourceTwi: row.text,
      responseExpected: Boolean(row.model_proposal.reply_twi && row.model_proposal.safety_level),
      synthesizedProposal,
      adjudicatorModel: judgeModel,
      adjudicatorSelectedProposalId: selectedProposalId,
      adjudicatorConfidence: typeof judge.confidence === "number" ? judge.confidence : undefined,
      adjudicatorDisagreementFields: Array.isArray(judge.material_disagreement_fields)
        ? stringArray(judge.material_disagreement_fields)
        : undefined,
      adjudicatorReviewReasons: Array.isArray(judge.review_reasons) && judge.review_reasons.length > 0
        ? ["judge_requested_review"]
        : undefined,
      adjudicatorRationale: authoritativeSource
        ? "Preserved the paired source meaning and original Twi; model alternatives remain as independent checks."
        : asString(judge.rationale),
    });
    recordsById.set(row.id, record);
    imported += 1;
  }

  const records: CorpusSynthesis[] = [...recordsById.values()].sort((left, right) => left.row_id.localeCompare(right.row_id));
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, `${records.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  console.log(JSON.stringify({
    schema_version: 1,
    prompt_version: corpusSynthesisPromptVersion,
    raw_rows: rawRows.length,
    imported,
    rejected: rejected.length,
    rejected_reasons: rejected.reduce<Record<string, number>>((acc, row) => {
      acc[row.reason] = (acc[row.reason] ?? 0) + 1;
      return acc;
    }, {}),
    total_synthesis_records: records.length,
    output,
  }, null, 2));
  if (rawRows.length > 0 && imported === 0) process.exit(1);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
