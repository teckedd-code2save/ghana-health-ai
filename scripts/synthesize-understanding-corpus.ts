import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";
import { researchIntentOntology } from "../src/lib/research-intents";
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
type AgentRole = "translator" | "semantic_annotator";

const root = process.cwd();
const defaultOut = path.join(root, "tmp", "understanding-corpus", "synthesis.v1.jsonl");
const promptVersion = corpusSynthesisPromptVersion;

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function numericArgValue(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : fallback;
}

function argSet(name: string) {
  return new Set(
    argValue(name)
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function compactJson(value: unknown) {
  if (typeof value === "string") return value.trim();
  if (value === undefined || value === null) return "{}";
  return JSON.stringify(value);
}

function stringArray(value: unknown) {
  return Array.isArray(value) ? value.map(asString).filter(Boolean) : [];
}

function isComputedReviewReason(value: string) {
  return value.startsWith("disagreement:") || [
    "low_recommendation_score",
    "close_candidate_scores",
    "missing_required_field",
    "invalid_intent",
    "intent_ontology_mismatch",
    "invalid_entities",
    "entity_type_conflict",
    "entity_value_misalignment",
    "incomplete_grounded_response",
    "ungrounded_response_added",
    "low_source_alignment",
    "low_twi_source_alignment",
  ].includes(value);
}

function parseJsonPayload(content: string): Json | null {
  const fenced = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const cleaned = (fenced?.[1] ?? content).trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(cleaned.slice(start, end + 1)) as Json;
  } catch {
    return null;
  }
}

function safeSafetyLevel(value: unknown): AnnotationProposal["safety_level"] {
  return value === "routine" || value === "same_day" || value === "urgent" || value === "emergency"
    ? value
    : "";
}

function proposalId(rowId: string, role: string, model: string) {
  return crypto.createHash("sha256").update(`${promptVersion}:${rowId}:${role}:${model}`).digest("hex").slice(0, 20);
}

async function chatWithRetries(
  messages: Parameters<typeof chatComplete>[0],
  options: NonNullable<Parameters<typeof chatComplete>[1]>,
) {
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const content = await chatComplete(messages, options);
    if (content) return content;
    if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
  }
  return null;
}

function proposalFromJson(input: {
  row: CorpusCandidate;
  json: Json;
  role: AgentRole | "adjudicator";
  model: string;
}): AnnotationProposal {
  return annotationProposalSchema.parse({
    proposal_id: proposalId(input.row.id, input.role, input.model),
    row_id: input.row.id,
    agent_role: input.role,
    model: input.model,
    prompt_version: promptVersion,
    generated_at: new Date().toISOString(),
    normalized_twi: asString(input.json.normalized_twi) || input.row.text,
    natural_english: asString(input.json.natural_english),
    literal_english: asString(input.json.literal_english),
    intent: asString(input.json.intent),
    entities: compactJson(input.json.entities),
    ambiguities: compactJson(input.json.ambiguities),
    reply_twi: asString(input.json.reply_twi),
    safety_level: safeSafetyLevel(input.json.safety_level),
    requires_clarification: input.json.requires_clarification === true,
  });
}

function sourceProposal(row: CorpusCandidate): AnnotationProposal | null {
  const source = row.model_proposal;
  if (source.status !== "draft" || !source.natural_english.trim()) return null;
  return annotationProposalSchema.parse({
    proposal_id: proposalId(row.id, "source", source.model || row.source),
    row_id: row.id,
    agent_role: "source",
    model: source.model || `source:${row.source}`,
    prompt_version: "source-import-v1",
    generated_at: new Date(0).toISOString(),
    normalized_twi: source.normalized_twi || row.text,
    natural_english: source.natural_english,
    literal_english: source.literal_english,
    intent: source.intent,
    entities: source.entities || "{}",
    ambiguities: semanticAmbiguities(source.ambiguities),
    reply_twi: source.reply_twi,
    safety_level: safeSafetyLevel(source.safety_level),
    requires_clarification: source.requires_clarification,
  });
}

function generationInstruction(role: AgentRole) {
  if (role === "translator") {
    return "Act as a Ghanaian Twi translation specialist. Preserve the utterance's exact meaning, tense, negation, quantities, code-switching, uncertainty, and discourse references. Prefer natural Ghanaian English meaning over a word-for-word guess.";
  }
  return "Act as a Ghanaian health and commerce semantic annotator. Focus on the correct intent, entities, negation, time, quantity, location, symptom and body-part references. Do not diagnose or infer facts that are absent.";
}

async function generateAgentChunk(
  rows: CorpusCandidate[],
  role: AgentRole,
  model: string,
) {
  const provider = llmProviderInfo(model);
  if (!provider) throw new Error("No LLM provider configured.");
  const modelId = `${provider.provider}:${provider.model}`;
  const content = await chatWithRetries(
    [
      {
        role: "system",
        content: `${generationInstruction(role)} Return valid JSON only. These are candidate annotations for adjudication, not gold labels. The original utterance is immutable evidence: normalized_twi may correct punctuation, spacing, and obvious orthography, but must not add, remove, or duplicate meaning-bearing words. When uncertain, preserve the original wording. Never add a medical response unless source_reply_twi is present; when it is present, preserve its medical claims and urgency without adding diagnosis, dosage, medicine names, or facilities.`,
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction: "Produce one complete annotation bundle per row. Keep every id unchanged.",
          allowed_intents: researchIntentOntology,
          rows: rows.map((row) => ({
            id: row.id,
            source: row.source,
            domain: row.domain,
            language: row.language,
            utterance: row.text,
            source_reply_twi: row.model_proposal.reply_twi,
            source_safety_level: row.model_proposal.safety_level,
          })),
          response_shape: {
            rows: [{
              id: "same id",
              normalized_twi: "clean natural Twi",
              natural_english: "faithful natural English meaning",
              literal_english: "literal gloss when useful",
              intent: "one value from allowed_intents; put symptom detail in entities",
              entities: { symptom: [], body_part: [], product: [], quantity: [], location: [], time: [], negated: [] },
              ambiguities: "short uncertainty note or empty string",
              reply_twi: "only a rendering of source_reply_twi, otherwise empty",
              safety_level: "only when source safety is present: routine|same_day|urgent|emergency",
              requires_clarification: false,
            }],
          },
        }),
      },
    ],
    { model, allowFallback: false, temperature: role === "translator" ? 0.1 : 0.2, maxTokens: Math.max(1400, rows.length * 520) },
  );
  if (!content) throw new Error(`${modelId} returned no annotation content.`);
  const payload = parseJsonPayload(content);
  const generatedRows = Array.isArray(payload?.rows) ? payload.rows as Json[] : [];
  const byId = new Map(generatedRows.map((row) => [asString(row.id), row]));
  return new Map(
    rows.flatMap((row) => {
      const generated = byId.get(row.id);
      return generated ? [[row.id, proposalFromJson({ row, json: generated, role, model: modelId })]] : [];
    }),
  );
}

async function adjudicateChunk(
  rows: CorpusCandidate[],
  proposalsByRow: Map<string, AnnotationProposal[]>,
  model: string,
) {
  const provider = llmProviderInfo(model);
  if (!provider) throw new Error("No adjudicator provider configured.");
  const modelId = `${provider.provider}:${provider.model}`;
  const content = await chatWithRetries(
    [
      {
        role: "system",
        content:
          "You adjudicate alternative Twi corpus annotations. Return valid JSON only. Select the most faithful complete bundle or synthesize a correction. A paired source English reference is evidence, not authority: flag a material conflict when it adds, removes, or contradicts meaning in the Twi. If both independent translations contradict the paired reference, do not select the paired reference unchanged. Do not favor a proposal because it came from the same model family as you. The original utterance is immutable evidence: normalized_twi may correct punctuation, spacing, and obvious orthography, but must not add, remove, or duplicate meaning-bearing words. If uncertain, preserve the original. Do not invent a medical response; reply_twi and safety_level must remain empty unless the source proposal already contains them. A material disagreement means contradictory, missing, added, or genuinely uncertain meaning; wording differences and compatible levels of detail are not material disagreements. List every material disagreement field and keep confidence below 0.85 when proposals materially disagree.",
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction: "For each row, compare all proposals and return a final coherent bundle, rationale, and whether you selected one unchanged or synthesized a correction.",
          allowed_intents: researchIntentOntology,
          rows: rows.map((row) => ({
            id: row.id,
            utterance: row.text,
            domain: row.domain,
            paired_source_english_reference: row.model_proposal.natural_english,
            source_reply_twi: row.model_proposal.reply_twi,
            proposals: proposalsByRow.get(row.id) ?? [],
          })),
          response_shape: {
            rows: [{
              id: "same id",
              selected_proposal_id: "proposal id or synthesized",
              confidence: 0.0,
              material_disagreement_fields: ["only fields whose meanings materially conflict"],
              review_reasons: ["specific reason human review is required, otherwise empty"],
              rationale: "brief evidence-based reason",
              normalized_twi: "final Twi",
              natural_english: "final meaning",
              literal_english: "final gloss",
              intent: "one value from allowed_intents; put symptom detail in entities",
              entities: {},
              ambiguities: "remaining uncertainty",
              reply_twi: "source-grounded only or empty",
              safety_level: "source-grounded only or empty",
              requires_clarification: false,
            }],
          },
        }),
      },
    ],
    { model, allowFallback: false, temperature: 0, maxTokens: Math.max(1600, rows.length * 600) },
  );
  if (!content) throw new Error(`${modelId} returned no adjudication content.`);
  const payload = parseJsonPayload(content);
  const generatedRows = Array.isArray(payload?.rows) ? payload.rows as Json[] : [];
  const byId = new Map(generatedRows.map((row) => [asString(row.id), row]));
  return {
    modelId,
    rows: new Map(rows.flatMap((row) => {
      const generated = byId.get(row.id);
      if (!generated) return [];
      return [[row.id, {
        proposal: proposalFromJson({ row, json: generated, role: "adjudicator", model: modelId }),
        selectedProposalId: asString(generated.selected_proposal_id),
        confidence: typeof generated.confidence === "number" ? generated.confidence : undefined,
        disagreementFields: Array.isArray(generated.material_disagreement_fields)
          ? stringArray(generated.material_disagreement_fields)
          : undefined,
        reviewReasons: Array.isArray(generated.review_reasons)
          ? stringArray(generated.review_reasons)
          : undefined,
        rationale: asString(generated.rationale),
      }]];
    })),
  };
}

async function readExisting(filePath: string) {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => corpusSynthesisSchema.parse(JSON.parse(line)));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function writeRecords(filePath: string, records: CorpusSynthesis[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tempPath, `${records.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.rename(tempPath, filePath);
}

async function main() {
  const output = argValue("--out", defaultOut);
  const sources = argSet("--source");
  const models = argValue("--models", "gpt-5.6-sol,gpt-5.4-mini").split(",").map((value) => value.trim()).filter(Boolean);
  if (models.length < 2) throw new Error("Use at least two independent proposal models with --models.");
  const adjudicatorModel = argValue("--adjudicator", models[0]!);
  const chunkSize = Math.max(1, Math.min(10, numericArgValue("--chunk-size", 4)));
  const maxNew = numericArgValue("--max-new", 20);
  const offset = numericArgValue("--offset", 0);
  const force = process.argv.includes("--force");
  const candidates = await readCorpusCandidates();
  const existing = await readExisting(output);
  const existingById = new Map(existing.map((row) => [row.row_id, row]));
  if (process.argv.includes("--rescore-only")) {
    const candidateById = new Map(candidates.map((row) => [row.id, row]));
    const rescored = existing.flatMap((record) => {
      const row = candidateById.get(record.row_id);
      if (!row) return [];
      return [buildSynthesisRecord({
        rowId: row.id,
        sourceHash: row.source_hash,
        promptVersion: record.prompt_version,
        proposals: record.proposals,
        sourceEnglish: row.model_proposal.natural_english,
        sourceTwi: row.text,
        responseExpected: Boolean(row.model_proposal.reply_twi && row.model_proposal.safety_level),
        synthesizedProposal: record.synthesized_proposal,
        adjudicatorModel: record.adjudication.model,
        adjudicatorSelectedProposalId: record.synthesized_proposal
          ? record.synthesized_proposal.proposal_id
          : record.adjudication.selected_proposal_id ?? undefined,
        adjudicatorConfidence: record.adjudication.confidence,
        adjudicatorDisagreementFields: record.adjudication.disagreement_fields,
        adjudicatorReviewReasons: record.adjudication.review_reasons.filter((reason) => !isComputedReviewReason(reason)),
        adjudicatorRationale: record.adjudication.rationale,
      })];
    });
    await writeRecords(output, rescored.sort((left, right) => left.row_id.localeCompare(right.row_id)));
    console.log(JSON.stringify({ rescored: rescored.length, output }, null, 2));
    return;
  }
  const eligible = candidates.filter((row) =>
    (sources.size === 0 || sources.has(row.source)) &&
    row.model_proposal.status === "draft" &&
    row.model_proposal.natural_english.trim(),
  );
  const targets = eligible.filter((row) => {
    const prior = existingById.get(row.id);
    return force || !prior || prior.source_hash !== row.source_hash || prior.prompt_version !== promptVersion;
  }).slice(offset, offset + maxNew);

  let completed = 0;
  let failedChunks = 0;
  for (let start = 0; start < targets.length; start += chunkSize) {
    const chunk = targets.slice(start, start + chunkSize);
    try {
      const proposalsByRow = new Map<string, AnnotationProposal[]>();
      for (const row of chunk) {
        const source = sourceProposal(row);
        proposalsByRow.set(row.id, source ? [source] : []);
      }
      const generated = [
        await generateAgentChunk(chunk, "translator", models[0]!),
        await generateAgentChunk(chunk, "semantic_annotator", models[1]!),
      ];
      for (const result of generated) {
        for (const [rowId, proposal] of result) proposalsByRow.get(rowId)?.push(proposal);
      }
      const adjudicated = await adjudicateChunk(chunk, proposalsByRow, adjudicatorModel);
      for (const row of chunk) {
        const proposals = proposalsByRow.get(row.id) ?? [];
        if (proposals.length < 2) continue;
        const adjudication = adjudicated.rows.get(row.id);
        const authoritativeSource = hasAuthoritativeSourceMeaning(row.source, row.model_proposal.model);
        const sourceGroundedSynthesis = authoritativeSource
          ? proposalFromJson({
              row,
              role: "adjudicator",
              model: `${adjudicated.modelId}:source-grounded-synthesis`,
              json: {
                normalized_twi: row.text,
                natural_english: row.model_proposal.natural_english,
                literal_english: row.model_proposal.literal_english,
                intent: row.model_proposal.intent,
                entities: row.model_proposal.entities || "{}",
                ambiguities: semanticAmbiguities(row.model_proposal.ambiguities),
                reply_twi: row.model_proposal.reply_twi,
                safety_level: row.model_proposal.safety_level,
                requires_clarification: row.model_proposal.requires_clarification,
              },
            })
          : null;
        const record = buildSynthesisRecord({
          rowId: row.id,
          sourceHash: row.source_hash,
          promptVersion,
          proposals,
          sourceEnglish: row.model_proposal.natural_english,
          sourceTwi: row.text,
          responseExpected: Boolean(row.model_proposal.reply_twi && row.model_proposal.safety_level),
          synthesizedProposal: sourceGroundedSynthesis ?? (
            adjudication?.selectedProposalId === "synthesized" ? adjudication.proposal : null
          ),
          adjudicatorModel: adjudicated.modelId,
          adjudicatorSelectedProposalId: adjudication?.selectedProposalId,
          adjudicatorConfidence: adjudication?.confidence,
          adjudicatorDisagreementFields: adjudication?.disagreementFields,
          adjudicatorReviewReasons: adjudication?.reviewReasons,
          adjudicatorRationale: authoritativeSource
            ? "Preserved the paired source meaning and original Twi; model alternatives remain as independent checks."
            : adjudication?.rationale,
        });
        existingById.set(row.id, record);
        completed += 1;
      }
      await writeRecords(output, [...existingById.values()].sort((left, right) => left.row_id.localeCompare(right.row_id)));
      console.log(JSON.stringify({ chunk: `${start + 1}-${Math.min(start + chunkSize, targets.length)}`, completed, output }));
    } catch (error) {
      failedChunks += 1;
      console.error(JSON.stringify({
        chunk: `${start + 1}-${Math.min(start + chunkSize, targets.length)}`,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }

  const records = [...existingById.values()];
  console.log(JSON.stringify({
    schema_version: 1,
    prompt_version: promptVersion,
    candidate_rows: candidates.length,
    eligible_rows: eligible.length,
    selected_rows: targets.length,
    newly_synthesized: completed,
    failed_chunks: failedChunks,
    total_synthesized: records.length,
    needs_human_review: records.filter((row) => row.adjudication.status === "needs_human_review").length,
    output,
    proposal_models: models,
    adjudicator_model: adjudicatorModel,
  }, null, 2));
  if (targets.length > 0 && completed === 0) process.exit(1);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
