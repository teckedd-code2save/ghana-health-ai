import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import "../src/config/load-env";
import { createMedicalAnnotationClient, MedicalAnnotationProviderError } from "../src/lib/medical-annotation-client";
import { annotationWorkPlan } from "../src/lib/annotation-work-plan";
import {
  buildMedicalResponseAnnotation,
  medicalResponseAnnotationSchema,
  medicalResponseIntents,
  medicalResponseTeacherPromptVersion,
  medicalResponseProposalSchema,
  medicalResponseSourceSchema,
  medicalResponseSynthesisSchema,
  medicalResponseTopics,
  normalizeMedicalModelFields,
  stableMedicalProposalId,
  type MedicalResponseAnnotation,
  type MedicalResponseProposal,
  type MedicalResponseSource,
  type MedicalResponseSynthesis,
} from "../src/lib/medical-response-annotations";

type Json = Record<string, unknown>;
type TeacherRole = "teacher_a" | "teacher_b";

const root = process.cwd();
const defaultInput = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-source.v1.jsonl");
const defaultOutput = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-annotations.v1.jsonl");
const defaultSummary = path.join(root, "data", "medical-response-corpus", "afrihealth-akan-annotations.v1.summary.json");
const activePromptVersion = medicalResponseTeacherPromptVersion;

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

function integerArg(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown) {
  return Array.isArray(value) ? value.map(asString).filter(Boolean) : [];
}

async function readJsonl<T>(filePath: string, parse: (value: unknown) => T) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => parse(JSON.parse(line)));
}

async function writeJsonl(filePath: string, rows: MedicalResponseAnnotation[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tempPath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.rename(tempPath, filePath);
}

const completeJson = createMedicalAnnotationClient(
  argValue("--cache-dir", `${argValue("--out", defaultOutput)}.requests`),
);

const responseShape = {
  rows: [{
    row_id: "source row id",
    question_twi_normalized: "same meaning, spelling normalized only",
    question_english: "faithful natural English meaning",
    answer_english: "faithful English meaning of the source answer",
    intent: medicalResponseIntents.join("|"),
    model_intent_raw: "same value the model originally chose",
    topics: medicalResponseTopics,
    model_topics_raw: ["same topic values the model originally chose"],
    normalization_flags: [],
    entities: {
      symptoms: [], conditions: [], medicines: [], body_parts: [], durations: [],
      quantities: [], persons: [], locations: [], other: [],
    },
    safety_level: "routine|same_day|urgent|emergency|needs_review",
    requires_clarification: false,
    source_answer_assessment: "supportable|needs_minor_edit|needs_expert_review|unsafe_or_incorrect",
    source_answer_issues: [],
    reply_twi: "concise spoken Twi answer grounded only in source answer",
    reply_english: "faithful English meaning of reply_twi",
    evidence_spans_twi: ["one to eight exact substrings copied from source question or answer"],
    confidence: 0.0,
  }],
};

async function runTeacher(rows: MedicalResponseSource[], role: TeacherRole, model: string) {
  const payload = await completeJson([
    {
      role: "system",
      content:
        `You are the ${role === "teacher_a" ? "meaning-first" : "safety-first"} independent Ghanaian Twi corpus annotator for a medical voice assistant. ` +
        "Return JSON only. Preserve the exact source meaning, tense, negation, quantities, uncertainty, and code-switching. " +
        "Interpret the question independently of the answer. The reference answer is not patient history. " +
        "Intent, entities and the question translation must describe only what the question says. " +
        "Every entity value must be an exact text span from the original question; leave unmentioned categories empty. " +
        "Do not copy symptoms, conditions, medicines, people, places or advice from the answer into entities. " +
        "A question about a condition is not evidence the speaker has that condition. " +
        "Do not add medical facts. Do not silently repair a questionable source answer: flag it. " +
        "A fluent answer is not necessarily medically supportable. " +
        (role === "teacher_a"
          ? "Prioritize complete translation and semantic fidelity before fluency."
          : "Independently challenge intent, entities, ambiguity, source-answer correctness, and medical risk."),
    },
    {
      role: "user",
      content: JSON.stringify({
        prompt_version: activePromptVersion,
        independent_role: role,
        instruction:
          "For each immutable Twi question-answer pair, produce faithful English meanings, semantic labels, and a concise natural Twi voice reply of at most four short sentences. The reply must use only claims present in the source answer. Use exact source substrings for evidence. If the answer is unsafe, contradictory, vague, culturally unsuitable, or medically doubtful, mark it for expert review. Never insert an emergency phone number.",
        ontology: { intents: medicalResponseIntents, topics: medicalResponseTopics },
        rows: rows.map((row) => ({
          row_id: row.id,
          question_twi_source: row.question_twi_source,
          answer_twi_source: row.answer_twi_source,
        })),
        response_shape: responseShape,
      }),
    },
  ], model, Math.max(4000, rows.length * 2600));
  const rawRows = Array.isArray(payload.rows) ? payload.rows : [];
  const byId = new Map(rawRows.map((value) => {
    const row = value as Json;
    return [asString(row.row_id), row];
  }));
  const modelId = `openai:${model}`;
  const generatedAt = new Date().toISOString();
  return rows.map((source): MedicalResponseProposal => {
    const raw = byId.get(source.id);
    if (!raw) throw new Error(`${model} omitted row ${source.id}.`);
    return medicalResponseProposalSchema.parse({
      proposal_id: stableMedicalProposalId(source.id, role, modelId, activePromptVersion),
      row_id: source.id,
      agent_role: role,
      model: modelId,
      prompt_version: activePromptVersion,
      generated_at: generatedAt,
      ...normalizeMedicalModelFields(raw),
    });
  });
}

type JudgeRow = Json & {
  selected_proposal?: unknown;
  material_disagreement_fields?: unknown;
  review_reasons?: unknown;
  rationale?: unknown;
};

async function runJudge(
  rows: MedicalResponseSource[],
  teacherA: MedicalResponseProposal[],
  teacherB: MedicalResponseProposal[],
  model: string,
) {
  const payload = await completeJson([
    {
      role: "system",
      content:
        "You are the strict adjudicator for a Ghanaian Twi medical corpus. Return JSON only. " +
        "Compare two independent annotations against the immutable Twi source. Prefer review over false confidence. " +
        "For question meaning, intent and entities, use the question alone, never facts introduced by the answer. " +
        "Every entity must be an exact span of the original question. Correct answer-only entities in a synthesis or request review. " +
        "The answer is reference material for assessment and reply generation, not additional patient history. " +
        "Do not introduce medical claims, diagnoses, doses, or emergency numbers.",
    },
    {
      role: "user",
      content: JSON.stringify({
        prompt_version: activePromptVersion,
        instruction:
          "For each row, select teacher_a or teacher_b only when it is fully faithful and safe. Use synthesized only to combine clearly supported strengths without adding claims. Use needs_human_review when meaning, medical safety, source quality, intent, or evidence remains uncertain. For a synthesized result, populate every annotation field.",
        rows: rows.map((source, index) => ({
          row_id: source.id,
          question_twi_source: source.question_twi_source,
          answer_twi_source: source.answer_twi_source,
          teacher_a: teacherA[index],
          teacher_b: teacherB[index],
        })),
        response_shape: {
          rows: [{
            row_id: "source row id",
            selected_proposal: "teacher_a|teacher_b|synthesized|needs_human_review",
            confidence: 0.0,
            rationale: "short evidence-based reason",
            material_disagreement_fields: [],
            review_reasons: [],
            synthesized: responseShape.rows[0],
          }],
        },
      }),
    },
  ], model, Math.max(5000, rows.length * 3000));
  const rawRows = Array.isArray(payload.rows) ? payload.rows as JudgeRow[] : [];
  const byId = new Map(rawRows.map((row) => [asString(row.row_id), row]));
  const modelId = `openai:${model}`;
  const generatedAt = new Date().toISOString();

  return rows.map((source, index) => {
    const raw = byId.get(source.id);
    if (!raw) throw new Error(`${model} omitted adjudication row ${source.id}.`);
    const selection = asString(raw.selected_proposal);
    const selectedProposalId = selection === "teacher_a"
      ? teacherA[index].proposal_id
      : selection === "teacher_b"
        ? teacherB[index].proposal_id
        : selection === "synthesized"
          ? stableMedicalProposalId(source.id, "adjudicator", modelId, activePromptVersion)
          : "needs_human_review";
    let synthesis: MedicalResponseSynthesis | null = null;
    if (selection === "synthesized") {
      const fields = raw.synthesized as Json | undefined;
      const parsedSynthesis = medicalResponseSynthesisSchema.safeParse({
        proposal_id: selectedProposalId,
        row_id: source.id,
        agent_role: "adjudicator",
        model: modelId,
        prompt_version: activePromptVersion,
        generated_at: generatedAt,
        ...normalizeMedicalModelFields(fields ?? {}),
      });
      if (parsedSynthesis.success) synthesis = parsedSynthesis.data;
    }
    const confidence = typeof raw.confidence === "number" ? raw.confidence : 0;
    const reviewReasons = stringArray(raw.review_reasons);
    if (selection === "synthesized" && !synthesis) reviewReasons.push("invalid_synthesis");
    if (selection === "needs_human_review") reviewReasons.push("adjudicator_requested_review");
    return buildMedicalResponseAnnotation({
      source,
      proposals: [teacherA[index], teacherB[index]],
      synthesis,
      selectedProposalId,
      adjudicatorModel: modelId,
      adjudicatorConfidence: confidence,
      adjudicatorRationale: asString(raw.rationale),
      adjudicatorDisagreements: stringArray(raw.material_disagreement_fields),
      adjudicatorReviewReasons: reviewReasons,
      promptVersion: activePromptVersion,
    });
  });
}

function hashSelect(rows: MedicalResponseSource[]) {
  return [...rows].sort((left, right) => left.record_source_hash.localeCompare(right.record_source_hash));
}

async function main() {
  const input = argValue("--input", defaultInput);
  const output = argValue("--out", defaultOutput);
  const summaryPath = argValue("--summary", defaultSummary);
  const limit = integerArg("--limit", 20);
  const offset = integerArg("--offset", 0);
  const chunkSize = Math.max(1, integerArg("--chunk-size", 2));
  const concurrency = Math.max(1, integerArg("--concurrency", 1));
  const selection = argValue("--selection", "hash");
  const teacherAModel = argValue("--teacher-a", "gpt-5.4-mini");
  const teacherBModel = argValue("--teacher-b", "gpt-5.5");
  const judgeModel = argValue("--judge", "gpt-5.6-sol");
  const force = process.argv.includes("--force");
  const rescoreOnly = process.argv.includes("--rescore-only");
  const trainOnly = process.argv.includes("--train-only");
  const referencePath = argValue("--reference", "");

  const sourceRows = await readJsonl(input, (value) => medicalResponseSourceSchema.parse(value));
  const existing = await fs.readFile(output, "utf8")
    .then((raw) => raw.split("\n").map((line) => line.trim()).filter(Boolean)
      .map((line) => medicalResponseAnnotationSchema.parse(JSON.parse(line))))
    .catch((error): MedicalResponseAnnotation[] => {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    });
  if (!rescoreOnly && existing.some((row) => row.prompt_version !== activePromptVersion ||
    row.adjudication.model !== `openai:${judgeModel}` ||
    row.proposals.some((proposal) => proposal.model !== `openai:${proposal.agent_role === "teacher_a" ? teacherAModel : teacherBModel}`))) {
    throw new Error("The output contains a different annotation pipeline. Choose a new --out file; do not mix prompt versions or models.");
  }
  const sourceById = new Map(sourceRows.map((row) => [row.id, row]));
  for (const row of existing) {
    if (sourceById.get(row.row_id)?.record_source_hash !== row.source_record_hash) {
      throw new Error(`Source changed for saved annotation ${row.row_id}; choose a new output file.`);
    }
  }
  if (rescoreOnly) {
    const computedReasons = new Set([
      "teacher_disagreement",
      "material_teacher_disagreement",
      "invalid_adjudicator_selection",
      "low_confidence",
      "ungrounded_evidence_span",
      "entities_not_grounded_in_question",
      "source_answer_requires_expert_review",
      "unresolved_safety_level",
      "foreign_or_unverified_emergency_number",
      "numeric_claim_not_in_source_answer",
      "high_risk_safety_review",
      "medication_review",
      "vulnerable_population_review",
      "repetitive_reply",
    ]);
    const rescored = existing.map((row) => {
      const source = sourceById.get(row.row_id);
      if (!source) throw new Error(`Missing source row ${row.row_id}.`);
      const proposals: [MedicalResponseProposal, MedicalResponseProposal] = [row.proposals[0], row.proposals[1]];
      return buildMedicalResponseAnnotation({
        source,
        proposals,
        synthesis: row.synthesized_proposal,
        selectedProposalId: row.recommended_proposal_id,
        adjudicatorModel: row.adjudication.model,
        adjudicatorConfidence: row.adjudication.confidence,
        adjudicatorRationale: row.adjudication.rationale,
        adjudicatorDisagreements: row.adjudication.disagreement_fields,
        adjudicatorReviewReasons: row.adjudication.review_reasons.filter((reason) => !computedReasons.has(reason)),
        promptVersion: row.prompt_version,
      });
    });
    await writeJsonl(output, rescored.sort((a, b) => a.row_id.localeCompare(b.row_id)));
    const statuses = rescored.reduce<Record<string, number>>((counts, row) => {
      counts[row.adjudication.status] = (counts[row.adjudication.status] ?? 0) + 1;
      return counts;
    }, {});
    const summary = {
      schema_version: 1,
      prompt_versions: [...new Set(rescored.map((row) => row.prompt_version))].sort(),
      source_rows: sourceRows.length,
      annotations: rescored.length,
      statuses,
      eligible_for_training: rescored.filter((row) => row.eligible_for_training).length,
      rescored_only: true,
      failures: [],
    };
    await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  const records = new Map(existing.map((row) => [row.row_id, row]));
  const trainingRows = trainOnly ? await readJsonl(
    argValue("--train-pool", path.join(root, "data", "medical-response-corpus", "afrihealth-ghana-response-train.v1.jsonl")),
    (value) => value as { language: string; source_record_id: string; source_split: string; eligible_for_research_training: boolean },
  ) : [];
  const trainingKeys = new Set(trainingRows
    .filter((row) => row.language === "tw" && row.source_split === "train" && row.eligible_for_research_training)
    .map((row) => row.source_record_id));
  const poolSources = trainOnly
    ? sourceRows.filter((row) => row.source_split === "train" && trainingKeys.has(row.source_record_id))
    : sourceRows;
  const referenceRows = referencePath
    ? await readJsonl(referencePath, (value) => medicalResponseAnnotationSchema.parse(value))
    : [];
  const referenceIds = new Set(referenceRows.map((row) => row.row_id));
  const eligibleSources = referencePath ? poolSources.filter((row) => referenceIds.has(row.id)) : poolSources;
  if (!eligibleSources.length) throw new Error("No source rows match the requested corpus selection.");
  const ordered = selection === "sequential" ? eligibleSources : hashSelect(eligibleSources);
  const selected = limit > 0 ? ordered.slice(offset, offset + limit) : ordered.slice(offset);
  const pending = selected.filter((row) => force || !records.has(row.id));
  const failures: Array<{ row_ids: string[]; error: string }> = [];
  await fs.mkdir(path.dirname(output), { recursive: true });
  const planned = await annotationWorkPlan({
    file: `${output}.plan.json`, cacheDirectory: argValue("--cache-dir", `${output}.requests`),
    signature: crypto.createHash("sha256").update(JSON.stringify({
      prompt: activePromptVersion, teacherAModel, teacherBModel, judgeModel,
      sources: selected.map((row) => [row.id, row.record_source_hash]),
    })).digest("hex"),
    selectedIds: selected.map((row) => row.id), pendingIds: pending.map((row) => row.id), chunkSize, force,
  });
  const pendingIds = new Set(pending.map((row) => row.id));
  const chunks = planned.filter((ids) => ids.some((id) => pendingIds.has(id)))
    .map((ids) => ids.map((id) => sourceById.get(id)!));
  if (process.argv.includes("--dry-run")) {
    console.log(JSON.stringify({ selected: selected.length, saved: records.size, pending: pending.length, planned_requests: chunks.length, paid_requests_made: 0 }));
    return;
  }
  let nextChunk = 0;
  let completed = 0;
  let saveQueue = Promise.resolve();
  let fatalError: Error | null = null;

  async function processChunk(chunk: MedicalResponseSource[]): Promise<void> {
    if (fatalError || (!force && chunk.every((row) => records.has(row.id)))) return;
    try {
      const teacherA = await runTeacher(chunk, "teacher_a", teacherAModel);
      if (fatalError) return;
      const teacherB = await runTeacher(chunk, "teacher_b", teacherBModel);
      if (fatalError) return;
      const annotations = await runJudge(chunk, teacherA, teacherB, judgeModel);
      for (const annotation of annotations) {
        if (force || !records.has(annotation.row_id)) {
          records.set(annotation.row_id, annotation);
          completed += 1;
        }
      }
      saveQueue = saveQueue.then(() =>
        writeJsonl(output, [...records.values()].sort((a, b) => a.row_id.localeCompare(b.row_id))),
      );
      await saveQueue;
      console.log(JSON.stringify({ completed, requested: pending.length, rows_saved: records.size }));
    } catch (error) {
      if (error instanceof MedicalAnnotationProviderError) {
        fatalError = error;
        failures.push({ row_ids: chunk.map((row) => row.id), error: error.message });
        return;
      }
      if (chunk.length > 1) {
        const middle = Math.ceil(chunk.length / 2);
        await processChunk(chunk.slice(0, middle));
        await processChunk(chunk.slice(middle));
        return;
      }
      failures.push({
        row_ids: chunk.map((row) => row.id),
        error: error instanceof Error ? error.message : String(error),
      });
      console.error(JSON.stringify(failures.at(-1)));
    }
  }

  async function worker() {
    while (!fatalError && nextChunk < chunks.length) {
      const chunk = chunks[nextChunk++];
      await processChunk(chunk);
    }
  }
  // Verify all three provider stages before starting the parallel batch.
  if (chunks.length) await processChunk(chunks[nextChunk++]);
  if (!fatalError) {
    await Promise.all(Array.from({ length: Math.min(concurrency, Math.max(1, chunks.length - nextChunk)) }, () => worker()));
  }
  await saveQueue;

  const annotations = [...records.values()].sort((a, b) => a.row_id.localeCompare(b.row_id));
  await writeJsonl(output, annotations);
  const statuses = annotations.reduce<Record<string, number>>((counts, row) => {
    counts[row.adjudication.status] = (counts[row.adjudication.status] ?? 0) + 1;
    return counts;
  }, {});
  const summary = {
    schema_version: 1,
    prompt_version: activePromptVersion,
    prompt_versions: [...new Set(annotations.map((row) => row.prompt_version))].sort(),
    source_rows: sourceRows.length,
    annotations: annotations.length,
    statuses,
    eligible_for_training: annotations.filter((row) => row.eligible_for_training).length,
    teacher_a: teacherAModel,
    teacher_b: teacherBModel,
    adjudicator: judgeModel,
    concurrency,
    train_only: trainOnly,
    reference_path: referencePath || null,
    eligible_source_rows: eligibleSources.length,
    requested: pending.length,
    completed,
    interrupted_by_provider: Boolean(fatalError),
    failures,
  };
  await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(summary, null, 2));
  if (failures.length) process.exitCode = 1;
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
