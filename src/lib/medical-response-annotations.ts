import crypto from "node:crypto";
import { z } from "zod";

export const medicalResponsePromptVersion = "afrihealth-akan-response-v1";
export const medicalResponseOpenPromptVersion = "afrihealth-akan-response-open-v2";
export const medicalResponsePivotPromptVersion = "afrihealth-akan-response-pivot-v1";
export const medicalResponseTeacherPromptVersion = "afrihealth-akan-response-teacher-v3";

export const medicalResponseIntents = [
  "health_information_request",
  "health_symptom_report",
  "health_prevention_question",
  "health_testing_question",
  "health_treatment_question",
  "health_medication_question",
  "health_service_question",
  "health_personal_safety_question",
  "health_emergency_report",
  "health_followup",
  "unclear_or_out_of_scope",
] as const;

export const medicalResponseTopics = [
  "sexual_reproductive_health",
  "maternal_pregnancy",
  "child_adolescent_health",
  "infectious_disease",
  "mental_health",
  "violence_abuse_consent",
  "substance_use",
  "nutrition",
  "medication",
  "health_services",
  "general_health",
  "other",
] as const;

export const medicalResponseSourceSchema = z.object({
  schema_version: z.literal(1),
  id: z.string().min(1),
  source_dataset: z.string().min(1),
  source_revision: z.string().min(1),
  source_split: z.enum(["train", "validation"]),
  source_record_id: z.string().min(1),
  language: z.literal("tw"),
  domain: z.literal("health"),
  question_twi_source: z.string().min(1),
  answer_twi_source: z.string().min(1),
  question_source_hash: z.string().min(1),
  answer_source_hash: z.string().min(1),
  record_source_hash: z.string().min(1),
});

export const medicalEntitiesSchema = z.object({
  symptoms: z.array(z.string()).default([]),
  conditions: z.array(z.string()).default([]),
  medicines: z.array(z.string()).default([]),
  body_parts: z.array(z.string()).default([]),
  durations: z.array(z.string()).default([]),
  quantities: z.array(z.string()).default([]),
  persons: z.array(z.string()).default([]),
  locations: z.array(z.string()).default([]),
  other: z.array(z.string()).default([]),
});

export function medicalQuestionEntityViolations(question: string, entities: z.infer<typeof medicalEntitiesSchema>) {
  const normalize = (value: string) => value.normalize("NFKC").toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const questionText = ` ${normalize(question)} `;
  return Object.entries(entities).flatMap(([category, values]) => values
    .filter((value) => !normalize(value) || !questionText.includes(` ${normalize(value)} `))
    .map((value) => ({ category, value })));
}

const annotationFields = {
  question_twi_normalized: z.string().min(1).max(1200),
  question_english: z.string().min(1).max(1800),
  answer_english: z.string().min(1).max(6000),
  intent: z.enum(medicalResponseIntents),
  model_intent_raw: z.string().default(""),
  topics: z.array(z.enum(medicalResponseTopics)).min(1),
  model_topics_raw: z.array(z.string()).default([]),
  normalization_flags: z.array(z.string()).default([]),
  entities: medicalEntitiesSchema,
  safety_level: z.enum(["routine", "same_day", "urgent", "emergency", "needs_review"]),
  requires_clarification: z.boolean(),
  source_answer_assessment: z.enum([
    "supportable",
    "needs_minor_edit",
    "needs_expert_review",
    "unsafe_or_incorrect",
  ]),
  source_answer_issues: z.array(z.string()),
  reply_twi: z.string().min(1).max(1400),
  reply_english: z.string().min(1).max(1800),
  evidence_spans_twi: z.array(z.string()).min(1).max(8),
  confidence: z.number().min(0).max(1),
};

export const medicalResponseProposalSchema = z.object({
  proposal_id: z.string().min(1),
  row_id: z.string().min(1),
  agent_role: z.enum(["teacher_a", "teacher_b"]),
  model: z.string().min(1),
  prompt_version: z.string().min(1),
  generated_at: z.string().min(1),
  ...annotationFields,
});

export const medicalResponseSynthesisSchema = z.object({
  proposal_id: z.string().min(1),
  row_id: z.string().min(1),
  agent_role: z.literal("adjudicator"),
  model: z.string().min(1),
  prompt_version: z.string().min(1),
  generated_at: z.string().min(1),
  ...annotationFields,
});

export const medicalResponseAnnotationSchema = z.object({
  schema_version: z.literal(1),
  row_id: z.string().min(1),
  source_record_hash: z.string().min(1),
  prompt_version: z.string().min(1),
  generated_at: z.string().min(1),
  proposals: z.array(medicalResponseProposalSchema).length(2),
  synthesized_proposal: medicalResponseSynthesisSchema.nullable(),
  recommended_proposal_id: z.string().min(1),
  adjudication: z.object({
    model: z.string().min(1),
    confidence: z.number().min(0).max(1),
    status: z.enum(["silver_consensus", "needs_human_review", "rejected"]),
    rationale: z.string(),
    disagreement_fields: z.array(z.string()),
    review_reasons: z.array(z.string()),
  }),
  eligible_for_training: z.boolean(),
  eligible_for_final_evaluation: z.literal(false),
});

export type MedicalResponseSource = z.infer<typeof medicalResponseSourceSchema>;
export type MedicalResponseProposal = z.infer<typeof medicalResponseProposalSchema>;
export type MedicalResponseSynthesis = z.infer<typeof medicalResponseSynthesisSchema>;
export type MedicalResponseAnnotation = z.infer<typeof medicalResponseAnnotationSchema>;

export function medicalAnnotationPipelineId(annotation: MedicalResponseAnnotation) {
  const profile = {
    prompt_version: annotation.prompt_version,
    teachers: annotation.proposals.map((proposal) => ({ role: proposal.agent_role, model: proposal.model }))
      .sort((left, right) => left.role.localeCompare(right.role)),
    adjudicator: annotation.adjudication.model,
  };
  return crypto.createHash("sha256").update(JSON.stringify(profile)).digest("hex");
}
type MedicalIntent = (typeof medicalResponseIntents)[number];
type MedicalTopic = (typeof medicalResponseTopics)[number];

type ProposalFields = Pick<
  MedicalResponseProposal,
  | "question_english"
  | "answer_english"
  | "intent"
  | "topics"
  | "safety_level"
  | "source_answer_assessment"
  | "reply_twi"
  | "evidence_spans_twi"
>;

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown) {
  return Array.isArray(value) ? value.map(asString).filter(Boolean) : [];
}

const intentSet = new Set<string>(medicalResponseIntents);
const topicSet = new Set<string>(medicalResponseTopics);

function normalizeIntent(value: unknown): { value: MedicalIntent; flag?: string } {
  const raw = asString(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  if (intentSet.has(raw)) return { value: raw as MedicalIntent };
  if (/emergency|crisis/.test(raw)) return { value: "health_emergency_report", flag: "intent_alias_normalized" };
  if (/medicine|medication|drug|dose|pharmacy/.test(raw)) {
    return { value: "health_medication_question", flag: "intent_alias_normalized" };
  }
  if (/symptom|pain|fever|feeling_unwell|illness_report/.test(raw)) {
    return { value: "health_symptom_report", flag: "intent_alias_normalized" };
  }
  if (/prevent|avoid|risk_reduction/.test(raw)) {
    return { value: "health_prevention_question", flag: "intent_alias_normalized" };
  }
  if (/test|screen|diagnos/.test(raw)) {
    return { value: "health_testing_question", flag: "intent_alias_normalized" };
  }
  if (/treat|management|manage|therapy|care_advice/.test(raw)) {
    return { value: "health_treatment_question", flag: "intent_alias_normalized" };
  }
  if (/service|facility|clinic|hospital|access|provider/.test(raw)) {
    return { value: "health_service_question", flag: "intent_alias_normalized" };
  }
  if (/safety|consent|abuse|violence|assault|relationship/.test(raw)) {
    return { value: "health_personal_safety_question", flag: "intent_alias_normalized" };
  }
  if (/follow_?up|continuation|context/.test(raw)) {
    return { value: "health_followup", flag: "intent_alias_normalized" };
  }
  if (/information|general|education|sexual|reproductive|pregnan|maternal|adolescent|mental|nutrition/.test(raw)) {
    return { value: "health_information_request", flag: "intent_alias_normalized" };
  }
  if (/unclear|unknown|out_of_scope/.test(raw)) return { value: "unclear_or_out_of_scope" };
  return { value: "unclear_or_out_of_scope", flag: "intent_out_of_ontology" };
}

function normalizeTopics(value: unknown): { values: MedicalTopic[]; raw: string[]; flags: string[] } {
  const raw = stringArray(value);
  const values = [...new Set(raw.filter((topic) => topicSet.has(topic)))] as MedicalTopic[];
  const flags: string[] = [];
  if (values.length !== raw.length) flags.push(values.length ? "topics_filtered_to_ontology" : "topics_out_of_ontology");
  return { values: values.length ? values : ["other"], raw, flags };
}

function normalizeSafety(value: unknown) {
  const raw = asString(value).toLowerCase();
  if (["routine", "same_day", "urgent", "emergency", "needs_review"].includes(raw)) return raw;
  if (/non.?urgent|self.?care|low/.test(raw)) return "routine";
  if (/same.?day|soon|moderate/.test(raw)) return "same_day";
  if (/high|immediate/.test(raw)) return "urgent";
  return "needs_review";
}

function normalizeAssessment(value: unknown) {
  const raw = asString(value).toLowerCase();
  if (["supportable", "needs_minor_edit", "needs_expert_review", "unsafe_or_incorrect"].includes(raw)) return raw;
  if (/unsafe|incorrect|harm/.test(raw)) return "unsafe_or_incorrect";
  if (/expert|review|uncertain/.test(raw)) return "needs_expert_review";
  if (/minor|edit/.test(raw)) return "needs_minor_edit";
  return "needs_expert_review";
}

export function normalizeMedicalModelFields(value: unknown) {
  const raw = typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
  const intentRaw = asString(raw.intent);
  const intent = normalizeIntent(intentRaw);
  const topics = normalizeTopics(raw.topics);
  return {
    ...raw,
    intent: intent.value,
    model_intent_raw: intentRaw,
    topics: topics.values,
    model_topics_raw: topics.raw,
    normalization_flags: [...topics.flags, ...(intent.flag ? [intent.flag] : [])],
    safety_level: normalizeSafety(raw.safety_level),
    source_answer_assessment: normalizeAssessment(raw.source_answer_assessment),
    requires_clarification: raw.requires_clarification === true,
    confidence: typeof raw.confidence === "number" ? raw.confidence : Number(raw.confidence) || 0,
    entities: medicalEntitiesSchema.parse(raw.entities ?? {}),
    source_answer_issues: stringArray(raw.source_answer_issues),
    evidence_spans_twi: stringArray(raw.evidence_spans_twi).slice(0, 8),
  };
}

function normalizedTokens(value: string) {
  return new Set(
    value
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter((token) => token.length > 1),
  );
}

function tokenSimilarity(left: string, right: string) {
  const a = normalizedTokens(left);
  const b = normalizedTokens(right);
  const union = new Set([...a, ...b]);
  if (!union.size) return 0;
  let overlap = 0;
  for (const token of a) if (b.has(token)) overlap += 1;
  return overlap / union.size;
}

function numberTokens(value: string) {
  return new Set(value.match(/\b\d+(?:\.\d+)?\b/g) ?? []);
}

function entityValuesDisagree(left: string[], right: string[]) {
  if (left.length === 0 || right.length === 0) return left.length !== right.length;
  return !left.some((leftValue) => right.some((rightValue) => tokenSimilarity(leftValue, rightValue) >= 0.34));
}

function hasRepeatedPhrase(value: string) {
  const sentences = value
    .toLowerCase()
    .split(/[.!?]+/)
    .map((sentence) => sentence.replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim())
    .filter((sentence) => sentence.split(" ").length >= 4);
  if (new Set(sentences).size !== sentences.length) return true;

  const tokens = value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
  const phrases = new Set<string>();
  for (let index = 0; index <= tokens.length - 6; index += 1) {
    const phrase = tokens.slice(index, index + 6).join(" ");
    if (phrases.has(phrase)) return true;
    phrases.add(phrase);
  }
  return false;
}

function sourceNeedsVulnerablePopulationReview(source: MedicalResponseSource, topics: readonly string[]) {
  const sourceText = `${source.question_twi_source}\n${source.answer_twi_source}`;
  const pregnancyMentioned = topics.includes("maternal_pregnancy") &&
    /(?:nyins[ɛe]n|pregnan|awoɔ? mu|awo bere|awo ber[ɛe])/i.test(sourceText);
  const youngChildMentioned = topics.includes("child_adolescent_health") &&
    /(?:akokoaa|abofra|newborn|infant|baby|toddler|young child|onnii? asram)/i.test(sourceText);
  return pregnancyMentioned || youngChildMentioned;
}

function evidenceIsGrounded(source: MedicalResponseSource, spans: string[]) {
  const evidence = `${source.question_twi_source}\n${source.answer_twi_source}`.toLowerCase();
  return spans.every((span) => span.trim().length >= 2 && evidence.includes(span.trim().toLowerCase()));
}

export function stableMedicalProposalId(
  rowId: string,
  role: string,
  model: string,
  promptVersion = medicalResponsePromptVersion,
) {
  return crypto
    .createHash("sha256")
    .update(`${promptVersion}:${rowId}:${role}:${model}`)
    .digest("hex")
    .slice(0, 20);
}

export function buildMedicalResponseAnnotation(input: {
  source: MedicalResponseSource;
  proposals: [MedicalResponseProposal, MedicalResponseProposal];
  synthesis: MedicalResponseSynthesis | null;
  selectedProposalId: string;
  adjudicatorModel: string;
  adjudicatorConfidence: number;
  adjudicatorRationale: string;
  adjudicatorDisagreements: string[];
  adjudicatorReviewReasons: string[];
  promptVersion?: string;
}): MedicalResponseAnnotation {
  const [left, right] = input.proposals;
  const selected = input.synthesis?.proposal_id === input.selectedProposalId
    ? input.synthesis
    : input.proposals.find((proposal) => proposal.proposal_id === input.selectedProposalId);
  const final = selected ?? input.synthesis ?? left;
  const disagreements = new Set(input.adjudicatorDisagreements);
  const criticalDisagreements = new Set<string>();
  const reviewReasons = new Set(input.adjudicatorReviewReasons);

  const markCritical = (field: string) => {
    disagreements.add(field);
    criticalDisagreements.add(field);
  };
  const markDisagreement = (field: string) => disagreements.add(field);
  if (left.intent !== right.intent) markCritical("intent");
  if (left.safety_level !== right.safety_level) markCritical("safety_level");
  if (left.requires_clarification !== right.requires_clarification) markCritical("requires_clarification");
  if (left.source_answer_assessment !== right.source_answer_assessment) {
    markDisagreement("source_answer_assessment");
    if ([left.source_answer_assessment, right.source_answer_assessment].includes("unsafe_or_incorrect")) {
      criticalDisagreements.add("source_answer_assessment");
    }
  }
  if (tokenSimilarity(left.question_english, right.question_english) < 0.45) markCritical("question_english");
  if (tokenSimilarity(left.answer_english, right.answer_english) < 0.35) markCritical("answer_english");
  if (!left.topics.some((topic) => right.topics.includes(topic))) markCritical("topics");
  for (const category of Object.keys(left.entities) as Array<keyof typeof left.entities>) {
    if (entityValuesDisagree(left.entities[category], right.entities[category])) {
      markCritical(`entities.${category}`);
    }
  }
  const adjudicatorRequestedReview = reviewReasons.has("adjudicator_requested_review");
  if (!selected && !adjudicatorRequestedReview) reviewReasons.add("invalid_adjudicator_selection");
  if (input.adjudicatorConfidence < 0.9 || final.confidence < 0.85) reviewReasons.add("low_confidence");
  const unresolvedSafetyConflict = criticalDisagreements.has("safety_level") ||
    criticalDisagreements.has("source_answer_assessment");
  const unresolvedSemanticConflict = criticalDisagreements.size > 0 && input.adjudicatorConfidence < 0.95;
  if (!unresolvedSafetyConflict && !unresolvedSemanticConflict) reviewReasons.delete("teacher_disagreement");
  if (unresolvedSafetyConflict || unresolvedSemanticConflict) reviewReasons.add("material_teacher_disagreement");
  if (!evidenceIsGrounded(input.source, final.evidence_spans_twi)) reviewReasons.add("ungrounded_evidence_span");
  if (medicalQuestionEntityViolations(input.source.question_twi_source, final.entities).length) {
    reviewReasons.add("entities_not_grounded_in_question");
  }
  if (["needs_expert_review", "unsafe_or_incorrect"].includes(final.source_answer_assessment)) {
    reviewReasons.add("source_answer_requires_expert_review");
  }
  if (final.safety_level === "needs_review") reviewReasons.add("unresolved_safety_level");
  if (["urgent", "emergency"].includes(final.safety_level)) reviewReasons.add("high_risk_safety_review");
  if (final.intent === "health_medication_question" || final.topics.includes("medication")) {
    reviewReasons.add("medication_review");
  }
  if (sourceNeedsVulnerablePopulationReview(input.source, final.topics)) {
    reviewReasons.add("vulnerable_population_review");
  }
  if (final.normalization_flags.some((flag) => flag.endsWith("_out_of_ontology"))) {
    reviewReasons.add("unresolved_ontology_value");
  }
  if (/\b(?:911|999|111)\b/.test(final.reply_twi)) reviewReasons.add("foreign_or_unverified_emergency_number");
  if (hasRepeatedPhrase(final.reply_twi)) reviewReasons.add("repetitive_reply");

  const sourceNumbers = numberTokens(input.source.answer_twi_source);
  const novelNumbers = [...numberTokens(final.reply_twi)].filter((value) => !sourceNumbers.has(value));
  if (novelNumbers.length) reviewReasons.add("numeric_claim_not_in_source_answer");

  const status = reviewReasons.size === 0 ? "silver_consensus" : "needs_human_review";
  return medicalResponseAnnotationSchema.parse({
    schema_version: 1,
    row_id: input.source.id,
    source_record_hash: input.source.record_source_hash,
    prompt_version: input.promptVersion ?? medicalResponsePromptVersion,
    generated_at: new Date().toISOString(),
    proposals: input.proposals,
    synthesized_proposal: input.synthesis,
    recommended_proposal_id: final.proposal_id,
    adjudication: {
      model: input.adjudicatorModel,
      confidence: input.adjudicatorConfidence,
      status,
      rationale: input.adjudicatorRationale,
      disagreement_fields: [...disagreements].sort(),
      review_reasons: [...reviewReasons].sort(),
    },
    eligible_for_training: status === "silver_consensus" && input.source.source_split === "train",
    eligible_for_final_evaluation: false,
  });
}

export type MedicalProposalFields = ProposalFields;
