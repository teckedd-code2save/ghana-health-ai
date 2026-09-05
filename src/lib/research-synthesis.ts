import { z } from "zod";
import { isResearchIntent } from "@/lib/research-intents";

export const corpusSynthesisPromptVersion = "understanding-synthesis-v9";

export const safetyLevelSchema = z.enum(["", "routine", "same_day", "urgent", "emergency"]);

export const annotationProposalSchema = z.object({
  proposal_id: z.string().min(1),
  row_id: z.string().min(1),
  agent_role: z.enum(["source", "translator", "semantic_annotator", "adjudicator"]),
  model: z.string().min(1),
  prompt_version: z.string().min(1),
  generated_at: z.string().min(1),
  normalized_twi: z.string().default(""),
  natural_english: z.string().default(""),
  literal_english: z.string().default(""),
  intent: z.string().default(""),
  entities: z.string().default(""),
  ambiguities: z.string().default(""),
  reply_twi: z.string().default(""),
  safety_level: safetyLevelSchema.default(""),
  requires_clarification: z.boolean().default(false),
});

export const scoredAnnotationProposalSchema = annotationProposalSchema.extend({
  score: z.object({
    total: z.number().min(0).max(1),
    completeness: z.number().min(0).max(1),
    agreement: z.number().min(0).max(1),
    source_alignment: z.number().min(0).max(1),
    source_twi_alignment: z.number().min(0).max(1).default(0),
    structure: z.number().min(0).max(1),
    response_grounding: z.number().min(0).max(1),
    flags: z.array(z.string()),
  }),
});

export const corpusSynthesisSchema = z.object({
  schema_version: z.literal(1),
  row_id: z.string().min(1),
  source_hash: z.string().min(1),
  prompt_version: z.string().min(1),
  generated_at: z.string().min(1),
  proposals: z.array(scoredAnnotationProposalSchema).min(2),
  recommended_proposal_id: z.string().min(1),
  synthesized_proposal: annotationProposalSchema.nullable(),
  adjudication: z.object({
    model: z.string().min(1),
    selected_proposal_id: z.string().nullable().default(null),
    confidence: z.number().min(0).max(1),
    status: z.enum(["recommended", "synthesized", "needs_human_review"]),
    rationale: z.string(),
    disagreement_fields: z.array(z.string()),
    review_reasons: z.array(z.string()),
  }),
});

export type AnnotationProposal = z.infer<typeof annotationProposalSchema>;
export type ScoredAnnotationProposal = z.infer<typeof scoredAnnotationProposalSchema>;
export type CorpusSynthesis = z.infer<typeof corpusSynthesisSchema>;

const authoritativeMeaningSources = new Set([
  "curated_prompt",
  "ghana_health_symptoms",
  "local_recording",
  "medical_qa_twi_draft",
  "medical_response_seed",
  "product_failure_seed",
]);

export function hasAuthoritativeSourceMeaning(source: string, proposalModel: string) {
  return authoritativeMeaningSources.has(source) && proposalModel === "source_seed_or_translation_draft";
}

export function semanticAmbiguities(value: string) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !/^(source|sources|source_twi|training_use|body_system|license|notes)=/i.test(line))
    .join("\n");
}

type ScoreContext = {
  sourceEnglish: string;
  sourceTwi: string;
  responseExpected: boolean;
};

function normalizedTokens(value: string) {
  return new Set(
    value
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 1),
  );
}

function tokenSimilarity(left: string, right: string) {
  if (!left.trim() || !right.trim()) return 0;
  const a = normalizedTokens(left);
  const b = normalizedTokens(right);
  const union = new Set([...a, ...b]);
  if (union.size === 0) return 0;
  let overlap = 0;
  for (const token of a) if (b.has(token)) overlap += 1;
  return overlap / union.size;
}

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function roundScore(value: number) {
  return Number(Math.max(0, Math.min(1, value)).toFixed(4));
}

function validEntities(value: string) {
  if (!value.trim()) return false;
  try {
    const parsed = JSON.parse(value) as unknown;
    return typeof parsed === "object" && parsed !== null;
  } catch {
    return false;
  }
}

function entityTypeConflict(value: string) {
  try {
    const parsed = JSON.parse(value) as { quantity?: unknown };
    const quantities = Array.isArray(parsed.quantity) ? parsed.quantity.map(String) : [];
    const timeWords = /\b(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|nnansa|nnawɔtwe|bosome)\b/i;
    return quantities.some((item) => timeWords.test(item));
  } catch {
    return false;
  }
}

function entityValueText(value: string) {
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    const ignored = new Set(["body_system", "person"]);
    return Object.entries(parsed)
      .filter(([key]) => !ignored.has(key))
      .flatMap(([, item]) => Array.isArray(item) ? item : [item])
      .filter((item): item is string | number => typeof item === "string" || typeof item === "number")
      .map(String)
      .join(" ");
  } catch {
    return "";
  }
}

function tokenCoverage(needle: string, haystack: string) {
  const expected = normalizedTokens(needle);
  const actual = normalizedTokens(haystack);
  if (expected.size === 0) return 1;
  let overlap = 0;
  for (const token of expected) if (actual.has(token)) overlap += 1;
  return overlap / expected.size;
}

function scoreOne(
  proposal: AnnotationProposal,
  peers: AnnotationProposal[],
  context: ScoreContext,
): ScoredAnnotationProposal {
  const required = [proposal.normalized_twi, proposal.natural_english, proposal.intent];
  const optional = [proposal.literal_english, proposal.entities];
  const completeness = average([
    ...required.map((value) => value.trim() ? 1 : 0),
    ...optional.map((value) => value.trim() ? 1 : 0.5),
  ]);
  const peerAgreement = peers
    .filter((peer) => peer.proposal_id !== proposal.proposal_id)
    .map((peer) => average([
      tokenSimilarity(proposal.normalized_twi, peer.normalized_twi),
      tokenSimilarity(proposal.natural_english, peer.natural_english),
      proposal.intent && proposal.intent === peer.intent ? 1 : 0,
      proposal.safety_level === peer.safety_level ? 1 : 0.5,
    ]));
  const agreement = average(peerAgreement);
  const sourceAlignment = context.sourceEnglish
    ? tokenSimilarity(proposal.natural_english, context.sourceEnglish)
    : agreement;
  const sourceTwiAlignment = context.sourceTwi
    ? tokenSimilarity(proposal.normalized_twi, context.sourceTwi)
    : agreement;
  const hasEntityTypeConflict = entityTypeConflict(proposal.entities);
  const entityText = entityValueText(proposal.entities);
  const entityValueAlignment = entityText ? tokenCoverage(entityText, proposal.natural_english) : 1;
  const hasEntityValueMisalignment = entityText.split(/\s+/).filter(Boolean).length >= 2 && entityValueAlignment < 0.25;
  const structure = average([
    /^[a-z][a-z0-9_]*$/.test(proposal.intent) ? 1 : 0,
    validEntities(proposal.entities) && !hasEntityTypeConflict ? 1 : 0,
    proposal.normalized_twi.length <= 1500 ? 1 : 0,
    proposal.natural_english.length <= 1500 ? 1 : 0,
    hasEntityValueMisalignment ? 0 : 1,
  ]);
  const responseGrounding = context.responseExpected
    ? average([proposal.reply_twi.trim() ? 1 : 0, proposal.safety_level ? 1 : 0])
    : proposal.reply_twi.trim() || proposal.safety_level ? 0 : 1;
  const flags: string[] = [];
  if (required.some((value) => !value.trim())) flags.push("missing_required_field");
  if (!/^[a-z][a-z0-9_]*$/.test(proposal.intent)) flags.push("invalid_intent");
  if (proposal.intent && !isResearchIntent(proposal.intent)) flags.push("intent_ontology_mismatch");
  if (!validEntities(proposal.entities)) flags.push("invalid_entities");
  if (hasEntityTypeConflict) flags.push("entity_type_conflict");
  if (hasEntityValueMisalignment) flags.push("entity_value_misalignment");
  if (context.sourceEnglish && sourceAlignment < 0.35) flags.push("low_source_alignment");
  if (context.sourceTwi && sourceTwiAlignment < 0.55) flags.push("low_twi_source_alignment");
  if (context.responseExpected && responseGrounding < 1) flags.push("incomplete_grounded_response");
  if (!context.responseExpected && (proposal.reply_twi.trim() || proposal.safety_level)) {
    flags.push("ungrounded_response_added");
  }
  const baseTotal = context.responseExpected
    ? completeness * 0.15 + agreement * 0.15 + sourceAlignment * 0.15 + sourceTwiAlignment * 0.15 + structure * 0.15 + responseGrounding * 0.25
    : completeness * 0.25 + agreement * 0.2 + sourceAlignment * 0.2 + sourceTwiAlignment * 0.2 + structure * 0.15;
  const penalty = flags.reduce((sum, flag) => sum + (
    flag === "ungrounded_response_added" ? 0.3 :
    flag === "missing_required_field" || flag === "invalid_intent" || flag === "invalid_entities" ? 0.15 :
    flag === "intent_ontology_mismatch" ? 0.12 :
    flag === "entity_type_conflict" || flag === "entity_value_misalignment" || flag === "incomplete_grounded_response" ? 0.08 :
    0.02
  ), 0);
  const total = baseTotal - penalty;

  return {
    ...proposal,
    score: {
      total: roundScore(total),
      completeness: roundScore(completeness),
      agreement: roundScore(agreement),
      source_alignment: roundScore(sourceAlignment),
      source_twi_alignment: roundScore(sourceTwiAlignment),
      structure: roundScore(structure),
      response_grounding: roundScore(responseGrounding),
      flags,
    },
  };
}

function disagreementFields(proposals: AnnotationProposal[]) {
  const fields: string[] = [];
  const distinct = (values: string[]) => new Set(values.map((value) => value.trim()).filter(Boolean)).size;
  if (distinct(proposals.map((row) => row.intent)) > 1) fields.push("intent");
  if (distinct(proposals.map((row) => row.safety_level)) > 1) fields.push("safety_level");
  const meaningAgreement = average(
    proposals.flatMap((left, index) =>
      proposals.slice(index + 1).map((right) => tokenSimilarity(left.natural_english, right.natural_english)),
    ),
  );
  const twiAgreement = average(
    proposals.flatMap((left, index) =>
      proposals.slice(index + 1).map((right) => tokenSimilarity(left.normalized_twi, right.normalized_twi)),
    ),
  );
  if (meaningAgreement < 0.55) fields.push("natural_english");
  if (twiAgreement < 0.65) fields.push("normalized_twi");
  if (proposals.some((row) => row.reply_twi) && distinct(proposals.map((row) => row.reply_twi)) > 1) {
    fields.push("reply_twi");
  }
  return fields;
}

function criticalFlags(proposal: ScoredAnnotationProposal) {
  return proposal.score.flags.filter((flag) =>
    flag !== "low_source_alignment" && flag !== "low_twi_source_alignment",
  );
}

export function scoreAnnotationProposals(
  proposals: AnnotationProposal[],
  context: ScoreContext,
) {
  return proposals
    .map((proposal) => scoreOne(proposal, proposals, context))
    .sort((left, right) => right.score.total - left.score.total);
}

export function buildSynthesisRecord(input: {
  rowId: string;
  sourceHash: string;
  promptVersion: string;
  proposals: AnnotationProposal[];
  sourceEnglish: string;
  sourceTwi: string;
  responseExpected: boolean;
  synthesizedProposal?: AnnotationProposal | null;
  adjudicatorModel: string;
  adjudicatorSelectedProposalId?: string;
  adjudicatorConfidence?: number;
  adjudicatorDisagreementFields?: string[];
  adjudicatorReviewReasons?: string[];
  adjudicatorRationale?: string;
}): CorpusSynthesis {
  const scored = scoreAnnotationProposals(input.proposals, {
    sourceEnglish: input.sourceEnglish,
    sourceTwi: input.sourceTwi,
    responseExpected: input.responseExpected,
  });
  const scoreWinner = scored[0]!;
  const validScoreWinner = scored.find((proposal) => criticalFlags(proposal).length === 0) ?? scoreWinner;
  const rawAdjudicatorSelection = scored.find(
    (proposal) => proposal.proposal_id === input.adjudicatorSelectedProposalId,
  );
  const selectedByAdjudicator = rawAdjudicatorSelection && criticalFlags(rawAdjudicatorSelection).length === 0
    ? rawAdjudicatorSelection
    : undefined;
  const recommended = selectedByAdjudicator ?? validScoreWinner;
  const runnerUp = scored.find((proposal) => proposal.proposal_id !== recommended.proposal_id);
  const heuristicDisagreements = disagreementFields(input.proposals);
  const disagreements = [...new Set([
    ...heuristicDisagreements,
    ...(input.adjudicatorDisagreementFields ?? []),
  ])];
  const synthesized = input.synthesizedProposal
    ? scoreOne(input.synthesizedProposal, input.proposals, {
        sourceEnglish: input.sourceEnglish,
        sourceTwi: input.sourceTwi,
        responseExpected: input.responseExpected,
      })
    : null;
  const finalProposal = synthesized ?? recommended;
  const finalCriticalFlags = criticalFlags(finalProposal);
  const adjudicatorConfidence = input.adjudicatorConfidence ?? 0;
  const rawSelectionFlags = rawAdjudicatorSelection ? criticalFlags(rawAdjudicatorSelection) : [];
  const sourceGroundedSynthesis = Boolean(
    synthesized?.model.endsWith(":source-grounded-synthesis"),
  );
  const confidentDeterministicOverride = Boolean(
    rawAdjudicatorSelection &&
    !selectedByAdjudicator &&
    adjudicatorConfidence >= 0.88 &&
    validScoreWinner.score.total >= 0.85 &&
    rawSelectionFlags.length > 0 &&
    rawSelectionFlags.every((flag) => flag === "entity_type_conflict") &&
    rawAdjudicatorSelection.intent === validScoreWinner.intent &&
    tokenSimilarity(rawAdjudicatorSelection.natural_english, validScoreWinner.natural_english) >= 0.45,
  );
  const confidentAdjudication = Boolean(
    (selectedByAdjudicator || confidentDeterministicOverride) &&
    adjudicatorConfidence >= (sourceGroundedSynthesis ? 0.78 : 0.88) &&
    (input.adjudicatorDisagreementFields?.length ?? 0) === 0 &&
    (input.adjudicatorReviewReasons?.length ?? 0) === 0,
  );
  const disagreementReviewReasons = confidentAdjudication
    ? []
    : disagreements.map((field) => `disagreement:${field}`);
  const reviewReasons = [...new Set([
    ...(finalProposal.score.total < 0.62 ? ["low_recommendation_score"] : []),
    ...(!synthesized && !selectedByAdjudicator && runnerUp && recommended.score.total - runnerUp.score.total < 0.04
      ? ["close_candidate_scores"]
      : []),
    ...finalCriticalFlags,
    ...(input.adjudicatorReviewReasons ?? []),
    ...disagreementReviewReasons,
  ])];
  const confidence = roundScore(
    input.adjudicatorConfidence === undefined
      ? finalProposal.score.total
      : (finalProposal.score.total * 2 + input.adjudicatorConfidence) / 3,
  );

  return corpusSynthesisSchema.parse({
    schema_version: 1,
    row_id: input.rowId,
    source_hash: input.sourceHash,
    prompt_version: input.promptVersion,
    generated_at: new Date().toISOString(),
    proposals: scored,
    recommended_proposal_id: finalProposal.proposal_id,
    synthesized_proposal: input.synthesizedProposal ?? null,
    adjudication: {
      model: input.adjudicatorModel,
      selected_proposal_id: finalProposal.proposal_id,
      confidence,
      status: reviewReasons.length > 0
        ? "needs_human_review"
        : input.synthesizedProposal
          ? "synthesized"
          : "recommended",
      rationale: rawAdjudicatorSelection && !selectedByAdjudicator
        ? `Deterministic validation rejected the adjudicator selection (${criticalFlags(rawAdjudicatorSelection).join(", ")}); the highest-scoring valid option was selected.`
        : input.adjudicatorRationale ?? "Ranked by agreement, source alignment, completeness, and schema validity.",
      disagreement_fields: disagreements,
      review_reasons: reviewReasons,
    },
  });
}
