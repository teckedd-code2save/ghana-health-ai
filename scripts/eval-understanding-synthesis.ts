import {
  buildSynthesisRecord,
  scoreAnnotationProposals,
  type AnnotationProposal,
} from "../src/lib/research-synthesis";

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function proposal(overrides: Partial<AnnotationProposal>): AnnotationProposal {
  return {
    proposal_id: "source",
    row_id: "row-1",
    agent_role: "source",
    model: "source:test",
    prompt_version: "test-v1",
    generated_at: "2026-09-05T00:00:00.000Z",
    normalized_twi: "Me yafunu yɛ me yaw efiri nnansa.",
    natural_english: "I have had stomach pain for three days.",
    literal_english: "My stomach hurts for three days.",
    intent: "health_symptom_report",
    entities: JSON.stringify({ symptom: ["stomach pain"], time: ["three days"], quantity: [] }),
    ambiguities: "",
    reply_twi: "",
    safety_level: "",
    requires_clarification: false,
    ...overrides,
  };
}

const source = proposal({});
const translator = proposal({
  proposal_id: "translator",
  agent_role: "translator",
  model: "open:test-translator",
  normalized_twi: "Nnansa ni me yafunu yɛ me yaw.",
  natural_english: "My stomach has been hurting for three days.",
});
const malformed = proposal({
  proposal_id: "malformed",
  agent_role: "semantic_annotator",
  model: "open:test-annotator",
  entities: JSON.stringify({ symptom: ["stomach pain"], quantity: ["three days"] }),
  reply_twi: "Fa aduru bi nom.",
});

const scored = scoreAnnotationProposals([source, translator, malformed], {
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
});
const malformedScore = scored.find((row) => row.proposal_id === malformed.proposal_id)!;
assert(malformedScore.score.flags.includes("entity_type_conflict"), "Time in quantity must be flagged.");
assert(malformedScore.score.flags.includes("ungrounded_response_added"), "Ungrounded response must be flagged.");

const selected = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, translator, malformed],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: translator.proposal_id,
  adjudicatorConfidence: 0.99,
  adjudicatorRationale: "Translator preserved the full temporal meaning.",
});
assert(selected.recommended_proposal_id === translator.proposal_id, "Adjudicator selection must be retained.");
assert(selected.adjudication.selected_proposal_id === translator.proposal_id, "Selection provenance must be retained.");
assert(selected.adjudication.confidence < 0.99, "Raw model confidence must be calibrated.");

const independentMt = proposal({
  proposal_id: "independent-mt",
  agent_role: "semantic_annotator",
  model: "open:test-mt",
  natural_english: "I have a headache.",
  intent: "health_general_question",
});
const confidentConsensus = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, translator, independentMt],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: source.proposal_id,
  adjudicatorConfidence: 0.94,
  adjudicatorDisagreementFields: [],
  adjudicatorReviewReasons: [],
});
assert(
  confidentConsensus.adjudication.status === "recommended",
  "One weak disagreement signal must not force human review when the adjudicated consensus is strong.",
);

const entityConflict = proposal({
  proposal_id: "entity-conflict",
  agent_role: "translator",
  model: "open:test-translator",
  natural_english: "I have had stomach pain for three days.",
  entities: JSON.stringify({ symptom: ["stomach pain"], quantity: ["three days"] }),
});
const validatedOverride = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, entityConflict, independentMt],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: entityConflict.proposal_id,
  adjudicatorConfidence: 0.94,
  adjudicatorDisagreementFields: [],
  adjudicatorReviewReasons: [],
});
assert(validatedOverride.recommended_proposal_id === source.proposal_id, "Invalid selected entities must be rejected.");
assert(validatedOverride.adjudication.status === "recommended", "A strong deterministic override should not require manual review.");

const synthesizedProposal = proposal({
  proposal_id: "synthesis",
  agent_role: "adjudicator",
  model: "open:test-judge",
  normalized_twi: "Nnansa ni na me yafunu yɛ me yaw.",
});
const synthesized = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, translator],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  synthesizedProposal,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: "synthesized",
});
assert(synthesized.recommended_proposal_id === synthesizedProposal.proposal_id, "Synthesis must be the recommendation.");
assert(synthesized.synthesized_proposal?.proposal_id === synthesizedProposal.proposal_id, "Synthesis must be retained.");

const sourceGrounded = proposal({
  proposal_id: "source-grounded",
  agent_role: "adjudicator",
  model: "open:test-judge:source-grounded-synthesis",
});
const sourceGroundedConsensus = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, translator, independentMt],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  synthesizedProposal: sourceGrounded,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: translator.proposal_id,
  adjudicatorConfidence: 0.8,
  adjudicatorDisagreementFields: [],
  adjudicatorReviewReasons: [],
});
assert(
  sourceGroundedConsensus.adjudication.status === "synthesized",
  "A source-grounded paraphrase consensus should not be sent to manual review.",
);

const sourceGroundedConflict = buildSynthesisRecord({
  rowId: "row-1",
  sourceHash: "source-hash",
  promptVersion: "test-v1",
  proposals: [source, translator, independentMt],
  sourceEnglish: source.natural_english,
  sourceTwi: source.normalized_twi,
  responseExpected: false,
  synthesizedProposal: sourceGrounded,
  adjudicatorModel: "open:test-judge",
  adjudicatorSelectedProposalId: source.proposal_id,
  adjudicatorConfidence: 0.84,
  adjudicatorDisagreementFields: ["natural_english"],
  adjudicatorReviewReasons: ["candidate_meaning_conflict"],
});
assert(
  sourceGroundedConflict.adjudication.status === "needs_human_review",
  "A source-grounded row with a material semantic conflict must stay in review.",
);

console.log(JSON.stringify({
  proposals_scored: scored.length,
  malformed_flags: malformedScore.score.flags,
  selected_proposal: selected.recommended_proposal_id,
  calibrated_confidence: selected.adjudication.confidence,
  confident_consensus: confidentConsensus.adjudication.status,
  validated_override: validatedOverride.recommended_proposal_id,
  synthesized_proposal: synthesized.recommended_proposal_id,
  source_grounded_consensus: sourceGroundedConsensus.adjudication.status,
  source_grounded_conflict: sourceGroundedConflict.adjudication.status,
}, null, 2));
