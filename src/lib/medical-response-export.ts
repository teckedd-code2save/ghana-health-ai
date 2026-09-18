import crypto from "node:crypto";
import { z } from "zod";
import {
  medicalAnnotationPipelineId,
  medicalEntitiesSchema,
  medicalQuestionEntityViolations,
  medicalResponseIntents,
  medicalResponseSourceSchema,
  type MedicalResponseAnnotation,
} from "./medical-response-annotations";

export const licensedMedicalSourceSchema = medicalResponseSourceSchema.extend({
  license: z.string().min(1),
  license_evidence_url: z.string().url(),
  attribution: z.string().min(1),
  source_url: z.string().url(),
});

export const medicalExportReviewSchema = z.object({
  id: z.string(),
  decision: z.enum(["unreviewed", "reviewed", "needs_second_review", "exclude"]),
  normalizedTwi: z.string().default(""),
  naturalEnglish: z.string().default(""),
  literalEnglish: z.string().default(""),
  intent: z.string().default(""),
  entities: z.string().default(""),
  ambiguities: z.string().default(""),
  replyTwi: z.string().default(""),
  replyEnglish: z.string().default(""),
  safetyLevel: z.string().default(""),
  requiresClarification: z.boolean().optional(),
  selectedProposalId: z.string().default(""),
  synthesisVersion: z.string().default(""),
  reviewer: z.string().default(""),
  updatedAt: z.string().optional(),
});

export type MedicalExportReview = z.infer<typeof medicalExportReviewSchema>;
export type LicensedMedicalSource = z.infer<typeof licensedMedicalSourceSchema>;
export const annotationCalibrationSchema = z.object({
  gate: z.enum(["pass", "fail"]),
  failures: z.array(z.string()),
  pipeline_ids: z.array(z.string()),
  candidate_sha256: z.string().length(64),
  reference_sha256: z.string().length(64),
});

const targetSchema = z.object({
  normalized_text: z.string().trim().min(1),
  natural_english: z.string().trim().min(1),
  intent: z.enum(medicalResponseIntents),
  entities: medicalEntitiesSchema,
  uncertainty_notes: z.string(),
  safety_level: z.enum(["routine", "same_day", "urgent", "emergency"]),
  reply: z.string().trim().min(1),
  reply_language: z.enum(["tw", "en"]),
  requires_clarification: z.boolean(),
});

export const medicalResponseTrainingSystem = "Interpret the utterance faithfully, preserve person, negation, quantities and uncertainty, and answer in its language. Return JSON with normalized_text, natural_english, intent, entities, uncertainty_notes, safety_level, reply, reply_language, and requires_clarification. Do not invent patient details, diagnoses, medicines, doses, or facts.";

const digest = (value: string) => crypto.createHash("sha256").update(value).digest("hex");
const textKey = (value: string) => value.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

export function buildMedicalResponseExport(input: {
  sources: LicensedMedicalSource[];
  annotations: MedicalResponseAnnotation[];
  reviews: MedicalExportReview[];
  trainRecordIds: Set<string>;
  evalRecordIds: Set<string>;
  heldOutQuestions: string[];
  calibration: z.infer<typeof annotationCalibrationSchema> | null;
}) {
  if (new Set(input.sources.map((source) => source.id)).size !== input.sources.length) {
    throw new Error("Duplicate source row IDs");
  }
  const sourceById = new Map(input.sources.map((source) => [source.id, source]));
  const reviewById = new Map(input.reviews.map((review) => [review.id, review]));
  const seenAnnotations = new Set<string>();
  const seenTexts = new Map<string, string>();
  const heldOutText = new Set([...input.heldOutQuestions, ...input.sources
    .filter((source) => source.source_split === "validation")
    .map((source) => source.question_twi_source)].map(textKey));
  const rejected: Array<{ row_id: string; reason: string }> = [];
  const rows = [];

  for (const annotation of input.annotations) {
    if (seenAnnotations.has(annotation.row_id)) throw new Error(`Duplicate annotation: ${annotation.row_id}`);
    seenAnnotations.add(annotation.row_id);
    const source = sourceById.get(annotation.row_id);
    const review = reviewById.get(annotation.row_id);
    const reject = (reason: string) => rejected.push({ row_id: annotation.row_id, reason });
    if (!source) { reject("missing_source"); continue; }
    if (annotation.source_record_hash !== source.record_source_hash ||
      digest(source.question_twi_source) !== source.question_source_hash ||
      digest(source.answer_twi_source) !== source.answer_source_hash) {
      reject("source_hash_mismatch"); continue;
    }
    const split = source.source_split === "train" ? "train" : "eval";
    const eligibleIds = split === "train" ? input.trainRecordIds : input.evalRecordIds;
    if (!eligibleIds.has(source.source_record_id)) { reject("source_outside_eligible_pool"); continue; }
    if (split === "train" && heldOutText.has(textKey(source.question_twi_source))) {
      reject("held_out_question_overlap"); continue;
    }
    if (review && ["exclude", "needs_second_review"].includes(review.decision)) {
      reject(`human_${review.decision}`); continue;
    }
    const humanReviewed = review?.decision === "reviewed";
    const pipelineId = medicalAnnotationPipelineId(annotation);
    if (!humanReviewed) {
      if (input.calibration?.gate !== "pass" || input.calibration.failures.length ||
        !input.calibration.pipeline_ids.includes(pipelineId)) {
        reject("annotation_pipeline_not_calibrated"); continue;
      }
      if (annotation.adjudication.status !== "silver_consensus" ||
        annotation.adjudication.review_reasons.length || annotation.adjudication.confidence < 0.9 ||
        (split === "train" && !annotation.eligible_for_training)) {
        reject("unresolved_annotation_review"); continue;
      }
    }
    const options = [...annotation.proposals, ...(annotation.synthesized_proposal ? [annotation.synthesized_proposal] : [])];
    const selectedId = humanReviewed ? review.selectedProposalId : annotation.recommended_proposal_id;
    const proposal = options.find((option) => option.proposal_id === selectedId);
    if (!proposal || proposal.row_id !== source.id) { reject("missing_selected_proposal"); continue; }
    if (!humanReviewed && medicalQuestionEntityViolations(source.question_twi_source, proposal.entities).length) {
      reject("entities_not_grounded_in_question"); continue;
    }
    if (!humanReviewed && (proposal.confidence < 0.85 ||
      ["needs_expert_review", "unsafe_or_incorrect"].includes(proposal.source_answer_assessment))) {
      reject("unresolved_proposal_quality"); continue;
    }
    if (humanReviewed && (review.synthesisVersion !== annotation.prompt_version || !review.reviewer.trim() || !review.updatedAt)) {
      reject("incomplete_or_stale_review_provenance"); continue;
    }

    try {
      const target = targetSchema.parse({
        normalized_text: humanReviewed ? review.normalizedTwi : proposal.question_twi_normalized,
        natural_english: humanReviewed ? review.naturalEnglish : proposal.question_english,
        intent: humanReviewed ? review.intent : proposal.intent,
        entities: humanReviewed ? JSON.parse(review.entities) : proposal.entities,
        uncertainty_notes: humanReviewed ? review.ambiguities : "",
        safety_level: humanReviewed ? review.safetyLevel : proposal.safety_level,
        reply: humanReviewed ? review.replyTwi : proposal.reply_twi,
        reply_language: "tw",
        requires_clarification: humanReviewed && typeof review.requiresClarification === "boolean"
          ? review.requiresClarification : proposal.requires_clarification,
      });
      const duplicateKey = textKey(source.question_twi_source);
      if (split === "train" && heldOutText.has(textKey(target.natural_english))) {
        reject("held_out_translation_overlap"); continue;
      }
      if (seenTexts.has(duplicateKey)) { reject("duplicate_source_question"); continue; }
      seenTexts.set(duplicateKey, source.id);
      const englishReply = humanReviewed
        ? review.replyEnglish.trim() || (review.replyTwi.trim() === proposal.reply_twi.trim() ? proposal.reply_english : "")
        : proposal.reply_english;
      const provenance = {
        row_id: source.id,
        source_dataset: source.source_dataset,
        source_revision: source.source_revision,
        source_record_id: source.source_record_id,
        source_record_hash: source.record_source_hash,
        source_url: source.source_url,
        license: source.license,
        license_evidence_url: source.license_evidence_url,
        attribution: source.attribution,
        annotation_sha256: digest(JSON.stringify(annotation)),
        pipeline_id: pipelineId,
        selected_proposal_id: proposal.proposal_id,
        selected_model: proposal.model,
        prompt_version: annotation.prompt_version,
        review_status: humanReviewed ? "human_reviewed" : "model_silver",
        reviewer: humanReviewed ? review.reviewer : null,
        reviewed_at: humanReviewed ? review.updatedAt : null,
        review_sha256: humanReviewed ? digest(JSON.stringify(review)) : null,
      };
      const views: Array<{ language: "tw" | "en"; utterance: string; target: z.infer<typeof targetSchema> }> = [
        { language: "tw", utterance: source.question_twi_source, target },
      ];
      if (englishReply) {
        views.push({
          language: "en",
          utterance: target.natural_english,
          target: targetSchema.parse({ ...target, normalized_text: target.natural_english, reply: englishReply, reply_language: "en" }),
        });
      } else {
        reject("english_view_needs_translation_of_human_edit");
      }
      for (const view of views) {
        rows.push({
          id: `${source.id}:${view.language}`,
          split,
          language: view.language,
          provenance,
          eligible_for_research_training: split === "train",
          eligible_for_final_evaluation: false,
          messages: [
            { role: "system", content: medicalResponseTrainingSystem },
            { role: "user", content: view.utterance },
            { role: "assistant", content: JSON.stringify(view.target) },
          ],
        });
      }
    } catch {
      reject("incomplete_or_invalid_target_fields");
    }
  }
  rows.sort((left, right) => left.id.localeCompare(right.id));
  return { rows, rejected };
}
