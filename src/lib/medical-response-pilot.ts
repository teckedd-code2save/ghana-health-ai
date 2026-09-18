import crypto from "node:crypto";
import { medicalQuestionEntityViolations, type MedicalResponseAnnotation, type MedicalResponseProposal, type MedicalResponseSynthesis } from "./medical-response-annotations";
import type { LicensedMedicalSource } from "./medical-response-export";

export const pilotSystem = "Interpret the user's Twi faithfully. Return only JSON with normalized_text, natural_english, intent, entities, reply, and reply_language. Entities must be exact spans of the user's text. Give your own concise Twi reply in reply; use tw for reply_language. Preserve uncertainty and negation. Do not invent patient details or diagnoses.";
export const digest = (value: string) => crypto.createHash("sha256").update(value).digest("hex");
const key = (text: string) => text.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
const shingles = (text: string) => {
  const words = key(text).split(" ");
  return new Set(words.length < 3 ? [words.join(" ")] : words.slice(2).map((_, i) => words.slice(i, i + 3).join(" ")));
};
const similar = (a: Set<string>, b: Set<string>) => {
  let overlap = 0;
  for (const value of a) if (b.has(value)) overlap++;
  return overlap / (a.size + b.size - overlap) >= 0.8;
};

export function buildMedicalPilot(input: {
  sources: LicensedMedicalSource[];
  annotations: MedicalResponseAnnotation[];
  trainRecordIds: Set<string>;
  heldOutQuestions: string[];
  humanBlockedIds?: Set<string>;
}) {
  const sourceMap = new Map(input.sources.map((source) => [source.id, source]));
  if (sourceMap.size !== input.sources.length) throw new Error("Duplicate source IDs");
  if (new Set(input.annotations.map((row) => row.row_id)).size !== input.annotations.length) throw new Error("Duplicate annotations");
  const locked = input.heldOutQuestions.map(shingles);
  const excluded: Array<{ id: string; reason: string }> = [];
  const candidates: Array<{ source: LicensedMedicalSource; annotation: MedicalResponseAnnotation; proposal: MedicalResponseProposal | MedicalResponseSynthesis; texts: Set<string>[] }> = [];
  for (const annotation of input.annotations) {
    const source = sourceMap.get(annotation.row_id);
    const proposal = [...annotation.proposals, annotation.synthesized_proposal]
      .find((p) => p?.proposal_id === annotation.recommended_proposal_id);
    const reject = (reason: string) => excluded.push({ id: annotation.row_id, reason });
    if (!source || !proposal) { reject("missing_source_or_selection"); continue; }
    if (input.humanBlockedIds?.has(source.id)) { reject("human_review_takes_precedence"); continue; }
    if (source.source_split !== "train" || !input.trainRecordIds.has(source.source_record_id)) { reject("not_in_training_source_pool"); continue; }
    if (annotation.source_record_hash !== source.record_source_hash || digest(source.question_twi_source) !== source.question_source_hash || digest(source.answer_twi_source) !== source.answer_source_hash) { reject("source_hash_mismatch"); continue; }
    if (annotation.prompt_version !== "afrihealth-akan-response-teacher-v3" || proposal.prompt_version !== annotation.prompt_version || proposal.row_id !== source.id) { reject("wrong_annotation_version"); continue; }
    if (annotation.adjudication.status !== "silver_consensus" || annotation.adjudication.review_reasons.length || !annotation.eligible_for_training || annotation.adjudication.confidence < 0.9 || proposal.confidence < 0.85) { reject("unresolved_annotation"); continue; }
    if (proposal.safety_level !== "routine" || !["supportable", "needs_minor_edit"].includes(proposal.source_answer_assessment)) { reject("outside_low_risk_pilot"); continue; }
    if (medicalQuestionEntityViolations(source.question_twi_source, proposal.entities).length) { reject("ungrounded_entities"); continue; }
    const texts = [source.question_twi_source, proposal.question_english].map(shingles);
    if (texts.some((text) => locked.some((heldout) => similar(text, heldout)))) { reject("locked_evaluation_near_duplicate"); continue; }
    candidates.push({ source, annotation, proposal, texts });
  }
  candidates.sort((a, b) => a.source.id.localeCompare(b.source.id));
  excluded.sort((a, b) => a.id.localeCompare(b.id));
  // Connected components keep near-duplicate questions and all translated views together.
  const parents = candidates.map((_, i) => i);
  const find = (i: number): number => parents[i] === i ? i : (parents[i] = find(parents[i]));
  for (let i = 0; i < candidates.length; i++) for (let j = 0; j < i; j++) {
    if (candidates[i].texts.some((a) => candidates[j].texts.some((b) => similar(a, b)))) parents[find(i)] = find(j);
  }
  const groups = new Map<number, typeof candidates>();
  candidates.forEach((row, i) => { const group = find(i); groups.set(group, [...(groups.get(group) ?? []), row]); });
  const ordered = [...groups.values()].map((group) => ({ group, hash: digest(group.map((r) => r.source.id).sort().join("\n")) }))
    .sort((a, b) => a.hash.localeCompare(b.hash));
  const holdout = new Set(ordered.slice(0, Math.max(1, Math.ceil(ordered.length * 0.2))).map((group) => group.hash));
  const rows = [];
  for (const { group, hash } of ordered) for (const { source, annotation, proposal } of group) {
    const split = holdout.has(hash) ? "pilot_holdout" : "pilot_train";
    const target = {
      normalized_text: proposal.question_twi_normalized, natural_english: proposal.question_english,
      intent: proposal.intent, entities: proposal.entities, reply: proposal.reply_twi, reply_language: "tw",
    };
    const provenance = {
      row_id: source.id, group_id: hash, source_record_id: source.source_record_id,
      source_record_hash: source.record_source_hash, source_dataset: source.source_dataset,
      source_revision: source.source_revision, license: source.license, attribution: source.attribution,
      source_url: source.source_url, license_evidence_url: source.license_evidence_url,
      annotation_sha256: digest(JSON.stringify(annotation)), selected_proposal_id: proposal.proposal_id,
      teacher_model: proposal.model, review_status: "uncalibrated_model_consensus", human_reviewed: false,
    };
    const views = [
      { task: "interpret_and_reply", language: "tw", question: source.question_twi_source, answer: JSON.stringify(target), system: pilotSystem },
      { task: "direct_reply", language: "tw", question: source.question_twi_source, answer: proposal.reply_twi, system: "" },
      { task: "direct_reply", language: "en", question: proposal.question_english, answer: proposal.reply_english, system: "" },
    ];
    for (const view of views) rows.push({
      id: `${source.id}:${view.language}:${view.task}`, split, task: view.task, language: view.language,
      provenance, eligible_for_research_training: false, eligible_for_production_training: false,
      eligible_for_final_evaluation: false, experimental_only: true,
      messages: [...(view.system ? [{ role: "system", content: view.system }] : []),
        { role: "user", content: view.question }, { role: "assistant", content: view.answer }],
      reference: target,
    });
  }
  return { rows, excluded, sourceGroups: ordered.length };
}
