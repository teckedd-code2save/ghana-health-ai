import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  medicalResponseAnnotationSchema,
  medicalResponseSourceSchema,
  type MedicalResponseAnnotation,
  type MedicalResponseProposal,
  type MedicalResponseSource,
  type MedicalResponseSynthesis,
} from "@/lib/medical-response-annotations";

const sourcePath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data",
  "medical-response-corpus",
  "afrihealth-akan-source.v1.jsonl",
);
const annotationPath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data",
  "medical-response-corpus",
  "afrihealth-akan-annotations.v1.jsonl",
);
const currentAnnotationPath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data", "medical-response-corpus", "afrihealth-teacher-v3-train.jsonl",
);

function parseJsonl<T>(raw: string, parse: (value: unknown) => T) {
  return raw.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => parse(JSON.parse(line)));
}

let sourceCache: Promise<MedicalResponseSource[]> | null = null;

export function readMedicalResponseSources() {
  sourceCache ??= readFile(sourcePath, "utf8")
    .then((raw) => parseJsonl(raw, (value) => medicalResponseSourceSchema.parse(value)))
    .catch((error) => {
      sourceCache = null;
      throw error;
    });
  return sourceCache;
}

export async function readMedicalResponseAnnotations(): Promise<MedicalResponseAnnotation[]> {
  for (const file of [currentAnnotationPath, annotationPath]) {
    try {
      const raw = await readFile(file, "utf8");
      return parseJsonl(raw, (value) => medicalResponseAnnotationSchema.parse(value));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return [];
}

type ReviewProposal = MedicalResponseProposal | MedicalResponseSynthesis;

export function toUnderstandingProposal(proposal: ReviewProposal) {
  return {
    proposal_id: proposal.proposal_id,
    agent_role: proposal.agent_role,
    model: proposal.model,
    normalized_twi: proposal.question_twi_normalized,
    natural_english: proposal.question_english,
    literal_english: proposal.answer_english,
    intent: proposal.intent,
    entities: JSON.stringify(proposal.entities),
    ambiguities: JSON.stringify({
      topics: proposal.topics,
      source_answer_assessment: proposal.source_answer_assessment,
      source_answer_issues: proposal.source_answer_issues,
      normalization_flags: proposal.normalization_flags,
    }),
    reply_twi: proposal.reply_twi,
    safety_level: proposal.safety_level === "needs_review" ? "" : proposal.safety_level,
    requires_clarification: proposal.requires_clarification,
    score: { total: proposal.confidence, flags: proposal.normalization_flags },
  };
}

export function recommendedMedicalProposal(annotation: MedicalResponseAnnotation) {
  if (annotation.synthesized_proposal?.proposal_id === annotation.recommended_proposal_id) {
    return annotation.synthesized_proposal;
  }
  return annotation.proposals.find((proposal) => proposal.proposal_id === annotation.recommended_proposal_id) ??
    annotation.proposals[0];
}
