import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import type { CorpusSynthesis } from "@/lib/research-synthesis";
import {
  readMedicalResponseAnnotations,
  readMedicalResponseSources,
  recommendedMedicalProposal,
  toUnderstandingProposal,
} from "@/lib/medical-response-store";
import {
  type CorpusCandidate,
  corpusStages,
  buildUnderstandingTrainingExport,
  getCandidateTrainingSplit,
  readBenchmarkSeeds,
  readCorpusCandidates,
  readCorpusSyntheses,
  readUnderstandingScorecard,
  readUnderstandingReviews,
  saveUnderstandingReview,
  sourceInventory,
  storageDecisions,
  understandingReviewInputSchema,
} from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";
type CorpusRowFilter =
  | "afrihealth_response"
  | "medical_large"
  | "language_sources"
  | "local_audio"
  | "product_text"
  | "all";

async function getReviewer() {
  const env = getEnv();
  const user = await getSessionUser();
  const role = user?.role?.toLowerCase();
  const allowed =
    env.NODE_ENV !== "production" ||
    env.RESEARCH_REVIEW_ENABLED ||
    role === "admin" ||
    role === "researcher";

  if (!allowed) return null;
  return user?.email ?? user?.phone ?? user?.id ?? "local_reviewer";
}

export async function GET(request: Request) {
  const reviewer = await getReviewer();
  if (!reviewer) return jsonError("Research review access is not enabled for this account.", 403);
  const { searchParams } = new URL(request.url);
  const summaryOnly = searchParams.get("summary") === "1" || searchParams.get("summary") === "true";
  const datasetOnly = searchParams.get("dataset") === "1" || searchParams.get("dataset") === "true";
  const rowLimitValue = Number(searchParams.get("limit") ?? "");
  const rowLimit = Number.isFinite(rowLimitValue) && rowLimitValue >= 0 ? Math.floor(rowLimitValue) : 300;
  const rowOffsetValue = Number(searchParams.get("offset") ?? "0");
  const rowOffset = Number.isFinite(rowOffsetValue) && rowOffsetValue >= 0 ? Math.floor(rowOffsetValue) : 0;
  const rowFilter = parseCorpusRowFilter(searchParams.get("filter"));

  if (rowFilter === "afrihealth_response") {
    const [sources, annotations, reviews] = await Promise.all([
      readMedicalResponseSources(),
      readMedicalResponseAnnotations(),
      readUnderstandingReviews(),
    ]);
    const sourceById = new Map(sources.map((source) => [source.id, source]));
    const reviewById = new Map(reviews.map((review) => [review.id, review]));
    const sorted = annotations
      .filter((annotation) => sourceById.has(annotation.row_id))
      .sort((left, right) => {
        const leftReviewed = reviewById.get(left.row_id)?.decision === "reviewed" ? 1 : 0;
        const rightReviewed = reviewById.get(right.row_id)?.decision === "reviewed" ? 1 : 0;
        if (leftReviewed !== rightReviewed) return leftReviewed - rightReviewed;
        const leftPriority = left.adjudication.status === "needs_human_review" ? 0 : 1;
        const rightPriority = right.adjudication.status === "needs_human_review" ? 0 : 1;
        return leftPriority - rightPriority || left.row_id.localeCompare(right.row_id);
      });
    const visible = sorted.slice(rowOffset, rowOffset + rowLimit).flatMap((annotation) => {
      const source = sourceById.get(annotation.row_id);
      if (!source) return [];
      const recommended = recommendedMedicalProposal(annotation);
      const modelProposal = toUnderstandingProposal(recommended);
      return [{
        kind: "corpus" as const,
        id: source.id,
        category: "health/afrihealth_response",
        domain: "health",
        text: source.question_twi_source,
        sourceAnswer: source.answer_twi_source,
        responseSource: true,
        review_status: annotation.adjudication.status,
        source: "afrihealth_response",
        sourceRecordId: source.source_record_id,
        split: source.source_split,
        trainingSplit: source.source_split === "validation" ? "dev" : "train",
        language: "tw",
        speakerId: null,
        audioArtifactId: null,
        consentScope: "dataset_license",
        modelProposal,
        annotationSet: {
          prompt_version: annotation.prompt_version,
          proposals: annotation.proposals.map(toUnderstandingProposal),
          recommended_proposal_id: annotation.recommended_proposal_id,
          synthesized_proposal: annotation.synthesized_proposal
            ? toUnderstandingProposal(annotation.synthesized_proposal)
            : null,
          adjudication: annotation.adjudication,
        },
        review: reviewById.get(source.id) ?? null,
      }];
    });
    const statuses = annotations.reduce<Record<string, number>>((counts, row) => {
      counts[row.adjudication.status] = (counts[row.adjudication.status] ?? 0) + 1;
      return counts;
    }, {});
    return jsonOk({
      reviewer,
      corpus: { sourceInventory: [], storageDecisions: [], stages: [] },
      benchmark: { rows: [], total: 0, completed: 0, needsSecondReview: 0, excluded: 0, scorecard: null },
      candidates: {
        rows: summaryOnly ? [] : visible,
        total: annotations.length,
        sourceTotal: sources.length,
        visible: visible.length,
        offset: rowOffset,
        limit: rowLimit,
        filter: rowFilter,
        completed: annotations.filter((row) => reviewById.get(row.row_id)?.decision === "reviewed").length,
        withAudio: 0,
        draftAnnotated: annotations.length,
        synthesized: annotations.filter((row) => row.synthesized_proposal).length,
        trainingReady: annotations.filter((row) => row.eligible_for_training).length,
        statuses,
      },
    });
  }

  const [seeds, candidates, syntheses, reviews, scorecard, trainingExport] = await Promise.all([
    readBenchmarkSeeds(),
    readCorpusCandidates(),
    readCorpusSyntheses(),
    readUnderstandingReviews(),
    readUnderstandingScorecard(),
    buildUnderstandingTrainingExport(),
  ]);
  const synthesisById = new Map(syntheses.map((synthesis) => [synthesis.row_id, synthesis]));
  const reviewById = new Map(reviews.map((review) => [review.id, review]));
  const rows = seeds.map((seed) => ({
    kind: "benchmark" as const,
    ...seed,
    review: reviewById.get(seed.id) ?? null,
  }));
  const filteredCandidates = candidates
    .filter((candidate) => matchesCorpusRowFilter(candidate, rowFilter));
  const candidateRows = filteredCandidates
    .sort((left, right) =>
      synthesisReviewPriority(synthesisById.get(left.id)) - synthesisReviewPriority(synthesisById.get(right.id)) ||
      reviewQueuePriority(left, rowFilter) - reviewQueuePriority(right, rowFilter),
    )
    .slice(rowOffset, rowOffset + rowLimit);
  const corpusRows = candidateRows.map((candidate) => ({
    kind: "corpus" as const,
    id: candidate.id,
    category: `${candidate.domain}/${candidate.source}`,
    domain: candidate.domain,
    text: candidate.text,
    review_status: candidate.review_status,
    source: candidate.source,
    sourceRecordId: candidate.source_record_id,
    split: candidate.split,
    trainingSplit: getCandidateTrainingSplit(candidate),
    language: candidate.language,
    speakerId: candidate.speaker_id,
    audioArtifactId: candidate.audio_artifact_id,
    consentScope: candidate.consent_scope,
    modelProposal: candidate.model_proposal,
    annotationSet: synthesisById.get(candidate.id) ?? null,
    review: reviewById.get(candidate.id) ?? null,
  }));
  const completed = rows.filter((row) => row.review?.decision === "reviewed").length;
  const needsSecondReview = rows.filter((row) => row.review?.decision === "needs_second_review").length;
  const excluded = rows.filter((row) => row.review?.decision === "exclude").length;
  const corpusCompleted = filteredCandidates.filter((row) => reviewById.get(row.id)?.decision === "reviewed").length;
  const candidateSplits = filteredCandidates.reduce(
    (acc, row) => {
      const split = getCandidateTrainingSplit(row);
      acc[split] += 1;
      return acc;
    },
    { train: 0, dev: 0, test: 0 },
  );
  const sourceSummary = filteredCandidates.reduce<
    Record<string, { total: number; draftAnnotated: number; reviewed: number; excluded: number }>
  >((acc, row) => {
    const source = row.source ?? "unknown";
    acc[source] ??= { total: 0, draftAnnotated: 0, reviewed: 0, excluded: 0 };
    acc[source].total += 1;
    if (row.model_proposal.status === "draft") acc[source].draftAnnotated += 1;
    const review = reviewById.get(row.id);
    if (review?.decision === "reviewed") acc[source].reviewed += 1;
    if (review?.decision === "exclude") acc[source].excluded += 1;
    return acc;
  }, {});

  return jsonOk({
    reviewer,
    corpus: {
      sourceInventory: datasetOnly ? [] : sourceInventory,
      storageDecisions: datasetOnly ? [] : storageDecisions,
      stages: datasetOnly ? [] : corpusStages,
    },
    benchmark: {
      rows: summaryOnly || datasetOnly ? [] : rows,
      total: rows.length,
      completed,
      needsSecondReview,
      excluded,
      scorecard: datasetOnly ? null : scorecard,
    },
    candidates: {
      rows: summaryOnly ? [] : corpusRows,
      total: filteredCandidates.length,
      visible: corpusRows.length,
      offset: rowOffset,
      limit: rowLimit,
      filter: rowFilter,
      completed: corpusCompleted,
      withAudio: filteredCandidates.filter((row) => row.audio_artifact_id).length,
      draftAnnotated: filteredCandidates.filter((row) => row.model_proposal.status === "draft").length,
      synthesized: filteredCandidates.filter((row) => synthesisById.has(row.id)).length,
      trainingReady: trainingExport.accepted,
      splits: trainingExport.splits,
      candidateSplits,
      sourceSummary,
      readiness: trainingExport.readiness,
    },
  });
}

function synthesisReviewPriority(synthesis: CorpusSynthesis | undefined) {
  if (synthesis?.adjudication.status === "needs_human_review") return 0;
  if (synthesis) return 1;
  return 2;
}

function parseCorpusRowFilter(value: string | null): CorpusRowFilter {
  if (
    value === "afrihealth_response" ||
    value === "medical_large" ||
    value === "language_sources" ||
    value === "local_audio" ||
    value === "product_text" ||
    value === "all"
  ) {
    return value;
  }
  return "all";
}

function reviewQueuePriority(candidate: CorpusCandidate, filter: CorpusRowFilter) {
  if (filter !== "product_text") return 0;
  const responseReady = candidate.model_proposal.reply_twi.trim() && candidate.model_proposal.safety_level.trim();
  if (responseReady) return 0;
  if (candidate.source === "product_failure_seed") return 1;
  return 2;
}

function matchesCorpusRowFilter(candidate: CorpusCandidate, filter: CorpusRowFilter) {
  if (filter === "all") return true;
  if (filter === "medical_large") return candidate.source === "ghana_health_symptoms";
  if (filter === "language_sources") {
    return candidate.source === "waxal" || candidate.source === "ghana_nlp_speech";
  }
  if (filter === "local_audio") return candidate.source === "local_recording";
  if (filter === "product_text") {
    return (
      candidate.source === "curated_prompt" ||
      candidate.source === "medical_response_seed" ||
      candidate.source === "medical_qa_twi_draft"
    );
  }
  return true;
}

export async function POST(request: Request) {
  const reviewer = await getReviewer();
  if (!reviewer) return jsonError("Research review access is not enabled for this account.", 403);

  const payload = await request.json().catch(() => null);
  const parsed = understandingReviewInputSchema.safeParse(payload);
  if (!parsed.success) {
    return jsonError("Review could not be saved. Check the fields and try again.", 400, {
      details: parsed.error.issues,
    });
  }

  try {
    const review = await saveUnderstandingReview(parsed.data, reviewer);
    return jsonOk({ review });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Review could not be saved.", 400);
  }
}
