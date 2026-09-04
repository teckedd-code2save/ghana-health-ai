import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import {
  type CorpusCandidate,
  corpusStages,
  buildUnderstandingTrainingExport,
  getCandidateTrainingSplit,
  readBenchmarkSeeds,
  readCorpusCandidates,
  readUnderstandingScorecard,
  readUnderstandingReviews,
  saveUnderstandingReview,
  sourceInventory,
  storageDecisions,
  understandingReviewInputSchema,
} from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";
type CorpusRowFilter =
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

  const [seeds, candidates, reviews, scorecard, trainingExport] = await Promise.all([
    readBenchmarkSeeds(),
    readCorpusCandidates(),
    readUnderstandingReviews(),
    readUnderstandingScorecard(),
    buildUnderstandingTrainingExport(),
  ]);
  const reviewById = new Map(reviews.map((review) => [review.id, review]));
  const rows = seeds.map((seed) => ({
    kind: "benchmark" as const,
    ...seed,
    review: reviewById.get(seed.id) ?? null,
  }));
  const candidateRows = candidates
    .filter((candidate) => matchesCorpusRowFilter(candidate, rowFilter))
    .slice(rowOffset, rowOffset + rowLimit);
  const corpusRows = candidateRows.map((candidate) => ({
    kind: "corpus" as const,
    id: candidate.id,
    category: `${candidate.domain}/${candidate.source}`,
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
    review: reviewById.get(candidate.id) ?? null,
  }));
  const completed = rows.filter((row) => row.review?.decision === "reviewed").length;
  const needsSecondReview = rows.filter((row) => row.review?.decision === "needs_second_review").length;
  const excluded = rows.filter((row) => row.review?.decision === "exclude").length;
  const corpusCompleted = candidates.filter((row) => reviewById.get(row.id)?.decision === "reviewed").length;
  const candidateSplits = candidates.reduce(
    (acc, row) => {
      const split = getCandidateTrainingSplit(row);
      acc[split] += 1;
      return acc;
    },
    { train: 0, dev: 0, test: 0 },
  );
  const sourceSummary = candidates.reduce<
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
      total: candidates.length,
      visible: corpusRows.length,
      offset: rowOffset,
      limit: rowLimit,
      filter: rowFilter,
      completed: corpusCompleted,
      withAudio: candidates.filter((row) => row.audio_artifact_id).length,
      draftAnnotated: candidates.filter((row) => row.model_proposal.status === "draft").length,
      trainingReady: trainingExport.accepted,
      splits: trainingExport.splits,
      candidateSplits,
      sourceSummary,
      readiness: trainingExport.readiness,
    },
  });
}

function parseCorpusRowFilter(value: string | null): CorpusRowFilter {
  if (
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
