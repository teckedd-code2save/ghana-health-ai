import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import {
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

export async function GET() {
  const reviewer = await getReviewer();
  if (!reviewer) return jsonError("Research review access is not enabled for this account.", 403);

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
  const corpusRows = candidates.map((candidate) => ({
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
  const corpusCompleted = corpusRows.filter((row) => row.review?.decision === "reviewed").length;
  const candidateSplits = corpusRows.reduce(
    (acc, row) => {
      if (row.trainingSplit) acc[row.trainingSplit] += 1;
      return acc;
    },
    { train: 0, dev: 0, test: 0 },
  );
  const sourceSummary = corpusRows.reduce<
    Record<string, { total: number; draftAnnotated: number; reviewed: number; excluded: number }>
  >((acc, row) => {
    const source = row.source ?? "unknown";
    acc[source] ??= { total: 0, draftAnnotated: 0, reviewed: 0, excluded: 0 };
    acc[source].total += 1;
    if (row.modelProposal.status === "draft") acc[source].draftAnnotated += 1;
    if (row.review?.decision === "reviewed") acc[source].reviewed += 1;
    if (row.review?.decision === "exclude") acc[source].excluded += 1;
    return acc;
  }, {});

  return jsonOk({
    reviewer,
    corpus: {
      sourceInventory,
      storageDecisions,
      stages: corpusStages,
    },
    benchmark: {
      rows,
      total: rows.length,
      completed,
      needsSecondReview,
      excluded,
      scorecard,
    },
    candidates: {
      rows: corpusRows,
      total: corpusRows.length,
      completed: corpusCompleted,
      withAudio: corpusRows.filter((row) => row.audioArtifactId).length,
      draftAnnotated: corpusRows.filter((row) => row.modelProposal.status === "draft").length,
      trainingReady: trainingExport.accepted,
      splits: trainingExport.splits,
      candidateSplits,
      sourceSummary,
      readiness: trainingExport.readiness,
    },
  });
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
