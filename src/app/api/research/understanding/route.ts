import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import {
  corpusStages,
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

  const [seeds, candidates, reviews, scorecard] = await Promise.all([
    readBenchmarkSeeds(),
    readCorpusCandidates(),
    readUnderstandingReviews(),
    readUnderstandingScorecard(),
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
