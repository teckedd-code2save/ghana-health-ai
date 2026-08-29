import "../src/config/load-env";
import {
  buildUnderstandingTrainingExportFromReviews,
  getCandidateTrainingSplit,
  readCorpusCandidates,
  selectMinimumTrainingReviewCandidates,
  type CorpusCandidate,
  type UnderstandingReview,
} from "../src/lib/research-understanding-store";

function reviewFromCandidate(candidate: CorpusCandidate): UnderstandingReview {
  return {
    id: candidate.id,
    normalizedTwi: candidate.model_proposal.normalized_twi || candidate.text,
    naturalEnglish: candidate.model_proposal.natural_english || "Synthetic gate check only.",
    literalEnglish: candidate.model_proposal.literal_english,
    intent: candidate.model_proposal.intent || "gate_check",
    entities: candidate.model_proposal.entities,
    ambiguities: candidate.model_proposal.ambiguities,
    decision: "reviewed",
    notes: "Synthetic gate validation row; not training data.",
    reviewer: "gate_check",
    createdAt: "2026-08-29T00:00:00.000Z",
    updatedAt: "2026-08-29T00:00:00.000Z",
  };
}

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

async function main() {
  const candidates = await readCorpusCandidates();
  const minimumPack = selectMinimumTrainingReviewCandidates(candidates, [], 20);
  const minimumSplits = minimumPack.reduce(
    (acc, candidate) => {
      acc[getCandidateTrainingSplit(candidate)] += 1;
      return acc;
    },
    { train: 0, dev: 0, test: 0 },
  );
  const minimumDomains = new Set(minimumPack.map((candidate) => candidate.domain));
  assert(minimumPack.length === 20, "Minimum review pack should include exactly 20 rows.");
  assert(minimumSplits.train > 0, "Minimum review pack should include train rows.");
  assert(minimumSplits.dev > 0, "Minimum review pack should include dev rows.");
  assert(minimumSplits.test > 0, "Minimum review pack should include test rows.");
  assert(minimumDomains.has("health"), "Minimum review pack should include health rows.");
  assert(minimumDomains.has("commerce"), "Minimum review pack should include commerce rows.");

  const empty = buildUnderstandingTrainingExportFromReviews(candidates, []);
  assert(!empty.readiness.ready, "Empty review export must not pass the readiness gate.");
  assert(empty.accepted === 0, "Empty review export must not accept rows.");

  const full = buildUnderstandingTrainingExportFromReviews(candidates, candidates.map(reviewFromCandidate));
  assert(full.accepted >= 20, "Reviewed export should include enough rows for the gate fixture.");
  assert(full.splits.train > 0, "Reviewed export should include train rows.");
  assert(full.splits.dev > 0, "Reviewed export should include dev rows.");
  assert(full.splits.test > 0, "Reviewed export should include test rows.");
  assert(full.readiness.ready, "Fully reviewed fixture should pass required readiness checks.");

  console.log(
    JSON.stringify(
      {
        empty: {
          accepted: empty.accepted,
          ready: empty.readiness.ready,
          required_passed: empty.readiness.required_passed,
          required_total: empty.readiness.required_total,
        },
        minimum_pack: {
          rows: minimumPack.length,
          splits: minimumSplits,
          domains: Array.from(minimumDomains).sort(),
        },
        reviewed: {
          accepted: full.accepted,
          splits: full.splits,
          ready: full.readiness.ready,
          required_passed: full.readiness.required_passed,
          required_total: full.readiness.required_total,
        },
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
