import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { prisma } from "@/db/prisma";
import { z } from "zod";

const seedPath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data",
  "understanding-benchmark",
  "seed.v0.jsonl",
);
const reviewDir = path.join(/* turbopackIgnore: true */ process.cwd(), "tmp", "understanding-review");
const reviewPath = path.join(reviewDir, "reviews.v0.jsonl");
const scorecardPath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "tmp",
  "understanding-results",
  "scorecard.v0.json",
);
const committedScorecardPath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data",
  "understanding-benchmark",
  "scorecard.v0.json",
);
const candidatePath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "tmp",
  "understanding-corpus",
  "candidates.v0.jsonl",
);
const committedCandidatePath = path.join(
  /* turbopackIgnore: true */ process.cwd(),
  "data",
  "understanding-corpus",
  "candidates.v0.jsonl",
);

export const sourceInventory = [
  {
    id: "ghananlp-parallel",
    name: "GhanaNLP Twi-English parallel",
    role: "translation baseline and training candidate",
    evidence: "text pairs",
    storage: "Hugging Face reference, pinned revision, hashes in Postgres",
    status: "licence and split audit needed",
  },
  {
    id: "waxal-akan",
    name: "WAXAL Akan ASR corpus",
    role: "Twi transcripts, ASR-noise experiments, ASR evaluation",
    evidence: "audio and source transcript",
    storage: "dataset reference plus private derived annotations",
    status: "needs English meaning review before understanding eval",
  },
  {
    id: "common-voice-twi",
    name: "Common Voice Twi",
    role: "speaker and acoustic diversity for ASR/adaptation",
    evidence: "licensed audio and transcript",
    storage: "upstream reference, split-aware import manifest",
    status: "use only according to licence and split policy",
  },
  {
    id: "local-recordings",
    name: "Local recorded corpus",
    role: "product-domain probes and consented training candidates",
    evidence: "consented audio, transcript, speaker metadata",
    storage: "private object storage for audio, Postgres references and reviews",
    status: "review and leakage-safe split required",
  },
  {
    id: "product-contributions",
    name: "Opt-in product contributions",
    role: "real-world corrections and domain expansion",
    evidence: "voice turn, ASR output, correction, meaning correction",
    storage: "consent-scoped private storage with deletion controls",
    status: "requires explicit research consent before collection",
  },
  {
    id: "synthetic-tts",
    name: "Reviewed synthetic TTS",
    role: "augmentation and stress tests, never final evaluation",
    evidence: "reviewed source text plus generated audio",
    storage: "artifact references with TTS model, date, checksum",
    status: "allowed only after source text review",
  },
] as const;

export const storageDecisions = [
  {
    material: "Public datasets",
    systemOfRecord: "Original repository",
    rule: "Pin immutable revision, store identifiers and checksums locally.",
  },
  {
    material: "Model weights",
    systemOfRecord: "Hugging Face model repository",
    rule: "Do not copy weights into the app; Modal cache is disposable.",
  },
  {
    material: "Benchmark jobs and reviews",
    systemOfRecord: "PostgreSQL",
    rule: "Store job metadata, reviewer decisions, labels, and artifact pointers.",
  },
  {
    material: "Private audio",
    systemOfRecord: "Private object storage",
    rule: "Keep deletion and withdrawal possible; store references, not blobs, in Postgres.",
  },
  {
    material: "Small reports",
    systemOfRecord: "Versioned JSON/Parquet artifact",
    rule: "Keep predictions reproducible with model revision and payload class.",
  },
] as const;

export const corpusStages = [
  "Inventory licences, revisions, splits, speaker IDs, and hashes.",
  "Import references only; preserve upstream transcript and audio identity.",
  "Run proposal models for normalized Twi, English meaning, semantics, and uncertainty.",
  "Route high-risk, uncertain, duplicate, and code-switched rows to review.",
  "Freeze eligible train, validation, and test splits without speaker or meaning leakage.",
  "Export only reviewed, eligible records for training or evaluation.",
] as const;

const benchmarkSeedSchema = z.object({
  id: z.string().min(1),
  category: z.string().min(1),
  text: z.string().min(1),
  review_status: z.string().default("unverified"),
});

const corpusCandidateSchema = z.object({
  id: z.string().min(1),
  record_id: z.string().min(1),
  source: z.string().min(1),
  source_record_id: z.string().min(1),
  source_path: z.string().min(1),
  language: z.string().min(1),
  dialect: z.string().default("unknown"),
  domain: z.string().min(1),
  split: z.string().default("unknown"),
  text: z.string().min(1),
  normalized_text: z.string().default(""),
  audio_artifact_id: z.string().nullable().default(null),
  speaker_id: z.string().nullable().default(null),
  duration_seconds: z.number().nullable().default(null),
  consent_scope: z.string().min(1),
  source_hash: z.string().min(1),
  duplicate_key: z.string().min(1),
  review_status: z.string().default("needs_review"),
  eligible_for_training: z.boolean().default(false),
  eligible_for_final_evaluation: z.boolean().default(false),
  model_proposal: z
    .object({
      normalized_twi: z.string().default(""),
      natural_english: z.string().default(""),
      literal_english: z.string().default(""),
      intent: z.string().default(""),
      entities: z.string().default(""),
      ambiguities: z.string().default(""),
      requires_clarification: z.boolean().default(false),
      model: z.string().default("none"),
      status: z.enum(["not_requested", "draft"]).default("not_requested"),
    })
    .default({
      normalized_twi: "",
      natural_english: "",
      literal_english: "",
      intent: "",
      entities: "",
      ambiguities: "",
      requires_clarification: false,
      model: "none",
      status: "not_requested",
    }),
});

const reviewSchema = z.object({
  id: z.string().min(1),
  normalizedTwi: z.string().max(1500).default(""),
  naturalEnglish: z.string().max(1500).default(""),
  literalEnglish: z.string().max(1500).default(""),
  intent: z.string().max(120).default(""),
  entities: z.string().max(2500).default(""),
  ambiguities: z.string().max(2000).default(""),
  decision: z.enum(["unreviewed", "reviewed", "needs_second_review", "exclude"]).default("unreviewed"),
  notes: z.string().max(2000).default(""),
  reviewer: z.string().max(200).default("local_reviewer"),
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
});

const scorecardSchema = z.object({
  schema_version: z.number(),
  created_at: z.string(),
  rubric: z.string(),
  scorecards: z.array(
    z.object({
      candidate: z.string(),
      model_id: z.string(),
      adapter_id: z.string().nullable().optional(),
      artifact: z.string(),
      cases_scored: z.number(),
      exact_cases: z.number(),
      checks: z.number(),
      passed: z.number(),
      score: z.number(),
      elapsed_seconds: z.number(),
      critical_failures: z.array(
        z.object({
          id: z.string(),
          category: z.string(),
          text: z.string(),
          prediction: z.string(),
          failures: z.array(z.string()),
        }),
      ),
    }),
  ),
  decision_hint: z.string(),
});

export type BenchmarkSeed = z.infer<typeof benchmarkSeedSchema>;
export type CorpusCandidate = z.infer<typeof corpusCandidateSchema>;
export type UnderstandingReview = z.infer<typeof reviewSchema>;
export type UnderstandingScorecard = z.infer<typeof scorecardSchema>;
export type UnderstandingTrainingRow = {
  id: string;
  split: "train" | "dev" | "test";
  domain: string;
  language: string;
  dialect: string;
  source: string;
  source_record_id: string;
  source_path: string;
  source_hash: string;
  duplicate_key: string;
  consent_scope: string;
  audio_artifact_id: string | null;
  speaker_id: string | null;
  duration_seconds: number | null;
  original_text: string;
  normalized_twi: string;
  natural_english: string;
  literal_english: string;
  intent: string;
  entities: unknown;
  ambiguities: string;
  reviewer: string;
  reviewed_at: string | null;
  eligible_for_training: true;
  eligible_for_final_evaluation: boolean;
};
export type UnderstandingReadinessCheck = {
  id: string;
  label: string;
  passed: boolean;
  value: number | string | boolean;
  required: number | string | boolean;
  severity: "required" | "warning";
};

const reviewSheetColumns = [
  "id",
  "domain",
  "language",
  "source",
  "source_record_id",
  "source_path",
  "audio_artifact_id",
  "speaker_id",
  "consent_scope",
  "original_text",
  "draft_normalized_twi",
  "draft_natural_english",
  "draft_literal_english",
  "draft_intent",
  "draft_entities",
  "draft_ambiguities",
  "review_normalized_twi",
  "review_natural_english",
  "review_literal_english",
  "review_intent",
  "review_entities",
  "review_ambiguities",
  "decision",
  "review_notes",
  "reviewer",
] as const;

export const understandingReviewInputSchema = reviewSchema.omit({
  reviewer: true,
  createdAt: true,
  updatedAt: true,
});

function parseJsonl<T>(raw: string, parse: (value: unknown) => T): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => parse(JSON.parse(line)));
}

export async function readBenchmarkSeeds(): Promise<BenchmarkSeed[]> {
  const raw = await readFile(seedPath, "utf8");
  return parseJsonl(raw, (value) => benchmarkSeedSchema.parse(value));
}

export async function readCorpusCandidates(): Promise<CorpusCandidate[]> {
  for (const filePath of [candidatePath, committedCandidatePath]) {
    try {
      const raw = await readFile(filePath, "utf8");
      return parseJsonl(raw, (value) => corpusCandidateSchema.parse(value));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return [];
}

export async function readUnderstandingReviews(): Promise<UnderstandingReview[]> {
  try {
    const rows = await prisma.researchUnderstandingReview.findMany({
      orderBy: { rowId: "asc" },
    });
    return rows.map((row) =>
      reviewSchema.parse({
        id: row.rowId,
        normalizedTwi: row.normalizedTwi,
        naturalEnglish: row.naturalEnglish,
        literalEnglish: row.literalEnglish,
        intent: row.intent,
        entities: row.entities,
        ambiguities: row.ambiguities,
        decision: row.decision,
        notes: row.notes,
        reviewer: row.reviewer,
        createdAt: row.createdAt.toISOString(),
        updatedAt: row.updatedAt.toISOString(),
      }),
    );
  } catch (error) {
    console.warn(
      "[research-understanding] database reviews unavailable; using local review file",
      error instanceof Error ? error.message : String(error),
    );
  }

  return readFileUnderstandingReviews();
}

export async function readFileUnderstandingReviews(): Promise<UnderstandingReview[]> {
  try {
    const raw = await readFile(reviewPath, "utf8");
    return parseJsonl(raw, (value) => reviewSchema.parse(value));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

export async function readUnderstandingScorecard(): Promise<UnderstandingScorecard | null> {
  for (const filePath of [scorecardPath, committedScorecardPath]) {
    try {
      const raw = await readFile(filePath, "utf8");
      return scorecardSchema.parse(JSON.parse(raw));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return null;
}

export async function saveUnderstandingReview(
  input: z.infer<typeof understandingReviewInputSchema>,
  reviewer: string,
) {
  const [seeds, candidates] = await Promise.all([readBenchmarkSeeds(), readCorpusCandidates()]);
  const known =
    seeds.some((seed) => seed.id === input.id) ||
    candidates.some((candidate) => candidate.id === input.id);
  if (!known) {
    throw new Error("Unknown research row");
  }

  const now = new Date().toISOString();
  const existing = await readUnderstandingReviews();
  const prior = existing.find((review) => review.id === input.id);
  const next = reviewSchema.parse({
    ...input,
    reviewer,
    createdAt: prior?.createdAt ?? now,
    updatedAt: now,
  });

  try {
    await prisma.researchUnderstandingReview.upsert({
      where: { rowId: input.id },
      update: {
        rowKind: seeds.some((seed) => seed.id === input.id) ? "benchmark" : "corpus",
        normalizedTwi: next.normalizedTwi,
        naturalEnglish: next.naturalEnglish,
        literalEnglish: next.literalEnglish,
        intent: next.intent,
        entities: next.entities,
        ambiguities: next.ambiguities,
        decision: next.decision,
        notes: next.notes,
        reviewer,
      },
      create: {
        rowId: input.id,
        rowKind: seeds.some((seed) => seed.id === input.id) ? "benchmark" : "corpus",
        normalizedTwi: next.normalizedTwi,
        naturalEnglish: next.naturalEnglish,
        literalEnglish: next.literalEnglish,
        intent: next.intent,
        entities: next.entities,
        ambiguities: next.ambiguities,
        decision: next.decision,
        notes: next.notes,
        reviewer,
      },
    });
    return next;
  } catch (error) {
    console.error("[research-understanding] falling back to file review save", error);
  }

  const merged = [...existing.filter((review) => review.id !== input.id), next].sort((a, b) =>
    a.id.localeCompare(b.id),
  );

  await mkdir(reviewDir, { recursive: true });
  await writeFile(reviewPath, `${merged.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  return next;
}

function stableBucket(id: string): "train" | "dev" | "test" {
  let hash = 0;
  for (const char of id) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  const mod = hash % 100;
  if (mod < 80) return "train";
  if (mod < 90) return "dev";
  return "test";
}

function hasRequiredReviewFields(review: UnderstandingReview) {
  return (
    review.normalizedTwi.trim().length > 0 &&
    review.naturalEnglish.trim().length > 0 &&
    review.intent.trim().length > 0
  );
}

function parseEntities(value: string): unknown {
  if (!value.trim()) return [];
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function csvCell(value: unknown) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function csvLine(values: readonly unknown[]) {
  return `${values.map(csvCell).join(",")}\n`;
}

export async function buildUnderstandingReviewSheetCsv() {
  const [candidates, reviews] = await Promise.all([
    readCorpusCandidates(),
    readUnderstandingReviews(),
  ]);
  return buildUnderstandingReviewSheetCsvFromReviews(candidates, reviews);
}

export function buildUnderstandingReviewSheetCsvFromReviews(
  candidates: CorpusCandidate[],
  reviews: UnderstandingReview[],
) {
  const reviewById = new Map(reviews.map((review) => [review.id, review]));
  let csv = csvLine(reviewSheetColumns);

  for (const candidate of candidates) {
    const review = reviewById.get(candidate.id);
    csv += csvLine([
      candidate.id,
      candidate.domain,
      candidate.language,
      candidate.source,
      candidate.source_record_id,
      candidate.source_path,
      candidate.audio_artifact_id ?? "",
      candidate.speaker_id ?? "",
      candidate.consent_scope,
      candidate.text,
      candidate.model_proposal.normalized_twi,
      candidate.model_proposal.natural_english,
      candidate.model_proposal.literal_english,
      candidate.model_proposal.intent,
      candidate.model_proposal.entities,
      candidate.model_proposal.ambiguities,
      review?.normalizedTwi ?? "",
      review?.naturalEnglish ?? "",
      review?.literalEnglish ?? "",
      review?.intent ?? "",
      review?.entities ?? "",
      review?.ambiguities ?? "",
      review?.decision ?? "unreviewed",
      review?.notes ?? "",
      review?.reviewer ?? "",
    ]);
  }

  return csv;
}

function buildReadinessChecks(rows: UnderstandingTrainingRow[]): UnderstandingReadinessCheck[] {
  const duplicateKeys = new Map<string, number>();
  const domains = new Set(rows.map((row) => row.domain));
  const splits = new Set(rows.map((row) => row.split));
  for (const row of rows) {
    duplicateKeys.set(row.duplicate_key, (duplicateKeys.get(row.duplicate_key) ?? 0) + 1);
  }
  const duplicateCount = Array.from(duplicateKeys.values()).filter((count) => count > 1).length;
  const rowsWithConsent = rows.filter((row) => row.consent_scope.trim().length > 0).length;

  return [
    {
      id: "minimum_rows",
      label: "At least 20 reviewed rows",
      passed: rows.length >= 20,
      value: rows.length,
      required: 20,
      severity: "required",
    },
    {
      id: "train_split",
      label: "Train split has rows",
      passed: splits.has("train"),
      value: splits.has("train"),
      required: true,
      severity: "required",
    },
    {
      id: "dev_split",
      label: "Dev split has rows",
      passed: splits.has("dev"),
      value: splits.has("dev"),
      required: true,
      severity: "required",
    },
    {
      id: "test_split",
      label: "Test split has rows",
      passed: splits.has("test"),
      value: splits.has("test"),
      required: true,
      severity: "required",
    },
    {
      id: "health_domain",
      label: "Health domain represented",
      passed: domains.has("health"),
      value: domains.has("health"),
      required: true,
      severity: "required",
    },
    {
      id: "no_duplicate_keys",
      label: "No duplicate meaning keys in export",
      passed: duplicateCount === 0,
      value: duplicateCount,
      required: 0,
      severity: "required",
    },
    {
      id: "consent_scope",
      label: "Every row has consent scope",
      passed: rows.length > 0 && rowsWithConsent === rows.length,
      value: `${rowsWithConsent}/${rows.length}`,
      required: "all",
      severity: "required",
    },
    {
      id: "commerce_domain",
      label: "Commerce domain represented",
      passed: domains.has("commerce"),
      value: domains.has("commerce"),
      required: true,
      severity: "warning",
    },
  ];
}

export async function buildUnderstandingTrainingExport() {
  const [candidates, reviews] = await Promise.all([
    readCorpusCandidates(),
    readUnderstandingReviews(),
  ]);
  return buildUnderstandingTrainingExportFromReviews(candidates, reviews);
}

export function buildUnderstandingTrainingExportFromReviews(
  candidates: CorpusCandidate[],
  reviews: UnderstandingReview[],
) {
  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const rejected: { id: string; reason: string }[] = [];
  const rows: UnderstandingTrainingRow[] = [];

  for (const review of reviews) {
    const candidate = candidateById.get(review.id);
    if (!candidate) {
      rejected.push({ id: review.id, reason: "review_without_candidate" });
      continue;
    }
    if (review.decision !== "reviewed") {
      rejected.push({ id: review.id, reason: `decision_${review.decision}` });
      continue;
    }
    if (!hasRequiredReviewFields(review)) {
      rejected.push({ id: review.id, reason: "missing_required_review_fields" });
      continue;
    }

    const split = stableBucket(candidate.speaker_id || candidate.duplicate_key || candidate.id);
    rows.push({
      id: candidate.id,
      split,
      domain: candidate.domain,
      language: candidate.language,
      dialect: candidate.dialect,
      source: candidate.source,
      source_record_id: candidate.source_record_id,
      source_path: candidate.source_path,
      source_hash: candidate.source_hash,
      duplicate_key: candidate.duplicate_key,
      consent_scope: candidate.consent_scope,
      audio_artifact_id: candidate.audio_artifact_id,
      speaker_id: candidate.speaker_id,
      duration_seconds: candidate.duration_seconds,
      original_text: candidate.text,
      normalized_twi: review.normalizedTwi.trim(),
      natural_english: review.naturalEnglish.trim(),
      literal_english: review.literalEnglish.trim(),
      intent: review.intent.trim(),
      entities: parseEntities(review.entities),
      ambiguities: review.ambiguities.trim(),
      reviewer: review.reviewer,
      reviewed_at: review.updatedAt ?? null,
      eligible_for_training: true,
      eligible_for_final_evaluation: split === "test",
    });
  }

  const splits = {
    train: rows.filter((row) => row.split === "train").length,
    dev: rows.filter((row) => row.split === "dev").length,
    test: rows.filter((row) => row.split === "test").length,
  };
  const rejectedReasons = rejected.reduce<Record<string, number>>((acc, row) => {
    acc[row.reason] = (acc[row.reason] ?? 0) + 1;
    return acc;
  }, {});
  const readinessChecks = buildReadinessChecks(rows);
  const requiredChecks = readinessChecks.filter((check) => check.severity === "required");

  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    candidates: candidates.length,
    reviews: reviews.length,
    accepted: rows.length,
    rejected: rejected.length,
    splits,
    rejected_reasons: rejectedReasons,
    readiness: {
      ready: requiredChecks.every((check) => check.passed),
      required_passed: requiredChecks.filter((check) => check.passed).length,
      required_total: requiredChecks.length,
      checks: readinessChecks,
    },
    rows,
  };
}
