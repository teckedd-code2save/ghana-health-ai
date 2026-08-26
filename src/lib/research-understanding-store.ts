import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { z } from "zod";

const root = process.cwd();
const seedPath = path.join(root, "data", "understanding-benchmark", "seed.v0.jsonl");
const reviewDir = path.join(root, "tmp", "understanding-review");
const reviewPath = path.join(reviewDir, "reviews.v0.jsonl");

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

export type BenchmarkSeed = z.infer<typeof benchmarkSeedSchema>;
export type UnderstandingReview = z.infer<typeof reviewSchema>;

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

export async function readUnderstandingReviews(): Promise<UnderstandingReview[]> {
  try {
    const raw = await readFile(reviewPath, "utf8");
    return parseJsonl(raw, (value) => reviewSchema.parse(value));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

export async function saveUnderstandingReview(
  input: z.infer<typeof understandingReviewInputSchema>,
  reviewer: string,
) {
  const seeds = await readBenchmarkSeeds();
  if (!seeds.some((seed) => seed.id === input.id)) {
    throw new Error("Unknown benchmark row");
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
  const merged = [...existing.filter((review) => review.id !== input.id), next].sort((a, b) =>
    a.id.localeCompare(b.id),
  );

  await mkdir(reviewDir, { recursive: true });
  await writeFile(reviewPath, `${merged.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  return next;
}
