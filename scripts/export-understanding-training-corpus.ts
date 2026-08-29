import fs from "node:fs/promises";
import path from "node:path";

type Candidate = {
  id: string;
  source: string;
  source_record_id: string;
  source_path: string;
  language: string;
  dialect: string;
  domain: string;
  split: string;
  text: string;
  normalized_text: string;
  audio_artifact_id: string | null;
  speaker_id: string | null;
  duration_seconds: number | null;
  consent_scope: string;
  source_hash: string;
  duplicate_key: string;
};

type Review = {
  id: string;
  normalizedTwi: string;
  naturalEnglish: string;
  literalEnglish: string;
  intent: string;
  entities: string;
  ambiguities: string;
  decision: "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
  notes: string;
  reviewer: string;
  updatedAt?: string;
};

const root = process.cwd();
const defaultCandidates = path.join(root, "data", "understanding-corpus", "candidates.v0.jsonl");
const defaultReviews = path.join(root, "tmp", "understanding-review", "reviews.v0.jsonl");
const defaultOutDir = path.join(root, "tmp", "understanding-corpus", "exports", "v0");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function parseJsonl<T>(raw: string): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

function stableBucket(id: string) {
  let hash = 0;
  for (const char of id) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  const mod = hash % 100;
  if (mod < 80) return "train";
  if (mod < 90) return "dev";
  return "test";
}

function required(value: string | undefined | null) {
  return typeof value === "string" && value.trim().length > 0;
}

function parseEntities(value: string) {
  if (!value.trim()) return [];
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

async function main() {
  const candidatePath = argValue("--candidates", defaultCandidates);
  const reviewPath = argValue("--reviews", defaultReviews);
  const outDir = argValue("--out-dir", defaultOutDir);
  const strict = process.argv.includes("--strict");

  const candidates = parseJsonl<Candidate>(await fs.readFile(candidatePath, "utf8"));
  let reviews: Review[] = [];
  try {
    reviews = parseJsonl<Review>(await fs.readFile(reviewPath, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const accepted = [];
  const rejected = [];

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
    const missing = [
      ["normalizedTwi", review.normalizedTwi],
      ["naturalEnglish", review.naturalEnglish],
      ["intent", review.intent],
    ]
      .filter(([, value]) => !required(value))
      .map(([key]) => key);
    if (missing.length) {
      rejected.push({ id: review.id, reason: `missing_${missing.join("_")}` });
      continue;
    }

    const split = stableBucket(candidate.speaker_id || candidate.duplicate_key || candidate.id);
    accepted.push({
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

  const bySplit = {
    train: accepted.filter((row) => row.split === "train"),
    dev: accepted.filter((row) => row.split === "dev"),
    test: accepted.filter((row) => row.split === "test"),
  };
  const summary = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    candidates: candidates.length,
    reviews: reviews.length,
    accepted: accepted.length,
    rejected: rejected.length,
    splits: Object.fromEntries(Object.entries(bySplit).map(([split, rows]) => [split, rows.length])),
    rejected_reasons: rejected.reduce<Record<string, number>>((acc, row) => {
      acc[row.reason] = (acc[row.reason] ?? 0) + 1;
      return acc;
    }, {}),
  };

  await fs.mkdir(outDir, { recursive: true });
  await Promise.all(
    Object.entries(bySplit).map(([split, rows]) =>
      fs.writeFile(
        path.join(outDir, `${split}.jsonl`),
        rows.length ? `${rows.map((row) => JSON.stringify(row)).join("\n")}\n` : "",
        "utf8",
      ),
    ),
  );
  await fs.writeFile(path.join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify({ outDir, ...summary }, null, 2));

  if (strict && accepted.length === 0) {
    throw new Error("No reviewed rows are eligible for training export yet.");
  }
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
