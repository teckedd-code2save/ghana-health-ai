import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const buckets = [
  "health_twi",
  "commerce_twi",
  "codeswitch_tw_en",
  "health_en",
  "phone_noise",
] as const;

const rowSchema = z
  .looseObject({
    id: z.string().min(3),
    bucket: z.enum(buckets),
    language: z.enum(["tw", "ak", "en", "tw-en", "ak-en", "ga"]),
    reference: z.string().min(1),
    speaker_label: z.string().min(3).optional(),
    speaker: z.string().min(2).optional(),
    consent: z.enum(["internal_eval", "public_release", "research_consent"]).optional(),
    duration_s: z.number().positive().optional(),
    sha256: z
      .string()
      .regex(/^[0-9a-f]{64}$/)
      .optional(),
    holdout: z.boolean().optional(),
    take: z.number().int().min(1).optional(),
    dialect: z.enum(["asante", "akuapem", "fante", "other"]).optional(),
    device: z.enum(["phone", "laptop", "other"]).optional(),
    environment: z.enum(["quiet", "street", "market", "home-noise"]).optional(),
    quality_flags: z.array(z.string()).optional(),
    quality: z
      .looseObject({
        rms: z.number().optional(),
        clipping_ratio: z.number().optional(),
        leading_silence_s: z.number().optional(),
        trailing_silence_s: z.number().optional(),
      })
      .optional(),
    domain_tags: z.array(z.string().min(1)).optional(),
    recording_tags: z.array(z.string().min(1)).optional(),
    audio_path: z.string().min(1).optional(),
    local_audio_path: z.string().min(1).optional(),
    filename: z.string().min(1).optional(),
    source_audio_file: z.string().min(1).optional(),
  })
  .refine((row) => row.speaker_label || row.speaker, {
    message: "speaker_label or speaker is required",
  });

type ManifestRow = z.infer<typeof rowSchema>;

const MIN_DURATION_S = 0.5;
const MAX_DURATION_S = 28;
const MIN_NORMALIZED_CHARS = 2;
const MIN_RMS = 0.008;

function normalizeText(text: string) {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function speakerKey(row: ManifestRow) {
  return (row.speaker_label || row.speaker || "").toLowerCase();
}

async function readJsonl(filePath: string) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

async function main() {
  const [newManifestPath, ...existingPaths] = process.argv.slice(2).filter((arg) => arg !== "--");
  if (!newManifestPath) {
    console.error(
      "Usage: node --import tsx scripts/validate-local-asr-import.ts <new-manifest.jsonl> [existing-manifest.jsonl ...]",
    );
    process.exit(1);
  }

  const problems: string[] = [];
  const warnings: string[] = [];

  // Existing manifests are read leniently: they only feed duplicate detection.
  const existingIds = new Set<string>();
  const existingSha256 = new Map<string, string>();
  const existingSpeakerRef = new Map<string, string>();
  let existingRowCount = 0;
  for (const existingPath of existingPaths) {
    const label = path.basename(existingPath);
    const lines = await readJsonl(existingPath);
    lines.forEach((line, index) => {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(line) as Record<string, unknown>;
      } catch (error) {
        problems.push(`${label} line ${index + 1}: invalid JSON (${String(error)})`);
        return;
      }
      existingRowCount += 1;
      if (typeof parsed.id === "string") existingIds.add(parsed.id);
      if (typeof parsed.sha256 === "string") existingSha256.set(parsed.sha256, `${label}#${parsed.id ?? index + 1}`);
      const speaker = String(parsed.speaker_label || parsed.speaker || "").toLowerCase();
      if (speaker && typeof parsed.reference === "string") {
        existingSpeakerRef.set(`${speaker}::${normalizeText(parsed.reference)}`, `${label}#${parsed.id ?? index + 1}`);
      }
    });
  }

  const rows: ManifestRow[] = [];
  const ids = new Set<string>();
  const sha256Seen = new Map<string, number>();
  const speakerRefSeen = new Map<string, { lineNo: number; take?: number }>();

  const lines = await readJsonl(newManifestPath);
  lines.forEach((line, index) => {
    const lineNo = index + 1;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      problems.push(`line ${lineNo}: invalid JSON (${String(error)})`);
      return;
    }

    const result = rowSchema.safeParse(parsed);
    if (!result.success) {
      problems.push(`line ${lineNo}: ${result.error.issues.map((issue) => issue.message).join("; ")}`);
      return;
    }
    const row = result.data;

    if (ids.has(row.id)) problems.push(`line ${lineNo}: duplicate id ${row.id}`);
    ids.add(row.id);
    if (existingIds.has(row.id)) problems.push(`line ${lineNo}: id ${row.id} already exists in an imported manifest`);

    if (row.duration_s !== undefined && (row.duration_s < MIN_DURATION_S || row.duration_s > MAX_DURATION_S)) {
      problems.push(
        `line ${lineNo}: duration_s ${row.duration_s} outside ${MIN_DURATION_S}-${MAX_DURATION_S}s bounds`,
      );
    }

    const normalized = normalizeText(row.reference);
    if (normalized.replace(/\s/g, "").length < MIN_NORMALIZED_CHARS) {
      problems.push(`line ${lineNo}: reference has fewer than ${MIN_NORMALIZED_CHARS} normalized characters`);
    }

    if (row.sha256) {
      const firstLine = sha256Seen.get(row.sha256);
      if (firstLine !== undefined) {
        problems.push(`line ${lineNo}: sha256 duplicates line ${firstLine} (exact audio duplicate)`);
      }
      sha256Seen.set(row.sha256, lineNo);
      const existingMatch = existingSha256.get(row.sha256);
      if (existingMatch) {
        problems.push(`line ${lineNo}: sha256 already imported as ${existingMatch} (exact audio duplicate)`);
      }
    }

    const nearKey = `${speakerKey(row)}::${normalized}`;
    const previous = speakerRefSeen.get(nearKey);
    if (previous) {
      const intentionalRetake = row.take !== undefined && previous.take !== undefined && row.take !== previous.take;
      if (!intentionalRetake) {
        problems.push(`line ${lineNo}: near-duplicate of line ${previous.lineNo} (same speaker + same reference)`);
      }
    }
    speakerRefSeen.set(nearKey, { lineNo, take: row.take });
    const existingNear = existingSpeakerRef.get(nearKey);
    if (existingNear && !(row.take !== undefined && row.take >= 2)) {
      problems.push(`line ${lineNo}: near-duplicate of ${existingNear} (same speaker + same reference)`);
    }

    if (row.quality_flags?.length) {
      warnings.push(`line ${lineNo}: quality flags [${row.quality_flags.join(", ")}]`);
    }
    if (row.quality?.rms !== undefined && row.quality.rms < MIN_RMS) {
      warnings.push(`line ${lineNo}: rms ${row.quality.rms} below serving threshold ${MIN_RMS}`);
    }

    rows.push(row);
  });

  const counts = Object.fromEntries(buckets.map((bucket) => [bucket, { rows: 0, holdout: 0 }])) as Record<
    (typeof buckets)[number],
    { rows: number; holdout: number }
  >;
  for (const row of rows) {
    counts[row.bucket].rows += 1;
    if (row.holdout) counts[row.bucket].holdout += 1;
  }

  console.log(`New manifest: ${newManifestPath} (${rows.length} valid row(s), ${lines.length} line(s))`);
  if (existingPaths.length) {
    console.log(`Existing manifests: ${existingPaths.length} file(s), ${existingRowCount} row(s) cross-checked`);
  }
  console.log("\n| bucket | rows | holdout |");
  console.log("| --- | ---: | ---: |");
  for (const bucket of buckets) {
    console.log(`| ${bucket} | ${counts[bucket].rows} | ${counts[bucket].holdout} |`);
  }
  const totalHoldout = rows.filter((row) => row.holdout).length;
  console.log(`\nHoldout rows: ${totalHoldout} of ${rows.length}`);

  if (problems.length) {
    console.log("\nProblems:");
    for (const problem of problems) console.log(`- ${problem}`);
  }
  if (warnings.length) {
    console.log("\nWarnings (quality only, non-blocking):");
    for (const warning of warnings) console.log(`- ${warning}`);
  }

  console.log(`\nImport ready: ${problems.length ? "no" : "yes"}`);
  if (problems.length) {
    throw new Error(`Import manifest has ${problems.length} validation problem(s)`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
