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

const promotionMinimums: Record<(typeof buckets)[number], number> = {
  health_twi: 100,
  commerce_twi: 70,
  codeswitch_tw_en: 60,
  health_en: 50,
  phone_noise: 20,
};

const rowSchema = z.looseObject({
  id: z.string().min(3),
  bucket: z.enum(buckets),
  language: z.enum(["tw", "ak", "en", "tw-en", "ak-en", "ga"]),
  reference: z.string().min(1),
  speaker_label: z.string().min(3).optional(),
  speaker: z.string().min(2).optional(),
  audio_path: z.string().min(1).optional(),
  local_audio_path: z.string().min(1).optional(),
  duration_s: z.number().positive().optional(),
  sha256: z.string().optional(),
  holdout: z.boolean().optional(),
  source: z.string().optional(),
  consent: z.string().optional(),
  domain_tags: z.array(z.string()).optional(),
  recording_tags: z.array(z.string()).optional(),
  quality_flags: z.array(z.string()).optional(),
});

type Row = z.infer<typeof rowSchema>;

type BucketStats = {
  rows: number;
  holdout: number;
  speakers: Set<string>;
  duration: number;
  synthetic: number;
  missingAudio: number;
};

type Args = {
  manifest: string;
  strict: boolean;
  checkFiles: boolean;
};

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (name: string, fallback = "") => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? argv[i + 1] : fallback;
  };
  return {
    manifest:
      get("manifest") ||
      process.env.ASR_CORPUS_MANIFEST ||
      path.join(process.cwd(), "tmp", "asr-local-train", "manifest.jsonl"),
    strict: argv.includes("--strict"),
    checkFiles: argv.includes("--check-files"),
  };
}

function normalizeText(text: string) {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function speakerOf(row: Row) {
  return row.speaker_label || row.speaker || "unknown";
}

function audioPathOf(row: Row) {
  return row.local_audio_path || row.audio_path || "";
}

async function exists(filePath: string) {
  if (!filePath || filePath.startsWith("MISSING_AUDIO")) return false;
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const args = parseArgs();
  const raw = await fs.readFile(args.manifest, "utf8");
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const rows: Row[] = [];
  const problems: string[] = [];
  const warnings: string[] = [];
  const ids = new Map<string, number>();
  const hashes = new Map<string, number>();
  const bySpeakerReference = new Map<string, number>();
  const byReference = new Map<string, number[]>();

  for (const [index, line] of lines.entries()) {
    const lineNo = index + 1;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      problems.push(`line ${lineNo}: invalid JSON (${String(error)})`);
      continue;
    }
    const result = rowSchema.safeParse(parsed);
    if (!result.success) {
      problems.push(`line ${lineNo}: ${result.error.issues.map((issue) => issue.message).join("; ")}`);
      continue;
    }
    const row = result.data;
    rows.push(row);

    const idLine = ids.get(row.id);
    if (idLine) problems.push(`line ${lineNo}: duplicate id ${row.id} first seen on line ${idLine}`);
    ids.set(row.id, lineNo);

    if (!row.speaker_label && !row.speaker) problems.push(`line ${lineNo}: missing speaker_label/speaker`);
    if (!audioPathOf(row)) problems.push(`line ${lineNo}: missing audio path`);
    if (row.duration_s !== undefined && (row.duration_s < 0.5 || row.duration_s > 28)) {
      warnings.push(`line ${lineNo}: duration ${row.duration_s}s is outside the 0.5-28s training target`);
    }
    if (row.holdout && row.source === "synthetic_tts") {
      problems.push(`line ${lineNo}: synthetic_tts row cannot be holdout evidence`);
    }

    if (row.sha256) {
      const firstHashLine = hashes.get(row.sha256);
      if (firstHashLine) problems.push(`line ${lineNo}: duplicate sha256 first seen on line ${firstHashLine}`);
      hashes.set(row.sha256, lineNo);
    }

    const normalized = normalizeText(row.reference);
    const referenceLines = byReference.get(normalized) || [];
    referenceLines.push(lineNo);
    byReference.set(normalized, referenceLines);

    const speakerRefKey = `${speakerOf(row).toLowerCase()}::${normalized}`;
    const firstSpeakerRef = bySpeakerReference.get(speakerRefKey);
    if (firstSpeakerRef) {
      warnings.push(`line ${lineNo}: same speaker/reference as line ${firstSpeakerRef}`);
    }
    bySpeakerReference.set(speakerRefKey, lineNo);
  }

  if (args.checkFiles) {
    for (const [index, row] of rows.entries()) {
      const filePath = audioPathOf(row);
      if (!(await exists(filePath))) {
        problems.push(`line ${index + 1}: audio file not found at ${filePath || "(missing)"}`);
      }
    }
  }

  const stats = Object.fromEntries(
    buckets.map((bucket) => [
      bucket,
      { rows: 0, holdout: 0, speakers: new Set<string>(), duration: 0, synthetic: 0, missingAudio: 0 },
    ]),
  ) as Record<(typeof buckets)[number], BucketStats>;

  const allSpeakers = new Set<string>();
  let holdoutRows = 0;
  let syntheticRows = 0;
  let totalDuration = 0;
  for (const row of rows) {
    const bucket = row.bucket;
    const speaker = speakerOf(row);
    stats[bucket].rows += 1;
    stats[bucket].speakers.add(speaker);
    allSpeakers.add(speaker);
    if (row.holdout) {
      stats[bucket].holdout += 1;
      holdoutRows += 1;
    }
    if (row.source === "synthetic_tts") {
      stats[bucket].synthetic += 1;
      syntheticRows += 1;
    }
    if (!audioPathOf(row) || audioPathOf(row).startsWith("MISSING_AUDIO")) stats[bucket].missingAudio += 1;
    if (row.duration_s) {
      stats[bucket].duration += row.duration_s;
      totalDuration += row.duration_s;
    }
  }

  const repeatedReferences = [...byReference.entries()]
    .filter(([, lineNos]) => lineNos.length > 1)
    .sort((a, b) => b[1].length - a[1].length);

  console.log(`Manifest: ${args.manifest}`);
  console.log(`Rows: ${rows.length}`);
  console.log(`Speakers: ${allSpeakers.size}`);
  console.log(`Holdout rows: ${holdoutRows}`);
  console.log(`Synthetic rows: ${syntheticRows}`);
  console.log(`Duration: ${(totalDuration / 60).toFixed(1)} min`);
  console.log("\n| bucket | rows | target | holdout | speakers | synthetic | missing audio | minutes | status |");
  console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
  for (const bucket of buckets) {
    const stat = stats[bucket];
    const target = promotionMinimums[bucket];
    const status = stat.rows >= target ? "COUNT_READY" : "COLLECT_MORE";
    console.log(
      `| ${bucket} | ${stat.rows} | ${target} | ${stat.holdout} | ${stat.speakers.size} | ${stat.synthetic} | ${stat.missingAudio} | ${(stat.duration / 60).toFixed(1)} | ${status} |`,
    );
  }

  console.log("\nPromotion gates:");
  console.log(`- bucket counts: ${buckets.every((bucket) => stats[bucket].rows >= promotionMinimums[bucket]) ? "pass" : "collect_more"}`);
  console.log(`- speaker diversity >= 6: ${allSpeakers.size >= 6 ? "pass" : "collect_more"}`);
  console.log(`- holdout rows >= 250: ${holdoutRows >= 250 ? "pass" : "collect_more"}`);
  console.log(`- synthetic holdout rows: ${rows.some((row) => row.holdout && row.source === "synthetic_tts") ? "fail" : "pass"}`);

  if (repeatedReferences.length) {
    console.log("\nRepeated references across speakers or takes:");
    for (const [reference, lineNos] of repeatedReferences.slice(0, 15)) {
      console.log(`- ${lineNos.length}x lines ${lineNos.join(", ")}: ${reference}`);
    }
    if (repeatedReferences.length > 15) console.log(`- ... ${repeatedReferences.length - 15} more`);
  }

  if (warnings.length) {
    console.log("\nWarnings:");
    for (const warning of warnings.slice(0, 30)) console.log(`- ${warning}`);
    if (warnings.length > 30) console.log(`- ... ${warnings.length - 30} more`);
  }

  if (problems.length) {
    console.log("\nProblems:");
    for (const problem of problems.slice(0, 30)) console.log(`- ${problem}`);
    if (problems.length > 30) console.log(`- ... ${problems.length - 30} more`);
  }

  const gateReady =
    buckets.every((bucket) => stats[bucket].rows >= promotionMinimums[bucket]) &&
    allSpeakers.size >= 6 &&
    holdoutRows >= 250 &&
    !rows.some((row) => row.holdout && row.source === "synthetic_tts") &&
    problems.length === 0;
  console.log(`\nTraining promotion-corpus ready: ${gateReady ? "yes" : "no"}`);

  if (problems.length || (args.strict && !gateReady)) {
    throw new Error(
      problems.length
        ? `Corpus has ${problems.length} problem(s)`
        : "Corpus does not meet strict promotion readiness",
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
