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

const minimums: Record<(typeof buckets)[number], number> = {
  health_twi: 100,
  commerce_twi: 50,
  codeswitch_tw_en: 50,
  health_en: 50,
  phone_noise: 50,
};

const rowSchema = z.object({
  id: z.string().min(3),
  bucket: z.enum(buckets),
  language: z.enum(["tw", "ak", "en", "tw-en", "ak-en", "ga"]),
  reference: z.string().min(2),
  audio_path: z.string().min(1),
  speaker_label: z.string().min(3),
  consent: z.enum(["internal_eval", "public_release", "research_consent"]),
  domain_tags: z.array(z.string().min(1)).min(1),
  recording_tags: z.array(z.string().min(1)).min(1),
  notes: z.string().optional(),
});

type ManifestRow = z.infer<typeof rowSchema>;

const manifestPath =
  process.env.ASR_PRODUCT_EVAL_MANIFEST ||
  path.join(process.cwd(), "data", "asr-product-eval", "manifest.example.jsonl");
const strict = process.argv.includes("--strict");

function isProbablyPii(text: string) {
  return (
    /\+?\d[\d\s().-]{6,}\d/.test(text) ||
    /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i.test(text)
  );
}

async function main() {
  const raw = await fs.readFile(manifestPath, "utf8");
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const rows: ManifestRow[] = [];
  const ids = new Set<string>();
  const problems: string[] = [];
  const warnings: string[] = [];

  for (const [index, line] of lines.entries()) {
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

    if (ids.has(result.data.id)) {
      problems.push(`line ${lineNo}: duplicate id ${result.data.id}`);
    }
    ids.add(result.data.id);

    if (isProbablyPii(result.data.reference) || isProbablyPii(result.data.notes || "")) {
      problems.push(`line ${lineNo}: possible PII in reference/notes`);
    }
    if (result.data.audio_path.startsWith("MISSING_AUDIO")) {
      warnings.push(`line ${lineNo}: missing audio path placeholder`);
    } else {
      const audioWarning = await localAudioWarning(result.data.audio_path);
      if (audioWarning) warnings.push(`line ${lineNo}: ${audioWarning}`);
    }
    rows.push(result.data);
  }

  const counts = Object.fromEntries(buckets.map((bucket) => [bucket, 0])) as Record<
    (typeof buckets)[number],
    number
  >;
  for (const row of rows) counts[row.bucket] += 1;

  console.log("| bucket | count | minimum | status |");
  console.log("| --- | ---: | ---: | --- |");
  for (const bucket of buckets) {
    const count = counts[bucket];
    const minimum = minimums[bucket];
    const status = count >= minimum ? "READY" : "COLLECT_MORE";
    console.log(`| ${bucket} | ${count} | ${minimum} | ${status} |`);
  }

  if (problems.length) {
    console.log("\nProblems:");
    for (const problem of problems) console.log(`- ${problem}`);
  }
  if (warnings.length) {
    console.log("\nWarnings:");
    for (const warning of warnings) console.log(`- ${warning}`);
  }

  const ready = buckets.every((bucket) => counts[bucket] >= minimums[bucket]);
  const audioReady = warnings.length === 0;
  console.log(`\nRows: ${rows.length}`);
  console.log(`Bucket-count ready: ${ready ? "yes" : "no"}`);
  console.log(`Audio-ready: ${audioReady ? "yes" : "no"}`);
  console.log(`Training-spend ready: ${ready && audioReady ? "yes" : "no"}`);

  if (problems.length || (strict && (!ready || !audioReady))) {
    throw new Error(
      problems.length
        ? `Manifest has ${problems.length} validation problem(s)`
        : "Manifest does not meet strict product-eval readiness",
    );
  }
}

async function localAudioWarning(audioPath: string) {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(audioPath)) return null;
  const resolved = path.isAbsolute(audioPath) ? audioPath : path.resolve(process.cwd(), audioPath);
  try {
    const stat = await fs.stat(resolved);
    if (!stat.isFile() || stat.size < 800) return `local audio path missing or too small: ${audioPath}`;
    return null;
  } catch {
    return `local audio path missing or too small: ${audioPath}`;
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
