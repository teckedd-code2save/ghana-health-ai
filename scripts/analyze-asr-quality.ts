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

const minimums: Record<Bucket, number> = {
  health_twi: 100,
  commerce_twi: 50,
  codeswitch_tw_en: 50,
  health_en: 50,
  phone_noise: 50,
};

const currentBaselines: Record<Bucket, number> = {
  health_twi: 30.44,
  commerce_twi: 30.44,
  codeswitch_tw_en: 30.44,
  health_en: 11.82,
  phone_noise: 30.44,
};

const rowSchema = z
  .object({
    id: z.string().min(3),
    bucket: z.enum(buckets),
    language: z.string().min(2),
    reference: z.string().min(2),
    audio_path: z.string().min(1),
    speaker_label: z.string().min(3),
    consent: z.string().min(3),
    domain_tags: z.array(z.string()).default([]),
    recording_tags: z.array(z.string()).default([]),
    notes: z.string().optional(),
    hypothesis: z.string().optional(),
    asr_transcript: z.string().optional(),
    original_transcript: z.string().optional(),
    model: z.string().optional(),
    asr_model: z.string().optional(),
  })
  .passthrough();

type Bucket = (typeof buckets)[number];
type ManifestRow = z.infer<typeof rowSchema>;

type BucketStats = {
  count: number;
  audioReady: number;
  scored: number;
  werErrors: number;
  werWords: number;
  cerErrors: number;
  cerChars: number;
};

type Edit = {
  type: "sub" | "ins" | "del";
  ref?: string;
  hyp?: string;
};

const manifestPath =
  process.env.ASR_QUALITY_MANIFEST ||
  process.env.ASR_PRODUCT_EVAL_MANIFEST ||
  path.join(process.cwd(), "tmp", "asr-feedback-export.jsonl");

function normalize(text: string) {
  return text
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[’]/g, "'")
    .replace(/[^\p{L}\p{N}'\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseOriginalFromNotes(notes?: string) {
  if (!notes) return undefined;
  const match = notes.match(/original=(".*?")(?:;|$)/);
  if (!match) return undefined;
  try {
    return JSON.parse(match[1]) as string;
  } catch {
    return undefined;
  }
}

function hypothesisFor(row: ManifestRow) {
  return (
    row.hypothesis ||
    row.asr_transcript ||
    row.original_transcript ||
    parseOriginalFromNotes(row.notes)
  )?.trim();
}

function audioIsReady(row: ManifestRow) {
  return !row.audio_path.startsWith("MISSING_AUDIO") && !/placeholder/i.test(row.notes || "");
}

function levenshtein<T>(ref: T[], hyp: T[]) {
  const dp = Array.from({ length: ref.length + 1 }, () => Array<number>(hyp.length + 1).fill(0));
  const back = Array.from({ length: ref.length + 1 }, () =>
    Array<"ok" | "sub" | "ins" | "del" | undefined>(hyp.length + 1).fill(undefined),
  );

  for (let i = 0; i <= ref.length; i++) {
    dp[i][0] = i;
    if (i > 0) back[i][0] = "del";
  }
  for (let j = 0; j <= hyp.length; j++) {
    dp[0][j] = j;
    if (j > 0) back[0][j] = "ins";
  }

  for (let i = 1; i <= ref.length; i++) {
    for (let j = 1; j <= hyp.length; j++) {
      const same = ref[i - 1] === hyp[j - 1];
      const candidates = [
        { cost: dp[i - 1][j - 1] + (same ? 0 : 1), op: same ? "ok" : "sub" },
        { cost: dp[i][j - 1] + 1, op: "ins" },
        { cost: dp[i - 1][j] + 1, op: "del" },
      ] as const;
      const best = candidates.reduce((a, b) => (b.cost < a.cost ? b : a));
      dp[i][j] = best.cost;
      back[i][j] = best.op;
    }
  }

  return { distance: dp[ref.length][hyp.length], back };
}

function wordEdits(reference: string, hypothesis: string) {
  const ref = normalize(reference).split(" ").filter(Boolean);
  const hyp = normalize(hypothesis).split(" ").filter(Boolean);
  const { distance, back } = levenshtein(ref, hyp);
  const edits: Edit[] = [];
  let i = ref.length;
  let j = hyp.length;

  while (i > 0 || j > 0) {
    const op = back[i][j];
    if (op === "ok") {
      i -= 1;
      j -= 1;
    } else if (op === "sub") {
      edits.push({ type: "sub", ref: ref[i - 1], hyp: hyp[j - 1] });
      i -= 1;
      j -= 1;
    } else if (op === "ins") {
      edits.push({ type: "ins", hyp: hyp[j - 1] });
      j -= 1;
    } else {
      edits.push({ type: "del", ref: ref[i - 1] });
      i -= 1;
    }
  }

  return { errors: distance, words: Math.max(ref.length, 1), edits };
}

function charDistance(reference: string, hypothesis: string) {
  const ref = Array.from(normalize(reference).replace(/\s+/g, ""));
  const hyp = Array.from(normalize(hypothesis).replace(/\s+/g, ""));
  return { errors: levenshtein(ref, hyp).distance, chars: Math.max(ref.length, 1) };
}

function pct(numerator: number, denominator: number) {
  if (!denominator) return "n/a";
  return `${((numerator / denominator) * 100).toFixed(2)}%`;
}

function increment(map: Map<string, number>, key: string) {
  map.set(key, (map.get(key) || 0) + 1);
}

function top(map: Map<string, number>, limit = 10) {
  return Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
}

async function readRows(filePath: string) {
  const raw = await fs.readFile(filePath, "utf8");
  const rows: ManifestRow[] = [];
  const problems: string[] = [];

  raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line, index) => {
      try {
        const parsed = rowSchema.parse(JSON.parse(line));
        rows.push(parsed);
      } catch (error) {
        problems.push(`line ${index + 1}: ${String(error)}`);
      }
    });

  return { rows, problems };
}

function recommendationFor(rows: ManifestRow[], stats: Record<Bucket, BucketStats>) {
  const missingBuckets = buckets.filter((bucket) => stats[bucket].count < minimums[bucket]);
  const audioReadyRows = rows.filter(audioIsReady).length;
  const scoredRows = rows.filter((row) => hypothesisFor(row)).length;
  const weakBuckets = buckets.filter((bucket) => {
    const stat = stats[bucket];
    if (!stat.scored) return false;
    return (stat.werErrors / stat.werWords) * 100 >= currentBaselines[bucket] - 1;
  });

  if (rows.length < 50 || missingBuckets.length) {
    return [
      "Spend credits on data generation/transcription first, not model training.",
      `Fill these eval buckets before a large run: ${missingBuckets.join(", ") || "none"}.`,
    ];
  }
  if (audioReadyRows < rows.length * 0.8) {
    return [
      "Attach consented audio to correction rows before acoustic training.",
      "Use text-only corrections now for prompt, language routing, and intent fixtures.",
    ];
  }
  if (scoredRows < rows.length * 0.5) {
    return [
      "Run v6, English Whisper, and DONDO against this manifest before training.",
      "The manifest has enough rows to benchmark, but not enough ASR hypotheses to diagnose errors.",
    ];
  }
  if (weakBuckets.length) {
    return [
      "Run targeted continuation, not another broad fine-tune.",
      `Weak buckets versus current baseline: ${weakBuckets.join(", ")}.`,
    ];
  }
  return [
    "Current route is holding on this eval set.",
    "Spend next credits on larger phone/noise and code-switch coverage before changing architecture.",
  ];
}

async function main() {
  const { rows, problems } = await readRows(manifestPath);
  const stats = Object.fromEntries(
    buckets.map((bucket) => [
      bucket,
      { count: 0, audioReady: 0, scored: 0, werErrors: 0, werWords: 0, cerErrors: 0, cerChars: 0 },
    ]),
  ) as Record<Bucket, BucketStats>;
  const substitutions = new Map<string, number>();
  const insertions = new Map<string, number>();
  const deletions = new Map<string, number>();
  const highRisk: string[] = [];

  for (const row of rows) {
    const stat = stats[row.bucket];
    stat.count += 1;
    if (audioIsReady(row)) stat.audioReady += 1;

    const hypothesis = hypothesisFor(row);
    if (!hypothesis) continue;

    stat.scored += 1;
    const word = wordEdits(row.reference, hypothesis);
    const char = charDistance(row.reference, hypothesis);
    stat.werErrors += word.errors;
    stat.werWords += word.words;
    stat.cerErrors += char.errors;
    stat.cerChars += char.chars;

    for (const edit of word.edits) {
      if (edit.type === "sub") increment(substitutions, `${edit.ref} -> ${edit.hyp}`);
      if (edit.type === "ins") increment(insertions, edit.hyp || "");
      if (edit.type === "del") increment(deletions, edit.ref || "");
    }

    const rowWer = (word.errors / word.words) * 100;
    const tags = [...row.domain_tags, ...row.recording_tags].join(" ");
    if (rowWer >= 50 || /medicine|emergency|malaria|pregnancy|shopping/i.test(tags)) {
      highRisk.push(`${row.id} (${row.bucket}, WER ${rowWer.toFixed(1)}%): "${hypothesis}" -> "${row.reference}"`);
    }
  }

  console.log(`# ASR Quality Report\n`);
  console.log(`Manifest: ${manifestPath}`);
  console.log(`Rows: ${rows.length}`);
  if (problems.length) {
    console.log(`\n## Problems`);
    for (const problem of problems) console.log(`- ${problem}`);
  }

  console.log(`\n## Bucket Readiness\n`);
  console.log(`| bucket | rows | minimum | audio-ready | scored | WER | CER |`);
  console.log(`| --- | ---: | ---: | ---: | ---: | ---: | ---: |`);
  for (const bucket of buckets) {
    const stat = stats[bucket];
    console.log(
      `| ${bucket} | ${stat.count} | ${minimums[bucket]} | ${stat.audioReady} | ${stat.scored} | ${pct(
        stat.werErrors,
        stat.werWords,
      )} | ${pct(stat.cerErrors, stat.cerChars)} |`,
    );
  }

  console.log(`\n## Frequent Errors\n`);
  console.log(`Substitutions: ${top(substitutions).map(([k, v]) => `${k} (${v})`).join(", ") || "n/a"}`);
  console.log(`Insertions: ${top(insertions).map(([k, v]) => `${k} (${v})`).join(", ") || "n/a"}`);
  console.log(`Deletions: ${top(deletions).map(([k, v]) => `${k} (${v})`).join(", ") || "n/a"}`);

  console.log(`\n## High-Risk Rows\n`);
  for (const item of highRisk.slice(0, 20)) console.log(`- ${item}`);
  if (!highRisk.length) console.log(`- n/a`);

  console.log(`\n## Next Credit Move\n`);
  for (const item of recommendationFor(rows, stats)) console.log(`- ${item}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
