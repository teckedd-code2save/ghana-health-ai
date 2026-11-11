import "../src/config/load-env";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { modalTranscribe } from "../src/lib/modal-asr";

const rowSchema = z
  .object({
    id: z.string().min(3),
    bucket: z.enum(["health_twi", "commerce_twi", "codeswitch_tw_en", "health_en", "phone_noise"]),
    language: z.enum(["tw", "ak", "en", "tw-en", "ak-en", "ga"]),
    reference: z.string().min(2),
    audio_path: z.string().min(1),
    speaker_label: z.string().min(3),
    consent: z.enum(["internal_eval", "public_release", "research_consent"]),
    domain_tags: z.array(z.string().min(1)).min(1),
    recording_tags: z.array(z.string().min(1)).min(1),
    notes: z.string().optional(),
  })
  .passthrough();

type Row = z.infer<typeof rowSchema>;

const manifestPath =
  process.env.ASR_PRODUCT_EVAL_MANIFEST ||
  path.join(process.cwd(), "tmp", "asr-product-eval-audio-ready.jsonl");
const outPath =
  process.env.ASR_SCORE_OUT ||
  path.join(process.cwd(), "tmp", "asr-product-eval-scored.jsonl");
const limit = Number(process.env.ASR_SCORE_LIMIT || 0);

function mimeFor(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".wav") return "audio/wav";
  if (ext === ".mp3") return "audio/mpeg";
  if (ext === ".m4a") return "audio/mp4";
  if (ext === ".ogg" || ext === ".oga") return "audio/ogg";
  return "audio/webm";
}

function routeLanguage(row: Row) {
  return row.language === "en" ? "en" : undefined;
}

async function readRows(filePath: string) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const parsed = rowSchema.safeParse(JSON.parse(line));
      if (!parsed.success) {
        throw new Error(`line ${index + 1}: ${parsed.error.issues.map((i) => i.message).join("; ")}`);
      }
      return parsed.data;
    });
}

async function assertAudio(row: Row) {
  if (row.audio_path.startsWith("MISSING_AUDIO")) {
    throw new Error(`${row.id}: audio placeholder cannot be scored`);
  }
  const resolved = path.isAbsolute(row.audio_path)
    ? row.audio_path
    : path.resolve(process.cwd(), row.audio_path);
  const stat = await fs.stat(resolved);
  if (!stat.isFile() || stat.size < 800) {
    throw new Error(`${row.id}: audio missing or too small (${row.audio_path})`);
  }
  return resolved;
}

async function main() {
  const allRows = await readRows(manifestPath);
  const rows = limit > 0 ? allRows.slice(0, limit) : allRows;
  const scored: unknown[] = [];

  for (const [index, row] of rows.entries()) {
    const audioPath = await assertAudio(row);
    const audio = await fs.readFile(audioPath);
    const result = await modalTranscribe(audio, {
      language: routeLanguage(row),
      contentType: mimeFor(audioPath),
      filename: path.basename(audioPath),
    });
    if (result.error && !result.text?.trim()) {
      throw new Error(`${row.id}: ASR failed: ${result.error}`);
    }
    const scoredRow = {
      ...row,
      hypothesis: result.text,
      asr_transcript: result.text,
      asr_model: result.model,
      asr_route: result.route?.name,
      asr_language: result.language,
      asr_language_probability: result.language_probability,
      asr_latency_ms: result.latency_ms,
      asr_duration_s: result.duration,
      asr_rms: result.rms,
      scored_at: new Date().toISOString(),
    };
    scored.push(scoredRow);
    console.log(
      `scored ${index + 1}/${rows.length} ${row.id} route=${result.route?.name ?? "unknown"} model=${result.model}`,
    );
  }

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, scored.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
  console.log(`wrote scored manifest -> ${outPath}`);
  console.log(`next: ASR_PRODUCT_EVAL_MANIFEST=${outPath} pnpm eval:asr-quality`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
