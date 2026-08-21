import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { z } from "zod";
import { resolveTtsRoute, speakableText, type TtsProvider } from "../src/lib/modal-tts";

const eligibleBuckets = ["health_twi", "commerce_twi", "codeswitch_tw_en"] as const;
const eligibleLanguages = ["tw", "tw-en", "ak"] as const;

const sourceRowSchema = z.looseObject({
  id: z.string().min(3),
  bucket: z.enum(eligibleBuckets),
  language: z.enum(eligibleLanguages),
  reference: z.string().min(1),
  en_reference: z.string().optional(),
  needs_review: z.boolean().optional(),
  source: z.string().optional(),
  domain_tags: z.array(z.string()).optional(),
  recording_tags: z.array(z.string()).optional(),
});

type SourceRow = z.infer<typeof sourceRowSchema>;

type Args = {
  input: string;
  outDir: string;
  manifest: string;
  provider: TtsProvider;
  limit: number;
  allowDrafts: boolean;
  dryRun: boolean;
  mock: boolean;
  voiceId: string;
};

type TtsResponse = {
  audio_base64?: string;
  audioBase64?: string;
  sample_rate?: number;
  sampleRate?: number;
  format?: string;
  duration?: number;
  latency_ms?: number;
  latencyMs?: number;
  model?: string;
  provider?: string;
  voice?: string;
  language?: string;
  error?: string;
};

export type SyntheticManifestRow = {
  id: string;
  bucket: string;
  language: string;
  reference: string;
  audio_path: string;
  speaker_label: string;
  source: "synthetic_tts";
  tts_model: string;
  tts_provider: TtsProvider;
  voice_id: string;
  duration_s: number;
  sample_rate: number;
  format: string;
  sha256: string;
  holdout: false;
  source_text_id: string;
  source_text_source?: string;
  source_en_reference?: string;
  domain_tags: string[];
  recording_tags: string[];
  generated_at: string;
};

function parseArgs(argv = process.argv.slice(2)): Args {
  const get = (name: string, fallback = "") => {
    const index = argv.indexOf(`--${name}`);
    return index >= 0 ? argv[index + 1] : fallback;
  };
  const has = (name: string) => argv.includes(`--${name}`);
  return {
    input: get("input"),
    outDir: get("out-dir", "tmp/synthetic-twi-voice-notes/audio"),
    manifest: get("manifest", "tmp/synthetic-twi-voice-notes/manifest.jsonl"),
    provider: (get("provider", "stable-twi") as TtsProvider) || "stable-twi",
    limit: Number(get("limit", "0")) || 0,
    allowDrafts: has("allow-drafts"),
    dryRun: has("dry-run"),
    mock: has("mock"),
    voiceId: get("voice-id", ""),
  };
}

async function readJsonl(filePath: string) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return { lineNo: index + 1, parsed: JSON.parse(line) as unknown };
      } catch (error) {
        throw new Error(`${filePath} line ${index + 1}: invalid JSON (${String(error)})`);
      }
    });
}

function isReviewed(row: SourceRow) {
  return row.needs_review !== true && row.source !== "llm_translation_draft";
}

function wavDurationSeconds(buffer: Buffer) {
  if (buffer.subarray(0, 4).toString("ascii") !== "RIFF" || buffer.subarray(8, 12).toString("ascii") !== "WAVE") {
    return 0;
  }
  let offset = 12;
  let byteRate = 0;
  let dataBytes = 0;
  while (offset + 8 <= buffer.length) {
    const id = buffer.subarray(offset, offset + 4).toString("ascii");
    const size = buffer.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    if (id === "fmt " && dataStart + 12 <= buffer.length) {
      byteRate = buffer.readUInt32LE(dataStart + 8);
    }
    if (id === "data") {
      dataBytes = size;
      break;
    }
    offset = dataStart + size + (size % 2);
  }
  return byteRate > 0 && dataBytes > 0 ? dataBytes / byteRate : 0;
}

function makeMockWav(durationS = 0.6, sampleRate = 16000) {
  const samples = Math.max(1, Math.round(durationS * sampleRate));
  const dataBytes = samples * 2;
  const buffer = Buffer.alloc(44 + dataBytes);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataBytes, 40);
  return buffer;
}

async function synthesize(row: SourceRow, args: Args, routeModel: string): Promise<TtsResponse> {
  if (args.mock) {
    return {
      audio_base64: makeMockWav().toString("base64"),
      sample_rate: 16000,
      format: "wav",
      duration: 0.6,
      model: routeModel,
      provider: args.provider,
      voice: args.voiceId || "mock",
      language: row.language,
      latency_ms: 0,
    };
  }

  const route = resolveTtsRoute("tw", args.provider);
  if (!route) throw new Error(`No TTS route configured for provider ${args.provider}`);
  const language = row.language === "tw-en" ? "tw-en" : "tw";
  const res = await fetch(route.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.MODAL_TTS_TOKEN ? { Authorization: `Bearer ${process.env.MODAL_TTS_TOKEN}` } : {}),
    },
    body: JSON.stringify({
      text: speakableText(row.reference, "tw"),
      language,
      provider: args.provider,
      model: route.modelLabel,
      voice: args.voiceId || undefined,
    }),
  });
  if (!res.ok) throw new Error(`TTS ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()) as TtsResponse;
}

export async function generateSyntheticVoiceNotes(args: Args) {
  if (!args.input) throw new Error("--input is required");
  if (args.provider !== "mms" && args.provider !== "stable-twi" && args.provider !== "nano-twi" && args.provider !== "qwen") {
    throw new Error(`Unsupported --provider ${args.provider}`);
  }

  const route = resolveTtsRoute("tw", args.provider);
  if (!args.mock && !args.dryRun && !route) throw new Error(`No TTS route configured for provider ${args.provider}`);
  const modelLabel = route?.modelLabel ?? `mock-${args.provider}`;
  const lines = await readJsonl(args.input);
  const rows: SourceRow[] = [];
  const skipped: string[] = [];

  for (const { lineNo, parsed } of lines) {
    const parsedRow = sourceRowSchema.safeParse(parsed);
    if (!parsedRow.success) {
      skipped.push(`line ${lineNo}: ${parsedRow.error.issues.map((issue) => issue.message).join("; ")}`);
      continue;
    }
    if (!args.allowDrafts && !isReviewed(parsedRow.data)) {
      skipped.push(`line ${lineNo}: ${parsedRow.data.id} is still a draft; review it or pass --allow-drafts`);
      continue;
    }
    rows.push(parsedRow.data);
    if (args.limit > 0 && rows.length >= args.limit) break;
  }

  if (args.dryRun) {
    return { rows: [], eligible: rows.length, skipped };
  }

  await fs.mkdir(args.outDir, { recursive: true });
  await fs.mkdir(path.dirname(args.manifest), { recursive: true });

  const generatedAt = new Date().toISOString();
  const manifestRows: SyntheticManifestRow[] = [];
  for (const row of rows) {
    const response = await synthesize(row, args, modelLabel);
    const audioBase64 = response.audio_base64 ?? response.audioBase64;
    if (response.error || !audioBase64) throw new Error(`${row.id}: TTS failed (${response.error || "missing audio"})`);

    const format = response.format || "wav";
    const audio = Buffer.from(audioBase64, "base64");
    const sha256 = crypto.createHash("sha256").update(audio).digest("hex");
    const duration = response.duration || (format === "wav" ? wavDurationSeconds(audio) : 0);
    const audioName = `${row.id}__synthetic_${args.provider}.${format}`;
    const audioPath = path.join(args.outDir, audioName);
    await fs.writeFile(audioPath, audio);

    manifestRows.push({
      id: `${row.id}__synthetic_${args.provider}`,
      bucket: row.bucket,
      language: row.language,
      reference: row.reference,
      audio_path: audioPath,
      speaker_label: `synthetic_${args.provider}`,
      source: "synthetic_tts",
      tts_model: response.model || modelLabel,
      tts_provider: args.provider,
      voice_id: response.voice || args.voiceId || (row.language === "tw-en" ? "mixed-default" : "default"),
      duration_s: Number(duration.toFixed(3)),
      sample_rate: response.sample_rate ?? response.sampleRate ?? 0,
      format,
      sha256,
      holdout: false,
      source_text_id: row.id,
      source_text_source: row.source,
      source_en_reference: row.en_reference,
      domain_tags: [...(row.domain_tags || []), "synthetic_tts"],
      recording_tags: [...(row.recording_tags || []), "synthetic"],
      generated_at: generatedAt,
    });
    console.log(`[synthetic-tts] ${manifestRows.length}/${rows.length} ${row.id} -> ${audioPath}`);
  }

  await fs.writeFile(args.manifest, manifestRows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
  return { rows: manifestRows, eligible: rows.length, skipped };
}

async function main() {
  const args = parseArgs();
  const result = await generateSyntheticVoiceNotes(args);
  if (args.dryRun) {
    console.log(`[synthetic-tts] dry run: ${result.eligible} eligible row(s), ${result.skipped.length} skipped`);
  } else {
    console.log(`[synthetic-tts] wrote ${result.rows.length} row(s) -> ${args.manifest}`);
    console.log("[synthetic-tts] reminder: synthetic audio is augmentation/stress-test data, not final held-out evidence.");
  }
  if (result.skipped.length) {
    console.log("\nSkipped:");
    for (const reason of result.skipped.slice(0, 20)) console.log(`- ${reason}`);
    if (result.skipped.length > 20) console.log(`- ... ${result.skipped.length - 20} more`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error("[synthetic-tts] failed:", error);
    process.exit(1);
  });
}
