import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const rowSchema = z.object({
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
});

type Row = z.infer<typeof rowSchema>;

const promptPath =
  process.env.ASR_COLLECTION_PROMPTS ||
  path.join(process.cwd(), "tmp", "asr-collection-pack", "prompts.jsonl");
const audioDir = process.env.ASR_AUDIO_DIR || path.join(process.cwd(), "tmp", "asr-recordings");
const outPath =
  process.env.ASR_AUDIO_MANIFEST_OUT ||
  path.join(process.cwd(), "tmp", "asr-product-eval-audio-ready.jsonl");
const speakerLabel = process.env.ASR_SPEAKER_LABEL;
const consent = (process.env.ASR_CONSENT || "internal_eval") as Row["consent"];
const allowedExts = [".wav", ".webm", ".m4a", ".mp3", ".ogg", ".oga"];

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

async function findAudio(id: string) {
  for (const ext of allowedExts) {
    const candidate = path.resolve(audioDir, `${id}${ext}`);
    try {
      const stat = await fs.stat(candidate);
      if (stat.isFile() && stat.size > 800) return { path: candidate, size: stat.size };
    } catch {
      // Keep checking other extensions.
    }
  }
  return null;
}

async function sha256(filePath: string) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

async function main() {
  const rows = await readRows(promptPath);
  const attached: Row[] = [];
  const missing: string[] = [];

  for (const row of rows) {
    const audio = await findAudio(row.id);
    if (!audio) {
      missing.push(row.id);
      continue;
    }
    const hash = await sha256(audio.path);
    attached.push({
      ...row,
      audio_path: audio.path,
      speaker_label: speakerLabel || row.speaker_label,
      consent,
      notes: [
        row.notes,
        `audio_sha256=${hash}`,
        `audio_bytes=${audio.size}`,
        `source_prompt_id=${row.id}`,
      ]
        .filter(Boolean)
        .join("; "),
    });
  }

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, attached.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");

  console.log(`attached ${attached.length}/${rows.length} audio files -> ${outPath}`);
  if (missing.length) {
    console.log(`missing ${missing.length}: ${missing.slice(0, 30).join(", ")}${missing.length > 30 ? "..." : ""}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
