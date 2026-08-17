import "../src/config/load-env";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { prisma } from "../src/db/prisma";

type Bucket =
  | "health_twi"
  | "commerce_twi"
  | "codeswitch_tw_en"
  | "health_en"
  | "phone_noise";

type ExportRow = {
  id: string;
  bucket: Bucket;
  language: "tw" | "en" | "ga" | "tw-en";
  reference: string;
  audio_path: string;
  speaker_label: string;
  consent: "internal_eval" | "user_shared_audio";
  domain_tags: string[];
  recording_tags: string[];
  notes: string;
};

type ExportOptions = {
  outPath?: string;
  minCorrectionChars?: number;
  limit?: number;
  includeUncorrected?: boolean;
};

function isMeaningfulCorrection(original: string, corrected: string, minCorrectionChars: number) {
  const a = normalize(original);
  const b = normalize(corrected);
  if (!a || !b || a === b) return false;
  return Math.abs(a.length - b.length) >= minCorrectionChars || wordDiffCount(a, b) > 0;
}

function normalize(text: string) {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

function wordDiffCount(a: string, b: string) {
  const aw = a.split(" ");
  const bw = b.split(" ");
  const max = Math.max(aw.length, bw.length);
  let diff = 0;
  for (let i = 0; i < max; i++) {
    if (aw[i] !== bw[i]) diff += 1;
  }
  return diff;
}

function hasEnglish(text: string) {
  return /\b(the|and|pain|fever|buy|order|price|tomato|medicine|delivery|stomach|headache)\b/i.test(
    text,
  );
}

function bucketFor(input: { focus: string | null; language: string; reference: string; notes?: string | null }) {
  const language = input.language;
  const text = `${input.reference} ${input.notes ?? ""}`;
  const phoneNoise = /phone|noise|noisy|mobile|field/i.test(text);
  if (phoneNoise) return "phone_noise" as const;
  if (language === "en") return "health_en" as const;
  if (language === "tw" && hasEnglish(input.reference)) return "codeswitch_tw_en" as const;
  if (input.focus === "commerce") return "commerce_twi" as const;
  return "health_twi" as const;
}

function languageFor(language: string, reference: string): ExportRow["language"] {
  if (language === "en") return "en";
  if (language === "ga") return "ga";
  if (hasEnglish(reference)) return "tw-en";
  return "tw";
}

function domainTagsFor(input: { focus: string | null; reference: string; notes?: string | null }) {
  const text = `${input.reference} ${input.notes ?? ""}`;
  const tags = new Set<string>();
  if (input.focus === "commerce") tags.add("shopping");
  if (/fever|hyew|temperature|malaria/i.test(text)) tags.add("fever");
  if (/pain|ya|headache|stomach|yam/i.test(text)) tags.add("symptom");
  if (/medicine|paracetamol|drug|aduro/i.test(text)) tags.add("medicine");
  if (/tomato|mako|rice|soap|buy|order|price|t[oɔ]/i.test(text)) tags.add("commerce");
  if (!tags.size) tags.add(input.focus === "commerce" ? "commerce" : "health");
  return Array.from(tags);
}

function recordingTagsFor(input: { notes?: string | null; asrRoute?: string | null }) {
  const text = `${input.notes ?? ""} ${input.asrRoute ?? ""}`;
  const tags = new Set<string>(["user_correction"]);
  if (/phone|mobile/i.test(text)) tags.add("phone");
  if (/noise|noisy|field/i.test(text)) tags.add("noise");
  return Array.from(tags);
}

export async function exportAsrFeedback(options: ExportOptions = {}) {
  const outPath =
    options.outPath ||
    process.env.ASR_FEEDBACK_EXPORT_PATH ||
    path.join(process.cwd(), "tmp", "asr-feedback-export.jsonl");
  const minCorrectionChars =
    options.minCorrectionChars ?? Number(process.env.ASR_FEEDBACK_MIN_CHARS || 2);
  const limit = options.limit ?? Number(process.env.ASR_FEEDBACK_EXPORT_LIMIT || 1000);
  const includeUncorrected =
    options.includeUncorrected ?? process.argv.includes("--include-uncorrected");

  const feedback = await prisma.asrFeedback.findMany({
    where: includeUncorrected
      ? {}
      : {
          correctedTranscript: { not: null },
        },
    orderBy: { createdAt: "desc" },
    take: limit,
  });

  const audioDir =
    process.env.ASR_FEEDBACK_AUDIO_DIR ||
    path.join(process.cwd(), "data", "asr-feedback-audio");

  const rows: ExportRow[] = [];
  for (const item of feedback) {
    const corrected = item.correctedTranscript?.trim();
    const reference = corrected || item.originalTranscript.trim();
    if (!reference) continue;
    if (
      !includeUncorrected &&
      (!corrected || !isMeaningfulCorrection(item.originalTranscript, corrected, minCorrectionChars))
    ) {
      continue;
    }

    const language = languageFor(item.language, reference);
    const bucket = bucketFor({
      focus: item.focus,
      language,
      reference,
      notes: item.notes,
    });
    const hasConsentedAudio = Boolean(item.audioConsent && item.audioPath);

    rows.push({
      id: `feedback_${item.id}`,
      bucket,
      language,
      reference,
      audio_path: hasConsentedAudio
        ? path.join(audioDir, item.audioPath as string)
        : `MISSING_AUDIO_FOR_FEEDBACK_${item.id}`,
      speaker_label: item.userId ? `user_${item.userId.slice(0, 8)}` : "anonymous_user",
      consent: hasConsentedAudio ? "user_shared_audio" : "internal_eval",
      domain_tags: domainTagsFor({
        focus: item.focus,
        reference,
        notes: item.notes,
      }),
      recording_tags: recordingTagsFor({
        notes: item.notes,
        asrRoute: item.asrRoute,
      }),
      notes: [
        `original=${JSON.stringify(item.originalTranscript)}`,
        item.asrModel ? `asr_model=${item.asrModel}` : undefined,
        item.asrRoute ? `asr_route=${item.asrRoute}` : undefined,
        item.rating ? `rating=${item.rating}` : undefined,
      ]
        .filter(Boolean)
        .join("; "),
    });
  }

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");

  const counts = rows.reduce<Record<string, number>>((acc, row) => {
    acc[row.bucket] = (acc[row.bucket] ?? 0) + 1;
    return acc;
  }, {});

  console.log(`exported ${rows.length} ASR feedback rows -> ${outPath}`);
  console.log(JSON.stringify(counts, null, 2));
  return { outPath, rows, counts };
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  exportAsrFeedback()
    .catch(async (error) => {
      console.error(error);
      await prisma.$disconnect();
      process.exit(1);
    })
    .finally(async () => {
      await prisma.$disconnect();
    });
}
