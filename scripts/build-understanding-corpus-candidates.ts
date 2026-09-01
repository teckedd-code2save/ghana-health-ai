import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";

type Json = Record<string, unknown>;

type Candidate = {
  id: string;
  record_id: string;
  source: "waxal" | "ghana_nlp_speech" | "local_recording" | "curated_prompt" | string;
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
  consent_scope: "dataset_license" | "research_audio" | "text_only";
  source_hash: string;
  duplicate_key: string;
  review_status: "needs_review";
  eligible_for_training: false;
  eligible_for_final_evaluation: false;
  model_proposal: {
    normalized_twi: string;
    natural_english: string;
    literal_english: string;
    intent: string;
    entities: string;
    ambiguities: string;
    requires_clarification: boolean;
    model: string;
    status: "not_requested" | "draft";
  };
};

const root = process.cwd();
const defaultOut = path.join(root, "tmp", "understanding-corpus", "candidates.v0.jsonl");
const speechLabRoot = path.join(
  root,
  "..",
  "akan-speech-lab",
);
const localDatasetRoot = path.join(root, "..", "gha-language-models", "datasets", "health", "twi");

const sources = [
  {
    source: "curated_prompt" as const,
    path: path.join(root, "tmp", "asr-collection-pack", "prompts.corpus-v2.health_twi.jsonl"),
    limit: 126,
    domain: "health",
  },
  {
    source: "curated_prompt" as const,
    path: path.join(root, "tmp", "asr-collection-pack", "prompts.corpus-v2.commerce_twi.jsonl"),
    limit: 74,
    domain: "commerce",
  },
  {
    source: "curated_prompt" as const,
    path: path.join(root, "tmp", "asr-collection-pack", "prompts.corpus-v2.codeswitch_tw_en.jsonl"),
    limit: 50,
    domain: "mixed",
  },
  {
    source: "medical_response_seed" as const,
    path: path.join(root, "data", "medical-response-corpus", "twi-drafts.v0.jsonl"),
    limit: 1000,
    domain: "health",
  },
  {
    source: "ghana_health_symptoms" as const,
    path: path.join(root, "data", "medical-response-corpus", "ghana-health-symptoms.v0.jsonl"),
    limit: 5000,
    domain: "health",
  },
  {
    source: "ghana_nlp_speech" as const,
    path: path.join(speechLabRoot, "data", "manifests", "ghana_nlp_twi.jsonl"),
    limit: 2500,
    domain: "general",
  },
  {
    source: "waxal" as const,
    path: path.join(speechLabRoot, "data", "manifests", "waxal_round2", "train.jsonl"),
    limit: 1800,
    domain: "general",
  },
  {
    source: "waxal" as const,
    path: path.join(speechLabRoot, "data", "manifests", "waxal_round2", "dev.jsonl"),
    limit: 350,
    domain: "general",
  },
  {
    source: "waxal" as const,
    path: path.join(speechLabRoot, "data", "manifests", "waxal_round2", "test.jsonl"),
    limit: 300,
    domain: "general",
  },
];

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function hash(value: string) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function normalize(value: string) {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseLines(raw: string) {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Json);
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function numericArgValue(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function localRecordingPath(promptId: string, bucket: string) {
  const match = promptId.match(/_u(\d+)$/);
  if (!match) return null;
  const utterance = match[1];
  return path.join(localDatasetRoot, `${bucket}_sp001_u${utterance}.webm`);
}

function portableSourcePath(sourcePath: string) {
  const relativeToRoot = path.relative(root, sourcePath);
  if (!relativeToRoot.startsWith("..")) return relativeToRoot;
  return path.join("..", path.relative(path.dirname(root), sourcePath));
}

function portableAudioArtifact(audioPath: string | null | undefined) {
  if (!audioPath) return null;
  const relativeToRoot = path.relative(root, audioPath);
  if (!relativeToRoot.startsWith("..")) return relativeToRoot;
  if (audioPath.startsWith("hf://")) return audioPath;
  return `local-research://${path.basename(audioPath)}`;
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function buildCandidate(input: {
  source: Candidate["source"];
  sourcePath: string;
  row: Json;
  index: number;
  domain: string;
  audioPath?: string | null;
}): Candidate | null {
  const text =
    asString(input.row.twi_user) ||
    asString(input.row.symptom_twi) ||
    asString(input.row.text) ||
    asString(input.row.reference) ||
    asString(input.row.normalized_text);
  if (!text) return null;
  const normalized = asString(input.row.normalized_text) || normalize(text);
  const sourceRecordId =
    asString(input.row.sample_id) ||
    asString(input.row.id) ||
    `${path.basename(input.sourcePath)}:${input.index}`;
  const sourceHash = hash(
    JSON.stringify({
      source: input.source,
      sourceRecordId,
      text,
      audio: input.audioPath ?? asString(input.row.audio_path),
    }),
  );
  const id = `${input.source}_${sourceHash.slice(0, 16)}`;
  const language =
    asString(input.row.language) ||
    (input.source === "curated_prompt" ||
    input.source === "medical_response_seed" ||
    input.source === "ghana_health_symptoms"
      ? "tw"
      : "aka");
  const audioArtifact = input.audioPath || asString(input.row.audio_path) || null;
  const split = asString(input.row.split) || asString(input.row.source_split) || "unknown";
  const speakerId = asString(input.row.speaker_id) || null;
  const initialEnglish =
    input.source === "curated_prompt"
      ? asString(input.row.en_reference)
      : asString(input.row.faithful_english_meaning) || asString(input.row.tag_en);
  const initialIntent =
    asString(input.row.intent) ||
    (input.source === "ghana_health_symptoms" ? "health_symptom_report" : "");
  const initialEntities =
    typeof input.row.entities === "string"
      ? input.row.entities
      : input.row.entities
        ? JSON.stringify(input.row.entities)
        : asString(input.row.body_system)
          ? JSON.stringify({ body_system: asString(input.row.body_system) })
        : "";
  const initialNotes = [
    asString(input.row.answer) ? `answer=${asString(input.row.answer)}` : "",
    asString(input.row.twi_answer) ? `twi_answer=${asString(input.row.twi_answer)}` : "",
    asString(input.row.safety_level) ? `safety_level=${asString(input.row.safety_level)}` : "",
    asString(input.row.body_system) ? `body_system=${asString(input.row.body_system)}` : "",
    asString(input.row.source_twi) ? `source_twi=${asString(input.row.source_twi)}` : "",
    Array.isArray(input.row.source_urls) ? `sources=${input.row.source_urls.join(" | ")}` : "",
    asString(input.row.source_url) ? `source=${asString(input.row.source_url)}` : "",
    asString(input.row.training_use) ? `training_use=${asString(input.row.training_use)}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  return {
    id,
    record_id: id,
    source: input.source,
    source_record_id: sourceRecordId,
    source_path: portableSourcePath(input.sourcePath),
    language,
    dialect: "unknown",
    domain: input.domain,
    split,
    text,
    normalized_text: normalized,
    audio_artifact_id: portableAudioArtifact(audioArtifact),
    speaker_id: speakerId,
    duration_seconds: asNumber(input.row.duration_seconds),
    consent_scope:
      input.source === "curated_prompt" || input.source === "medical_response_seed"
        ? "text_only"
        : "dataset_license",
    source_hash: sourceHash,
    duplicate_key: hash(normalized).slice(0, 16),
    review_status: "needs_review",
    eligible_for_training: false,
    eligible_for_final_evaluation: false,
    model_proposal: {
      normalized_twi: normalized,
      natural_english: initialEnglish,
      literal_english: "",
      intent: initialIntent,
      entities: initialEntities,
      ambiguities: initialNotes,
      requires_clarification: false,
      model: initialEnglish ? "source_seed_or_translation_draft" : "none",
      status: initialEnglish ? "draft" : "not_requested",
    },
  };
}

async function propose(candidate: Candidate): Promise<Candidate> {
  const provider = llmProviderInfo();
  if (!provider) return candidate;
  const content = await chatComplete(
    [
      {
        role: "system",
        content:
          "You prepare draft labels for Ghanaian-language understanding research. Return only JSON. Do not answer the speaker. Preserve uncertainty.",
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction:
            "Given this Twi/Akan/code-switched transcript, draft normalized_twi, natural_english, literal_english, intent, entities, ambiguities, and requires_clarification. These are drafts for human review, not gold labels.",
          transcript: candidate.text,
          normalized_hint: candidate.normalized_text,
          domain: candidate.domain,
          source: candidate.source,
        }),
      },
    ],
    { temperature: 0, maxTokens: 500 },
  );
  if (!content) return candidate;
  try {
    const json = JSON.parse(content.replace(/^```json\s*|\s*```$/g, "")) as Json;
    return {
      ...candidate,
      model_proposal: {
        normalized_twi: asString(json.normalized_twi) || candidate.model_proposal.normalized_twi,
        natural_english: asString(json.natural_english) || candidate.model_proposal.natural_english,
        literal_english: asString(json.literal_english) || candidate.model_proposal.literal_english,
        intent: asString(json.intent),
        entities:
          typeof json.entities === "string"
            ? json.entities
            : JSON.stringify(json.entities ?? [], null, 0),
        ambiguities:
          typeof json.ambiguities === "string"
            ? json.ambiguities
            : JSON.stringify(json.ambiguities ?? [], null, 0),
        requires_clarification: json.requires_clarification === true,
        model: `${provider.provider}:${provider.model}`,
        status: "draft",
      },
    };
  } catch {
    return {
      ...candidate,
      model_proposal: {
        ...candidate.model_proposal,
        ambiguities: "Draft model returned non-JSON output.",
        model: `${provider.provider}:${provider.model}`,
        status: "not_requested",
      },
    };
  }
}

async function readCandidates(limit: number, annotateLimit: number) {
  const candidates: Candidate[] = [];
  const seen = new Set<string>();
  let annotated = 0;

  for (const source of sources) {
    if (candidates.length >= limit) break;
    if (!(await exists(source.path))) continue;
    const rows = parseLines(await fs.readFile(source.path, "utf8"));
    let accepted = 0;
    for (let index = 0; index < rows.length; index += 1) {
      if (accepted >= source.limit || candidates.length >= limit) break;
      const row = rows[index];
      const bucket = asString(row.bucket);
      const promptId = asString(row.id);
      const localAudio = bucket && promptId ? localRecordingPath(promptId, bucket) : null;
      const hasLocalAudio = localAudio && (await exists(localAudio));
      const candidate = buildCandidate({
        source: hasLocalAudio ? "local_recording" : source.source,
        sourcePath: source.path,
        row,
        index,
        domain: source.domain,
        audioPath: hasLocalAudio ? localAudio : null,
      });
      if (!candidate || seen.has(candidate.duplicate_key)) continue;
      seen.add(candidate.duplicate_key);
      if (annotated < annotateLimit) {
        candidates.push(await propose(candidate));
        annotated += 1;
      } else {
        candidates.push(candidate);
      }
      accepted += 1;
    }
  }

  return candidates;
}

async function main() {
  const limit = numericArgValue("--limit", 5000);
  const annotateLimit = process.argv.includes("--annotate")
    ? numericArgValue("--annotate-limit", limit)
    : 0;
  const out = argValue("--out", defaultOut);
  const candidates = await readCandidates(limit, annotateLimit);
  await fs.mkdir(path.dirname(out), { recursive: true });
  await fs.writeFile(out, `${candidates.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");

  const bySource = candidates.reduce<Record<string, number>>((acc, row) => {
    acc[row.source] = (acc[row.source] ?? 0) + 1;
    return acc;
  }, {});
  const byDomain = candidates.reduce<Record<string, number>>((acc, row) => {
    acc[row.domain] = (acc[row.domain] ?? 0) + 1;
    return acc;
  }, {});
  const bySplit = candidates.reduce<Record<string, number>>((acc, row) => {
    acc[row.split] = (acc[row.split] ?? 0) + 1;
    return acc;
  }, {});
  const annotated = candidates.filter((row) => row.model_proposal.status === "draft").length;
  const withAudio = candidates.filter((row) => row.audio_artifact_id).length;
  console.log(
    JSON.stringify(
      {
        out,
        rows: candidates.length,
        withAudio,
        bySource,
        byDomain,
        bySplit,
        annotated,
        annotateLimit,
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
