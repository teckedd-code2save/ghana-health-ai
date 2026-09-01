import fs from "node:fs/promises";
import path from "node:path";

type CorpusCandidate = {
  id: string;
  source: string;
  source_record_id: string;
  language: string;
  domain: string;
  text: string;
  normalized_text: string;
  consent_scope: string;
  duplicate_key: string;
  speaker_id: string | null;
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

type SilverRow = {
  id: string;
  split: "train" | "dev" | "test";
  domain: string;
  language: string;
  source: string;
  source_record_id: string;
  consent_scope: string;
  training_lane: "medical_research_silver" | "language_coverage_silver";
  license_policy: string;
  original_text: string;
  normalized_twi: string;
  natural_english: string;
  literal_english: string;
  intent: string;
  entities: unknown;
  ambiguities: string;
  requires_clarification: boolean;
  label_source: string;
  messages: { role: "system" | "user" | "assistant"; content: string }[];
};

const root = process.cwd();
const defaultInput = path.join(root, "data", "understanding-corpus", "candidates.v0.jsonl");
const defaultOutDir = path.join(root, "tmp", "understanding-corpus", "silver", "v0");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function numericArgValue(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function parseEntities(value: string): unknown {
  if (!value.trim()) return {};
  try {
    return JSON.parse(value);
  } catch {
    return { raw: value };
  }
}

function parseJsonl(raw: string): CorpusCandidate[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as CorpusCandidate);
}

function stableBucket(id: string): "train" | "dev" | "test" {
  let hash = 0;
  for (const char of id) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  const mod = hash % 100;
  if (mod < 80) return "train";
  if (mod < 90) return "dev";
  return "test";
}

function getCandidateTrainingSplit(candidate: CorpusCandidate): "train" | "dev" | "test" {
  return stableBucket(candidate.speaker_id || candidate.duplicate_key || candidate.id);
}

function isResearchOnly(candidate: CorpusCandidate) {
  return candidate.source === "ghana_health_symptoms";
}

function shouldIncludeSource(candidate: CorpusCandidate) {
  const includeLanguageCoverage = process.argv.includes("--include-language-coverage");
  if (candidate.source === "ghana_health_symptoms") return true;
  if (includeLanguageCoverage && ["ghana_nlp_speech", "waxal"].includes(candidate.source)) return true;
  return false;
}

function isUseful(candidate: CorpusCandidate) {
  if (!shouldIncludeSource(candidate)) return false;
  const proposal = candidate.model_proposal;
  if (proposal.status !== "draft") return false;
  if (!proposal.normalized_twi.trim()) return false;
  if (!proposal.natural_english.trim()) return false;
  if (!proposal.intent.trim()) return false;
  if (proposal.requires_clarification) return false;
  return true;
}

function systemPrompt(row: CorpusCandidate) {
  if (row.domain === "health") {
    return [
      "You are Ghana Health AI's semantic recovery model.",
      "Given a Twi/Akan or code-switched user utterance, output faithful structured understanding.",
      "Do not diagnose. Do not invent missing symptoms. Preserve uncertainty.",
    ].join(" ");
  }
  if (row.domain === "commerce") {
    return [
      "You are Ghana Health AI's commerce understanding model.",
      "Extract the shopping intent, item, quantity, delivery or pickup need, and location from Twi/Akan or code-switched text.",
      "Do not invent products, prices, or stores.",
    ].join(" ");
  }
  return "You are Ghana Health AI's semantic recovery model. Output faithful structured understanding without answering the user.";
}

function toSilverRow(candidate: CorpusCandidate): SilverRow {
  const proposal = candidate.model_proposal;
  const split = getCandidateTrainingSplit(candidate);
  const assistant = {
    normalized_twi: proposal.normalized_twi,
    natural_english: proposal.natural_english,
    literal_english: proposal.literal_english,
    intent: proposal.intent,
    entities: parseEntities(proposal.entities),
    ambiguities: proposal.ambiguities,
    requires_clarification: proposal.requires_clarification,
  };
  const researchOnly = isResearchOnly(candidate);
  return {
    id: candidate.id,
    split,
    domain: candidate.domain,
    language: candidate.language,
    source: candidate.source,
    source_record_id: candidate.source_record_id,
    consent_scope: candidate.consent_scope,
    training_lane: researchOnly ? "medical_research_silver" : "language_coverage_silver",
    license_policy: researchOnly
      ? "noncommercial_research_only"
      : "source_license_or_project_controlled",
    original_text: candidate.text,
    normalized_twi: proposal.normalized_twi,
    natural_english: proposal.natural_english,
    literal_english: proposal.literal_english,
    intent: proposal.intent,
    entities: assistant.entities,
    ambiguities: proposal.ambiguities,
    requires_clarification: proposal.requires_clarification,
    label_source: proposal.model,
    messages: [
      { role: "system", content: systemPrompt(candidate) },
      { role: "user", content: candidate.text },
      { role: "assistant", content: JSON.stringify(assistant) },
    ],
  };
}

async function writeJsonl(filePath: string, rows: SilverRow[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, rows.length ? `${rows.map((row) => JSON.stringify(row)).join("\n")}\n` : "", "utf8");
}

async function main() {
  const outDir = argValue("--out-dir", defaultOutDir);
  const input = argValue("--input", defaultInput);
  const limit = numericArgValue("--limit", 0);
  const candidates = parseJsonl(await fs.readFile(input, "utf8"));
  const rows = candidates
    .filter(isUseful)
    .map(toSilverRow)
    .sort((a, b) => {
      const priority = (row: SilverRow) =>
        row.source === "ghana_health_symptoms"
          ? 0
          : row.source === "ghana_nlp_speech"
            ? 1
            : row.source === "waxal"
              ? 2
              : 3;
      return priority(a) - priority(b) || a.id.localeCompare(b.id);
    });
  const selected = limit > 0 ? rows.slice(0, limit) : rows;
  const bySplit = {
    train: selected.filter((row) => row.split === "train"),
    dev: selected.filter((row) => row.split === "dev"),
    test: selected.filter((row) => row.split === "test"),
  };
  await Promise.all(
    Object.entries(bySplit).map(([split, splitRows]) =>
      writeJsonl(path.join(outDir, `${split}.jsonl`), splitRows),
    ),
  );
  await writeJsonl(path.join(outDir, "all.jsonl"), selected);

  const summary = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    outDir,
    candidates: candidates.length,
    selected: selected.length,
    splits: Object.fromEntries(Object.entries(bySplit).map(([split, splitRows]) => [split, splitRows.length])),
    bySource: selected.reduce<Record<string, number>>((acc, row) => {
      acc[row.source] = (acc[row.source] ?? 0) + 1;
      return acc;
    }, {}),
    byTrainingLane: selected.reduce<Record<string, number>>((acc, row) => {
      acc[row.training_lane] = (acc[row.training_lane] ?? 0) + 1;
      return acc;
    }, {}),
    excludedPolicy: {
      curated_prompt: "excluded from silver export because sampled rows showed weak synthetic quality",
      local_recording: "excluded from this corpus because user requested large corpus only",
      medical_response_seed: "excluded from this corpus because it is a small seed",
      medical_qa_twi_draft: "excluded from this corpus until the translated QA set is scaled and audited",
      general_waxal_ghana_nlp: "included only with --include-language-coverage after machine annotation",
      requires_clarification: "excluded from first training run",
    },
  };
  await fs.writeFile(path.join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(summary, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
