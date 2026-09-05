import fs from "node:fs/promises";
import path from "node:path";
import { isResearchIntent } from "../src/lib/research-intents";

type Split = "train" | "dev" | "test";

type SilverRow = {
  id: string;
  split: Split;
  language: string;
  domain: string;
  source: string;
  original_text: string;
  normalized_twi: string;
  natural_english: string;
  intent: string;
  entities: unknown;
  ambiguities: string;
  requires_clarification: boolean;
  quality_tier: string;
  verification_status: string;
  messages: Array<{ role: string; content: string }>;
};

const root = process.cwd();
const defaultDir = path.join(root, "data", "understanding-corpus", "silver-medical-paired-v2");
const expectedRows = 7_000;
const metadataPattern = /(?:https?:\/\/|training_use=|body_system=|license=|source=)/i;

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

function parseJsonl(raw: string): SilverRow[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as SilverRow);
}

function fingerprint(value: string) {
  return value
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function countDuplicates(values: string[]) {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) duplicates.add(value);
    seen.add(value);
  }
  return duplicates.size;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

async function main() {
  const directory = argValue("--dir", defaultDir);
  const [all, train, dev, test] = await Promise.all(
    (["all", "train", "dev", "test"] as const).map(async (name) =>
      parseJsonl(await fs.readFile(path.join(directory, `${name}.jsonl`), "utf8")),
    ),
  );
  const errors: string[] = [];
  const idDuplicates = countDuplicates(all.map((row) => row.id));
  const textDuplicates = countDuplicates(all.map((row) => fingerprint(row.original_text)));
  if (all.length !== expectedRows) errors.push(`expected_${expectedRows}_rows_got_${all.length}`);
  if (idDuplicates > 0) errors.push(`duplicate_ids:${idDuplicates}`);
  if (textDuplicates > 0) errors.push(`duplicate_texts:${textDuplicates}`);

  const splitFiles = { train, dev, test } satisfies Record<Split, SilverRow[]>;
  const splitIds = new Set<string>();
  for (const [split, rows] of Object.entries(splitFiles) as Array<[Split, SilverRow[]]>) {
    for (const row of rows) {
      if (row.split !== split) errors.push(`${row.id}:split_file_mismatch`);
      if (splitIds.has(row.id)) errors.push(`${row.id}:split_leakage`);
      splitIds.add(row.id);
    }
  }
  if (splitIds.size !== all.length) errors.push("split_files_do_not_cover_all_rows_once");

  const bodySystems = new Set<string>();
  for (const row of all) {
    if (!row.id || !row.original_text.trim() || !row.natural_english.trim() || !row.intent.trim()) {
      errors.push(`${row.id || "unknown"}:missing_required_field`);
    }
    if (row.normalized_twi !== row.original_text) errors.push(`${row.id}:destructive_twi_normalization`);
    if (row.source !== "ghana_health_symptoms") errors.push(`${row.id}:unexpected_source:${row.source}`);
    if (row.language !== "tw" || row.domain !== "health") errors.push(`${row.id}:unexpected_lane`);
    if (row.quality_tier !== "paired_source_silver") errors.push(`${row.id}:missing_quality_tier`);
    if (row.verification_status !== "source_paired_unreviewed") errors.push(`${row.id}:invalid_verification_status`);
    if (!isResearchIntent(row.intent)) errors.push(`${row.id}:invalid_intent:${row.intent}`);
    if (row.requires_clarification) errors.push(`${row.id}:unexpected_clarification_row`);
    if (metadataPattern.test(row.ambiguities)) errors.push(`${row.id}:metadata_in_ambiguities`);

    const entities = objectValue(row.entities);
    if (!entities) {
      errors.push(`${row.id}:invalid_entities`);
    } else if (typeof entities.body_system === "string" && entities.body_system.trim()) {
      bodySystems.add(entities.body_system);
    }

    if (row.messages.length !== 3 || row.messages[0]?.role !== "system" || row.messages[1]?.role !== "user" || row.messages[2]?.role !== "assistant") {
      errors.push(`${row.id}:invalid_messages`);
      continue;
    }
    if (row.messages[1].content !== row.original_text) errors.push(`${row.id}:user_message_mismatch`);
    try {
      const assistant = JSON.parse(row.messages[2].content) as Record<string, unknown>;
      if (assistant.normalized_twi !== row.normalized_twi) errors.push(`${row.id}:assistant_twi_mismatch`);
      if (assistant.natural_english !== row.natural_english) errors.push(`${row.id}:assistant_english_mismatch`);
      if (assistant.intent !== row.intent) errors.push(`${row.id}:assistant_intent_mismatch`);
      if (metadataPattern.test(String(assistant.ambiguities ?? ""))) errors.push(`${row.id}:assistant_metadata_contamination`);
    } catch {
      errors.push(`${row.id}:invalid_assistant_json`);
    }
  }
  if (bodySystems.size < 10) errors.push(`insufficient_body_system_coverage:${bodySystems.size}`);

  const summary = {
    directory,
    rows: all.length,
    splits: { train: train.length, dev: dev.length, test: test.length },
    duplicate_ids: idDuplicates,
    duplicate_texts: textDuplicates,
    body_systems: bodySystems.size,
    errors: errors.length,
    error_examples: errors.slice(0, 20),
    passed: errors.length === 0,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (errors.length > 0) process.exit(1);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
