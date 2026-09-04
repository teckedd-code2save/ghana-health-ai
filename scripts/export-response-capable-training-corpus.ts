import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  buildUnderstandingTrainingExport,
  type UnderstandingTrainingRow,
} from "../src/lib/research-understanding-store";

const root = process.cwd();
const outDir = path.join(root, "tmp", "response-capable-corpus", "v1");

const system = [
  "You are Ghana Health AI. Reply naturally in the speaker's requested language.",
  "First recover the meaning faithfully. Do not invent symptoms, diagnoses, medicines, prices, or facts.",
  "For health, give short safe guidance and ask only the next necessary question when details are missing.",
  "Return JSON only with normalized_twi, natural_english, intent, entities, ambiguities, safety_level, reply_twi, and requires_clarification.",
].join(" ");

type DirectRow = UnderstandingTrainingRow & {
  reply_twi: string;
  safety_level: Exclude<UnderstandingTrainingRow["safety_level"], "">;
};

function isDirectResponseRow(row: UnderstandingTrainingRow): row is DirectRow {
  return (
    row.domain === "health" &&
    row.reply_twi.trim().length > 0 &&
    row.safety_level !== ""
  );
}

function asMessages(row: DirectRow) {
  return {
    id: row.id,
    source: row.source,
    source_record_id: row.source_record_id,
    consent_scope: row.consent_scope,
    split: row.split,
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: `language=${row.language}\nfocus=health\nrecent_history=[]\nutterance=${row.original_text}`,
      },
      {
        role: "assistant",
        content: JSON.stringify({
          normalized_twi: row.normalized_twi,
          natural_english: row.natural_english,
          intent: row.intent,
          entities: row.entities,
          ambiguities: row.ambiguities,
          safety_level: row.safety_level,
          reply_twi: row.reply_twi,
          requires_clarification: Boolean(row.ambiguities.trim()),
        }),
      },
    ],
  };
}

async function writeJsonl(filePath: string, rows: unknown[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}${rows.length ? "\n" : ""}`, "utf8");
}

async function main() {
  const strict = process.argv.includes("--strict");
  const exportData = await buildUnderstandingTrainingExport();
  const directRows = exportData.rows.filter(isDirectResponseRow);
  const splits = {
    train: directRows.filter((row) => row.split === "train"),
    dev: directRows.filter((row) => row.split === "dev"),
    test: directRows.filter((row) => row.split === "test"),
  };

  await Promise.all([
    writeJsonl(path.join(outDir, "train.jsonl"), splits.train.map(asMessages)),
    writeJsonl(path.join(outDir, "dev.jsonl"), splits.dev.map(asMessages)),
    writeJsonl(path.join(outDir, "test.jsonl"), splits.test.map(asMessages)),
  ]);

  const manifest = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    purpose: "Direct Twi health response SFT. Every row is human-reviewed and source-linked.",
    rows: directRows.length,
    splits: Object.fromEntries(Object.entries(splits).map(([key, value]) => [key, value.length])),
    excluded_without_reviewed_reply: exportData.rows.length - directRows.length,
    promotion_gate: "Do not train or deploy if any split is empty or rows are below the minimum.",
  };
  await fs.writeFile(path.join(outDir, "manifest.v1.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(manifest, null, 2));

  if (strict && (directRows.length < 100 || !splits.train.length || !splits.dev.length || !splits.test.length)) {
    throw new Error("Response corpus is not ready: require 100 reviewed rows across train, dev, and test.");
  }
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
