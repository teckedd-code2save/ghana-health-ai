import fs from "node:fs/promises";
import path from "node:path";

type HfRow = {
  row_idx: number;
  row: {
    instruction?: string;
    input?: string;
    output?: string;
  };
};

type HfRowsResponse = {
  rows?: HfRow[];
};

const root = process.cwd();
const dataset = "Malikeh1375/medical-question-answering-datasets";
const defaultOut = path.join(root, "data", "medical-response-corpus", "english-medical-qa.v0.jsonl");

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

function configs() {
  return argValue("--configs", "medical_meadow_wikidoc_patient_information,medical_meadow_mediqa")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

async function fetchRows(config: string, offset: number, length: number) {
  const params = new URLSearchParams({
    dataset,
    config,
    split: "train",
    offset: String(offset),
    length: String(length),
  });
  const url = `https://datasets-server.huggingface.co/rows?${params.toString()}`;
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    const res = await fetch(url);
    if (res.ok) {
      const data = (await res.json()) as HfRowsResponse;
      return data.rows ?? [];
    }
    if (res.status !== 429 || attempt === 6) {
      throw new Error(`Hugging Face rows request failed: ${res.status} ${await res.text()}`);
    }
    await new Promise((resolve) => setTimeout(resolve, attempt * 2500));
  }
  return [];
}

async function writeRows(out: string, rows: unknown[]) {
  await fs.mkdir(path.dirname(out), { recursive: true });
  const tmpPath = `${out}.${process.pid}.tmp`;
  await fs.writeFile(tmpPath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.rename(tmpPath, out);
}

function clean(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

async function main() {
  const out = argValue("--out", defaultOut);
  const limit = numericArgValue("--limit", 500);
  const pageSize = Math.min(100, Math.max(1, numericArgValue("--page-size", 50)));
  const delayMs = numericArgValue("--delay-ms", 800);
  const maxAnswerChars = numericArgValue("--max-answer-chars", 1800);
  const rows: unknown[] = [];

  for (const config of configs()) {
    for (let offset = 0; rows.length < limit; offset += pageSize) {
      const page = await fetchRows(config, offset, Math.min(pageSize, limit - rows.length));
      if (page.length === 0) break;
      for (const item of page) {
        const question = clean(item.row.input || item.row.instruction || "");
        const answer = clean(item.row.output || "");
        if (!question || !answer || answer.length > maxAnswerChars) continue;
        rows.push({
          id: `medical_qa_${config}_${String(item.row_idx).padStart(6, "0")}`,
          source_dataset: dataset,
          source_config: config,
          source_url: "https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets",
          source_row_index: item.row_idx,
          license: "mit",
          english_user: question,
          faithful_english_meaning: question,
          english_answer: answer,
          intent: question.toLowerCase().includes("symptoms")
            ? "medical_symptom_information_request"
            : question.toLowerCase().includes("urgent")
              ? "urgent_care_guidance_request"
              : "medical_information_request",
          entities: {},
          translation_status: "needs_twi_translation",
          review_status: "needs_review",
        });
        if (rows.length >= limit) break;
      }
      await writeRows(out, rows);
      console.log(JSON.stringify({ imported: rows.length, config, offset: offset + page.length, out }));
      if (rows.length >= limit) break;
      if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    if (rows.length >= limit) break;
  }

  await writeRows(out, rows);
  console.log(JSON.stringify({ out, rows: rows.length, dataset, license: "mit" }, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
