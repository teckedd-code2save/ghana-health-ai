import fs from "node:fs/promises";
import path from "node:path";

type HuggingFaceRow = {
  row_idx: number;
  row: {
    symptom_twi?: string;
    ipa_twi?: string;
    tag_en?: string;
    body_system?: string;
    source_twi?: string;
    row_id?: string;
  };
};

type HuggingFaceRowsResponse = {
  rows?: HuggingFaceRow[];
};

const root = process.cwd();
const dataset = "ghananlpcommunity/ghana-health-symptoms";
const defaultOut = path.join(root, "data", "medical-response-corpus", "ghana-health-symptoms.v0.jsonl");

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

async function fetchRows(offset: number, length: number) {
  const params = new URLSearchParams({
    dataset,
    config: "default",
    split: "train",
    offset: String(offset),
    length: String(length),
  });
  const url = `https://datasets-server.huggingface.co/rows?${params.toString()}`;
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    const res = await fetch(url);
    if (res.ok) {
      const data = (await res.json()) as HuggingFaceRowsResponse;
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

async function main() {
  const out = argValue("--out", defaultOut);
  const limit = numericArgValue("--limit", 5000);
  const pageSize = Math.min(100, Math.max(1, numericArgValue("--page-size", 100)));
  const delayMs = numericArgValue("--delay-ms", 500);
  const rows: unknown[] = [];

  for (let offset = 0; rows.length < limit; offset += pageSize) {
    const page = await fetchRows(offset, Math.min(pageSize, limit - rows.length));
    if (page.length === 0) break;
    for (const item of page) {
      const symptomTwi = item.row.symptom_twi?.trim();
      const tagEn = item.row.tag_en?.trim();
      if (!symptomTwi || !tagEn) continue;
      rows.push({
        id: `ghana_health_symptoms_${item.row.row_id || String(item.row_idx).padStart(5, "0")}`,
        row_index: item.row_idx,
        text: symptomTwi,
        symptom_twi: symptomTwi,
        faithful_english_meaning: tagEn,
        tag_en: tagEn,
        intent: "health_symptom_report",
        entities: {
          body_system: item.row.body_system || "unknown",
        },
        body_system: item.row.body_system || "unknown",
        ipa_twi: item.row.ipa_twi || "",
        source_twi: item.row.source_twi || "",
        source_dataset: dataset,
        source_url: "https://huggingface.co/datasets/ghananlpcommunity/ghana-health-symptoms",
        license: "cc-by-nc-4.0",
        consent_scope: "dataset_license",
        training_use: "noncommercial_research_only",
      });
    }
    await writeRows(out, rows);
    console.log(JSON.stringify({ imported: rows.length, offset: offset + page.length, out }));
    if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  await writeRows(out, rows);
  console.log(JSON.stringify({ out, rows: rows.length, dataset, license: "cc-by-nc-4.0" }, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
