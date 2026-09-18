import fs from "node:fs/promises";
import path from "node:path";
import { corpusDigest, corpusTextKey, sourceCorpusHash, sourceCorpusSchema, type SourceCorpusRow } from "../src/lib/source-corpus-annotations";

async function main() {
  const root = process.cwd();
  const directory = path.join(root, "data", "annotated-source-corpus");
  const inputPaths = ["data/medical-response-corpus/afrihealth-ghana-response-train.v1.jsonl", "data/medical-response-corpus/afrihealth-ghana-response-eval.v1.jsonl", "data/understanding-corpus/candidates.v0.jsonl"];
  const rows: SourceCorpusRow[] = [];
  const inputs = [];
  for (const input of inputPaths) {
    const raw = await fs.readFile(path.join(root, input), "utf8");
    inputs.push({ path: input, sha256: corpusDigest(raw) });
    for (const line of raw.split("\n").filter((line) => line.trim())) {
      const row = JSON.parse(line);
      let source;
      if (input.includes("afrihealth")) {
        if (row.language !== "en") continue;
        source = {
          id: row.id, source: row.source_dataset, source_record_id: row.source_record_id,
          source_revision: row.source_revision, source_url: row.source_url,
          license: row.license, license_evidence_url: row.license_evidence_url, attribution: row.attribution,
          split: row.source_split, language: "en", task: "understanding_and_response",
          utterance: row.question, reference_answer: row.answer, speaker_id: null,
        };
      } else {
        if (!["waxal", "ghana_nlp_speech"].includes(row.source)) continue;
        if (!['train', 'validation', 'dev', 'test'].includes(row.split)) throw new Error(`Unknown split: ${row.id}`);
        const waxal = row.source === "waxal";
        const url = waxal ? "https://huggingface.co/datasets/google/WaxalNLP" : "https://huggingface.co/datasets/ghanaopenai/twi-speech-text-multispeaker-16k";
        source = {
          id: row.id, source: row.source, source_record_id: row.source_record_id,
          source_revision: `local-manifest-sha256:${corpusDigest(raw)}`, source_url: url,
          license: waxal ? "cc-by-4.0" : "cc-by-nc-4.0", license_evidence_url: url,
          attribution: waxal ? "University of Ghana / WAXAL" : "GhanaNLP / Ghana Open AI",
          split: row.split === "dev" ? "validation" : row.split, language: "tw", task: "understanding",
          utterance: row.text, reference_answer: null, speaker_id: row.speaker_id ?? null,
        };
      }
      const parsed = sourceCorpusSchema.parse({ ...source, source_hash: "0".repeat(64) });
      parsed.source_hash = sourceCorpusHash(parsed);
      rows.push(parsed);
    }
  }
  if (new Set(rows.map((row) => row.id)).size !== rows.length) throw new Error("Duplicate source IDs");
  // Held-out text wins duplicate conflicts, never the training copy.
  const ordered = rows.sort((a, b) => Number(a.split === "train") - Number(b.split === "train") || a.id.localeCompare(b.id));
  const seen = new Set<string>();
  const excluded = [];
  const accepted = [];
  for (const row of ordered) {
    const key = corpusTextKey(row.utterance);
    if (seen.has(key)) { excluded.push({ id: row.id, reason: "duplicate_or_held_out_utterance", split: row.split }); continue; }
    seen.add(key);
    accepted.push(row);
  }
  await fs.mkdir(directory, { recursive: true });
  for (const [name, contents] of [["sources.v1.jsonl", accepted], ["excluded.v1.jsonl", excluded]] as const) {
    await fs.writeFile(path.join(directory, name), contents.map((row) => JSON.stringify(row)).join("\n") + "\n");
  }
  const counts: Record<string, number> = {};
  for (const row of accepted) { const key = `${row.source}:${row.split}`; counts[key] = (counts[key] || 0) + 1; }
  const report = { schema_version: 1, created_at: new Date().toISOString(), inputs, source_rows: rows.length, unique_rows: accepted.length, excluded: excluded.length, counts, notes: "Original English AfriHealth QA plus existing WAXAL/GhanaNLP text manifests. No audio uploaded. No source labels or synthetic seeds reused. Source text and original split retained; exact held-out overlap removed from training. Unknown original speech Hub revisions are identified honestly by local manifest hash. GhanaNLP remains noncommercial research-only." };
  await fs.writeFile(path.join(directory, "sources.v1.summary.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
