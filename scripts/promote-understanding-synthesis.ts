import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import {
  corpusSynthesisPromptVersion,
  corpusSynthesisSchema,
  type CorpusSynthesis,
} from "../src/lib/research-synthesis";
import { readCorpusCandidates } from "../src/lib/research-understanding-store";

const root = process.cwd();
const defaultInput = path.join(root, "tmp", "understanding-corpus", "synthesis.v1.jsonl");
const defaultOutput = path.join(root, "data", "understanding-corpus", "synthesis.v1.jsonl");

function argValue(name: string, fallback: string) {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

async function main() {
  const input = argValue("--input", defaultInput);
  const output = argValue("--out", defaultOutput);
  const candidates = await readCorpusCandidates();
  const candidateById = new Map(candidates.map((row) => [row.id, row]));
  const raw = await fs.readFile(input, "utf8");
  const records = raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => corpusSynthesisSchema.parse(JSON.parse(line)))
    .filter((row) => row.prompt_version === corpusSynthesisPromptVersion);
  const valid: CorpusSynthesis[] = [];
  const rejected: Array<{ row_id: string; reason: string }> = [];

  for (const record of records) {
    const candidate = candidateById.get(record.row_id);
    if (!candidate) {
      rejected.push({ row_id: record.row_id, reason: "missing_candidate" });
      continue;
    }
    if (candidate.source_hash !== record.source_hash) {
      rejected.push({ row_id: record.row_id, reason: "source_hash_mismatch" });
      continue;
    }
    if (!record.proposals.some((proposal) => proposal.proposal_id === record.recommended_proposal_id) &&
        record.synthesized_proposal?.proposal_id !== record.recommended_proposal_id) {
      rejected.push({ row_id: record.row_id, reason: "missing_recommended_proposal" });
      continue;
    }
    valid.push(record);
  }

  if (valid.length === 0) throw new Error("No current, source-matched synthesis records are available to promote.");
  if (rejected.length > 0) throw new Error(`Synthesis promotion rejected ${rejected.length} rows: ${JSON.stringify(rejected.slice(0, 10))}`);
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, `${valid.sort((left, right) => left.row_id.localeCompare(right.row_id)).map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  console.log(JSON.stringify({
    schema_version: 1,
    prompt_version: corpusSynthesisPromptVersion,
    promoted: valid.length,
    needs_human_review: valid.filter((row) => row.adjudication.status === "needs_human_review").length,
    auto_recommended: valid.filter((row) => row.adjudication.status !== "needs_human_review").length,
    output,
  }, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
