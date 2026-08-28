import fs from "node:fs/promises";
import path from "node:path";

type Candidate = {
  source: string;
  domain: string;
  audio_artifact_id: string | null;
  model_proposal?: { status?: string; model?: string };
};

type BenchmarkResult = {
  model_id: string;
  tokenizer_id?: string;
  resolved_revision?: string;
  case_count: number;
  elapsed_seconds: number;
  predictions: Array<{ id: string; category: string; text: string; prediction: string; latency_ms_approx?: number }>;
};

const root = process.cwd();
const candidatePath = path.join(root, "tmp", "understanding-corpus", "candidates.v0.jsonl");
const benchmarkDir = path.join(root, "tmp", "understanding-results", "understanding");
const reportPath = path.join(root, "tmp", "understanding-research-report.md");

function parseJsonl<T>(raw: string): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

async function readCandidates() {
  try {
    return parseJsonl<Candidate>(await fs.readFile(candidatePath, "utf8"));
  } catch {
    return [];
  }
}

async function findBenchmarkResults(dir: string): Promise<BenchmarkResult[]> {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const nested = await Promise.all(
      entries.map(async (entry) => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) return findBenchmarkResults(fullPath);
        if (!entry.name.endsWith(".json")) return [];
        try {
          return [JSON.parse(await fs.readFile(fullPath, "utf8")) as BenchmarkResult];
        } catch {
          return [];
        }
      }),
    );
    return nested.flat();
  } catch {
    return [];
  }
}

function table(rows: string[][]) {
  if (!rows.length) return "";
  const header = rows[0];
  const body = rows.slice(1);
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...body.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
}

async function main() {
  const candidates = await readCandidates();
  const benchmarkResults = await findBenchmarkResults(benchmarkDir);
  const bySource = candidates.reduce<Record<string, number>>((acc, row) => {
    acc[row.source] = (acc[row.source] ?? 0) + 1;
    return acc;
  }, {});
  const byDomain = candidates.reduce<Record<string, number>>((acc, row) => {
    acc[row.domain] = (acc[row.domain] ?? 0) + 1;
    return acc;
  }, {});
  const withAudio = candidates.filter((row) => row.audio_artifact_id).length;
  const draftAnnotated = candidates.filter((row) => row.model_proposal?.status === "draft").length;

  const latestByModel = new Map<string, BenchmarkResult>();
  for (const result of benchmarkResults) {
    latestByModel.set(result.model_id, result);
  }

  const report = `# Understanding research report

## Corpus candidate queue

- Candidate rows: ${candidates.length}
- Rows with audio references: ${withAudio}
- Rows with model draft annotations: ${draftAnnotated}
- Review status: drafts only; no row is training-eligible until human review.

${table([
  ["Source", "Rows"],
  ...Object.entries(bySource).map(([source, count]) => [source, String(count)]),
])}

${table([
  ["Domain", "Rows"],
  ...Object.entries(byDomain).map(([domain, count]) => [domain, String(count)]),
])}

## Benchmark outputs

${
  latestByModel.size
    ? table([
        ["Model", "Cases", "Elapsed", "Tokenizer"],
        ...Array.from(latestByModel.values()).map((result) => [
          result.model_id,
          String(result.case_count),
          `${result.elapsed_seconds}s`,
          result.tokenizer_id ?? "model default",
        ]),
      ])
    : "No pulled benchmark result artifacts found yet."
}

## Decision

Use benchmark outputs only to choose draft annotators. Use corpus candidate rows
only after human review/correction. The next training export must include
provenance, consent scope, reviewer decision, and split policy.
`;

  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, report, "utf8");
  console.log(reportPath);
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
