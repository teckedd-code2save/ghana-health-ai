import fs from "node:fs/promises";
import path from "node:path";

type RubricCase = {
  must_include_any?: string[][];
  must_not_include_any?: string[][];
};

type Rubric = {
  schema_version: number;
  cases: Record<string, RubricCase>;
};

type Prediction = {
  id: string;
  category: string;
  text: string;
  prediction: string;
  latency_ms_approx?: number;
};

type BenchmarkResult = {
  created_at?: string;
  model_id: string;
  adapter_id?: string | null;
  tokenizer_id?: string;
  resolved_revision?: string;
  resolved_adapter_revision?: string | null;
  case_count: number;
  elapsed_seconds: number;
  predictions: Prediction[];
};

const root = process.cwd();
const rubricPath = path.join(root, "data", "understanding-benchmark", "rubric.v0.json");
const defaultResultsDir = path.join(root, "tmp", "understanding-results", "understanding");
const defaultOut = path.join(root, "data", "understanding-benchmark", "scorecard.v0.json");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

async function findJsonFiles(dir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const nested = await Promise.all(
      entries.map(async (entry) => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) return findJsonFiles(fullPath);
        return entry.name.endsWith(".json") ? [fullPath] : [];
      }),
    );
    return nested.flat();
  } catch {
    return [];
  }
}

function includesAny(value: string, terms: string[]) {
  const lower = normalizeForMatch(value);
  return terms.some((term) => lower.includes(normalizeForMatch(term)));
}

function normalizeForMatch(value: string) {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}₵$]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function scorePrediction(prediction: Prediction, rubricCase: RubricCase) {
  const failures: string[] = [];
  let checks = 0;
  let passed = 0;

  for (const group of rubricCase.must_include_any ?? []) {
    checks += 1;
    if (includesAny(prediction.prediction, group)) {
      passed += 1;
    } else {
      failures.push(`missing one of: ${group.join(" | ")}`);
    }
  }

  for (const group of rubricCase.must_not_include_any ?? []) {
    checks += 1;
    if (!includesAny(prediction.prediction, group)) {
      passed += 1;
    } else {
      failures.push(`contains forbidden: ${group.join(" | ")}`);
    }
  }

  return {
    id: prediction.id,
    category: prediction.category,
    text: prediction.text,
    prediction: prediction.prediction,
    checks,
    passed,
    score: checks ? passed / checks : 0,
    ok: checks > 0 && passed === checks,
    failures,
  };
}

function candidateName(result: BenchmarkResult) {
  return result.adapter_id ? `${result.model_id} + ${result.adapter_id}` : result.model_id;
}

async function main() {
  const resultsDir = argValue("--results-dir", defaultResultsDir);
  const out = argValue("--out", defaultOut);
  const rubric = JSON.parse(await fs.readFile(rubricPath, "utf8")) as Rubric;
  const resultFiles = await findJsonFiles(resultsDir);
  const resultByCandidate = new Map<string, { filePath: string; result: BenchmarkResult }>();

  for (const filePath of resultFiles) {
    const result = JSON.parse(await fs.readFile(filePath, "utf8")) as BenchmarkResult;
    if (!Array.isArray(result.predictions)) continue;
    const key = candidateName(result);
    const prior = resultByCandidate.get(key);
    if (
      prior &&
      (prior.result.case_count > result.case_count ||
        (prior.result.case_count === result.case_count &&
          (prior.result.created_at ?? "") >= (result.created_at ?? "")))
    ) {
      continue;
    }
    resultByCandidate.set(key, { filePath, result });
  }

  const scorecards = [];

  for (const { filePath, result } of resultByCandidate.values()) {

    const rows = result.predictions
      .filter((prediction) => rubric.cases[prediction.id])
      .map((prediction) => scorePrediction(prediction, rubric.cases[prediction.id]));
    const checks = rows.reduce((sum, row) => sum + row.checks, 0);
    const passed = rows.reduce((sum, row) => sum + row.passed, 0);
    const exact = rows.filter((row) => row.ok).length;
    const score = checks ? passed / checks : 0;
    const criticalFailures = rows
      .filter((row) => !row.ok && (row.category === "health" || row.category === "commerce"))
      .slice(0, 12)
      .map((row) => ({
        id: row.id,
        category: row.category,
        text: row.text,
        prediction: row.prediction,
        failures: row.failures,
      }));

    scorecards.push({
      candidate: candidateName(result),
      model_id: result.model_id,
      adapter_id: result.adapter_id ?? null,
      tokenizer_id: result.tokenizer_id ?? null,
      resolved_revision: result.resolved_revision ?? null,
      resolved_adapter_revision: result.resolved_adapter_revision ?? null,
      artifact: path.relative(root, filePath),
      cases_scored: rows.length,
      exact_cases: exact,
      checks,
      passed,
      score: Number(score.toFixed(4)),
      elapsed_seconds: result.elapsed_seconds,
      critical_failures: criticalFailures,
    });
  }

  scorecards.sort((a, b) => b.score - a.score);
  const payload = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    rubric: path.relative(root, rubricPath),
    scorecards,
    decision_hint:
      "Candidates with health or commerce critical failures are draft assistants only. Promote only after human review confirms meaning preservation.",
  };
  await fs.mkdir(path.dirname(out), { recursive: true });
  await fs.writeFile(out, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(payload, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
