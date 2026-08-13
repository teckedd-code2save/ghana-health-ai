import fs from "node:fs/promises";
import path from "node:path";

type AsrResult = {
  model_id?: string;
  base_model?: string;
  run_name?: string;
  status?: string;
  architecture?: string;
  dataset?: string;
  dataset_name?: string;
  dataset_config?: string | null;
  split?: string;
  language?: string | null;
  n?: number;
  num_beams?: number;
  wer?: number;
  cer?: number;
  wer_pct?: number;
  cer_pct?: number;
  val_wer?: number;
  val_cer?: number;
  promote?: boolean;
  hub?: string | null;
  push_repo?: string;
  full_test?: {
    dataset?: string;
    n?: number;
    wer?: number;
    cer?: number;
    wer_pct?: number;
    cer_pct?: number;
    beam5?: {
      wer?: number;
      cer?: number;
      wer_pct?: number;
      cer_pct?: number;
      n?: number;
      num_beams?: number;
    };
  };
};

type GateStatus = "PASS" | "FAIL" | "MISSING";

type Gate = {
  name: string;
  status: GateStatus;
  evidence: string;
  metric?: number;
};

type ModelAudit = {
  model: string;
  gates: Gate[];
};

const resultsDir =
  process.env.ASR_RESULTS_DIR || path.join(process.cwd(), "tmp", "asr-results");
const strict = process.argv.includes("--strict");

const twiPromoteWerPct = Number(process.env.ASR_TWI_PROMOTE_WER_PCT || 28);
const englishMaxWerPct = Number(process.env.ASR_ENGLISH_MAX_WER_PCT || 17);
const englishMaxDeltaPct = Number(process.env.ASR_ENGLISH_MAX_DELTA_PCT || 5);
const minRepresentativeSamples = Number(process.env.ASR_MIN_REPRESENTATIVE_N || 100);
const preferredEnglishBaseline = process.env.ASR_ENGLISH_BASELINE_MODEL || "openai/whisper-small";
const knownCardModels = new Set(
  (
    process.env.ASR_KNOWN_CARD_MODELS ||
    [
      "teckedd/gha-whisper-small-twi-v6",
      "teckedd/gha-whisper-small-twi-en-balanced-v7-lite",
      "teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen",
      "teckedd/gha-dondo-w2v-bert-twi-v1",
    ].join(",")
  )
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean),
);

function pct(result: AsrResult, key: "wer" | "cer" = "wer") {
  const pctKey = `${key}_pct` as "wer_pct" | "cer_pct";
  const valKey = `val_${key}` as "val_wer" | "val_cer";
  const direct = Number(result[pctKey]);
  if (Number.isFinite(direct)) return direct;
  const raw = Number(result[valKey] ?? result[key]);
  return Number.isFinite(raw) ? Math.round(raw * 10000) / 100 : undefined;
}

function modelName(result: AsrResult) {
  return (
    result.model_id ||
    result.push_repo ||
    result.run_name ||
    result.base_model ||
    result.architecture ||
    "unknown"
  );
}

function isEnglish(result: AsrResult) {
  return result.language === "en" || result.dataset_config === "en";
}

function isTwiWaxal(result: AsrResult) {
  const dataset = `${result.dataset ?? ""} ${result.dataset_name ?? ""} ${result.dataset_config ?? ""} ${result.language ?? ""}`;
  return /WaxalNLP|aka_asr|Asante Twi|Twi/i.test(dataset) && !isEnglish(result);
}

function isHealthDomain(result: AsrResult) {
  const dataset = `${result.dataset ?? ""} ${result.dataset_name ?? ""}`;
  return /health|symptom|medicine|malaria|maternal|clinical/i.test(dataset);
}

function isCodeSwitch(result: AsrResult) {
  const dataset = `${result.dataset ?? ""} ${result.dataset_name ?? ""} ${result.language ?? ""}`;
  return /code[-_ ]?switch|twi[-_ ]?english|tw[-_ ]?en/i.test(dataset);
}

function isPhoneNoise(result: AsrResult) {
  const dataset = `${result.dataset ?? ""} ${result.dataset_name ?? ""}`;
  return /phone|mobile|noise|noisy|field/i.test(dataset);
}

function resultEvidence(result: AsrResult) {
  const wer = pct(result);
  const cer = pct(result, "cer");
  const parts = [
    result.dataset || [result.dataset_name, result.dataset_config, result.split].filter(Boolean).join("/"),
    result.n ? `n=${result.n}` : undefined,
    result.num_beams ? `beam=${result.num_beams}` : undefined,
    wer !== undefined ? `WER=${wer}%` : undefined,
    cer !== undefined ? `CER=${cer}%` : undefined,
    result.hub ? `hub=${result.hub}` : undefined,
  ].filter(Boolean);
  return parts.join(", ") || "result present";
}

function bestByWer(results: AsrResult[]) {
  return [...results]
    .filter((result) => pct(result) !== undefined)
    .sort((a, b) => Number(pct(a)) - Number(pct(b)))[0];
}

function hasValidCardEvidence(result: AsrResult) {
  return typeof result.hub === "string" && result.hub.includes("+card") && !result.hub.includes("push_failed");
}

async function loadResults() {
  const files = (await fs.readdir(resultsDir)).filter((file) => file.endsWith(".json"));
  const rows: AsrResult[] = [];
  for (const file of files) {
    const raw = await fs.readFile(path.join(resultsDir, file), "utf8");
    rows.push(...expandResult(JSON.parse(raw) as AsrResult));
  }
  return rows;
}

function expandResult(result: AsrResult) {
  const results = [result];
  if (result.full_test) {
    results.push({
      ...result,
      dataset: result.full_test.dataset || "google/WaxalNLP/aka_asr:test",
      n: result.full_test.n,
      wer: result.full_test.wer,
      cer: result.full_test.cer,
      wer_pct: result.full_test.wer_pct,
      cer_pct: result.full_test.cer_pct,
      num_beams: 1,
    });
    if (result.full_test.beam5) {
      results.push({
        ...result,
        dataset: result.full_test.dataset || "google/WaxalNLP/aka_asr:test",
        n: result.full_test.beam5.n || result.full_test.n,
        wer: result.full_test.beam5.wer,
        cer: result.full_test.beam5.cer,
        wer_pct: result.full_test.beam5.wer_pct,
        cer_pct: result.full_test.beam5.cer_pct,
        num_beams: result.full_test.beam5.num_beams || 5,
      });
    }
  }
  return results;
}

function gateFromBest(
  name: string,
  result: AsrResult | undefined,
  pass: (result: AsrResult, wer: number) => boolean,
  missing: string,
) {
  if (!result) {
    return { name, status: "MISSING" as const, evidence: missing };
  }
  const wer = pct(result);
  if (wer === undefined) {
    return { name, status: "MISSING" as const, evidence: `no WER in ${resultEvidence(result)}` };
  }
  return {
    name,
    status: pass(result, wer) ? ("PASS" as const) : ("FAIL" as const),
    metric: wer,
    evidence: resultEvidence(result),
  };
}

function auditModel(model: string, results: AsrResult[], englishBaselineWer?: number): ModelAudit {
  const modelResults = results.filter((result) => modelName(result) === model);
  const twi = bestByWer(modelResults.filter(isTwiWaxal));
  const english = bestByWer(modelResults.filter(isEnglish));
  const health = bestByWer(modelResults.filter(isHealthDomain));
  const codeSwitch = bestByWer(modelResults.filter(isCodeSwitch));
  const phoneNoise = bestByWer(modelResults.filter(isPhoneNoise));
  const cardResult = modelResults.find(hasValidCardEvidence);
  const hasKnownCard = knownCardModels.has(model);

  const gates: Gate[] = [
    gateFromBest(
      "Twi representative WER",
      twi,
      (result, wer) => wer <= twiPromoteWerPct && Number(result.n ?? 0) >= minRepresentativeSamples,
      "no Twi Waxal or equivalent result",
    ),
    gateFromBest(
      "English retention",
      english,
      (_result, wer) => {
        if (englishBaselineWer === undefined) return false;
        return wer <= englishMaxWerPct && wer - englishBaselineWer <= englishMaxDeltaPct;
      },
      "no English retention result",
    ),
    gateFromBest(
      "Health-domain Twi",
      health,
      (_result, wer) => wer <= 25,
      "no health-domain ASR result",
    ),
    gateFromBest(
      "Code-switch",
      codeSwitch,
      (_result, wer) => wer <= 30,
      "no Twi-English code-switch ASR result",
    ),
    gateFromBest(
      "Phone/noise",
      phoneNoise,
      (_result, wer) => wer <= 35,
      "no phone/noisy-audio ASR result",
    ),
    {
      name: "HF model card",
      status: cardResult || hasKnownCard ? "PASS" : "MISSING",
      evidence: cardResult
        ? resultEvidence(cardResult)
        : hasKnownCard
          ? "verified by pnpm eval:hf-model-cards"
          : "no pushed result with +card marker",
    },
  ];

  return { model, gates };
}

function printAudit(audits: ModelAudit[]) {
  console.log("| model | promote | failing or missing gates | best evidence |");
  console.log("| --- | --- | --- | --- |");
  for (const audit of audits) {
    const blockers = audit.gates.filter((gate) => gate.status !== "PASS");
    const promote = blockers.length === 0 ? "YES" : "NO";
    const blockerText = blockers.map((gate) => `${gate.name}: ${gate.status}`).join("; ") || "none";
    const evidenceText = audit.gates
      .filter((gate) => gate.metric !== undefined || gate.status === "PASS")
      .map((gate) => `${gate.name} (${gate.evidence})`)
      .join("<br>");
    console.log(`| ${audit.model} | ${promote} | ${blockerText} | ${evidenceText || "n/a"} |`);
  }
}

async function main() {
  const results = await loadResults();
  if (!results.length) throw new Error(`No ASR result JSON files in ${resultsDir}`);

  const englishBaseline = bestByWer(
    results.filter((result) => modelName(result) === preferredEnglishBaseline && isEnglish(result)),
  );
  const englishBaselineWer = englishBaseline ? pct(englishBaseline) : undefined;

  const candidateModels = Array.from(
    new Set(
      results
        .filter((result) => pct(result) !== undefined)
        .map(modelName)
        .filter((name) => name !== "unknown"),
    ),
  ).sort();

  const audits = candidateModels.map((model) => auditModel(model, results, englishBaselineWer));
  printAudit(audits);

  const promoted = audits.filter((audit) => audit.gates.every((gate) => gate.status === "PASS"));
  if (promoted.length) {
    console.log(`\nPromotion candidates: ${promoted.map((audit) => audit.model).join(", ")}`);
  } else {
    console.log("\nPromotion candidates: none");
  }

  if (englishBaselineWer === undefined) {
    console.log(`English baseline: missing ${preferredEnglishBaseline}`);
  } else {
    console.log(`English baseline: ${preferredEnglishBaseline} WER=${englishBaselineWer}%`);
  }

  if (strict && promoted.length === 0) {
    throw new Error("No ASR model passes all promotion gates");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
