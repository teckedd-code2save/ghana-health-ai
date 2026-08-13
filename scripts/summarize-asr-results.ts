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
  dataset_config?: string;
  language?: string;
  n?: number;
  num_beams?: number;
  streaming?: boolean;
  wer_pct?: number;
  cer_pct?: number;
  val_wer?: number;
  val_cer?: number;
  promote?: boolean;
  hub?: string | null;
};

const resultsDir =
  process.env.ASR_RESULTS_DIR || path.join(process.cwd(), "tmp", "asr-results");
const promoteWer = Number(process.env.ASR_PROMOTE_WER_PCT || 28);
const holdWer = Number(process.env.ASR_HOLD_WER_PCT || 32);
const includeAll = process.argv.includes("--all");

function decision(result: AsrResult) {
  const wer = Number(result.wer_pct ?? (Number.isFinite(Number(result.val_wer)) ? Number(result.val_wer) * 100 : undefined));
  if (!Number.isFinite(wer)) return "UNKNOWN";
  if (result.run_name) {
    if (result.promote && wer <= promoteWer) return "TRAIN_PROMOTE_CANDIDATE";
    if (result.promote && wer <= holdWer) return "TRAIN_HOLD_AND_VALIDATE";
    return "TRAIN_DO_NOT_PROMOTE";
  }
  if (result.language === "en" || result.dataset_config === "en") {
    return "ENGLISH_RETENTION_CHECK";
  }
  if (wer <= promoteWer) return "PROMOTE_CANDIDATE";
  if (wer <= holdWer) return "HOLD_AND_VALIDATE";
  return "DO_NOT_PROMOTE";
}

async function main() {
  const files = (await fs.readdir(resultsDir)).filter((file) => file.endsWith(".json"));
  const rows: AsrResult[] = [];
  for (const file of files) {
    const raw = await fs.readFile(path.join(resultsDir, file), "utf8");
    rows.push(JSON.parse(raw) as AsrResult);
  }

  const scored = includeAll
    ? rows
    : rows.filter((row) => Number.isFinite(Number(row.wer_pct)));
  scored.sort((a, b) => {
    const aWer = Number(a.wer_pct ?? (Number.isFinite(Number(a.val_wer)) ? Number(a.val_wer) * 100 : 999));
    const bWer = Number(b.wer_pct ?? (Number.isFinite(Number(b.val_wer)) ? Number(b.val_wer) * 100 : 999));
    return aWer - bWer;
  });
  if (!scored.length) throw new Error(`No scored ASR result JSON files in ${resultsDir}`);

  console.log("| decision | WER | CER | n | beams | lang | model/run | dataset/sources | hub |");
  console.log("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |");
  for (const row of scored) {
    const werPct = row.wer_pct ?? (Number.isFinite(Number(row.val_wer)) ? roundPct(Number(row.val_wer)) : "n/a");
    const cerPct = row.cer_pct ?? (Number.isFinite(Number(row.val_cer)) ? roundPct(Number(row.val_cer)) : "n/a");
    console.log(
      `| ${decision(row)} | ${werPct} | ${cerPct} | ${
        row.n ?? "n/a"
      } | ${row.num_beams ?? "n/a"} | ${row.language ?? row.dataset_config ?? "n/a"} | ${
        row.run_name ?? row.model_id ?? row.base_model ?? row.architecture ?? "unknown"
      } | ${
        row.dataset ?? row.language ?? "unknown"
      } | ${row.hub ?? "n/a"} |`,
    );
  }
}

function roundPct(value: number) {
  return Math.round(value * 10000) / 100;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

export {};
