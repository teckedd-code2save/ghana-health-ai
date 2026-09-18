import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";

type Prediction = {
  id: string; set: string; language: string; prediction: string;
  prompt: Array<{ role: string; content: string }>; reference?: string;
};

async function main() {
  const directory = process.argv[2];
  if (!directory) throw new Error("Provide the downloaded completed run directory");
  const read = async (name: string) => JSON.parse(await fs.readFile(path.join(directory, name), "utf8"));
  const report = await read("evaluation.json");
  const manifest = await read("corpus_manifest.json");
  const training = await read("training_metrics.json");
  const base: Prediction[] = await read("base_reply_predictions.json");
  const adapter: Prediction[] = await read("adapter_reply_predictions.json");
  const byId = new Map(base.map((row) => [row.id, row]));
  if (base.length !== adapter.length || adapter.some((row) => !byId.has(row.id))) throw new Error("Incomplete base/adapter comparison");
  const b = report.comparisons.base;
  const a = report.comparisons.adapter;
  const structured = await read("adapter_structured_predictions.json") as Array<{ reference: { intent: string } }>;
  const trainText = await fs.readFile("tmp/medical-response-pilot/v1/train.jsonl", "utf8");
  if (createHash("sha256").update(trainText).digest("hex") !== manifest.artifacts.find((row: { file: string }) => row.file === "train.jsonl")?.sha256) throw new Error("Training source artifact no longer matches this model");
  const train = trainText.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const intents: Record<string, number> = {};
  for (const row of train.filter((row) => row.task === "interpret_and_reply")) intents[row.reference.intent] = (intents[row.reference.intent] ?? 0) + 1;
  const majorityIntent = Object.entries(intents).sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0][0];
  const majorityAgreement = structured.filter((row) => row.reference.intent === majorityIntent).length / structured.length;
  const number = (value: number) => value.toFixed(2);
  const quote = (text: string) => text.split("\n").map((line) => `> ${line}`).join("\n");
  const lines = [
    "# Saved-Annotation LLM Pilot", "",
    "Research output, not medical advice or a production model. No GPT fallback produced these replies.", "",
    `Run: ${report.run_id}`, `Base: ${report.base_model}`, "",
    `Trained on ${manifest.training_sources} unique sources (${manifest.training_examples} task views). Held back ${manifest.holdout_sources} source groups. No human-verified gold labels are claimed.`, "",
    "## Measured Results", "",
    "| Check | Base | Adapter |", "| --- | ---: | ---: |",
    `| Valid interpretation JSON | ${b.structured.valid_json_schema}/${b.structured.count} | ${a.structured.valid_json_schema}/${a.structured.count} |`,
    `| Intent agreement with model labels | ${number(b.structured.intent_agreement * 100)}% | ${number(a.structured.intent_agreement * 100)}% |`,
    `| Question-grounded entities, all rows | ${number(b.structured.grounded_entities_rate * 100)}% | ${number(a.structured.grounded_entities_rate * 100)}% |`,
    ...["pilot_holdout_tw", "pilot_holdout_en", "locked_source_tw", "locked_source_en"].map((key) => `| ${key} reply chrF++ | ${number(b.replies[key].chrf_pp)} | ${number(a.replies[key].chrf_pp)} |`),
    `| Critical product lexical checks | ${b.product_lexical_proxies.filter((row: { critical: boolean; lexical_check_pass: boolean }) => row.critical && row.lexical_check_pass).length}/8 | ${a.product_lexical_proxies.filter((row: { critical: boolean; lexical_check_pass: boolean }) => row.critical && row.lexical_check_pass).length}/8 |`,
    "", `Always predicting the most common training intent (${majorityIntent}) gives ${number(majorityAgreement * 100)}% held-out intent agreement. Use this simple baseline when interpreting the adapter's score.`, "",
    "chrF++ measures wording similarity, not meaning accuracy or medical safety. JSON format success is not proof of understanding. The saved model-assisted targets and source references can contain errors. The structured test uses greedy decoding, which can cause repetition in this base; free-text results use its documented sampling settings.", "",
    `Training loss: ${number(training.train_loss)}. Total measured worker time: ${number(report.elapsed_seconds / 60)} minutes. GPU-only estimated cost: $${number(report.gpu_cost_estimate_usd)}; this is not the final bill.`, "",
    "## Replies", "",
    "All free-text evaluation responses are below, including failures. Product fixtures are diagnostics, not clinical validation.", "",
  ];
  for (const row of [...adapter].sort((left, right) => left.set.localeCompare(right.set) || left.id.localeCompare(right.id))) {
    lines.push(`### ${row.id}`, "", `Set: ${row.set}; language: ${row.language}`, "", "**Conversation**", "",
      ...row.prompt.flatMap((message) => [`${message.role}:`, quote(message.content), ""]),
      "**Base model**", "", quote(byId.get(row.id)!.prediction), "",
      "**Trained adapter**", "", quote(row.prediction), "");
    if (row.reference) lines.push("**Reference (not human gold)**", "", quote(row.reference), "");
  }
  const destination = path.join(directory, "comparison.md");
  await fs.writeFile(destination, lines.join("\n"));
  const summary = { ...report, training_sources: manifest.training_sources, holdout_sources: manifest.holdout_sources,
    training_examples: manifest.training_examples, training_loss: training.train_loss,
    annotation_sha256: manifest.annotation_sha256,
    majority_intent_baseline: { intent: majorityIntent, agreement: majorityAgreement },
    conclusions: ["Format and wording metrics do not establish semantic competence",
      a.structured.intent_agreement > majorityAgreement ? "Intent agreement exceeds the majority-class baseline" : "Intent agreement does not exceed the majority-class baseline",
      a.replies.locked_source_tw.chrf_pp > b.replies.locked_source_tw.chrf_pp ? "Locked Twi reference similarity increased" : "Locked Twi reference similarity did not improve",
      a.replies.locked_source_en.chrf_pp > b.replies.locked_source_en.chrf_pp ? "Locked English reference similarity increased" : "Locked English reference similarity did not improve",
      "Product checks are lexical proxies, not clinical validation",
      "Not promoted or deployed; no public HF upload"],
  };
  await fs.writeFile("data/medical-response-corpus/afrihealth-annotated-pilot.v1.summary.json", JSON.stringify(summary, null, 2) + "\n");
  console.log(destination);
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
