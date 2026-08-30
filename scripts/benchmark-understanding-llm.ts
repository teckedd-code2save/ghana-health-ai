import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";

type Seed = {
  id: string;
  category: string;
  text: string;
  review_status: string;
};

type Json = Record<string, unknown>;

const root = process.cwd();
const seedPath = path.join(root, "data", "understanding-benchmark", "seed.v0.jsonl");
const defaultOutDir = path.join(root, "tmp", "understanding-results", "understanding");

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function parseJsonl<T>(raw: string): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function extractJson(value: string) {
  const fenced = value.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const raw = fenced?.[1] ?? value;
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start >= 0 && end > start) return raw.slice(start, end + 1);
  return raw;
}

async function main() {
  const limit = Number(argValue("--limit", "0"));
  const outDir = argValue("--out-dir", defaultOutDir);
  const modelOverride = argValue("--model", "");
  if (modelOverride) {
    process.env.OPENAI_LANGUAGE_MODEL = modelOverride;
    process.env.LLM_MODEL = modelOverride;
  }
  const provider = llmProviderInfo();
  if (!provider) throw new Error("No LLM provider configured.");

  const seeds = parseJsonl<Seed>(await fs.readFile(seedPath, "utf8"));
  const selected = limit > 0 ? seeds.slice(0, limit) : seeds;
  const predictions = [];
  const started = performance.now();

  for (const seed of selected) {
    const turnStarted = performance.now();
    const content = await chatComplete(
      [
        {
          role: "system",
          content:
            "You are evaluating Twi/Akan and code-switched understanding for a Ghanaian health and commerce voice product. Return JSON only. Do not answer the user. Preserve these product-critical Twi meanings: m'ani kum, mani kum, and ani kum mean eye pain/eye ache in health context, not sleepiness, sadness, or indifference; abɔ waw and abo waw mean cough/coughing; mehome yɛ den and ahome yɛ den mean difficulty breathing; me koko mu yɛ me yaw means chest pain, not heartburn.",
        },
        {
          role: "user",
          content: JSON.stringify({
            instruction:
              "Translate the utterance into faithful English meaning and extract intent/entities. Preserve uncertainty. Do not infer unavailable facts.",
            category: seed.category,
            utterance: seed.text,
            output_schema: {
              natural_english: "faithful English meaning",
              literal_english: "literal English gloss when useful",
              intent: "health|commerce|general|unclear",
              entities: "important symptoms, body parts, items, quantities, locations, dates, negation",
              ambiguity: "what remains uncertain, or empty string",
            },
          }),
        },
      ],
      { temperature: 0, maxTokens: 500 },
    );
    const elapsed = Math.round(performance.now() - turnStarted);
    let parsed: Json = {};
    if (content) {
      try {
        parsed = JSON.parse(extractJson(content)) as Json;
      } catch {
        parsed = { natural_english: content, ambiguity: "Model returned non-JSON output." };
      }
    }
    predictions.push({
      ...seed,
      prediction: asString(parsed.natural_english) || asString(parsed.literal_english) || content || "",
      structured_prediction: parsed,
      latency_ms_approx: elapsed,
    });
  }

  const seedSha256 = crypto
    .createHash("sha256")
    .update(await fs.readFile(seedPath))
    .digest("hex");
  const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  const modelId = `${provider.provider}:${provider.model}`;
  const safeModelId = modelId.replace(/[/:]/g, "--");
  const outputDir = path.join(outDir, safeModelId);
  const outputPath = path.join(outputDir, `${timestamp}.json`);
  const payload = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    model_id: modelId,
    tokenizer_id: null,
    adapter_id: null,
    requested_revision: null,
    resolved_revision: null,
    source_language: "twi_Latn",
    target_language: "eng_Latn",
    seed_sha256: seedSha256,
    case_count: predictions.length,
    elapsed_seconds: Number(((performance.now() - started) / 1000).toFixed(3)),
    device: "remote_llm_api",
    review_status: "unverified",
    predictions,
  };
  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(JSON.stringify({ outputPath, modelId, caseCount: predictions.length }, null, 2));
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
