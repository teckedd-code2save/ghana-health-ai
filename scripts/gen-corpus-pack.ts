/**
 * Generate a corpus prompt pack by translating curated English source lines
 * into Twi (or Twi-English code-switch) via the configured LLM.
 *
 * Output rows are DRAFTS: `needs_review: true` — a native speaker must
 * correct them in the recorder (editable prompt mode) before training use.
 *
 * Usage:
 *   sec -- pnpm corpus:gen -- --source tmp/corpus-source/health-en.txt \
 *     --bucket health_twi --language tw --pack v2
 *
 *   # code-switch pack (model is asked to keep natural Twi-English mix):
 *   sec -- pnpm corpus:gen -- --source tmp/corpus-source/commerce-en.txt \
 *     --bucket codeswitch_tw_en --language tw-en --pack v2
 */

import fs from "node:fs";
import path from "node:path";

type Args = {
  source: string;
  bucket: string;
  language: "tw" | "tw-en";
  pack: string;
  limit: number;
};

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (name: string, dflt = "") => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? argv[i + 1] : dflt;
  };
  return {
    source: get("source"),
    bucket: get("bucket", "health_twi"),
    language: (get("language", "tw") as "tw" | "tw-en") || "tw",
    pack: get("pack", "v2"),
    limit: Number(get("limit", "0")) || 0,
  };
}

function resolveLlm() {
  const groq = process.env.GROQ_API_KEY;
  if (groq) {
    return {
      apiKey: groq,
      baseUrl: "https://api.groq.com/openai/v1",
      model: process.env.LLM_MODEL || "llama-3.3-70b-versatile",
    };
  }
  const openai = process.env.OPENAI_API_KEY;
  if (openai) {
    return {
      apiKey: openai,
      baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      model: process.env.LLM_MODEL || "gpt-4o-mini",
    };
  }
  throw new Error("No LLM key — run via `sec --` (needs GROQ_API_KEY or OPENAI_API_KEY)");
}

async function chat(messages: { role: string; content: string }[]): Promise<string> {
  const llm = resolveLlm();
  const res = await fetch(`${llm.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${llm.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: llm.model, messages, temperature: 0.2, max_tokens: 2000 }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  return data.choices?.[0]?.message?.content?.trim() || "";
}

function extractJsonArray(raw: string): string[] {
  const match = raw.match(/\[[\s\S]*\]/);
  if (!match) throw new Error(`no JSON array in model output: ${raw.slice(0, 200)}`);
  const parsed = JSON.parse(match[0]);
  if (!Array.isArray(parsed)) throw new Error("model output is not an array");
  return parsed.map((x) => String(x).trim()).filter(Boolean);
}

async function main() {
  const args = parseArgs();
  if (!args.source) throw new Error("--source is required");
  const lines = fs
    .readFileSync(args.source, "utf8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
  const targets = args.limit > 0 ? lines.slice(0, args.limit) : lines;
  console.log(`[corpus-gen] ${targets.length} source lines → bucket=${args.bucket} lang=${args.language}`);

  const styleNote =
    args.language === "tw-en"
      ? "Translate each sentence into natural spoken Ghanaian style mixing Twi and English the way a real person in Accra or Kumasi speaks — keep common English words (medicine names, shop words, numbers) in English inside Twi grammar. Keep each line short enough to say in one breath."
      : "Translate each sentence into natural, correct Asante Twi, the way a fluent native speaker would say it aloud. Keep medicine/product names in their common form. Keep each line short enough to say in one breath.";

  const rows: Record<string, unknown>[] = [];
  const chunkSize = 10;
  for (let i = 0; i < targets.length; i += chunkSize) {
    const chunk = targets.slice(i, i + chunkSize);
    const raw = await chat([
      {
        role: "system",
        content: `You are a professional English↔Twi translator for a Ghana health voice app. ${styleNote} Return ONLY a JSON array of strings, one translation per input line, same order.`,
      },
      { role: "user", content: JSON.stringify(chunk) },
    ]);
    const translated = extractJsonArray(raw);
    for (let j = 0; j < chunk.length; j++) {
      const n = rows.length + 1;
      rows.push({
        id: `corpus_${args.pack}_${args.bucket}_u${String(n).padStart(4, "0")}`,
        bucket: args.bucket,
        language: args.language,
        reference: translated[j] || "",
        en_reference: chunk[j],
        needs_review: true,
        source: "llm_translation_draft",
        domain_tags: ["curated_corpus", `pack_${args.pack}`],
        recording_tags: [],
      });
    }
    console.log(`[corpus-gen] ${Math.min(i + chunkSize, targets.length)}/${targets.length}`);
  }

  const dropped = rows.filter((r) => !r.reference).length;
  const outPath = path.join(
    "tmp",
    "asr-collection-pack",
    `prompts.corpus-${args.pack}.${args.bucket}.jsonl`,
  );
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(
    outPath,
    rows.filter((r) => r.reference).map((r) => JSON.stringify(r)).join("\n") + "\n",
    "utf8",
  );
  console.log(`[corpus-gen] wrote ${rows.length - dropped} rows -> ${outPath} (dropped ${dropped} empty)`);
  console.log("[corpus-gen] REMINDER: rows are needs_review drafts — correct them in the recorder before training.");
}

main().catch((err) => {
  console.error("[corpus-gen] failed:", err);
  process.exit(1);
});
