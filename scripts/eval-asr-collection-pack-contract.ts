import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

type Row = {
  id: string;
  bucket: string;
  language: string;
  reference: string;
  audio_path: string;
  speaker_label: string;
};

const outDir = path.join(process.cwd(), "tmp", "asr-collection-pack-contract");
const expectedCounts: Record<string, number> = {
  health_twi: 100,
  commerce_twi: 50,
  codeswitch_tw_en: 50,
  health_en: 50,
  phone_noise: 50,
};

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  await fs.rm(outDir, { recursive: true, force: true });
  const result = spawnSync("pnpm", ["asr:collection-pack"], {
    cwd: process.cwd(),
    env: { ...process.env, ASR_COLLECTION_OUT_DIR: outDir },
    encoding: "utf8",
  });
  assertOk(result.status === 0, `collection pack failed: ${result.stderr || result.stdout}`);

  const promptsRaw = await fs.readFile(path.join(outDir, "prompts.jsonl"), "utf8");
  const rows = promptsRaw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Row);
  const recorder = await fs.readFile(path.join(outDir, "recorder.html"), "utf8");
  const csv = await fs.readFile(path.join(outDir, "prompts.csv"), "utf8");

  assertOk(rows.length === 300, `expected 300 prompts, got ${rows.length}`);
  assertOk(recorder.includes("navigator.mediaDevices.getUserMedia"), "recorder missing mic capture");
  assertOk(recorder.includes("${prompt.id}.webm"), "recorder missing prompt-id download naming");
  assertOk(csv.includes("speaker_label"), "CSV missing speaker label");

  const counts = rows.reduce<Record<string, number>>((acc, row) => {
    acc[row.bucket] = (acc[row.bucket] || 0) + 1;
    return acc;
  }, {});
  for (const [bucket, expected] of Object.entries(expectedCounts)) {
    assertOk(counts[bucket] === expected, `${bucket}: expected ${expected}, got ${counts[bucket] || 0}`);
  }

  const ids = new Set<string>();
  for (const row of rows) {
    assertOk(!ids.has(row.id), `duplicate id ${row.id}`);
    ids.add(row.id);
    assertOk(
      /^[a-z_]+_sp\d{3}_u\d{4}$/.test(row.id),
      `id does not match recorder filename contract: ${row.id}`,
    );
    assertOk(row.audio_path === `MISSING_AUDIO_${row.id}`, `bad audio placeholder for ${row.id}`);
    assertOk(row.speaker_label.startsWith("speaker_"), `bad speaker label for ${row.id}`);
    assertOk(row.reference.length > 4, `short reference for ${row.id}`);
  }

  console.log("ok asr-collection-pack-contract");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
