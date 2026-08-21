import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { generateSyntheticVoiceNotes } from "./synthesize-twi-voice-notes";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "gha-synthetic-tts-"));
  const input = path.join(dir, "input.jsonl");
  const manifest = path.join(dir, "manifest.jsonl");
  const outDir = path.join(dir, "audio");
  await fs.writeFile(
    input,
    [
      JSON.stringify({
        id: "reviewed_health_twi_u0001",
        bucket: "health_twi",
        language: "tw",
        reference: "Me ti yɛ me ya paa.",
        en_reference: "My head hurts badly.",
        needs_review: false,
        source: "human_reviewed",
        domain_tags: ["headache"],
      }),
      JSON.stringify({
        id: "draft_health_twi_u0002",
        bucket: "health_twi",
        language: "tw",
        reference: "Draft text",
        needs_review: true,
        source: "llm_translation_draft",
      }),
    ].join("\n") + "\n",
    "utf8",
  );

  const strict = await generateSyntheticVoiceNotes({
    input,
    outDir,
    manifest,
    provider: "stable-twi",
    limit: 0,
    allowDrafts: false,
    dryRun: true,
    mock: true,
    voiceId: "contract",
  });
  assertOk(strict.eligible === 1, `expected one reviewed eligible row, got ${strict.eligible}`);
  assertOk(strict.skipped.some((reason) => reason.includes("still a draft")), "expected draft skip reason");

  const generated = await generateSyntheticVoiceNotes({
    input,
    outDir,
    manifest,
    provider: "stable-twi",
    limit: 1,
    allowDrafts: false,
    dryRun: false,
    mock: true,
    voiceId: "contract",
  });
  assertOk(generated.rows.length === 1, `expected one generated row, got ${generated.rows.length}`);
  const row = generated.rows[0];
  assertOk(row.source === "synthetic_tts", `unexpected source ${row.source}`);
  assertOk(row.holdout === false, "synthetic row must not be holdout");
  assertOk(row.speaker_label === "synthetic_stable-twi", `unexpected speaker ${row.speaker_label}`);
  assertOk(row.tts_model === "ghananlpcommunity/stable-twi-tts", `unexpected model ${row.tts_model}`);
  assertOk(row.voice_id === "contract", `unexpected voice ${row.voice_id}`);
  assertOk(row.duration_s > 0, `expected duration, got ${row.duration_s}`);
  assertOk(/^[0-9a-f]{64}$/.test(row.sha256), `bad sha256 ${row.sha256}`);

  const audio = await fs.readFile(row.audio_path);
  assertOk(audio.subarray(0, 4).toString("ascii") === "RIFF", "expected WAV audio");
  const rawManifest = await fs.readFile(manifest, "utf8");
  assertOk(rawManifest.includes("synthetic_tts"), "manifest missing synthetic source");

  await fs.rm(dir, { recursive: true, force: true });
  console.log("ok synthetic-voice-notes");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
