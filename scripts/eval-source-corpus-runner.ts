import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { sourceAnnotationSchema, sourceCorpusHash, type SourceCorpusRow } from "../src/lib/source-corpus-annotations";

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "source-runner-contract-"));
  let calls = 0;
  const server = http.createServer(async (request, response) => {
    try {
      let raw = "";
      for await (const chunk of request) raw += chunk;
      const body = JSON.parse(raw);
      const payload = JSON.parse(body.messages.at(-1).content);
      calls += 1;
      const rows = payload.rows.map((row: { row_id: string; utterance: string; reference_answer: string | null; teacher_a?: unknown }) => row.teacher_a
        ? { row_id: row.row_id, choice: "teacher_a", confidence: 0.99, rationale: "Contract fixture", review_reasons: [], synthesized: null }
        : { row_id: row.row_id, normalized_text: row.utterance, natural_english: row.utterance,
          intent: row.reference_answer ? "health_service_question" : "general_statement", entities: {},
          uncertainty_notes: "", requires_clarification: false, safety_level: "routine",
          source_answer_assessment: row.reference_answer ? "supportable" : "not_applicable", source_answer_issues: [],
          reply: row.reference_answer, evidence_spans: [row.reference_answer || row.utterance], confidence: 0.99 });
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ model: body.model, choices: [{ finish_reason: "stop", message: { content: JSON.stringify({ rows }) } }], usage: { total_tokens: 1 } }));
    } catch { response.writeHead(500); response.end(); }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const port = (server.address() as { port: number }).port;
    const sources: SourceCorpusRow[] = ["response", "meaning"].map((kind) => {
      const row: SourceCorpusRow = {
        id: `contract_${kind}`, source: "contract_fixture", source_record_id: kind, source_revision: "test",
        source_url: "https://example.com/fixture", license: "cc0-1.0", license_evidence_url: "https://example.com/fixture", attribution: "Automated test only",
        split: "train", language: "en", task: kind === "response" ? "understanding_and_response" : "understanding",
        utterance: kind === "response" ? "Where can I find a clinic?" : "I saw a house.", reference_answer: kind === "response" ? "Ask the local health service." : null,
        source_hash: "", speaker_id: null,
      };
      row.source_hash = sourceCorpusHash(row);
      return row;
    });
    const input = path.join(directory, "sources.jsonl");
    const output = path.join(directory, "annotations.jsonl");
    await fs.writeFile(input, sources.map((row) => JSON.stringify(row)).join("\n"));
    const run = () => new Promise<number | null>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "scripts/annotate-source-corpus.ts", "--provider", "mock", "--input", input, "--out", output, "--limit", "0", "--chunk-size", "2"], {
        env: { ...process.env, OPENAI_API_KEY: crypto.randomUUID(), OPENAI_BASE_URL: `http://127.0.0.1:${port}/v1` },
        stdio: ["ignore", "ignore", "pipe"],
      });
      let errors = "";
      child.stderr.on("data", (chunk) => { errors += chunk; });
      child.on("error", reject);
      child.on("close", (code) => { if (code !== 0) reject(new Error(errors)); else resolve(code); });
    });
    assert.equal(await run(), 0);
    assert.equal(calls, 3, "Both teachers and adjudicator must run");
    const annotations = (await fs.readFile(output, "utf8")).trim().split("\n").map((line) => sourceAnnotationSchema.parse(JSON.parse(line)));
    assert.equal(annotations.length, 2);
    assert.ok(annotations.every((row) => row.status === "model_consensus" && !row.eligible_for_production_training));
    assert.equal(annotations.find((row) => row.row_id === "contract_meaning")!.selected.reply, null);
    assert.equal(await run(), 0);
    assert.equal(calls, 3, "Completed annotations must not cause more provider requests on resume");
    console.log("Source runner: isolated mock-provider three-stage generation, task separation, saved annotations and zero-call resume passed. No real API calls or corpus labels generated.");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await fs.rm(directory, { recursive: true, force: true });
  }
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
