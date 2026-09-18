import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createMedicalAnnotationClient, MedicalAnnotationProviderError } from "../src/lib/medical-annotation-client";

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "gha-annotation-client-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-fixture-key";
  try {
    let calls = 0;
    const payload = { rows: [{ row_id: "source-1", question_english: "Original source meaning" }] };
    const client = createMedicalAnnotationClient(directory);
    const messages = [{ role: "user" as const, content: "source-1" }];
    globalThis.fetch = async () => {
      calls += 1;
      return Response.json({
        model: "teacher-snapshot",
        usage: { total_tokens: 20 },
        choices: [{ finish_reason: "stop", message: { content: JSON.stringify(payload) } }],
      });
    };
    assert.deepEqual(await client(messages, "teacher", 100), payload);
    assert.deepEqual(await client(messages, "teacher", 100), payload);
    assert.equal(calls, 1, "Completed stage must resume without another provider call");
    const audits = await fs.readdir(directory);
    assert.equal(audits.length, 2, "Save both raw attempt and reusable checkpoint");
    const audit = JSON.parse(await fs.readFile(path.join(directory, audits[0]), "utf8"));
    assert.equal(audit.returned_model, "teacher-snapshot");
    assert.equal(audit.usage.total_tokens, 20);

    calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return Response.json({ error: { code: "invalid_api_key", message: "secret-must-not-appear" } }, { status: 401 });
    };
    await assert.rejects(client(messages, "unavailable", 100), (error: unknown) => (
      error instanceof MedicalAnnotationProviderError && error.fatal && !error.message.includes("secret-must-not-appear")
    ));
    assert.equal(calls, 1, "Invalid credentials must fail immediately");

    calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return Response.json({ error: { code: "credit_balance_exhausted" } }, { status: 429 });
    };
    await assert.rejects(client(messages, "unfunded", 100), (error: unknown) => (
      error instanceof MedicalAnnotationProviderError && error.fatal && error.code === "credit_balance_exhausted"
    ));
    assert.equal(calls, 1, "Credit exhaustion must stop immediately rather than retrying as a rate limit");

    calls = 0;
    const budgets: number[] = [];
    globalThis.fetch = async (_input, init) => {
      calls += 1;
      budgets.push(JSON.parse(String(init?.body)).max_completion_tokens);
      return Response.json({ choices: [{
        finish_reason: calls === 1 ? "length" : "stop",
        message: { content: JSON.stringify({ response_shape: payload }) },
      }] });
    };
    assert.deepEqual(await client(messages, "truncated", 100), payload);
    assert.deepEqual(budgets, [100, 200], "Truncated responses must be retried with sufficient output budget");
    console.log("Annotation client: cache resume, audit provenance, credential failure, and truncation recovery passed.");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = originalKey;
    await fs.rm(directory, { recursive: true, force: true });
  }
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
