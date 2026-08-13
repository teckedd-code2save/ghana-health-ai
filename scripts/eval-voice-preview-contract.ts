const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const form = new FormData();
  form.append("language", "en");

  const res = await fetch(`${baseUrl}/api/voice/preview`, {
    method: "POST",
    body: form,
  });
  const data = (await res.json().catch(() => ({}))) as { error?: string };
  assertOk(res.status === 400, `expected 400 for missing audio, got ${res.status}`);
  assertOk(typeof data.error === "string" && data.error.length > 0, "missing error message");
  console.log(`ok voice-preview-contract status=${res.status}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

export {};
