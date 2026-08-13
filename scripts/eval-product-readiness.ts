type ProductReadiness = {
  ok?: boolean;
  status?: "ready" | "degraded" | "blocked";
  checks?: Array<{
    key?: string;
    level?: "pass" | "degraded" | "fail";
    required?: boolean;
    detail?: string;
  }>;
};

const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");
const allowDegraded = process.env.EVAL_ALLOW_DEGRADED !== "0";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const res = await fetch(`${baseUrl}/api/readiness`);
  const contentType = res.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? ((await res.json().catch(() => ({}))) as ProductReadiness)
    : ({} as ProductReadiness);

  assertOk(contentType.includes("application/json"), `readiness did not return JSON from ${baseUrl}`);
  assertOk(res.ok, `readiness HTTP ${res.status} from ${baseUrl}`);
  assertOk(body.ok === true, `readiness not ok: ${body.status}`);
  assertOk(body.status === "ready" || (allowDegraded && body.status === "degraded"), `bad status ${body.status}`);
  assertOk(Array.isArray(body.checks) && body.checks.length > 0, "missing readiness checks");

  const requiredFailures = body.checks.filter((check) => check.required && check.level === "fail");
  assertOk(
    requiredFailures.length === 0,
    `required readiness failures: ${requiredFailures.map((check) => check.key).join(", ")}`,
  );

  for (const key of ["db", "asr_twi", "asr_english", "tts", "response_fallback"]) {
    const check = body.checks.find((item) => item.key === key);
    assertOk(check, `missing check ${key}`);
    assertOk(check.level === "pass", `${key} is ${check.level}: ${check.detail}`);
  }

  const llm = body.checks.find((item) => item.key === "llm");
  assertOk(llm, "missing check llm");
  assertOk(
    llm.level === "pass" || (allowDegraded && llm.level === "degraded"),
    `llm is ${llm.level}: ${llm.detail}`,
  );

  console.log(`ok product-readiness status=${body.status}`);
  for (const check of body.checks) {
    console.log(`${check.level} ${check.key}: ${check.detail}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
