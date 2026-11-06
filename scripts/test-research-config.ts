import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { compileFunction } from "node:vm";
import { test } from "node:test";
import ts from "typescript";

function configFixture(available: boolean) {
  let healthChecks = 0;
  const dependencies: Record<string, unknown> = {
    "@/lib/api": { jsonOk: (body: unknown, init?: ResponseInit) => Response.json(body, init) },
    "@/lib/research-r3": { verifiedR3Config: async () => { healthChecks++; return available ? { token: "must-not-be-public", url: "must-not-be-public" } : null; } },
    "@/lib/paystack": { isPaystackConfigured: () => false, getPaystackPublicKey: () => "" },
    "@/lib/llm": { isLlmConfigured: () => true },
    "@/lib/modal-asr": { isModalAsrConfigured: () => true, resolveModalAsrRoute: (lang: string) => ({ name: lang === "en" ? "english" : "v6" }) },
    "@/lib/modal-tts": { isModalTtsConfigured: () => true, resolveTtsRoute: () => ({ provider: "fixture", modelLabel: "fixture" }) },
    "@/lib/embed": { isAbenaConfigured: () => false, isEmbedConfigured: () => true },
  };
  const code = ts.transpileModule(readFileSync("src/app/api/config/route.ts", "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const loaded = { exports: {} };
  compileFunction(code, ["exports", "require", "module", "process"], { filename: "config-route-fixture" })(
    loaded.exports, (name: string) => { assert(Object.hasOwn(dependencies, name)); return dependencies[name]; }, loaded, { env: {} });
  return { get: (loaded.exports as { GET: (request: Request) => Promise<Response> }).GET, checks: () => healthChecks };
}

test("ordinary app configuration never checks research gateway or changes Stable flags", async () => {
  const f = configFixture(true);
  const response = await f.get(new Request("https://fixture.test/api/config"));
  const body = await response.json();
  assert.equal(f.checks(), 0); assert.equal(body.llm, true); assert.equal(body.modalAsr, true);
  assert.equal(body.researchR3, undefined); assert.equal(body.appUrl, "https://fixture.test");
});
for (const available of [false, true]) test(`separate research query returns only no-store availability=${available}`, async () => {
  const f = configFixture(available);
  const response = await f.get(new Request("https://fixture.test/api/config?research=r3"));
  assert.equal(f.checks(), 1);
  assert.equal(response.headers.get("cache-control"), "private, no-store");
  assert.deepEqual(await response.json(), { researchR3: { available } });
});
