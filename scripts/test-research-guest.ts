import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { compileFunction } from "node:vm";
import { test } from "node:test";
import ts from "typescript";
import { z } from "zod";
import * as core from "../src/lib/research-guest-core";

const secret = "fixture-only-not-a-deployment-secret";
function load(file: string, dependencies: Record<string, unknown>) {
  const code = ts.transpileModule(readFileSync(file, "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const loaded = { exports: {} };
  compileFunction(code, ["exports", "require", "module"], { filename: file })(loaded.exports, (id: string) => {
    assert(Object.hasOwn(dependencies, id), `Unexpected dependency ${id}`); return dependencies[id];
  }, loaded);
  return loaded.exports;
}

test("guest token is scoped, tamper resistant, expiring and not an account JWT", () => {
  const now = Date.now();
  const token = core.newGuestToken(secret, now);
  const owner = core.guestOwner(token, secret, now);
  assert.match(owner!, /^[a-f0-9]{64}$/);
  assert.equal(core.guestOwner(token, secret, now + 1000), owner);
  for (const bad of [undefined, "", token + "a", token.replace(/.$/, token.endsWith("a") ? "b" : "a"), "eyJ.eyJ.signature"]) {
    assert.equal(core.guestOwner(bad, secret, now), undefined);
  }
  assert.equal(core.guestOwner(token, "different", now), undefined);
  assert.equal(core.guestOwner(token, secret, now + core.GUEST_MAX_AGE * 1000), undefined);
  assert.equal(core.guestOwner(token, secret, now - 1000), undefined);
  assert.notEqual(core.guestOwner(core.newGuestToken(secret, now), secret, now), owner);
});

test("account, guest and legacy conversation ownership have distinct boundaries", () => {
  const guest = { userId: null, metadata: { researchGuestOwner: "owner" } };
  assert(core.canAccessChat(guest, undefined, "owner"));
  assert(!core.canAccessChat(guest, undefined, "other"));
  assert(!core.canAccessChat(guest, "signed-in-account"));
  assert(!core.canAccessChat({ userId: "account" }, undefined, "account"));
  assert(core.canAccessChat({ userId: "account" }, "account"));
  assert(core.canAccessChat({ userId: null, metadata: null }));
  assert(!core.canAccessChat({ userId: null, metadata: { researchGuestOwner: null } }));
});

test("guest cookie is HttpOnly, same-site, and creation is feature-gated", async () => {
  const prior = process.env.RESEARCH_R3_GUEST_ENABLED;
  let stored: string | undefined;
  let options: Record<string, unknown> = {};
  const api = load("src/lib/research-guest.ts", {
    "next/headers": { cookies: async () => ({ get: () => stored ? { value: stored } : undefined,
      set: (_name: string, value: string, config: Record<string, unknown>) => { stored = value; options = config; } }) },
    "@/config/env": { getEnv: () => ({ JWT_SECRET: secret }) }, "@/lib/research-guest-core": core,
  }) as { researchGuest: (req: Request, create?: boolean) => Promise<string | undefined>; researchGuestIp: (req: Request) => string };
  const req = new Request("https://example.test/api/chat", { headers: { host: "example.test", origin: "https://example.test" } });
  try {
    process.env.RESEARCH_R3_GUEST_ENABLED = "false";
    assert.equal(await api.researchGuest(req, true), undefined);
    process.env.RESEARCH_R3_GUEST_ENABLED = "true";
    assert.equal(await api.researchGuest(req), undefined);
    const owner = await api.researchGuest(req, true);
    assert(owner);
    assert.equal(options.httpOnly, true); assert.equal(options.sameSite, "lax"); assert.equal(options.path, "/");
    assert.equal(await api.researchGuest(req), owner);
    for (const headers of [{ origin: "https://evil.test" }, { "sec-fetch-site": "cross-site" }] as Record<string, string>[]) {
      assert.equal(await api.researchGuest(new Request(req.url, { headers }), true), undefined);
    }
    assert.equal(api.researchGuestIp(new Request(req.url, { headers: { "x-forwarded-for": "forged, 203.0.113.9" } })), "203.0.113.9");
  } finally {
    if (prior === undefined) delete process.env.RESEARCH_R3_GUEST_ENABLED;
    else process.env.RESEARCH_R3_GUEST_ENABLED = prior;
  }
});

test("guest read/list/delete routes deny cross-browser access and retain normal owner access", async () => {
  const id = "11111111-1111-4111-8111-111111111111";
  let owner: string | undefined = "guest-a";
  let user: { id: string } | null = null;
  let reads = 0, deletes = 0;
  const conversation = { id, userId: null, metadata: { researchGuestOwner: "guest-a" }, title: "Private fixture" };
  const api = load("src/app/api/chat/route.ts", {
    zod: { z }, "@/lib/auth": { getSessionUser: async () => user },
    "@/lib/api": { jsonOk: (x: unknown) => Response.json(x), jsonError: (error: string, status = 400) => Response.json({ error }, { status }) },
    "@/lib/rate-limit": {}, "@/lib/conversation-turn": {}, "@/lib/public-errors": {},
    "@/lib/model-modes": { MODEL_MODES: ["shadow", "r3_response"] },
    "@/lib/research-guest-core": core, "@/lib/research-guest": { researchGuest: async () => owner },
    "@/db/prisma": { prisma: { conversation: {
      findUnique: async () => conversation, findMany: async () => [conversation], delete: async () => { deletes++; },
    }, message: { findMany: async () => { reads++; return [{ content: "Private answer" }]; } } } },
  }) as { GET: (r: Request) => Promise<Response>; DELETE: (r: Request) => Promise<Response> };
  const req = new Request(`https://example.test/api/chat?conversationId=${id}`);
  assert.equal((await api.GET(req)).status, 200);
  for (const candidate of [undefined, "guest-b"]) {
    owner = candidate;
    assert.equal((await api.GET(req)).status, 404);
    assert.equal((await api.DELETE(req)).status, 404);
    assert.deepEqual((await (await api.GET(new Request(`https://example.test/api/chat?ids=${id}`))).json()).conversations, []);
  }
  user = { id: "different-account" };
  assert.equal((await api.GET(req)).status, 404);
  assert.equal(reads, 1); assert.equal(deletes, 0);
  user = null; owner = "guest-a";
  const listing = (await (await api.GET(new Request(`https://example.test/api/chat?ids=${id}`))).json()).conversations;
  assert.equal(listing.length, 1); assert.equal(listing[0].metadata, undefined);
  assert.equal((await api.DELETE(req)).status, 200); assert.equal(deletes, 1);
});
