import assert from "node:assert/strict";

process.env.OPENAI_API_KEY = "contract-test-key";
process.env.LLM_MODEL = "gpt-4o-mini";
delete process.env.OPENAI_LANGUAGE_MODEL;

const requests: Array<Record<string, unknown>> = [];
let responseQueue: Array<{ ok: boolean; status: number; body: unknown }> = [];

globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
  requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
  const next = responseQueue.shift();
  assert(next, "Unexpected LLM request");
  return new Response(JSON.stringify(next.body), {
    status: next.status,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

async function main() {
const { chatComplete, llmProviderInfo } = await import("../src/lib/llm");
const { understandUtterance } = await import("../src/lib/understand");

assert.equal(llmProviderInfo()?.model, "gpt-5.4-mini");

responseQueue = [
  { ok: false, status: 404, body: { error: "model unavailable" } },
  {
    ok: true,
    status: 200,
    body: { choices: [{ message: { content: "fallback worked" } }] },
  },
];
assert.equal(
  await chatComplete([{ role: "user", content: "hello" }], { maxTokens: 123 }),
  "fallback worked",
);
assert.deepEqual(
  requests.slice(0, 2).map((request) => request.model),
  ["gpt-5.4-mini", "gpt-4o-mini"],
);
assert.equal(requests[0].max_completion_tokens, 123);
assert.equal(requests[0].max_tokens, undefined);
assert.equal(requests[1].max_tokens, 123);

requests.length = 0;
responseQueue = [
  {
    ok: true,
    status: 200,
    body: {
      choices: [
        {
          message: {
            content: JSON.stringify({
              understood: true,
              understoodMeaning: null,
              uncertaintyReason: null,
              reply: "Ɛyɛ me yaw sɛ wo ti yɛ wo yaw. Bere bɛn na efii ase?",
              intent: "HEALTH",
              severity: "LOW",
              escalate: false,
            }),
          },
        },
      ],
    },
  },
];

const result = await understandUtterance({
  text: "Me ti y3 me yaw paa",
  language: "tw",
  history: [
    { role: "user", content: "Wote Twi?" },
    {
      role: "assistant",
      content:
        "Mante nea wokae no ase yie sɛ memmua. Mesrɛ wo, san ka no wɔ ɔkwan foforo so.",
    },
  ],
});

assert.equal(result.comprehension?.understood, true);
assert.equal(result.comprehension?.meaning, null);
assert.notEqual(
  result.reply,
  "Mante nea wokae no ase yie sɛ memmua. Mesrɛ wo, san ka no wɔ ɔkwan foforo so.",
);

const sentMessages = requests[0].messages as Array<{
  role: string;
  content: string;
}>;
assert(sentMessages.some((message) => message.content === "Wote Twi?"));
assert(
  !sentMessages.some((message) =>
    message.content.toLowerCase().includes("mante nea wokae no ase"),
  ),
);
assert.equal(sentMessages.at(-1)?.content, "Me ti y3 me yaw paa");
const systemInstruction = sentMessages.find(
  (message) => message.role === "system",
)?.content;
assert(systemInstruction?.includes("Start immediately with the useful answer"));
assert(systemInstruction?.includes("Never restate the user's request in first person"));
assert(systemInstruction?.includes('a short message such as "MacBook" adds detail'));

console.log("language response contract: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
