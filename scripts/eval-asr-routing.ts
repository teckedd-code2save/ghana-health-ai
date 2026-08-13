import { DEFAULT_MODAL_ASR_EN_URL, resolveModalAsrRoute } from "../src/lib/modal-asr";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function withEnv<T>(env: Record<string, string | undefined>, fn: () => T): T {
  const prev = new Map<string, string | undefined>();
  for (const key of Object.keys(env)) {
    prev.set(key, process.env[key]);
    if (env[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = env[key];
    }
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of prev.entries()) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }
}

withEnv(
  {
    MODAL_ASR_URL: "https://twi.example.test/",
    MODAL_ASR_EN_URL: "https://en.example.test/",
  },
  () => {
    const tw = resolveModalAsrRoute("tw");
    const ga = resolveModalAsrRoute("ga");
    const en = resolveModalAsrRoute("en");

    assertOk(tw?.name === "twi-default", `expected twi-default, got ${tw?.name}`);
    assertOk(tw.url === "https://twi.example.test", `unexpected tw URL ${tw.url}`);
    assertOk(ga?.name === "twi-default", `expected Ga default route, got ${ga?.name}`);
    assertOk(en?.name === "english", `expected english route, got ${en?.name}`);
    assertOk(en.url === "https://en.example.test", `unexpected English URL ${en.url}`);
  },
);

withEnv(
  {
    MODAL_ASR_URL: "https://twi.example.test/",
    MODAL_ASR_EN_URL: undefined,
  },
  () => {
    const en = resolveModalAsrRoute("en");
    assertOk(en?.name === "english", `expected default english route, got ${en?.name}`);
    assertOk(en.routedLanguage === "en", `expected routed language en, got ${en.routedLanguage}`);
    assertOk(en.url === DEFAULT_MODAL_ASR_EN_URL, `unexpected default English URL ${en.url}`);
  },
);

withEnv(
  {
    MODAL_ASR_URL: undefined,
    MODAL_ASR_EN_URL: "https://en.example.test/",
  },
  () => {
    const tw = resolveModalAsrRoute("tw");
    const en = resolveModalAsrRoute("en");
    assertOk(tw === null, "Twi route should be unconfigured without MODAL_ASR_URL");
    assertOk(en?.name === "english", `expected English route, got ${en?.name}`);
    assertOk(en.url === "https://en.example.test", `unexpected English URL ${en.url}`);
  },
);

console.log("ok asr-routing");

export {};
