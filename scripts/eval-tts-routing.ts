import { resolveTtsRoute, speakableText } from "../src/lib/modal-tts";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function withEnv<T>(patch: Record<string, string | undefined>, fn: () => T): T {
  const previous = new Map<string, string | undefined>();
  for (const key of Object.keys(patch)) {
    previous.set(key, process.env[key]);
    const value = patch[key];
    if (typeof value === "undefined") delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of previous) {
      if (typeof value === "undefined") delete process.env[key];
      else process.env[key] = value;
    }
  }
}

withEnv(
  {
    TTS_PROVIDER: undefined,
    TTS_TWI_PROVIDER: undefined,
    MODAL_TTS_URL: undefined,
    STABLE_TWI_TTS_URL: undefined,
    NANO_TWI_TTS_URL: undefined,
    QWEN_TTS_URL: undefined,
  },
  () => {
    const tw = resolveTtsRoute("tw");
    assertOk(tw?.provider === "mms", `expected default Twi provider mms, got ${tw?.provider}`);
    assertOk(
      tw.modelLabel === "facebook/mms-tts-aka",
      `expected default Twi model label, got ${tw.modelLabel}`,
    );

    const en = resolveTtsRoute("en");
    assertOk(en?.provider === "mms", `expected English provider mms, got ${en?.provider}`);
    assertOk(
      en.modelLabel === "facebook/mms-tts-eng",
      `expected default English model label, got ${en.modelLabel}`,
    );
  },
);

withEnv(
  {
    TTS_TWI_PROVIDER: "stable-twi",
    STABLE_TWI_TTS_URL: "https://example.invalid/stable/speak",
  },
  () => {
    const route = resolveTtsRoute("tw");
    assertOk(route?.provider === "stable-twi", `expected stable-twi, got ${route?.provider}`);
    assertOk(
      route.modelLabel === "ghananlpcommunity/stable-twi-tts",
      `unexpected stable model label ${route.modelLabel}`,
    );
    assertOk(route.url === "https://example.invalid/stable/speak", `unexpected URL ${route.url}`);
  },
);

withEnv(
  {
    TTS_TWI_PROVIDER: "nano-twi",
    NANO_TWI_TTS_URL: "https://example.invalid/nano",
  },
  () => {
    const route = resolveTtsRoute("tw");
    assertOk(route?.provider === "nano-twi", `expected nano-twi, got ${route?.provider}`);
    assertOk(route.url === "https://example.invalid/nano/speak", `unexpected URL ${route.url}`);
  },
);

withEnv(
  {
    TTS_TWI_PROVIDER: "qwen",
    QWEN_TTS_URL: "https://example.invalid/qwen/speak",
  },
  () => {
    const route = resolveTtsRoute("tw");
    assertOk(route?.provider === "qwen", `expected qwen, got ${route?.provider}`);
    assertOk(route.modelLabel === "qwen3-tts-12hz", `unexpected qwen label ${route.modelLabel}`);
  },
);

assertOk(
  !speakableText("Call CHW about MoMo 25kg", "tw").includes("CHW"),
  "speakable text should expand health acronyms",
);

console.log("ok tts-routing providers=mms,stable-twi,nano-twi,qwen");
