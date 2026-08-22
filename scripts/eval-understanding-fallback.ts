import { understandUtterance } from "../src/lib/understand";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  const previousGroq = process.env.GROQ_API_KEY;
  const previousOpenai = process.env.OPENAI_API_KEY;
  delete process.env.GROQ_API_KEY;
  delete process.env.OPENAI_API_KEY;

  try {
    const health = await understandUtterance({
      text: "me ba no ho yɛ hyew na ɔyɛ mmerɛ",
      language: "tw",
      focus: "health",
      transcript: {
        mode: "asr",
        duration: 3.2,
        rms: 0.05,
        languageProbability: 0.82,
      },
    });
    assertOk(health.engine === "fallback", `expected fallback engine, got ${health.engine}`);
    assertOk(health.intent === "HEALTH", `expected HEALTH, got ${health.intent}`);
    assertOk(health.health, "expected health plan");
    assertOk(health.reply.trim().length > 12, "expected health fallback reply");
    assertOk(health.comprehension?.understood === false, "fallback must be marked not understood");
    assertOk(!/rest|drink fluids|gye w'ahome|nom nsuo/i.test(health.reply), "fallback must not simulate health advice");
    console.log(`ok fallback-health intent=${health.intent} plan=${health.health.plan.status}`);

    const weak = await understandUtterance({
      text: "ɛɛ me hmm",
      language: "tw",
      focus: "health",
      transcript: {
        mode: "asr",
        duration: 1.1,
        rms: 0.012,
        languageProbability: 0.36,
      },
    });
    assertOk(weak.engine === "fallback", `expected fallback engine, got ${weak.engine}`);
    assertOk(weak.health?.plan.status === "needs_clarification", "expected weak ASR clarification");
    assertOk(weak.reply.includes("?") || weak.reply.toLowerCase().includes("san ka"), "expected clarifying reply");
    console.log(`ok fallback-weak plan=${weak.health?.plan.status}`);

    const commerce = await understandUtterance({
      text: "mepɛ sɛ metɔ tomatoes kilo 2 wɔ Madina",
      language: "tw",
      focus: "commerce",
    });
    assertOk(commerce.engine === "fallback", `expected fallback engine, got ${commerce.engine}`);
    assertOk(commerce.intent === "ECOMMERCE", `expected ECOMMERCE, got ${commerce.intent}`);
    const commerceUnderstanding = commerce.commerce;
    assertOk(commerceUnderstanding, "expected commerce understanding");
    assertOk(commerceUnderstanding.item?.toLowerCase().includes("tomatoes"), "expected commerce item");
    assertOk(commerceUnderstanding.plan, "expected commerce plan");
    assertOk(commerce.comprehension?.understood === false, "fallback commerce must be marked not understood");
    console.log(
      `ok fallback-commerce action=${commerceUnderstanding.action} next=${commerceUnderstanding.plan.nextAction}`,
    );
  } finally {
    if (previousGroq) process.env.GROQ_API_KEY = previousGroq;
    if (previousOpenai) process.env.OPENAI_API_KEY = previousOpenai;
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
