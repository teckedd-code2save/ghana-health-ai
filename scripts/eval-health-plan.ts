import { buildHealthUnderstanding } from "../src/lib/health-plan";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const hotline = "112";

const emergency = buildHealthUnderstanding({
  text: "My father has crushing chest pain",
  severity: "EMERGENCY",
  escalate: true,
  hotline,
});
assertOk(emergency.plan.status === "urgent_referral", "emergency status mismatch");
assertOk(emergency.plan.nextAction === "recommend_urgent_care", "emergency next action mismatch");
assertOk(emergency.plan.hotline === hotline, "emergency hotline missing");
console.log(`ok emergency status=${emergency.plan.status}`);

const weak = buildHealthUnderstanding({
  text: "ɛɛ me hmm",
  severity: "LOW",
  escalate: false,
  transcript: { duration: 1.1, rms: 0.012, languageProbability: 0.36 },
  hotline,
});
assertOk(weak.plan.status === "needs_clarification", "weak transcript status mismatch");
assertOk(weak.plan.nextAction === "ask_clarifying_question", "weak transcript action mismatch");
console.log(`ok weak-transcript status=${weak.plan.status}`);

const routine = buildHealthUnderstanding({
  text: "I have mild headache",
  severity: "LOW",
  escalate: false,
  transcript: { duration: 3.2, rms: 0.05, languageProbability: 0.9 },
  hotline,
});
assertOk(routine.plan.status === "self_care", "routine status mismatch");
assertOk(routine.plan.nextAction === "give_general_guidance", "routine action mismatch");
console.log(`ok routine status=${routine.plan.status}`);

export {};
