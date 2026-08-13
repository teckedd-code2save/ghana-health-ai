export type HealthActionPlan = {
  status: "needs_clarification" | "self_care" | "clinic_recommended" | "urgent_referral";
  nextAction:
    | "ask_clarifying_question"
    | "give_general_guidance"
    | "recommend_clinic"
    | "recommend_urgent_care";
  urgency: "routine" | "soon" | "urgent" | "emergency";
  reason: string;
  hotline?: string;
};

export type HealthUnderstanding = {
  plan: HealthActionPlan;
  source: "deterministic";
};

export function buildHealthUnderstanding(input: {
  text: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  escalate: boolean;
  transcript?: {
    duration?: number;
    rms?: number;
    languageProbability?: number;
  };
  hotline: string;
}): HealthUnderstanding {
  if (isWeakTranscript(input.transcript, input.text)) {
    return {
      source: "deterministic",
      plan: {
        status: "needs_clarification",
        nextAction: "ask_clarifying_question",
        urgency: "routine",
        reason: "Transcript quality is weak or the utterance is too unclear for safe health guidance.",
      },
    };
  }

  if (input.escalate || input.severity === "EMERGENCY") {
    return {
      source: "deterministic",
      plan: {
        status: "urgent_referral",
        nextAction: "recommend_urgent_care",
        urgency: "emergency",
        reason: "Emergency or danger-sign pattern detected.",
        hotline: input.hotline,
      },
    };
  }

  if (input.severity === "HIGH") {
    return {
      source: "deterministic",
      plan: {
        status: "clinic_recommended",
        nextAction: "recommend_clinic",
        urgency: "urgent",
        reason: "High-severity symptoms should be assessed by a clinician.",
      },
    };
  }

  if (input.severity === "MEDIUM") {
    return {
      source: "deterministic",
      plan: {
        status: "clinic_recommended",
        nextAction: "recommend_clinic",
        urgency: "soon",
        reason: "Symptoms may need clinic follow-up if persistent or worsening.",
      },
    };
  }

  return {
    source: "deterministic",
    plan: {
      status: "self_care",
      nextAction: "give_general_guidance",
      urgency: "routine",
      reason: "Low-severity health request with enough transcript signal for general guidance.",
    },
  };
}

function isWeakTranscript(
  transcript: { duration?: number; rms?: number; languageProbability?: number } | undefined,
  text: string,
) {
  const lower = text.toLowerCase();
  const veryShort = typeof transcript?.duration === "number" && transcript.duration < 1.4;
  const quiet = typeof transcript?.rms === "number" && transcript.rms > 0 && transcript.rms < 0.015;
  const lowLanguageConfidence =
    typeof transcript?.languageProbability === "number" && transcript.languageProbability < 0.55;
  const fragment = /\b(ɛɛ|hmm|mm)\b/.test(lower) && lower.split(/\s+/).length <= 6;
  return veryShort || quiet || lowLanguageConfidence || fragment;
}
