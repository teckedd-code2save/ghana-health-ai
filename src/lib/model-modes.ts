export const MODEL_MODES = ["shadow", "assist", "assist_v1", "r1_base", "r1_meaning", "r1_response", "r3_response"] as const;
export type UnderstandingModelMode = typeof MODEL_MODES[number];
export type ResearchR1Mode = "r1_base" | "r1_meaning" | "r1_response";

export function isModelMode(value: unknown): value is UnderstandingModelMode {
  return typeof value === "string" && MODEL_MODES.some((mode) => mode === value);
}

export function isResearchR1Mode(value: unknown): value is ResearchR1Mode {
  return value === "r1_base" || value === "r1_meaning" || value === "r1_response";
}

export function isResearchModelMode(value: unknown): value is ResearchR1Mode | "r3_response" {
  return isResearchR1Mode(value) || value === "r3_response";
}

export type ResearchR1Metadata = {
  variant: string;
  model: string;
  revision: string;
  executionSha: string;
  adapterSha: string | null;
  finishReason: string;
  complete: boolean;
};

export class ResearchR1Error extends Error {
  constructor(message: string, public status = 503) {
    super(message);
    this.name = "ResearchR1Error";
  }
}

export type ResearchR3Metadata = ResearchR1Metadata & { variant: "r3_response"; bundleSha: string };
export type ResearchMetadata = ResearchR1Metadata | ResearchR3Metadata;

export class ResearchR3Error extends Error {
  constructor(message: string, public status = 503) {
    super(message);
    this.name = "ResearchR3Error";
  }
}
