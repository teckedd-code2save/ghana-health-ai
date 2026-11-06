import { ResearchR1Error, ResearchR3Error } from "@/lib/model-modes";

export type FailurePhase = "chat" | "voice" | "transcription";

const STARTING_PATTERN =
  /ECONNREFUSED|Can't reach database|database is not reachable|PrismaClientInitialization/i;
const TIMEOUT_PATTERN = /timeout|timed out|ETIMEDOUT|AbortError/i;
const STORAGE_PATTERN = /No space left on device|database system is in recovery|could not extend file|53100|57P03/i;

export function publicFailure(error: unknown, phase: FailurePhase) {
  if (error instanceof ResearchR1Error || error instanceof ResearchR3Error) return { message: error.message, status: error.status };
  const detail = error instanceof Error ? error.message : String(error ?? "");
  const cause = error instanceof Error && error.cause && typeof error.cause === "object"
    ? error.cause as { originalCode?: unknown; code?: unknown; message?: unknown } : undefined;
  if (STORAGE_PATTERN.test([detail, cause?.originalCode, cause?.code, cause?.message].join(" "))) {
    return { message: "The service cannot save this conversation right now. Your microphone is not the problem. Please try again shortly.", status: 503 };
  }

  if (STARTING_PATTERN.test(detail)) {
    return {
      message: "The service is still getting ready. Please try again in a moment.",
      status: 503,
    };
  }

  if (TIMEOUT_PATTERN.test(detail)) {
    return {
      message: "That took longer than expected. Please try again.",
      status: 504,
    };
  }

  if (phase === "transcription") {
    return {
      message: "I couldn’t process that recording. Please try again a little closer to the microphone.",
      status: 500,
    };
  }

  if (phase === "voice") return { message: "The voice conversation could not finish. Please try again shortly.", status: 503 };

  return {
    message: "I couldn’t complete that response. Please try again.",
    status: 500,
  };
}

export const unclearRecording = {
  message: "I couldn’t hear that clearly. Please try again a little closer to the microphone.",
  status: 422,
};
