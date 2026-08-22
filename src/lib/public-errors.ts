export type FailurePhase = "chat" | "voice" | "transcription";

const STARTING_PATTERN =
  /ECONNREFUSED|Can't reach database|database is not reachable|PrismaClientInitialization/i;
const TIMEOUT_PATTERN = /timeout|timed out|ETIMEDOUT|AbortError/i;

export function publicFailure(error: unknown, phase: FailurePhase) {
  const detail = error instanceof Error ? error.message : String(error ?? "");

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

  if (phase === "transcription" || phase === "voice") {
    return {
      message: "I couldn’t process that recording. Please try again a little closer to the microphone.",
      status: 500,
    };
  }

  return {
    message: "I couldn’t complete that response. Please try again.",
    status: 500,
  };
}

export const unclearRecording = {
  message: "I couldn’t hear that clearly. Please try again a little closer to the microphone.",
  status: 422,
};
