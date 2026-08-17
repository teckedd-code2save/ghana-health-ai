-- Add consented-audio capture to AsrFeedback: audio is only stored when the
-- user explicitly opts in at correction time (audioConsent = true).
ALTER TABLE "asr_feedback" ADD COLUMN "audio_consent" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "asr_feedback" ADD COLUMN "audio_path" TEXT;
