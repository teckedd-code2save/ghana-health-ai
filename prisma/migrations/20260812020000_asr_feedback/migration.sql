CREATE TABLE "asr_feedback" (
    "id" UUID NOT NULL,
    "user_id" UUID,
    "conversation_id" UUID NOT NULL,
    "message_id" UUID,
    "language" "LanguageCode" NOT NULL DEFAULT 'tw',
    "focus" TEXT,
    "original_transcript" TEXT NOT NULL,
    "corrected_transcript" TEXT,
    "rating" INTEGER,
    "notes" TEXT,
    "asr_model" TEXT,
    "asr_route" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "asr_feedback_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "asr_feedback_user_id_idx" ON "asr_feedback"("user_id");
CREATE INDEX "asr_feedback_conversation_id_idx" ON "asr_feedback"("conversation_id");
CREATE INDEX "asr_feedback_message_id_idx" ON "asr_feedback"("message_id");
CREATE INDEX "asr_feedback_language_created_at_idx" ON "asr_feedback"("language", "created_at");

ALTER TABLE "asr_feedback"
ADD CONSTRAINT "asr_feedback_user_id_fkey"
FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

ALTER TABLE "asr_feedback"
ADD CONSTRAINT "asr_feedback_conversation_id_fkey"
FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "asr_feedback"
ADD CONSTRAINT "asr_feedback_message_id_fkey"
FOREIGN KEY ("message_id") REFERENCES "messages"("id") ON DELETE SET NULL ON UPDATE CASCADE;
