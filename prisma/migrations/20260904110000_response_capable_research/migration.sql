ALTER TABLE "research_understanding_reviews"
  ADD COLUMN "reply_twi" TEXT NOT NULL DEFAULT '',
  ADD COLUMN "safety_level" TEXT NOT NULL DEFAULT '';

CREATE TABLE "research_corpus_recordings" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "row_id" TEXT NOT NULL,
    "speaker_id" TEXT NOT NULL,
    "language" "LanguageCode" NOT NULL DEFAULT 'tw',
    "transcript" TEXT NOT NULL,
    "mime_type" TEXT NOT NULL,
    "audio_bytes" BYTEA NOT NULL,
    "size_bytes" INTEGER NOT NULL,
    "duration_ms" INTEGER,
    "reviewer" TEXT NOT NULL DEFAULT 'local_reviewer',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "research_corpus_recordings_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "research_corpus_recordings_row_id_created_at_idx"
  ON "research_corpus_recordings"("row_id", "created_at");
CREATE INDEX "research_corpus_recordings_speaker_id_created_at_idx"
  ON "research_corpus_recordings"("speaker_id", "created_at");
