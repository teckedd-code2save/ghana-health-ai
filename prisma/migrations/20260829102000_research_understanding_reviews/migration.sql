CREATE TABLE "research_understanding_reviews" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "row_id" TEXT NOT NULL,
    "row_kind" TEXT,
    "normalized_twi" TEXT NOT NULL DEFAULT '',
    "natural_english" TEXT NOT NULL DEFAULT '',
    "literal_english" TEXT NOT NULL DEFAULT '',
    "intent" TEXT NOT NULL DEFAULT '',
    "entities" TEXT NOT NULL DEFAULT '',
    "ambiguities" TEXT NOT NULL DEFAULT '',
    "decision" TEXT NOT NULL DEFAULT 'unreviewed',
    "notes" TEXT NOT NULL DEFAULT '',
    "reviewer" TEXT NOT NULL DEFAULT 'local_reviewer',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "research_understanding_reviews_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "research_understanding_reviews_row_id_key" ON "research_understanding_reviews"("row_id");
CREATE INDEX "research_understanding_reviews_decision_idx" ON "research_understanding_reviews"("decision");
CREATE INDEX "research_understanding_reviews_updated_at_idx" ON "research_understanding_reviews"("updated_at");
