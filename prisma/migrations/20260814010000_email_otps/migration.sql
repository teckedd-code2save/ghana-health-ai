CREATE TABLE "email_otps" (
    "id" UUID NOT NULL,
    "email" TEXT NOT NULL,
    "code_hash" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "used_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" UUID,

    CONSTRAINT "email_otps_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "email_otps_email_created_at_idx" ON "email_otps"("email", "created_at");
CREATE INDEX "email_otps_expires_at_idx" ON "email_otps"("expires_at");

ALTER TABLE "email_otps" ADD CONSTRAINT "email_otps_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
