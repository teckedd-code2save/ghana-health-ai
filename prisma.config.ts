import path from "node:path";
import { defineConfig } from "prisma/config";
import "./src/config/load-env";

// DATABASE_URL is injected at runtime via Infisical (`sec --`) or local .env.local.
export default defineConfig({
  schema: path.join("prisma", "schema.prisma"),
  datasource: process.env.DATABASE_URL
    ? { url: process.env.DATABASE_URL }
    : undefined,
  migrations: {
    seed: "tsx prisma/seed.ts",
  },
});
