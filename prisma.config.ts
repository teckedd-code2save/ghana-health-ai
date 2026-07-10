import path from "node:path";
import fs from "node:fs";
import { defineConfig } from "prisma/config";

// Inline env load so this file works in Docker deps stage (no src/ yet)
// and under Infisical `sec --` (env already injected).
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const dotenv = require("dotenv") as typeof import("dotenv");
  const root = process.cwd();
  for (const file of [".env.local", ".env"]) {
    const p = path.join(root, file);
    if (fs.existsSync(p)) dotenv.config({ path: p, override: false });
  }
} catch {
  // dotenv may be absent in minimal images; process.env is fine
}

export default defineConfig({
  schema: path.join("prisma", "schema.prisma"),
  datasource: process.env.DATABASE_URL
    ? { url: process.env.DATABASE_URL }
    : undefined,
  migrations: {
    seed: "tsx prisma/seed.ts",
  },
});
