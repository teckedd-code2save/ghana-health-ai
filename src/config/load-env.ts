import { config } from "dotenv";
import path from "node:path";
import fs from "node:fs";

/**
 * Load env from the monorepo/workspace root regardless of cwd.
 * Prefer Infisical via `sec --` in real usage; this is a local fallback
 * for prisma generate / migrate when a .env.local is present (gitignored).
 */
const root = path.resolve(__dirname, "../..");
const candidates = [
  path.join(root, ".env.local"),
  path.join(root, ".env"),
  path.join(process.cwd(), ".env.local"),
  path.join(process.cwd(), ".env"),
];

for (const file of candidates) {
  if (fs.existsSync(file)) {
    config({ path: file, override: false });
  }
}
