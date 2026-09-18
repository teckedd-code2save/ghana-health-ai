import { createRequire } from "node:module";
import { mkdir, copyFile, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
const require = createRequire(import.meta.url);
const { build } = createRequire(require.resolve("tsx"))("esbuild");
const [out, release] = process.argv.slice(2);
if (!out || !/^[a-zA-Z0-9_-]{1,100}$/.test(release ?? "")) throw new Error("Output and release required");
await mkdir(path.join(out, "public"), { recursive: true });
await build({ entryPoints: ["scripts/corpus-review/main.tsx"], bundle: true, minify: true,
  outfile: path.join(out, "public/app.js"), jsx: "automatic", platform: "browser",
  define: { CORPUS_RELEASE: JSON.stringify(release), "process.env.NODE_ENV": '"production"' } });
await build({ entryPoints: ["scripts/corpus-review/server.ts"], bundle: true,
  outfile: path.join(out, "server.cjs"), platform: "node", target: "node20", external: ["@/db/prisma"] });
const directory = path.join(out, "public");
const fingerprint = createHash("sha256").update(await readFile(path.join(directory, "app.js")))
  .update(await readFile(path.join(directory, "app.css"))).digest("hex").slice(0, 12);
for (const type of ["js", "css"]) await copyFile(path.join(directory, "app." + type), path.join(directory, `app-${fingerprint}.${type}`));
const html = (await readFile("scripts/corpus-review/index.html", "utf8"))
  .replace("/app.js", `/app-${fingerprint}.js`).replace("/app.css", `/app-${fingerprint}.css`);
await writeFile(path.join(directory, "index.html"), html);
