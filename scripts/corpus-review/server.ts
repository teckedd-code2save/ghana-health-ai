import http from "node:http";
import { createHash, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { GET, POST } from "../../src/app/api/research/corpus-release/route";

const release = process.env.CORPUS_RELEASE!;
const origin = process.env.REVIEW_ORIGIN!;
const password = process.env.REVIEW_PASSWORD!;
if (!/^[a-zA-Z0-9_-]{1,100}$/.test(release ?? "") ||
    !/^https:\/\/[a-z0-9.-]+$/.test(origin ?? "") || (password?.length ?? 0) < 24) {
  throw new Error("Explicit release, HTTPS origin and strong review password required");
}
// Only this isolated server can bridge authenticated HTTPS requests to the
// existing loopback-only handlers. It never uses application database secrets.
delete process.env.DATABASE_URL;
process.env.CORPUS_STANDALONE_REVIEW = "1";
const credentials = createHash("sha256").update(`reviewer:${password}`).digest();
const failures = new Map<string, { count: number; until: number }>();
const root = process.cwd();
const headers = {
  "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
  "X-Robots-Tag": "noindex, nofollow, noarchive", "Referrer-Policy": "no-referrer",
  "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
};

async function handle(req: http.IncomingMessage, res: http.ServerResponse) {
  for (const [name, value] of Object.entries(headers)) res.setHeader(name, value);
  const url = new URL(req.url ?? "/", origin);
  if (url.pathname === "/healthz" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "text/plain" }); res.end("ok"); return;
  }
  if (!url.pathname.startsWith("/research/corpus")) { res.writeHead(404); res.end(); return; }
  const ip = String(req.headers["x-forwarded-for"] ?? req.socket.remoteAddress).slice(0, 100);
  const now = Date.now();
  for (const [key, value] of failures) if (value.until < now) failures.delete(key);
  if ((failures.get(ip)?.count ?? 0) >= 8) { res.writeHead(429, { "Retry-After": "60" }); res.end(); return; }
  const authorization = req.headers.authorization ?? "";
  const supplied = authorization.startsWith("Basic ") ? Buffer.from(authorization.slice(6), "base64") : Buffer.alloc(0);
  if (!timingSafeEqual(credentials, createHash("sha256").update(supplied).digest())) {
    if (failures.size >= 10000 && !failures.has(ip)) { res.writeHead(429); res.end(); return; }
    failures.set(ip, { count: (failures.get(ip)?.count ?? 0) + 1, until: now + 60000 });
    res.writeHead(401, { "WWW-Authenticate": 'Basic realm="Private corpus review", charset="UTF-8"' }); res.end(); return;
  }
  failures.delete(ip);
  if (url.pathname === "/research/corpus/api") {
    if (!["GET", "POST"].includes(req.method ?? "")) { res.writeHead(405); res.end(); return; }
    if (req.method === "POST" && req.headers.origin !== origin) { res.writeHead(403); res.end(); return; }
    let body = "";
    for await (const chunk of req) {
      body += chunk.toString();
      if (Buffer.byteLength(body) > 100000) { res.writeHead(413); res.end(); return; }
    }
    const input = req.method === "POST" ? JSON.parse(body) : Object.fromEntries(url.searchParams);
    if (input.release !== release) { res.writeHead(404); res.end(); return; }
    const target = new Request("http://localhost/api/research/corpus-release" + url.search,
      { method: req.method, headers: { origin: "http://localhost", "Content-Type": "application/json" },
        ...(req.method === "POST" ? { body } : {}) });
    const response = await (req.method === "POST" ? POST(target) : GET(target));
    res.writeHead(response.status, { "Content-Type": "application/json" });
    res.end(await response.text()); return;
  }
  if (req.method !== "GET") { res.writeHead(405); res.end(); return; }
  const assets: Record<string, [string, string]> = {
    "/research/corpus": ["index.html", "text/html; charset=utf-8"],
    "/research/corpus/": ["index.html", "text/html; charset=utf-8"],
    "/research/corpus/app.js": ["app.js", "text/javascript; charset=utf-8"],
    "/research/corpus/app.css": ["app.css", "text/css; charset=utf-8"],
  };
  const versioned = /^\/research\/corpus\/(app-[a-f0-9]{12}\.(js|css))$/.exec(url.pathname);
  const asset = assets[url.pathname] ?? (versioned ? [versioned[1], versioned[2] === "js" ? "text/javascript; charset=utf-8" : "text/css; charset=utf-8"] : null);
  if (!asset) { res.writeHead(404); res.end(); return; }
  const bytes = await readFile(path.join(root, "public", asset[0]));
  res.writeHead(200, { "Content-Type": asset[1] }); res.end(bytes);
}

const server = http.createServer((req, res) => {
  handle(req, res).catch(() => { if (!res.headersSent) res.writeHead(400); res.end("Request could not be completed."); });
});
server.requestTimeout = 30000;
server.headersTimeout = 10000;
server.listen(Number(process.env.PORT ?? "13121"), "127.0.0.1", () => console.log("Private review service ready"));
