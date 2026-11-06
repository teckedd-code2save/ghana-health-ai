import { cookies } from "next/headers";
import { getEnv } from "@/config/env";
import { guestOwner, GUEST_MAX_AGE, newGuestToken } from "@/lib/research-guest-core";

const COOKIE = "gha_research_guest";

export async function researchGuest(req: Request, create = false): Promise<string | undefined> {
  const origin = req.headers.get("origin");
  const host = req.headers.get("host") || new URL(req.url).host;
  if (req.headers.get("sec-fetch-site") === "cross-site") return;
  if (origin) {
    try { if (new URL(origin).host !== host) return; } catch { return; }
  }
  const jar = await cookies();
  const secret = getEnv().JWT_SECRET;
  const existing = guestOwner(jar.get(COOKIE)?.value, secret);
  if (existing) return existing;
  if (!create || process.env.RESEARCH_R3_GUEST_ENABLED !== "true") return;
  const token = newGuestToken(secret);
  jar.set(COOKIE, token, { httpOnly: true, secure: process.env.NODE_ENV === "production", sameSite: "lax", path: "/", maxAge: GUEST_MAX_AGE });
  return guestOwner(token, secret);
}

export function researchGuestIp(req: Request) {
  // Caddy is the only public ingress; use its appended peer, not a caller's first entry.
  return req.headers.get("x-forwarded-for")?.split(",").at(-1)?.trim().slice(0, 128) || "unknown";
}
