import crypto from "node:crypto";
import { cookies } from "next/headers";
import { getEnv } from "@/config/env";

const STATE_COOKIE = "gha_google_oauth_state";

function callbackUrl(req: Request) {
  return new URL("/api/auth/google/callback", req.url).toString();
}

function redirectTo(req: Request, path: string) {
  return Response.redirect(new URL(path, req.url).toString());
}

export async function GET(req: Request) {
  const env = getEnv();
  if (!env.GOOGLE_CLIENT_ID || !env.GOOGLE_CLIENT_SECRET) {
    return redirectTo(req, "/login?error=google_not_configured");
  }

  const state = crypto.randomBytes(24).toString("hex");
  const jar = await cookies();
  jar.set(STATE_COOKIE, state, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 10 * 60,
  });

  const url = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  url.searchParams.set("client_id", env.GOOGLE_CLIENT_ID);
  url.searchParams.set("redirect_uri", callbackUrl(req));
  url.searchParams.set("response_type", "code");
  url.searchParams.set("scope", "openid email profile");
  url.searchParams.set("state", state);
  url.searchParams.set("prompt", "select_account");

  return Response.redirect(url.toString());
}
