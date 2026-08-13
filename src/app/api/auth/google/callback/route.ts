import { cookies } from "next/headers";
import { prisma } from "@/db/prisma";
import { getEnv } from "@/config/env";
import { createSessionToken, setSessionCookie } from "@/lib/auth";

const STATE_COOKIE = "gha_google_oauth_state";

type GoogleTokenResponse = {
  access_token?: string;
  id_token?: string;
  error?: string;
  error_description?: string;
};

type GoogleProfile = {
  email?: string;
  email_verified?: boolean;
  name?: string;
};

function redirectTo(req: Request, path: string) {
  return Response.redirect(new URL(path, req.url).toString());
}

function callbackUrl(req: Request) {
  return new URL("/api/auth/google/callback", req.url).toString();
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const jar = await cookies();
  const expectedState = jar.get(STATE_COOKIE)?.value;
  jar.delete(STATE_COOKIE);

  if (!code || !state || !expectedState || state !== expectedState) {
    return redirectTo(req, "/login?error=google_state");
  }

  const env = getEnv();
  if (!env.GOOGLE_CLIENT_ID || !env.GOOGLE_CLIENT_SECRET) {
    return redirectTo(req, "/login?error=google_not_configured");
  }

  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: env.GOOGLE_CLIENT_ID,
      client_secret: env.GOOGLE_CLIENT_SECRET,
      redirect_uri: callbackUrl(req),
      grant_type: "authorization_code",
    }),
  });
  const token = (await tokenRes.json().catch(() => ({}))) as GoogleTokenResponse;
  if (!tokenRes.ok || !token.access_token) {
    return redirectTo(req, `/login?error=${encodeURIComponent(token.error_description || token.error || "google_token")}`);
  }

  const profileRes = await fetch("https://openidconnect.googleapis.com/v1/userinfo", {
    headers: { Authorization: `Bearer ${token.access_token}` },
  });
  const profile = (await profileRes.json().catch(() => ({}))) as GoogleProfile;
  if (!profileRes.ok || !profile.email || profile.email_verified === false) {
    return redirectTo(req, "/login?error=google_email");
  }

  const email = profile.email.toLowerCase();
  const user = await prisma.user.upsert({
    where: { email },
    update: {
      displayName: profile.name || undefined,
    },
    create: {
      email,
      displayName: profile.name,
      preferredLang: "tw",
      consentHealth: false,
      consentVoice: false,
    },
  });

  const session = await createSessionToken(user.id);
  await setSessionCookie(session);
  return redirectTo(req, "/");
}
