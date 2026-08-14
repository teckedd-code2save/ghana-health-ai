import { getEnv } from "@/config/env";

export async function sendLoginCode(email: string, code: string) {
  const env = getEnv();
  if (!env.RESEND_API_KEY || !env.EMAIL_FROM) {
    throw new Error("Email login is not configured yet.");
  }

  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: env.EMAIL_FROM,
      to: email,
      subject: "Your Ghana Health sign-in code",
      text: `Your Ghana Health sign-in code is ${code}. It expires in 10 minutes.`,
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(body || "Could not send sign-in code.");
  }
}
