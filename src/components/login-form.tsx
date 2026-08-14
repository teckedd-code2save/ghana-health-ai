"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { KeyRound, Mail } from "lucide-react";

type LoginFormProps = {
  initialError?: string | null;
};

const oauthErrors: Record<string, string> = {
  google_not_configured: "Google sign-in is not configured yet.",
  google_state: "Google sign-in expired. Try again.",
  google_email: "Google did not return a verified email.",
};

export function LoginForm({ initialError }: LoginFormProps) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [codeSent, setCodeSent] = useState(false);
  const [error, setError] = useState<string | null>(
    initialError ? (oauthErrors[initialError] ?? "Google sign-in could not continue.") : null,
  );
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [socialBusy, setSocialBusy] = useState(false);

  async function requestCode(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await fetch("/api/auth/otp/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Couldn’t continue");
      setCodeSent(true);
      setNotice("We sent a 6-digit code to your email.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn’t send code");
    } finally {
      setBusy(false);
    }
  }

  async function verifyCode(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/otp/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Couldn’t verify code");
      router.push("/");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn’t continue");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.refresh();
  }

  return (
    <div className="auth-card fade-up">
      <div className="auth-social">
        <button
          type="button"
          className="auth-social__button"
          disabled={socialBusy}
          onClick={() => {
            setError(null);
            setSocialBusy(true);
            window.location.href = "/api/auth/google";
          }}
        >
          <span className="auth-social__google" aria-hidden>
            G
          </span>
          Continue with Google
        </button>
      </div>

      <div className="auth-divider">
        <span>Email code</span>
      </div>

      <form onSubmit={(e) => void (codeSent ? verifyCode(e) : requestCode(e))} className="space-y-3">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          className="field rounded-[var(--radius-sm)]"
          autoComplete="email"
          disabled={codeSent}
        />
        {codeSent && (
          <input
            inputMode="numeric"
            value={code}
            onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
            placeholder="6-digit code"
            required
            minLength={6}
            maxLength={6}
            className="field rounded-[var(--radius-sm)]"
            autoComplete="one-time-code"
          />
        )}
        {notice && <p className="text-sm text-[var(--fg-muted)]">{notice}</p>}
        {error && <p className="text-sm text-[var(--coral)]">{error}</p>}
        <button type="submit" disabled={busy} className="auth-submit">
          {codeSent ? <KeyRound className="h-4 w-4" /> : <Mail className="h-4 w-4" />}
          {busy ? "Please wait..." : codeSent ? "Verify code" : "Send email code"}
        </button>
      </form>
      {codeSent && (
        <button
          type="button"
          className="btn btn-ghost mt-3 w-full text-xs"
          onClick={() => {
            setCodeSent(false);
            setCode("");
            setNotice(null);
            setError(null);
          }}
        >
          Use another email
        </button>
      )}
      <button type="button" onClick={() => void logout()} className="btn btn-ghost mt-4 w-full text-xs">
        Sign out
      </button>
    </div>
  );
}
