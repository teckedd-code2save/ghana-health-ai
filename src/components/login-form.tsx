"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Mail } from "lucide-react";

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
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [consentHealth, setConsentHealth] = useState(true);
  const [consentVoice, setConsentVoice] = useState(true);
  const [error, setError] = useState<string | null>(
    initialError ? (oauthErrors[initialError] ?? "Google sign-in could not continue.") : null,
  );
  const [busy, setBusy] = useState(false);
  const [socialBusy, setSocialBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const path = mode === "login" ? "/api/auth/login" : "/api/auth/register";
      const body =
        mode === "login"
          ? { email, password }
          : { email, password, displayName, consentHealth, consentVoice, preferredLang: "tw" };
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Couldn’t continue");
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
        <span>Email</span>
      </div>

      <div className="auth-mode-toggle">
        {(["login", "register"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={mode === m ? "auth-mode-toggle__item auth-mode-toggle__item--active" : "auth-mode-toggle__item"}
          >
            {m === "login" ? "Sign in" : "Register"}
          </button>
        ))}
      </div>
      <form onSubmit={(e) => void submit(e)} className="space-y-3">
        {mode === "register" && (
          <input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Your name"
            className="field rounded-[var(--radius-sm)]"
          />
        )}
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          className="field rounded-[var(--radius-sm)]"
          autoComplete="email"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
          minLength={8}
          className="field rounded-[var(--radius-sm)]"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
        />
        {mode === "register" && (
          <div className="space-y-2 pt-1 text-sm text-[var(--fg-muted)]">
            <label className="flex items-start gap-2.5">
              <input
                type="checkbox"
                className="mt-1"
                checked={consentHealth}
                onChange={(e) => setConsentHealth(e.target.checked)}
              />
              I agree to store health conversation data carefully
            </label>
            <label className="flex items-start gap-2.5">
              <input
                type="checkbox"
                className="mt-1"
                checked={consentVoice}
                onChange={(e) => setConsentVoice(e.target.checked)}
              />
              I agree to save my voice for recognition
            </label>
          </div>
        )}
        {error && <p className="text-sm text-[var(--coral)]">{error}</p>}
        <button type="submit" disabled={busy} className="auth-submit">
          <Mail className="h-4 w-4" />
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>
      <button type="button" onClick={() => void logout()} className="btn btn-ghost mt-4 w-full text-xs">
        Sign out
      </button>
    </div>
  );
}
