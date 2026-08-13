"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function LoginForm() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [consentHealth, setConsentHealth] = useState(true);
  const [consentVoice, setConsentVoice] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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
      router.push("/voice");
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
    <div className="fade-up surface mx-auto max-w-md rounded-[var(--radius)] p-6 md:p-8">
      <div className="mb-6 flex gap-1 rounded-full bg-black/25 p-1">
        {(["login", "register"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`flex-1 rounded-full px-4 py-2 text-sm capitalize transition ${
              mode === m
                ? "bg-[var(--accent)] font-medium text-[#1a1400]"
                : "text-[var(--fg-muted)]"
            }`}
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
        <button type="submit" disabled={busy} className="btn btn-teal w-full py-3">
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>
      <button type="button" onClick={() => void logout()} className="btn btn-ghost mt-4 w-full text-xs">
        Sign out
      </button>
    </div>
  );
}
