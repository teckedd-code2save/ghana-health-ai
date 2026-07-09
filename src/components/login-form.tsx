"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function LoginForm() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("demo@ghanahealth.ai");
  const [password, setPassword] = useState("demo1234");
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
      if (!res.ok) throw new Error(data.error || "Failed");
      router.push("/chat");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.refresh();
  }

  return (
    <div className="glass mx-auto max-w-md rounded-[var(--radius)] p-6">
      <div className="mb-4 flex gap-2">
        {(["login", "register"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded-full px-4 py-1.5 text-sm capitalize ${
              mode === m ? "bg-[var(--accent)] text-[#1a1400]" : "bg-white/5 text-[var(--fg-muted)]"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      <form onSubmit={(e) => void submit(e)} className="space-y-3">
        {mode === "register" && (
          <input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Display name"
            className="w-full rounded-2xl border border-white/10 bg-black/20 px-3 py-2.5 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
        )}
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          className="w-full rounded-2xl border border-white/10 bg-black/20 px-3 py-2.5 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
          minLength={8}
          className="w-full rounded-2xl border border-white/10 bg-black/20 px-3 py-2.5 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
        />
        {mode === "register" && (
          <div className="space-y-2 text-sm text-[var(--fg-muted)]">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={consentHealth} onChange={(e) => setConsentHealth(e.target.checked)} />
              Health data consent (Ghana DPA)
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={consentVoice} onChange={(e) => setConsentVoice(e.target.checked)} />
              Voice ID consent
            </label>
          </div>
        )}
        {error && <p className="text-sm text-[var(--coral)]">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-2xl bg-[var(--teal)] py-3 text-sm font-semibold text-[#062419] disabled:opacity-50"
        >
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>
      <button type="button" onClick={() => void logout()} className="mt-3 w-full text-center text-xs text-[var(--fg-muted)]">
        Sign out
      </button>
    </div>
  );
}
