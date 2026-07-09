"use client";

import { useEffect, useState } from "react";
import { Mic, ShieldCheck, UserRound } from "lucide-react";

export function VoicePanel() {
  const [user, setUser] = useState<{ id: string; displayName?: string | null; consentVoice?: boolean } | null>(
    null,
  );
  const [passphrase, setPassphrase] = useState("Me din de Ama Mensah twi");
  const [enrollment, setEnrollment] = useState<{ id: string; enrolledAt: string } | null>(null);
  const [verifyResult, setVerifyResult] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void (async () => {
      const me = await fetch("/api/auth/me");
      if (me.ok) {
        const data = await me.json();
        setUser(data.user);
        const en = await fetch("/api/voice/enroll");
        if (en.ok) {
          const e = await en.json();
          setEnrollment(e.enrollment);
        }
      }
    })();
  }, []);

  async function enroll() {
    setBusy(true);
    setStatus(null);
    try {
      const res = await fetch("/api/voice/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passphrase, language: "tw", sampleDurationS: 30 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setEnrollment(data.enrollment);
      setStatus("Voice ID enrolled (stub embedding).");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Enroll failed");
    } finally {
      setBusy(false);
    }
  }

  async function verify() {
    setBusy(true);
    setVerifyResult(null);
    try {
      const res = await fetch("/api/voice/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passphrase }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setVerifyResult(
        data.verified
          ? `Verified · score ${data.score} ≥ ${data.threshold}`
          : `Rejected · score ${data.score} < ${data.threshold}`,
      );
    } catch (e) {
      setVerifyResult(e instanceof Error ? e.message : "Verify failed");
    } finally {
      setBusy(false);
    }
  }

  async function stubListen() {
    setBusy(true);
    try {
      const res = await fetch("/api/voice/transcribe", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setTranscript(
        `${data.transcript.text}  ·  ${data.transcript.speaker}  ·  ${data.transcript.latencyMs}ms`,
      );
    } catch (e) {
      setTranscript(e instanceof Error ? e.message : "Transcribe failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="glass rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <Mic className="h-5 w-5 text-[var(--teal)]" />
          <h2 className="font-[family-name:var(--font-display)] text-xl">Live ASR (stub)</h2>
        </div>
        <p className="mb-4 text-sm text-[var(--fg-muted)]">
          Phase 0 uses a deterministic stub. Production swaps to Modal Parakeet multi-talker +
          Sortformer diarization under <code className="text-[var(--accent-soft)]">/modal</code>.
        </p>
        <button
          onClick={() => void stubListen()}
          disabled={busy}
          className={`inline-flex items-center gap-2 rounded-full bg-[var(--teal)] px-5 py-3 text-sm font-semibold text-[#062419] ${
            busy ? "mic-pulse opacity-80" : ""
          }`}
        >
          <Mic className="h-4 w-4" /> Simulate utterance
        </button>
        {transcript && (
          <p className="mt-4 rounded-2xl bg-black/20 px-4 py-3 text-sm leading-relaxed">{transcript}</p>
        )}
      </section>

      <section className="glass rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-[var(--accent)]" />
          <h2 className="font-[family-name:var(--font-display)] text-xl">Voice ID</h2>
        </div>
        {!user ? (
          <p className="text-sm text-[var(--fg-muted)]">
            <UserRound className="mr-1 inline h-4 w-4" />
            Log in and grant voice consent to enroll. Demo:{" "}
            <span className="text-[var(--accent-soft)]">demo@ghanahealth.ai / demo1234</span>
          </p>
        ) : (
          <>
            <label className="mb-1 block text-xs text-[var(--fg-muted)]">Twi passphrase (30–60s in prod)</label>
            <textarea
              value={passphrase}
              onChange={(e) => setPassphrase(e.target.value)}
              rows={3}
              className="mb-3 w-full rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
            <div className="flex flex-wrap gap-2">
              <button
                disabled={busy}
                onClick={() => void enroll()}
                className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[#1a1400]"
              >
                Enroll
              </button>
              <button
                disabled={busy}
                onClick={() => void verify()}
                className="rounded-full border border-white/15 px-4 py-2 text-sm"
              >
                Verify
              </button>
            </div>
            {enrollment && (
              <p className="mt-3 text-xs text-[var(--fg-muted)]">
                Active enrollment · {new Date(enrollment.enrolledAt).toLocaleString()}
              </p>
            )}
            {status && <p className="mt-2 text-sm text-[var(--accent-soft)]">{status}</p>}
            {verifyResult && <p className="mt-2 text-sm">{verifyResult}</p>}
          </>
        )}
      </section>
    </div>
  );
}
