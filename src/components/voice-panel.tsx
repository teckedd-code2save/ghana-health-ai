"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, ShieldCheck, Square, Volume2 } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { blobToPcmB64, recordUntilSilence, startLiveRecorder } from "@/lib/browser-audio";

type AsrPreview = {
  text: string;
  model?: string;
  latencyMs?: number;
  duration?: number;
};

type ReplyResult = {
  reply: string;
  intent?: string;
  severity?: string;
  engine?: string;
  tts?: { audioBase64: string; model?: string } | null;
  conversationId?: string;
};

function playWavBase64(b64: string) {
  const audio = new Audio(`data:audio/wav;base64,${b64}`);
  void audio.play();
  return audio;
}

export function VoicePanel() {
  const { lang } = useLang();
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [status, setStatus] = useState<string | null>(null);
  const [speak, setSpeak] = useState(true);
  const [asrPreview, setAsrPreview] = useState<AsrPreview | null>(null);
  const [editedText, setEditedText] = useState("");
  const [reply, setReply] = useState<ReplyResult | null>(null);

  const [vidBusy, setVidBusy] = useState(false);
  const [vidStatus, setVidStatus] = useState<string | null>(null);
  const [enrolled, setEnrolled] = useState(false);
  const [vidRecording, setVidRecording] = useState<"enroll" | "verify" | null>(null);
  const vidRecRef = useRef<Awaited<ReturnType<typeof startLiveRecorder>> | null>(null);

  useEffect(() => {
    void fetch("/api/voice/enroll")
      .then((r) => r.json())
      .then((d) => setEnrolled(Boolean(d.enrollment)))
      .catch(() => setEnrolled(false));
  }, []);

  async function listenAndTranscribe() {
    setStatus(null);
    setAsrPreview(null);
    setReply(null);
    setEditedText("");
    setRecording(true);
    setVadState("listening");
    try {
      const { blob, durationMs, peakLevel } = await recordUntilSilence({
        silenceMs: 1100,
        maxMs: 20_000,
        minSpeechMs: 450,
        onLevel: setLevel,
        onState: setVadState,
      });
      setRecording(false);
      setVadState(null);
      setLevel(0);

      if (blob.size < 400 || peakLevel < 0.01) {
        throw new Error("Too quiet — speak closer to the mic");
      }
      if (durationMs < 400) throw new Error("Recording too short");

      setBusy(true);
      setStatus("Transcribing…");

      const form = new FormData();
      form.append("audio", blob, "utterance.webm");
      form.append("language", lang);

      const res = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        const err = data.error || data.transcript?.error || "Transcription failed";
        throw new Error(
          err === "audio_too_quiet_or_silent"
            ? "Too quiet — try again closer to the mic"
            : err === "asr_hallucination_or_noise"
              ? "Couldn’t detect clear speech — please try again"
              : String(err),
        );
      }

      const text = (data.transcript?.text as string | undefined)?.trim();
      if (!text) throw new Error("Empty transcription — try speaking more clearly");

      const preview: AsrPreview = {
        text,
        model: data.transcript?.model,
        latencyMs: data.transcript?.latencyMs,
        duration: data.transcript?.duration,
      };
      setAsrPreview(preview);
      setEditedText(text);
      setStatus("Check the transcript, then Confirm to get a reply");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
      setRecording(false);
      setVadState(null);
      setLevel(0);
    }
  }

  async function confirmAndReply() {
    const text = editedText.trim();
    if (!text || busy) return;
    setBusy(true);
    setStatus("Understanding…");
    setReply(null);
    try {
      // Text path — same brain as chat; avoids re-running ASR
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversationId,
          language: lang,
          speak,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Reply failed");

      setConversationId(data.conversationId);
      setReply({
        reply: data.message?.content ?? "",
        intent: data.message?.intent,
        severity: data.message?.metadata?.severity as string | undefined,
        engine: data.message?.metadata?.engine as string | undefined,
        tts: data.tts
          ? { audioBase64: data.tts.audioBase64, model: data.tts.model }
          : null,
        conversationId: data.conversationId,
      });
      setStatus("Done");
      if (data.tts?.audioBase64) playWavBase64(data.tts.audioBase64);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function startVidCapture(mode: "enroll" | "verify") {
    setVidStatus(null);
    try {
      vidRecRef.current = await startLiveRecorder();
      setVidRecording(mode);
      setVidStatus(mode === "enroll" ? "Recording enrollment…" : "Recording verify sample…");
    } catch (e) {
      setVidStatus(e instanceof Error ? e.message : "Mic denied");
    }
  }

  async function stopVidCapture() {
    const mode = vidRecording;
    const rec = vidRecRef.current;
    vidRecRef.current = null;
    setVidRecording(null);
    if (!rec || !mode) return;

    setVidBusy(true);
    try {
      const blob = await rec.stop();
      const { pcmB64, sampleRate, durationS } = await blobToPcmB64(blob);
      if (durationS < 0.8) throw new Error("Speak for at least 1–2 seconds");

      if (mode === "enroll") {
        const res = await fetch("/api/voice/enroll", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pcmB64,
            sampleRate,
            language: lang,
            sampleDurationS: durationS,
            phraseHint: "spoken-enroll",
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Enroll failed");
        setEnrolled(true);
        setVidStatus(
          `Enrolled from ${durationS.toFixed(1)}s audio · threshold ${data.enrollment?.threshold ?? 0.82}`,
        );
      } else {
        const res = await fetch("/api/voice/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pcmB64, sampleRate }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Verify failed");
        setVidStatus(
          data.verified
            ? `Verified ✓ score ${data.score} (threshold ${data.threshold})`
            : `Not verified · score ${data.score} (need ≥ ${data.threshold})`,
        );
      }
    } catch (e) {
      setVidStatus(e instanceof Error ? e.message : "Voice ID failed");
    } finally {
      setVidBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="glass rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <Mic className="h-5 w-5 text-[var(--teal)]" />
          <h2 className="font-[family-name:var(--font-display)] text-xl">Live conversation</h2>
        </div>
        <p className="mb-4 text-sm text-[var(--fg-muted)]">
          Speak freely — we auto-stop when you pause, show the transcript first, then reply only after
          you confirm.
        </p>

        <label className="mb-4 flex items-center gap-2 text-sm text-[var(--fg-muted)]">
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
          Speak reply (TTS)
        </label>

        {recording && (
          <div className="mb-3 space-y-1 text-xs text-[var(--fg-muted)]">
            <p className="text-[var(--coral)]">
              {vadState === "speech"
                ? "Hearing you…"
                : vadState === "silence"
                  ? "Pause — wrapping up…"
                  : "Listening…"}
            </p>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full bg-[var(--teal)] transition-all duration-75"
                style={{ width: `${Math.min(100, level * 800)}%` }}
              />
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void listenAndTranscribe()}
            disabled={busy || recording}
            className="inline-flex items-center gap-2 rounded-full bg-[var(--teal)] px-5 py-3 text-sm font-semibold text-[#062419] disabled:opacity-50"
          >
            {busy || recording ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
            {recording ? "Listening…" : busy ? "Working…" : "Talk (auto-stop)"}
          </button>
        </div>

        {status && <p className="mt-4 text-xs text-[var(--accent-soft)]">{status}</p>}

        {asrPreview && (
          <div className="mt-4 space-y-3 rounded-2xl bg-black/20 p-4 text-sm">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
                You said (edit if wrong)
              </p>
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                rows={3}
                className="mt-2 w-full rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
              />
              <p className="mt-1 text-[10px] text-[var(--fg-muted)]">
                {asrPreview.model}
                {asrPreview.latencyMs != null ? ` · ${asrPreview.latencyMs}ms` : ""}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                disabled={busy || !editedText.trim()}
                onClick={() => void confirmAndReply()}
                className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[#1a1400] disabled:opacity-50"
              >
                Confirm &amp; reply
              </button>
              <button
                type="button"
                disabled={busy || recording}
                onClick={() => void listenAndTranscribe()}
                className="rounded-full border border-white/15 px-4 py-2 text-sm"
              >
                Re-record
              </button>
            </div>
          </div>
        )}

        {reply && (
          <div className="mt-4 space-y-2 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm">
            <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
              Reply
              {reply.intent ? ` · ${reply.intent}` : ""}
              {reply.severity ? ` · ${reply.severity}` : ""}
            </p>
            <p className="whitespace-pre-wrap">{reply.reply}</p>
            {reply.tts?.audioBase64 && (
              <button
                type="button"
                onClick={() => playWavBase64(reply.tts!.audioBase64)}
                className="inline-flex items-center gap-2 rounded-full border border-white/15 px-3 py-1.5 text-xs"
              >
                <Volume2 className="h-3.5 w-3.5" /> Replay voice
              </button>
            )}
          </div>
        )}
      </section>

      <section className="glass rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-[var(--accent)]" />
          <h2 className="font-[family-name:var(--font-display)] text-xl">Voice ID</h2>
        </div>
        <p className="mb-3 text-sm text-[var(--fg-muted)]">
          Enroll and verify with real mic audio. Login + voice consent required.
          {enrolled ? " · Enrollment on file." : " · Not enrolled yet."}
        </p>

        <div className="flex flex-wrap gap-2">
          {!vidRecording ? (
            <>
              <button
                disabled={vidBusy}
                onClick={() => void startVidCapture("enroll")}
                className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[#1a1400] disabled:opacity-50"
              >
                {vidBusy ? "Working…" : "Enroll (record)"}
              </button>
              <button
                disabled={vidBusy || !enrolled}
                onClick={() => void startVidCapture("verify")}
                className="rounded-full border border-white/15 px-4 py-2 text-sm disabled:opacity-40"
              >
                Verify (record)
              </button>
            </>
          ) : (
            <button
              onClick={() => void stopVidCapture()}
              className="mic-pulse inline-flex items-center gap-2 rounded-full bg-[var(--coral)] px-4 py-2 text-sm font-semibold text-white"
            >
              <Square className="h-3.5 w-3.5" /> Stop {vidRecording}
            </button>
          )}
        </div>
        {vidStatus && <p className="mt-3 text-sm text-[var(--accent-soft)]">{vidStatus}</p>}
      </section>
    </div>
  );
}
