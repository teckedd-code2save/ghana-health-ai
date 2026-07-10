"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Fingerprint, RotateCcw, Volume2, X } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { blobToPcmB64, recordUntilSilence, startLiveRecorder } from "@/lib/browser-audio";
import { VoiceOrb, modeLabel, type OrbMode } from "@/components/voice-orb";

type AsrPreview = { text: string; model?: string; latencyMs?: number };

export function VoicePanel() {
  const { lang } = useLang();
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [status, setStatus] = useState<string | null>(null);
  const [asrPreview, setAsrPreview] = useState<AsrPreview | null>(null);
  const [editedText, setEditedText] = useState("");
  const [reply, setReply] = useState<string | null>(null);
  const [ttsB64, setTtsB64] = useState<string | null>(null);
  const [showId, setShowId] = useState(false);

  const [vidBusy, setVidBusy] = useState(false);
  const [vidStatus, setVidStatus] = useState<string | null>(null);
  const [enrolled, setEnrolled] = useState(false);
  const [vidRecording, setVidRecording] = useState<"enroll" | "verify" | null>(null);
  const vidRecRef = useRef<Awaited<ReturnType<typeof startLiveRecorder>> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const orbMode: OrbMode = recording
    ? "listening"
    : speaking
      ? "speaking"
      : busy
        ? "thinking"
        : "idle";

  useEffect(() => {
    void fetch("/api/voice/enroll")
      .then((r) => r.json())
      .then((d) => setEnrolled(Boolean(d.enrollment)))
      .catch(() => setEnrolled(false));
  }, []);

  function playWav(b64: string) {
    audioRef.current?.pause();
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audioRef.current = audio;
    setSpeaking(true);
    audio.onended = () => setSpeaking(false);
    audio.onerror = () => setSpeaking(false);
    void audio.play().catch(() => setSpeaking(false));
  }

  async function listenAndTranscribe() {
    if (busy || speaking) return;
    setStatus(null);
    setAsrPreview(null);
    setReply(null);
    setTtsB64(null);
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

      if (blob.size < 400 || peakLevel < 0.01) throw new Error("Too quiet — try closer");
      if (durationMs < 400) throw new Error("Too short");

      setBusy(true);
      setStatus("Transcribing…");

      const form = new FormData();
      form.append("audio", blob, "utterance.webm");
      form.append("language", lang);

      const res = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        const err = data.error || "Transcription failed";
        throw new Error(
          err === "audio_too_quiet_or_silent"
            ? "Too quiet"
            : err === "asr_hallucination_or_noise"
              ? "Couldn’t catch clear speech"
              : String(err),
        );
      }

      const text = (data.transcript?.text as string | undefined)?.trim();
      if (!text) throw new Error("No speech heard");

      setAsrPreview({
        text,
        model: data.transcript?.model,
        latencyMs: data.transcript?.latencyMs,
      });
      setEditedText(text);
      setStatus(null);
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
    setStatus(null);
    setReply(null);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversationId,
          language: lang,
          speak: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Reply failed");

      setConversationId(data.conversationId);
      setReply(data.message?.content ?? "");
      if (data.tts?.audioBase64) {
        setTtsB64(data.tts.audioBase64);
        playWav(data.tts.audioBase64);
      }
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
      setVidStatus(mode === "enroll" ? "Recording enrollment…" : "Recording verify…");
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
      if (durationS < 0.8) throw new Error("Speak 1–2 seconds");

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
        setVidStatus("Enrolled");
      } else {
        const res = await fetch("/api/voice/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pcmB64, sampleRate }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Verify failed");
        setVidStatus(data.verified ? `Verified · ${data.score}` : `Not verified · ${data.score}`);
      }
    } catch (e) {
      setVidStatus(e instanceof Error ? e.message : "Voice ID failed");
    } finally {
      setVidBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <section className="glass flex flex-col items-center rounded-[calc(var(--radius)+6px)] px-6 py-10">
        <VoiceOrb
          mode={orbMode}
          level={level}
          size="lg"
          onClick={() => {
            if (!recording && !busy && !speaking) void listenAndTranscribe();
          }}
          label={status || modeLabel(orbMode, vadState)}
        />

        <div className="mt-4 flex items-center gap-2">
          <span
            className={
              recording
                ? "status-chip status-chip--live"
                : speaking
                  ? "status-chip status-chip--speak"
                  : "status-chip"
            }
          >
            {recording
              ? "Listening"
              : speaking
                ? "Speaking"
                : busy
                  ? "Thinking"
                  : "Ready"}
          </span>
          <button
            type="button"
            className="icon-action"
            title="Voice ID"
            aria-label="Voice ID"
            onClick={() => setShowId((v) => !v)}
          >
            <Fingerprint className="h-4 w-4" />
          </button>
        </div>
      </section>

      {asrPreview && (
        <section className="glass space-y-3 rounded-[var(--radius)] p-4">
          <p className="text-[10px] uppercase tracking-wider text-[var(--accent-soft)]">I heard</p>
          <textarea
            value={editedText}
            onChange={(e) => setEditedText(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-2xl border border-white/10 bg-black/25 px-3 py-2.5 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              className="icon-action"
              title="Discard"
              aria-label="Discard"
              onClick={() => {
                setAsrPreview(null);
                setEditedText("");
                setReply(null);
              }}
            >
              <X className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="icon-action"
              title="Re-record"
              aria-label="Re-record"
              disabled={busy || recording}
              onClick={() => void listenAndTranscribe()}
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="icon-action icon-action--accent"
              title="Confirm reply"
              aria-label="Confirm and reply"
              disabled={busy || !editedText.trim()}
              onClick={() => void confirmAndReply()}
            >
              <Check className="h-4 w-4" />
            </button>
          </div>
        </section>
      )}

      {reply && (
        <section className="glass space-y-3 rounded-[var(--radius)] p-4">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">Assistant</p>
            {ttsB64 && (
              <button
                type="button"
                className="icon-action"
                title="Replay"
                aria-label="Replay voice"
                onClick={() => playWav(ttsB64)}
              >
                <Volume2 className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{reply}</p>
        </section>
      )}

      {showId && (
        <section className="glass space-y-3 rounded-[var(--radius)] p-4">
          <p className="text-sm text-[var(--fg-muted)]">
            Voice ID · {enrolled ? "enrolled" : "not enrolled"} · login required
          </p>
          <div className="flex gap-2">
            {!vidRecording ? (
              <>
                <button
                  type="button"
                  className="icon-action icon-action--accent"
                  title="Enroll"
                  aria-label="Enroll voice"
                  disabled={vidBusy}
                  onClick={() => void startVidCapture("enroll")}
                >
                  <Fingerprint className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  className="icon-action"
                  title="Verify"
                  aria-label="Verify voice"
                  disabled={vidBusy || !enrolled}
                  onClick={() => void startVidCapture("verify")}
                >
                  <Check className="h-4 w-4" />
                </button>
              </>
            ) : (
              <button
                type="button"
                className="icon-action icon-action--danger"
                title="Stop"
                aria-label="Stop recording"
                onClick={() => void stopVidCapture()}
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          {vidStatus && <p className="text-xs text-[var(--accent-soft)]">{vidStatus}</p>}
        </section>
      )}
    </div>
  );
}
