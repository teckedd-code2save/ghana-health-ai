"use client";

import { useEffect, useRef, useState } from "react";
import { Check, RotateCcw, Send, X } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { enqueueOffline } from "@/lib/offline-queue";
import { recordUntilSilence } from "@/lib/browser-audio";
import { VoiceOrb, modeLabel, type OrbMode } from "@/components/voice-orb";

type ChatMessage = {
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM" | "local-user" | "local-assistant";
  content: string;
  phase?: "heard" | "status" | "error";
};

export function ChatPanel() {
  const { lang } = useLang();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [online, setOnline] = useState(true);
  const [pendingTranscript, setPendingTranscript] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const orbMode: OrbMode = recording
    ? "listening"
    : speaking
      ? "speaking"
      : loading
        ? "thinking"
        : "idle";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, pendingTranscript, speaking]);

  useEffect(() => {
    const sync = () => setOnline(navigator.onLine);
    sync();
    window.addEventListener("online", sync);
    window.addEventListener("offline", sync);
    return () => {
      window.removeEventListener("online", sync);
      window.removeEventListener("offline", sync);
    };
  }, []);

  function playTts(b64: string) {
    audioRef.current?.pause();
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audioRef.current = audio;
    setSpeaking(true);
    audio.onended = () => setSpeaking(false);
    audio.onerror = () => setSpeaking(false);
    void audio.play().catch(() => setSpeaking(false));
  }

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading || recording) return;
    setInput("");
    setPendingTranscript(null);
    setMessages((m) => [
      ...m.filter((x) => x.phase !== "heard" && x.phase !== "status"),
      { id: crypto.randomUUID(), role: "local-user", content: trimmed },
    ]);
    setLoading(true);
    const payload = { message: trimmed, conversationId, language: lang };
    try {
      if (!navigator.onLine) {
        enqueueOffline("chat", payload);
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: "local-assistant",
            content: "You’re offline — we’ll send this when you’re back.",
            phase: "status",
          },
        ]);
        return;
      }
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, speak: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Couldn’t send");
      setConversationId(data.conversationId);
      setMessages((m) => [
        ...m,
        {
          id: data.message.id,
          role: "ASSISTANT",
          content: data.message.content,
        },
      ]);
      if (data.tts?.audioBase64) playTts(data.tts.audioBase64);
    } catch (e) {
      enqueueOffline("chat", payload);
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: e instanceof Error ? e.message : "Something went wrong",
          phase: "error",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function startMic() {
    if (loading || speaking) return;
    setPendingTranscript(null);
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
        throw new Error("A bit quiet — try again closer");
      }
      if (durationMs < 400) throw new Error("That was too short — try again");

      setLoading(true);
      const form = new FormData();
      form.append("audio", blob, "chat-utterance.webm");
      form.append("language", lang);

      const res = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        throw new Error("Couldn’t catch that — try once more");
      }

      const text = (data.transcript?.text as string | undefined)?.trim();
      if (!text) throw new Error("No speech heard — try again");

      setPendingTranscript(text);
      setInput(text);
      setMessages((m) => [
        ...m.filter((x) => x.phase !== "heard"),
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: text,
          phase: "heard",
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: e instanceof Error ? e.message : "Voice failed",
          phase: "error",
        },
      ]);
    } finally {
      setLoading(false);
      setRecording(false);
      setVadState(null);
      setLevel(0);
    }
  }

  return (
    <div className="fade-up surface flex h-[min(78vh,720px)] flex-col overflow-hidden rounded-[calc(var(--radius)+6px)]">
      <div className="flex flex-col items-center gap-3 border-b border-white/[0.05] px-4 py-6">
        <VoiceOrb
          mode={orbMode}
          level={level}
          size="md"
          disabled={loading && !recording}
          onClick={() => {
            if (!recording && !loading && !speaking) void startMic();
          }}
          label={modeLabel(orbMode, vadState)}
        />
        <span
          className={
            recording
              ? "status-dot status-dot--live"
              : speaking
                ? "status-dot status-dot--speak"
                : "status-dot status-dot--ok"
          }
        >
          {recording
            ? "Listening"
            : speaking
              ? "Speaking"
              : loading
                ? "Thinking"
                : online
                  ? "Ready"
                  : "Offline"}
        </span>
      </div>

      <div className="chat-scroll flex-1 space-y-3 overflow-y-auto px-4 py-5">
        {messages.length === 0 && (
          <p className="mx-auto max-w-xs pt-8 text-center text-sm leading-relaxed text-[var(--fg-muted)]">
            Tap the mic and speak, or type below. We’ll show what we heard before answering.
          </p>
        )}
        {messages.map((m) => {
          const isUser = m.role === "USER" || m.role === "local-user";
          const isHeard = m.phase === "heard";
          return (
            <div
              key={m.id}
              className={`flex ${isUser || isHeard ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[88%] rounded-[1.15rem] px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                  isUser
                    ? "bg-[var(--teal)] text-[#062419]"
                    : isHeard
                      ? "border border-[var(--accent)]/30 bg-[var(--accent)]/10"
                      : m.phase === "error"
                        ? "border border-[var(--coral)]/25 bg-[var(--coral)]/10"
                        : "border border-white/[0.05] bg-white/[0.04]"
                }`}
              >
                {isHeard && (
                  <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-[var(--accent-soft)]">
                    You said
                  </p>
                )}
                {m.content}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {pendingTranscript && (
        <div className="flex items-center justify-between gap-3 border-t border-white/[0.05] bg-black/20 px-4 py-3">
          <p className="min-w-0 flex-1 truncate text-sm text-[var(--fg-muted)]">
            {pendingTranscript}
          </p>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              className="icon-action"
              aria-label="Discard"
              onClick={() => {
                setPendingTranscript(null);
                setInput("");
                setMessages((m) => m.filter((x) => x.phase !== "heard"));
              }}
            >
              <X className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="icon-action"
              aria-label="Speak again"
              disabled={loading || recording}
              onClick={() => void startMic()}
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="icon-action icon-action--accent"
              aria-label="Send"
              disabled={loading}
              onClick={() => void send(input || pendingTranscript)}
            >
              <Check className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <form
        className="border-t border-white/[0.05] p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={pendingTranscript ? "Edit what you said…" : "Type a message…"}
            className="field min-h-11 flex-1"
            disabled={recording}
          />
          <button
            type="submit"
            disabled={loading || recording || !input.trim()}
            className="icon-action icon-action--accent"
            aria-label="Send"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
