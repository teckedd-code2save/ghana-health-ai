"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, Send, Sparkles, Square } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { enqueueOffline } from "@/lib/offline-queue";
import { recordUntilSilence } from "@/lib/browser-audio";

type ChatMessage = {
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM" | "local-user" | "local-assistant";
  content: string;
  intent?: string;
  metadata?: Record<string, unknown>;
  kind?: "transcript-pending";
};

export function ChatPanel() {
  const { lang } = useLang();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [online, setOnline] = useState(true);
  const [pendingTranscript, setPendingTranscript] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortVadRef = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, pendingTranscript]);

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

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setInput("");
    setPendingTranscript(null);
    const localId = crypto.randomUUID();
    setMessages((m) => [...m, { id: localId, role: "local-user", content: trimmed }]);
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
            content: "You're offline — message queued and will send when you're back online.",
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
      if (!res.ok) throw new Error(data.error || "Chat failed");
      setConversationId(data.conversationId);
      setMessages((m) => [
        ...m,
        {
          id: data.message.id,
          role: "ASSISTANT",
          content: data.message.content,
          intent: data.message.intent,
          metadata: data.message.metadata,
        },
      ]);
      if (data.tts?.audioBase64) {
        const audio = new Audio(`data:audio/wav;base64,${data.tts.audioBase64}`);
        void audio.play();
      }
    } catch (e) {
      enqueueOffline("chat", payload);
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content:
            (e instanceof Error ? e.message : "Something went wrong") +
            " — queued for retry when online.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function startMic() {
    setPendingTranscript(null);
    setRecording(true);
    setVadState("listening");
    abortVadRef.current = false;
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
        throw new Error("Too quiet — speak closer to the mic and try again");
      }
      if (durationMs < 400) {
        throw new Error("Recording too short");
      }

      setLoading(true);
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: "Transcribing what you said…",
          kind: "transcript-pending",
        },
      ]);

      const form = new FormData();
      form.append("audio", blob, "chat-utterance.webm");
      form.append("language", lang);

      const res = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const data = await res.json();

      // Clear "transcribing…" bubble
      setMessages((m) => m.filter((x) => x.kind !== "transcript-pending"));

      if (!res.ok) {
        const err = data.error || data.transcript?.error || "Transcription failed";
        throw new Error(
          err === "audio_too_quiet_or_silent"
            ? "Mic was too quiet — try again closer to the phone"
            : err === "asr_hallucination_or_noise"
              ? "Couldn’t hear clear speech — please try again"
              : String(err),
        );
      }

      const text = (data.transcript?.text as string | undefined)?.trim();
      if (!text) {
        throw new Error(
          data.transcript?.error === "audio_too_quiet_or_silent"
            ? "Too quiet — speak louder and try again"
            : "Empty transcription — try speaking more clearly",
        );
      }

      // Show transcript first — do NOT auto-reply until user confirms/sends
      setPendingTranscript(text);
      setInput(text);
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: `I heard: “${text}”\n\nEdit if needed, then press Send (or Send as-is).`,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m.filter((x) => x.kind !== "transcript-pending"),
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: e instanceof Error ? e.message : "Voice input failed",
        },
      ]);
    } finally {
      setLoading(false);
      setRecording(false);
      setVadState(null);
      setLevel(0);
    }
  }

  function cancelMic() {
    // recordUntilSilence has no external abort; user can wait for max or we reload UX
    abortVadRef.current = true;
    setRecording(false);
    setVadState(null);
  }

  return (
    <div className="glass flex h-[min(72vh,720px)] flex-col overflow-hidden rounded-[var(--radius)]">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-[var(--accent)]" />
          <div>
            <p className="text-sm font-medium">Health & Market Companion</p>
            <p className="text-xs text-[var(--fg-muted)]">
              Lang · {lang.toUpperCase()} · {online ? "online" : "offline queue"} · mic auto-stops
              when you finish
            </p>
          </div>
        </div>
        {conversationId && (
          <span className="rounded-full bg-white/5 px-2.5 py-1 text-[10px] text-[var(--fg-muted)]">
            {conversationId.slice(0, 8)}
          </span>
        )}
      </div>

      <div className="chat-scroll flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <p className="text-sm text-[var(--fg-muted)]">
            Tap the mic, speak, pause when done — we show what we heard before answering.
          </p>
        )}
        {messages.map((m) => {
          const isUser = m.role === "USER" || m.role === "local-user";
          return (
            <div key={m.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                  isUser
                    ? "bg-[var(--teal)] text-[#062419]"
                    : "bg-white/5 text-[var(--fg)] border border-white/5"
                }`}
              >
                {m.content}
                {m.intent && !isUser && (
                  <p className="mt-2 text-[10px] uppercase tracking-wide text-[var(--accent-soft)]">
                    intent · {m.intent}
                  </p>
                )}
              </div>
            </div>
          );
        })}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-[var(--fg-muted)]">
            <Loader2 className="h-4 w-4 animate-spin" /> Working…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {pendingTranscript && (
        <div className="flex items-center gap-2 border-t border-white/5 bg-black/20 px-3 py-2 text-xs">
          <span className="text-[var(--fg-muted)]">Ready to send transcript</span>
          <button
            type="button"
            className="rounded-full bg-[var(--accent)] px-3 py-1 font-medium text-[#1a1400]"
            disabled={loading}
            onClick={() => void send(input || pendingTranscript)}
          >
            Send as heard
          </button>
          <button
            type="button"
            className="rounded-full border border-white/15 px-3 py-1"
            onClick={() => {
              setPendingTranscript(null);
              setInput("");
            }}
          >
            Discard
          </button>
        </div>
      )}

      <form
        className="border-t border-white/5 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        {recording && (
          <div className="mb-2 flex items-center gap-3 text-xs text-[var(--fg-muted)]">
            <span className="mic-pulse text-[var(--coral)]">
              {vadState === "speech"
                ? "Hearing you…"
                : vadState === "silence"
                  ? "Pause detected — finishing…"
                  : "Listening…"}
            </span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full bg-[var(--teal)] transition-all duration-75"
                style={{ width: `${Math.min(100, level * 800)}%` }}
              />
            </div>
          </div>
        )}
        <div className="flex items-end gap-2">
          {!recording ? (
            <button
              type="button"
              onClick={() => void startMic()}
              disabled={loading}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--teal-deep)] text-white transition hover:bg-[var(--teal)] disabled:opacity-50"
              aria-label="Start voice input"
            >
              <Mic className="h-5 w-5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={cancelMic}
              className="mic-pulse flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--coral)] text-white"
              aria-label="Stop voice"
              title="Wait for auto-stop, or keep talking"
            >
              <Square className="h-4 w-4" />
            </button>
          )}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={1}
            placeholder="Ka asɛm… or use the mic"
            className="min-h-11 flex-1 resize-none rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5 text-sm outline-none ring-[var(--accent)] placeholder:text-[var(--fg-muted)] focus:ring-1"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[#1a1400] transition hover:bg-[var(--accent-soft)] disabled:opacity-50"
            aria-label="Send"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
