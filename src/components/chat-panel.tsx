"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, Send, Sparkles, Square } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { enqueueOffline } from "@/lib/offline-queue";
import { startLiveRecorder } from "@/lib/browser-audio";

type ChatMessage = {
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM" | "local-user" | "local-assistant";
  content: string;
  intent?: string;
  metadata?: Record<string, unknown>;
};

export function ChatPanel() {
  const { lang } = useLang();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [online, setOnline] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const recorderRef = useRef<Awaited<ReturnType<typeof startLiveRecorder>> | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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
    try {
      recorderRef.current = await startLiveRecorder();
      setRecording(true);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: e instanceof Error ? e.message : "Mic permission denied",
        },
      ]);
    }
  }

  async function stopMicAndTranscribe() {
    const rec = recorderRef.current;
    recorderRef.current = null;
    setRecording(false);
    if (!rec) return;
    setLoading(true);
    try {
      const blob = await rec.stop();
      if (blob.size < 200) throw new Error("Recording too short");

      const form = new FormData();
      form.append("audio", blob, "chat-utterance.webm");
      form.append("language", lang);

      const res = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Transcription failed");
      const text = (data.transcript?.text as string | undefined)?.trim();
      if (!text) throw new Error("Empty transcription — try speaking more clearly");
      setLoading(false);
      await send(text);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-assistant",
          content: e instanceof Error ? e.message : "Voice input failed",
        },
      ]);
      setLoading(false);
    }
  }

  return (
    <div className="glass flex h-[min(72vh,720px)] flex-col overflow-hidden rounded-[var(--radius)]">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-[var(--accent)]" />
          <div>
            <p className="text-sm font-medium">Health & Market Companion</p>
            <p className="text-xs text-[var(--fg-muted)]">
              Lang · {lang.toUpperCase()} · {online ? "online" : "offline queue"} · LLM + real ASR
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
            Type or hold the mic — answers come from the live model, not canned scripts.
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
            <Loader2 className="h-4 w-4 animate-spin" /> Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        className="border-t border-white/5 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
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
              onClick={() => void stopMicAndTranscribe()}
              className="mic-pulse flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--coral)] text-white"
              aria-label="Stop and send voice"
            >
              <Square className="h-4 w-4" />
            </button>
          )}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={1}
            placeholder="Ka asɛm… e.g. Me ti yɛ me ya / How much is rice?"
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
