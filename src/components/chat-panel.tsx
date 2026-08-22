"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, Send } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { useAsrModel } from "@/lib/asr-model-store";
import { enqueueOffline } from "@/lib/offline-queue";
import { recordUntilSilence } from "@/lib/browser-audio";
import { VoiceOrb, modeLabel, type OrbMode } from "@/components/voice-orb";
import {
  CONVERSATION_EVENT,
  registerConversation,
  type ConversationChange,
} from "@/lib/conversation-store";

type ChatMessage = {
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM" | "local-user" | "local-assistant";
  content: string;
  phase?: "heard" | "status" | "error";
  meta?: {
    intent?: string;
    engine?: string;
    retrieve?: string;
    reviewed?: boolean;
    latencyMs?: number;
    synthesisMode?: "live_model" | "degraded_fallback";
    model?: string;
    understood?: boolean;
  };
};

type StoredMessage = {
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM";
  content: string;
};

type StreamEvent = {
  event: string;
  data: unknown;
};

type ChatTurnData = {
  conversationId?: string;
  message?: {
    id?: string;
    content?: string;
  };
  understanding?: {
    reply?: string;
    intent?: string;
    engine?: string;
    synthesis?: {
      mode?: "live_model" | "degraded_fallback";
      model?: string;
    };
    comprehension?: { understood?: boolean };
  };
  tts?: {
    audioBase64?: string;
  } | null;
  stage?: {
    retrieveEngine?: string;
    review?: boolean;
    totalLatencyMs?: number;
  };
  error?: string;
  totalLatencyMs?: number;
};

function pipelineLabel(name?: string, detail?: string) {
  if (name === "accepted") return "Accepted";
  if (name === "conversation") return "Conversation";
  if (name === "user_message") return "Saved";
  if (name === "understanding") {
    if (detail === "understood") return "Meaning understood";
    if (detail === "not_understood") return "Meaning unclear";
    return "Checking understanding";
  }
  if (name === "assistant_message") return "Answer generated";
  if (name === "tts") return detail === "started" ? "Preparing voice" : "Speech ready";
  if (name === "audit") return "Answered";
  return name || "Working";
}

async function readChatStream(
  res: Response,
  onStage: (stage: { name?: string; detail?: string }) => void,
  onReplyDelta: (chunk: string) => void,
): Promise<ChatTurnData> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData: ChatTurnData | null = null;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const eventLine = part.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = part.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const parsed: StreamEvent = {
        event: eventLine.slice("event:".length).trim(),
        data: JSON.parse(dataLine.slice("data:".length).trim()),
      };
      if (parsed.event === "stage") {
        onStage(parsed.data as { name?: string; detail?: string });
      } else if (parsed.event === "reply_delta") {
        const delta = parsed.data as { chunk?: string };
        if (delta.chunk) onReplyDelta(delta.chunk);
      } else if (parsed.event === "error") {
        const err = parsed.data as { error?: string };
        throw new Error(err.error || "Chat failed");
      } else if (parsed.event === "final") {
        finalData = parsed.data as ChatTurnData;
      }
    }
  }

  if (!finalData) throw new Error("No model response");
  return finalData;
}

export function ChatPanel() {
  const { lang } = useLang();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [voicePending, setVoicePending] = useState(false);
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<string[]>([]);
  const [online, setOnline] = useState(true);
  const [asrModel, changeAsrModel] = useAsrModel();
  const bottomRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recordAbortRef = useRef<AbortController | null>(null);

  const orbMode: OrbMode = recording
    ? "listening"
    : speaking || voicePending
      ? "speaking"
      : loading
        ? "thinking"
        : "idle";
  const statusLabel = recording
    ? "Listening"
    : voicePending
      ? "Preparing voice"
      : speaking
        ? "Speaking"
        : loading
          ? "Thinking"
          : online
            ? "Ready"
            : "Offline";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, speaking]);

  useEffect(() => {
    let cancelled = false;
    const storedConversationId = window.localStorage.getItem("gha:active-conversation");
    if (!storedConversationId) return;

    void fetch(`/api/chat?conversationId=${encodeURIComponent(storedConversationId)}`)
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Could not load session");
        return data.messages as StoredMessage[];
      })
      .then((storedMessages) => {
        if (cancelled) return;
        setConversationId(storedConversationId);
        registerConversation(storedConversationId);
        setMessages(
          storedMessages
            .filter((message) => message.role === "USER" || message.role === "ASSISTANT")
            .map((message) => ({
              id: message.id,
              role: message.role,
              content: message.content,
            })),
        );
      })
      .catch(() => {
        if (!cancelled) window.localStorage.removeItem("gha:active-conversation");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onConversationChange = (event: Event) => {
      const detail = (event as CustomEvent<ConversationChange>).detail;
      if (detail?.type === "new") {
        setConversationId(undefined);
        setMessages([]);
        setPipeline([]);
        setInput("");
      }
      if (detail?.type === "select") window.location.reload();
      if (detail?.type === "delete" && detail.conversationId === conversationId) {
        setConversationId(undefined);
        setMessages([]);
        setPipeline([]);
      }
    };
    window.addEventListener(CONVERSATION_EVENT, onConversationChange);
    return () => window.removeEventListener(CONVERSATION_EVENT, onConversationChange);
  }, [conversationId]);

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
    setMessages((m) => [
      ...m.filter((x) => x.phase !== "status"),
      { id: crypto.randomUUID(), role: "local-user", content: trimmed },
    ]);
    setLoading(true);
    setVoicePending(false);
    setPipeline(["Sending", "Checking understanding"]);
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
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, speak: true }),
      });
      if (!res.ok) throw new Error("Couldn’t start live turn");
      const assistantDraftId = crypto.randomUUID();
      const data = await readChatStream(res, (stage) => {
          if (stage.name === "tts" && stage.detail === "started") {
            setLoading(false);
            setVoicePending(true);
          }
          setPipeline((steps) => {
            const next = pipelineLabel(stage.name, stage.detail);
            return steps[steps.length - 1] === next ? steps : [...steps.slice(-4), next];
          });
      }, (chunk) => {
        setMessages((m) => {
          const existing = m.find((msg) => msg.id === assistantDraftId);
          if (existing) {
            return m.map((msg) =>
              msg.id === assistantDraftId
                ? { ...msg, content: `${msg.content}${chunk}` }
                : msg,
            );
          }
          return [
            ...m,
            {
              id: assistantDraftId,
              role: "ASSISTANT",
              content: chunk,
            },
          ];
        });
      });
      const messageId = data.message?.id ?? crypto.randomUUID();
      const messageContent = data.message?.content ?? data.understanding?.reply ?? "";
      if (!messageContent.trim()) throw new Error("No model response");
      setConversationId(data.conversationId);
      registerConversation(data.conversationId);
      setMessages((m) => {
        const finalMessage: ChatMessage = {
          id: messageId,
          role: "ASSISTANT",
          content: messageContent,
          meta: {
            intent: data.understanding?.intent,
            engine: data.understanding?.engine,
            retrieve: data.stage?.retrieveEngine,
            reviewed: data.stage?.review,
            latencyMs: data.stage?.totalLatencyMs,
            synthesisMode: data.understanding?.synthesis?.mode,
            model: data.understanding?.synthesis?.model,
            understood: data.understanding?.comprehension?.understood,
          },
        };
        return m.some((msg) => msg.id === assistantDraftId)
          ? m.map((msg) => (msg.id === assistantDraftId ? finalMessage : msg))
          : [...m, finalMessage];
      });
      if (data.tts?.audioBase64) playTts(data.tts.audioBase64);
      setVoicePending(false);
      setPipeline([
        data.understanding?.comprehension?.understood ? "Meaning understood" : "Meaning unclear",
        "Answer generated",
        data.tts?.audioBase64 ? "Speaking" : "Answered",
      ]);
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
      setPipeline([]);
    } finally {
      setLoading(false);
      setVoicePending(false);
    }
  }

  async function startMic() {
    if (loading || voicePending || speaking) return;
    setRecording(true);
    setVoicePending(false);
    setVadState("listening");
    setPipeline(["Listening"]);
    const aborter = new AbortController();
    recordAbortRef.current = aborter;
    try {
      const { blob, durationMs, peakLevel, speechDetected } = await recordUntilSilence({
        silenceMs: 1100,
        maxMs: 20_000,
        minSpeechMs: 450,
        onLevel: setLevel,
        onState: setVadState,
        signal: aborter.signal,
      });
      recordAbortRef.current = null;
      setRecording(false);
      setVadState(null);
      setLevel(0);
      setPipeline(["Heard speech", "Transcribing"]);

      if (!speechDetected) {
        setPipeline([]);
        return;
      }
      if (blob.size < 400 || peakLevel < 0.01) {
        throw new Error("A bit quiet — try again closer");
      }
      if (durationMs < 400) throw new Error("That was too short — try again");

      setLoading(true);
      const form = new FormData();
      form.append("audio", blob, "chat-utterance.webm");
      form.append("language", lang);
      if (asrModel === "dondo") {
        form.append("asrModel", "dondo");
      }

      if (conversationId) form.append("conversationId", conversationId);
      form.append("speak", "true");

      const res = await fetch("/api/voice/converse", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Voice turn failed");
      }

      const text = (data.asr?.text as string | undefined)?.trim();
      if (!text) throw new Error("No speech heard — try again");

      setConversationId(data.conversationId);
      registerConversation(data.conversationId);
      setInput("");
      setPipeline([
        "Transcribed",
        data.understanding?.comprehension?.understood ? "Meaning understood" : "Meaning unclear",
        "Answer generated",
        data.tts?.audioBase64 ? "Speaking" : "Answered",
      ]);
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "local-user",
          content: text,
        },
        {
          id: data.message?.id ?? crypto.randomUUID(),
          role: "ASSISTANT",
          content: data.message?.content ?? data.understanding?.reply ?? "",
          meta: {
            intent: data.understanding?.intent,
            engine: data.understanding?.engine,
            retrieve: data.stage?.retrieveEngine,
            reviewed: data.stage?.review,
            latencyMs: data.stage?.totalLatencyMs ?? data.totalLatencyMs,
            synthesisMode: data.understanding?.synthesis?.mode,
            model: data.understanding?.synthesis?.model,
            understood: data.understanding?.comprehension?.understood,
          },
        },
      ]);
      if (data.tts?.audioBase64) playTts(data.tts.audioBase64);
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
      setPipeline([]);
    } finally {
      recordAbortRef.current = null;
      setLoading(false);
      setVoicePending(false);
      setRecording(false);
      setVadState(null);
      setLevel(0);
    }
  }

  return (
    <div className="chat-panel fade-up">
      <div className="flex flex-col items-center gap-3 border-b border-[var(--line)] px-4 py-6">
        <VoiceOrb
          mode={orbMode}
          level={level}
          size="md"
          disabled={(loading || voicePending) && !recording}
          onClick={() => {
            if (recording) {
              recordAbortRef.current?.abort();
              return;
            }
            if (!loading && !voicePending && !speaking) void startMic();
          }}
          label={voicePending ? "Preparing voice..." : modeLabel(orbMode, vadState)}
        />
        <span
          className={
            recording
              ? "status-dot status-dot--live"
              : speaking || voicePending
                ? "status-dot status-dot--speak"
                : "status-dot status-dot--ok"
          }
        >
          {statusLabel}
        </span>
      </div>

      {pipeline.length > 0 && <div className="chat-activity">{pipeline[pipeline.length - 1]}</div>}

      <div className="chat-scroll flex-1 space-y-3 overflow-y-auto px-4 py-5">
        {messages.map((m) => {
          const isUser = m.role === "USER" || m.role === "local-user";
          return (
            <div
              key={m.id}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`chat-bubble ${
                  isUser
                    ? "chat-bubble--user"
                    : m.phase === "error"
                        ? "chat-bubble--error"
                        : "chat-bubble--assistant"
                }`}
              >
                {m.content}
                {!isUser && m.meta?.synthesisMode && (
                  <span
                    className={`response-provenance response-provenance--${m.meta.synthesisMode}`}
                    title={m.meta.model ? `Generated with ${m.meta.model}; deterministic safety checks applied` : undefined}
                  >
                    {m.meta.understood === false
                      ? "Not understood"
                      : m.meta.synthesisMode === "live_model"
                        ? "Live model"
                        : "Safety fallback"}
                  </span>
                )}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      <form
        className="chat-composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="chat-composer__surface">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message…"
            className="chat-composer__input"
            disabled={recording}
          />
          <div className="chat-composer__actions">
            <label className="voice-model-picker voice-model-picker--compact" aria-label="Speech model">
              <span>Voice</span>
              <select value={asrModel} onChange={(event) => changeAsrModel(event.target.value as typeof asrModel)}>
                <option value="dondo">DONDO</option>
                <option value="v6">v6</option>
              </select>
            </label>
            <span className="chat-composer__spacer" />
            <button
              type="button"
              disabled={loading || voicePending || speaking}
              className={recording ? "icon-action chat-mic-action chat-mic-action--live" : "icon-action chat-mic-action"}
              aria-label="Record voice"
              onClick={() => {
                if (recording) {
                  recordAbortRef.current?.abort();
                  return;
                }
                void startMic();
              }}
            >
              <Mic className="h-4 w-4" />
            </button>
            <button
              type="submit"
              disabled={loading || recording || !input.trim()}
              className="icon-action chat-send-action"
              aria-label="Send"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
