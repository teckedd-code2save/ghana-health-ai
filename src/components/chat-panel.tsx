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
  asr?: { text?: string };
  userMessage?: {
    id?: string;
    content?: string;
  };
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

async function readChatStream(
  res: Response,
  onStage: (stage: { name?: string; detail?: string }) => void,
  onReplyDelta: (chunk: string) => void,
  onAsr: (text: string) => void = () => {},
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
      } else if (parsed.event === "asr") {
        const asr = parsed.data as { text?: string };
        if (asr.text?.trim()) onAsr(asr.text.trim());
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
  const [threadScrolled, setThreadScrolled] = useState(false);
  const [level, setLevel] = useState(0);
  const [vadState, setVadState] = useState<string | null>(null);
  const [asrModel, changeAsrModel] = useAsrModel();
  const bottomRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioSafetyTimerRef = useRef<number | null>(null);
  const recordAbortRef = useRef<AbortController | null>(null);

  const orbMode: OrbMode = recording
    ? "listening"
    : loading || voicePending
        ? "thinking"
        : "idle";
  const compactOrb = threadScrolled;
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, speaking]);

  useEffect(() => () => {
    if (audioSafetyTimerRef.current) window.clearTimeout(audioSafetyTimerRef.current);
    audioRef.current?.pause();
  }, []);

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
        setInput("");
      }
      if (detail?.type === "select") window.location.reload();
      if (detail?.type === "delete" && detail.conversationId === conversationId) {
        setConversationId(undefined);
        setMessages([]);
      }
    };
    window.addEventListener(CONVERSATION_EVENT, onConversationChange);
    return () => window.removeEventListener(CONVERSATION_EVENT, onConversationChange);
  }, [conversationId]);

  function playTts(b64: string) {
    audioRef.current?.pause();
    if (audioSafetyTimerRef.current) window.clearTimeout(audioSafetyTimerRef.current);
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audioRef.current = audio;
    setSpeaking(true);
    const finish = () => {
      if (audioSafetyTimerRef.current) window.clearTimeout(audioSafetyTimerRef.current);
      audioSafetyTimerRef.current = null;
      setSpeaking(false);
    };
    audio.onended = finish;
    audio.onerror = finish;
    audioSafetyTimerRef.current = window.setTimeout(finish, 45_000);
    void audio.play().catch(finish);
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
      setVoicePending(false);
    }
  }

  async function startMic() {
    if (loading || voicePending || speaking) return;
    setRecording(true);
    setVoicePending(false);
    setVadState("listening");
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

      if (!speechDetected) {
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

      const res = await fetch("/api/voice/converse/stream", { method: "POST", body: form });
      if (!res.ok) throw new Error("Couldn’t start voice turn");

      const localUserId = crypto.randomUUID();
      const localAssistantId = crypto.randomUUID();
      let streamedTranscript = "";
      let streamedReply = "";
      const data = await readChatStream(
        res,
        () => {},
        (chunk) => {
          streamedReply += chunk;
          setMessages((current) => {
            const existing = current.some((message) => message.id === localAssistantId);
            const assistant = {
              id: localAssistantId,
              role: "ASSISTANT" as const,
              content: streamedReply,
            };
            return existing
              ? current.map((message) => message.id === localAssistantId ? assistant : message)
              : [...current, assistant];
          });
        },
        (transcript) => {
          streamedTranscript = transcript;
          setMessages((current) => current.some((message) => message.id === localUserId)
            ? current
            : [...current, { id: localUserId, role: "local-user", content: transcript }]);
        },
      );

      const text = data.asr?.text?.trim() || streamedTranscript;
      if (!text) throw new Error("No speech heard — try again");

      setConversationId(data.conversationId);
      registerConversation(data.conversationId);
      setInput("");
      setMessages((current) => {
        const finalUser: ChatMessage = {
          id: data.userMessage?.id ?? localUserId,
          role: "local-user",
          content: text,
        };
        const finalAssistant: ChatMessage = {
          id: data.message?.id ?? localAssistantId,
          role: "ASSISTANT",
          content: data.message?.content ?? data.understanding?.reply ?? streamedReply,
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
        };
        const withoutDrafts = current.filter(
          (message) => message.id !== localUserId && message.id !== localAssistantId,
        );
        return [...withoutDrafts, finalUser, finalAssistant];
      });
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
      <div
        className="chat-conversation-scroll"
        onScroll={(event) => setThreadScrolled(event.currentTarget.scrollTop > 18)}
      >
        <div className={`chat-orb-header ${compactOrb ? "chat-orb-header--compact" : ""}`}>
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
            showLabel={false}
          />
        </div>

        <div className="chat-scroll space-y-3 px-4 py-5">
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
                </div>
              </div>
            );
          })}
          {loading && (
            <div className="chat-typing" aria-label="Responding">
              <span className="typing-dots" aria-hidden="true"><span /><span /><span /></span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
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
