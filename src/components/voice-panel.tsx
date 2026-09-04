"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Mic, Pencil, Send, ShoppingCart, Volume2, X } from "lucide-react";
import { UnderstandingDetails, type UnderstandingDetailsData } from "@/components/understanding-details";
import { useLang } from "@/components/lang-provider";
import { useAsrModel } from "@/lib/asr-model-store";
import { useUnderstandingModelMode } from "@/lib/understanding-model-store";
import { recordUntilSilence } from "@/lib/browser-audio";
import { VoiceOrb, type OrbMode } from "@/components/voice-orb";
import {
  CONVERSATION_EVENT,
  registerConversation,
  type ConversationChange,
} from "@/lib/conversation-store";

type VoiceFocus = "health" | "commerce";

type VoiceTurnData = {
  conversationId?: string;
  asr?: { text?: string };
  understanding?: UnderstandingDetailsData & {
    reply?: string;
    commerceExecution?: CommerceExecution;
    synthesis?: { mode?: "live_model" | "degraded_fallback"; model?: string };
  };
  userMessage?: { id?: string; content?: string };
  message?: { id?: string; content?: string };
  tts?: { audioBase64?: string } | null;
  error?: string;
};

type CommerceExecution = {
  mode?: "none" | "local_catalog_search" | "order_draft";
  status?: "not_applicable" | "needs_clarification" | "ready" | "no_matches";
  products?: Array<{
    id: string;
    nameEn: string;
    nameTw: string;
    priceGhs: number;
    unit: string;
    stock: number;
  }>;
  draft?: {
    item: string;
    quantity?: string;
    location?: string;
    fulfillment?: "delivery" | "pickup" | "unknown";
    requiresConfirmation?: boolean;
  };
};

type VoiceStreamEvent = VoiceTurnData & {
  name?: string;
  detail?: string;
  text?: string;
  chunk?: string;
};

type StoredMessage = {
  metadata?: UnderstandingDetailsData;
  id: string;
  role: "USER" | "ASSISTANT" | "SYSTEM";
  content: string;
};

type VoiceMessage = {
  understanding?: UnderstandingDetailsData;
  id: string;
  role: "USER" | "ASSISTANT";
  content: string;
};

function upsertVoiceMessage(messages: VoiceMessage[], next: VoiceMessage, replaceId?: string) {
  let replaced = false;
  const updated = messages.map((message) => {
    if (message.id === next.id || (replaceId && message.id === replaceId)) {
      replaced = true;
      return next;
    }
    return message;
  });
  return replaced ? updated : [...updated, next];
}

function friendlyVoiceError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error || "");
  if (
    message.includes("ECONNREFUSED") ||
    message.includes("Prisma") ||
    message.includes("TURBOPACK") ||
    message.includes("Can't reach database") ||
    message.includes("database is not reachable")
  ) {
    return "The service is still getting ready. Please try again in a moment.";
  }
  if (message && !/Error:|Prisma|MODAL_|ECONN|HTTP \d|failed$/i.test(message)) return message;
  return "I couldn’t process that recording. Please try again a little closer to the microphone.";
}

async function readVoiceStream(
  res: Response,
  onAsr: (text: string) => void,
  onReplyDelta: (chunk: string) => void,
  onStage: (stage: { name: string; detail?: string }) => void,
): Promise<VoiceTurnData> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");

  const decoder = new TextDecoder();
  let buffer = "";
  let finalData: VoiceTurnData | null = null;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const eventLine = chunk.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = chunk.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const event = eventLine.slice("event:".length).trim();
      const parsed = JSON.parse(dataLine.slice("data:".length).trim()) as VoiceStreamEvent;

      if (event === "stage" && parsed.name) {
        onStage({ name: parsed.name, detail: parsed.detail });
      }
      if (event === "asr") {
        const text = parsed.asr?.text ?? parsed.text;
        if (text?.trim()) onAsr(text.trim());
      }
      if (event === "reply_delta" && parsed.chunk) onReplyDelta(parsed.chunk);
      if (event === "error") throw new Error(parsed.error || "Voice turn failed");
      if (event === "final") finalData = parsed;
    }
  }

  if (!finalData) throw new Error("No model response");
  return finalData;
}

function focusInstruction(focus: VoiceFocus) {
  if (focus === "commerce") {
    return "Commerce mode: understand the shopping request, item, quantity, location, and delivery or pickup need. No live store search is connected yet, so do not invent prices, stores, or availability.";
  }
  return "Health mode: answer as Ghana Health AI with safe health guidance. Do not diagnose or give drug dosage.";
}

function ThinkingDots() {
  return (
    <div className="live-thinking" aria-label="Working" aria-live="polite">
      <span className="typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

function CommerceAction({
  execution,
  lang,
  busy,
  status,
  onConfirm,
}: {
  execution: CommerceExecution | null;
  lang: string;
  busy: boolean;
  status: string | null;
  onConfirm: (productId: string, quantity: number) => Promise<void>;
}) {
  const product = execution?.products?.[0];
  if (!execution || execution.status !== "ready" || !product) {
    return status ? <p className="commerce-action-status">{status}</p> : null;
  }

  const quantity = parseQuantity(execution.draft?.quantity);
  const name = lang === "en" ? product.nameEn : product.nameTw || product.nameEn;
  const actionLabel = lang === "en" ? "Add to cart" : "Fa ka cart ho";
  const priceLabel = `GH₵ ${product.priceGhs.toFixed(2)} / ${product.unit}`;

  return (
    <div className="commerce-action">
      <div className="commerce-action__text">
        <span>{name}</span>
        <small>{priceLabel}</small>
      </div>
      <button
        type="button"
        className="commerce-action__button"
        disabled={busy}
        onClick={() => onConfirm(product.id, quantity)}
      >
        <ShoppingCart className="h-4 w-4" />
        {busy ? (lang === "en" ? "Adding" : "Ɛreka ho") : actionLabel}
      </button>
      {status && <span className="commerce-action-status">{status}</span>}
    </div>
  );
}

function parseQuantity(value?: string) {
  if (!value) return 1;
  const match = value.match(/\d+/);
  if (!match) return 1;
  const n = Number(match[0]);
  return Number.isFinite(n) && n > 0 ? Math.min(99, Math.round(n)) : 1;
}

export function VoicePanel() {
  const { lang } = useLang();
  const [focus, setFocus] = useState<VoiceFocus>("health");
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [level, setLevel] = useState(0);
  const [, setVadState] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [heard, setHeard] = useState<string | null>(null);
  const [reply, setReply] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [ttsB64, setTtsB64] = useState<string | null>(null);
  const [userMessageId, setUserMessageId] = useState<string | undefined>();
  const [correctionOpen, setCorrectionOpen] = useState(false);
  const [correctionText, setCorrectionText] = useState("");
  const [shareAudioConsent, setShareAudioConsent] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [commerceExecution, setCommerceExecution] = useState<CommerceExecution | null>(null);
  const [commerceStatus, setCommerceStatus] = useState<string | null>(null);
  const [confirmingCommerce, setConfirmingCommerce] = useState(false);
  // A/B: Whisper v6 (default, production) vs DONDO CTC (research endpoint)
  const [asrModel, changeAsrModel] = useAsrModel();
  const [understandingModelMode, changeUnderstandingModelMode] = useUnderstandingModelMode();

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastAudioBlobRef = useRef<Blob | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const recordAbortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const recordingRef = useRef(false);
  const speakingRef = useRef(false);
  const listenRef = useRef<() => void>(() => undefined);
  const homeSessionKey = `gha:home-voice:${focus}:${lang}`;

  const orbMode: OrbMode = recording
    ? "listening"
    : speaking
      ? "speaking"
      : busy
        ? "thinking"
        : "idle";

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);

  useEffect(() => {
    speakingRef.current = speaking;
  }, [speaking]);

  useEffect(() => {
    let cancelled = false;
    const hydrate = async () => {
      const storedConversationId = window.localStorage.getItem(homeSessionKey);
      await Promise.resolve();
      if (cancelled) return;
      setConversationId(storedConversationId ?? undefined);
      setHeard(null);
      setReply(null);
      setMessages([]);
      setStatus(null);
      setTtsB64(null);
      setUserMessageId(undefined);
      setCorrectionOpen(false);
      setCorrectionText("");
      setFeedbackStatus(null);
      setFeedbackSent(false);
      setCommerceExecution(null);
      setCommerceStatus(null);
      setConfirmingCommerce(false);
      if (!storedConversationId) return;

      try {
        const res = await fetch(`/api/chat?conversationId=${encodeURIComponent(storedConversationId)}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Could not load session");
        const messages = data.messages as StoredMessage[];
        if (cancelled) return;
        registerConversation(storedConversationId);
        const lastUser = [...messages].reverse().find((message) => message.role === "USER");
        const lastAssistant = [...messages].reverse().find((message) => message.role === "ASSISTANT");
        setMessages(
          messages.flatMap((message) =>
            message.role === "USER" || message.role === "ASSISTANT"
              ? [{ id: message.id, role: message.role, content: message.content, understanding: message.metadata }]
              : [],
          ),
        );
        setHeard(lastUser?.content ?? null);
        setCorrectionText(lastUser?.content ?? "");
        setUserMessageId(lastUser?.id);
        setReply(lastAssistant?.content ?? null);
      } catch {
        if (cancelled) return;
        window.localStorage.removeItem(homeSessionKey);
        setConversationId(undefined);
        setMessages([]);
      }
    };

    void hydrate();

    return () => {
      cancelled = true;
    };
  }, [homeSessionKey]);

  useEffect(() => {
    const onConversationChange = (event: Event) => {
      const detail = (event as CustomEvent<ConversationChange>).detail;
      if (detail?.type === "new" || (detail?.type === "delete" && detail.conversationId === conversationId)) {
        setConversationId(undefined);
        setMessages([]);
        setHeard(null);
        setReply(null);
        setStatus(null);
      }
    };
    window.addEventListener(CONVERSATION_EVENT, onConversationChange);
    return () => window.removeEventListener(CONVERSATION_EVENT, onConversationChange);
  }, [conversationId]);

  function stopPlaybackMeter() {
    if (playbackFrameRef.current != null) {
      window.cancelAnimationFrame(playbackFrameRef.current);
      playbackFrameRef.current = null;
    }
    const ctx = playbackCtxRef.current;
    playbackCtxRef.current = null;
    if (ctx && ctx.state !== "closed") void ctx.close();
    setLevel(0);
  }

  function startPlaybackMeter(audio: HTMLAudioElement) {
    stopPlaybackMeter();
    const ctx = new AudioContext();
    const source = ctx.createMediaElementSource(audio);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.62;
    source.connect(analyser);
    analyser.connect(ctx.destination);
    playbackCtxRef.current = ctx;

    const data = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const sample = data[i] ?? 0;
        sum += sample * sample;
      }
      setLevel(Math.sqrt(sum / data.length));
      playbackFrameRef.current = window.requestAnimationFrame(tick);
    };
    tick();
  }

  async function playWav(b64: string) {
    audioRef.current?.pause();
    stopPlaybackMeter();
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audioRef.current = audio;
    setSpeaking(true);
    audio.onended = () => {
      stopPlaybackMeter();
      setSpeaking(false);
    };
    audio.onerror = () => {
      stopPlaybackMeter();
      setSpeaking(false);
    };
    startPlaybackMeter(audio);
    await playbackCtxRef.current?.resume().catch(() => undefined);
    void audio.play().catch(() => {
      stopPlaybackMeter();
      setSpeaking(false);
    });
  }

  function stopPlayback() {
    audioRef.current?.pause();
    audioRef.current = null;
    stopPlaybackMeter();
    setSpeaking(false);
  }

  function stopRecording() {
    recordAbortRef.current?.abort();
  }

  async function listenAndRespond() {
    if (busyRef.current || speakingRef.current || recordingRef.current) return;
    setStatus(null);
    setHeard(null);
    setReply(null);
    setTtsB64(null);
    setUserMessageId(undefined);
    setCorrectionOpen(false);
    setCorrectionText("");
    setFeedbackStatus(null);
    setFeedbackSent(false);
    setCommerceExecution(null);
    setCommerceStatus(null);
    setConfirmingCommerce(false);
    setRecording(true);
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

      if (!speechDetected) return;
      if (blob.size < 400 || peakLevel < 0.01) throw new Error("Move closer and try again");
      if (durationMs < 400) throw new Error("Say a little more");
      lastAudioBlobRef.current = blob;

      setBusy(true);

      const form = new FormData();
      form.append("audio", blob, "utterance.webm");
      form.append("language", lang);
      form.append("speak", "true");
      form.append("focus", focus);
      form.append("instruction", focusInstruction(focus));
      form.append("understandingModelMode", understandingModelMode);
      if (asrModel === "dondo") form.append("asrModel", "dondo");
      if (conversationId) form.append("conversationId", conversationId);

      const res = await fetch("/api/voice/converse/stream", { method: "POST", body: form });
      if (!res.ok) throw new Error("Couldn’t start voice turn");

      let streamedReply = "";
      const localUserId = crypto.randomUUID();
      const localAssistantId = crypto.randomUUID();
      const data = await readVoiceStream(
        res,
        (text) => {
          setHeard(text);
          setMessages((current) =>
            upsertVoiceMessage(current, { id: localUserId, role: "USER", content: text }),
          );
        },
        (chunk) => {
          streamedReply += chunk;
          setReply(streamedReply);
          setMessages((current) =>
            upsertVoiceMessage(current, {
              id: localAssistantId,
              role: "ASSISTANT",
              content: streamedReply,
            }),
          );
        },
        () => {},
      );

      const text = data.asr?.text?.trim();
      if (!text) throw new Error("I didn’t catch that");

      setConversationId(data.conversationId);
      if (data.conversationId) {
        window.localStorage.setItem(homeSessionKey, data.conversationId);
        registerConversation(data.conversationId);
      }
      setUserMessageId(data.userMessage?.id);
      setHeard(text);
      setCorrectionText(text);
      const finalReply = data.message?.content ?? data.understanding?.reply ?? streamedReply;
      setReply(finalReply);
      setMessages((current) => {
        let next = upsertVoiceMessage(
          current,
          { id: data.userMessage?.id ?? localUserId, role: "USER", content: text },
          localUserId,
        );
        if (finalReply.trim()) {
          next = upsertVoiceMessage(
            next,
            { id: data.message?.id ?? localAssistantId, role: "ASSISTANT", content: finalReply, understanding: data.understanding },
            localAssistantId,
          );
        }
        return next;
      });
      setCommerceExecution(data.understanding?.commerceExecution ?? null);
      if (data.tts?.audioBase64) {
        setTtsB64(data.tts.audioBase64);
        playWav(data.tts.audioBase64);
      }
    } catch (e) {
      setStatus(friendlyVoiceError(e));
    } finally {
      recordAbortRef.current = null;
      setBusy(false);
      setRecording(false);
      setVadState(null);
      setLevel(0);
    }
  }

  useEffect(() => {
    listenRef.current = () => {
      void listenAndRespond();
    };
  });

  async function sendText() {
    const text = input.trim();
    if (!text || busy || recording || speaking) return;
    const localUserId = crypto.randomUUID();
    const localAssistantId = crypto.randomUUID();
    setInput("");
    setStatus(null);
    setHeard(text);
    setReply(null);
    setMessages((current) => [...current, { id: localUserId, role: "USER", content: text }]);
    setBusy(true);
    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversationId,
          language: lang,
          speak: true,
          understandingModelMode,
        }),
      });
      if (!res.ok) throw new Error("Couldn’t start chat turn");
      let streamedReply = "";
      const data = await readVoiceStream(
        res,
        () => {},
        (chunk) => {
          streamedReply += chunk;
          setReply(streamedReply);
          setMessages((current) => upsertVoiceMessage(current, {
            id: localAssistantId,
            role: "ASSISTANT",
            content: streamedReply,
          }));
        },
        () => {},
      );
      const finalReply = data.message?.content ?? data.understanding?.reply ?? streamedReply;
      setConversationId(data.conversationId);
      if (data.conversationId) {
        window.localStorage.setItem(homeSessionKey, data.conversationId);
        registerConversation(data.conversationId);
      }
      setReply(finalReply);
      setMessages((current) => upsertVoiceMessage(current, {
        id: data.message?.id ?? localAssistantId,
        role: "ASSISTANT",
        content: finalReply,
        understanding: data.understanding,
      }, localAssistantId));
      if (data.tts?.audioBase64) {
        setTtsB64(data.tts.audioBase64);
        playWav(data.tts.audioBase64);
      }
    } catch (error) {
      setStatus(friendlyVoiceError(error));
    } finally {
      setBusy(false);
    }
  }

  async function submitTranscriptCorrection() {
    const corrected = correctionText.trim();
    if (!conversationId || !heard || !corrected || corrected === heard.trim()) {
      setCorrectionOpen(false);
      return;
    }
    setFeedbackStatus(null);
    try {
      const blob = lastAudioBlobRef.current;
      let res: Response;
      if (shareAudioConsent && blob) {
        // Explicit opt-in: send the clip so corrections carry real audio.
        const form = new FormData();
        form.append("audio", blob, "correction.webm");
        form.append("audioConsent", "true");
        form.append("conversationId", conversationId);
        if (userMessageId) form.append("messageId", userMessageId);
        form.append("originalTranscript", heard);
        form.append("correctedTranscript", corrected);
        form.append("language", lang);
        form.append("focus", focus);
        form.append("rating", "3");
        res = await fetch("/api/voice/feedback", { method: "POST", body: form });
      } else {
        res = await fetch("/api/voice/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversationId,
            messageId: userMessageId,
            originalTranscript: heard,
            correctedTranscript: corrected,
            language: lang,
            focus,
            rating: 3,
          }),
        });
      }
      if (!res.ok) throw new Error("Correction failed");
      setHeard(corrected);
      setMessages((current) =>
        current.map((message) =>
          message.id === userMessageId ? { ...message, content: corrected } : message,
        ),
      );
      setCorrectionOpen(false);
      setFeedbackSent(true);
      setFeedbackStatus("Saved");
      setShareAudioConsent(false);
    } catch (error) {
      setFeedbackStatus(friendlyVoiceError(error));
    }
  }

  const latestAssistantId = [...messages].reverse().find((message) => message.role === "ASSISTANT")?.id;
  const showListening = recording && !status;
  const showTranscribing = busy && !heard && !status;
  const showThinking = busy && Boolean(heard) && !reply && !status;
  const showConversation =
    showListening || showTranscribing || showThinking || messages.length > 0 || heard || reply || status;

  return (
    <section className={`live-shell fade-up ${showConversation ? "live-shell--active" : ""}`}>
      <div className="live-tabs" role="tablist" aria-label="Conversation focus">
        {(["health", "commerce"] as const).map((item) => (
          <button
            key={item}
            type="button"
            role="tab"
            aria-selected={focus === item}
            className={`live-tab ${focus === item ? "live-tab--active" : ""}`}
            onClick={() => {
              setFocus(item);
              setHeard(null);
              setReply(null);
              setMessages([]);
              setStatus(null);
              setUserMessageId(undefined);
              setCorrectionOpen(false);
              setCorrectionText("");
              setFeedbackStatus(null);
              setFeedbackSent(false);
              setCommerceExecution(null);
              setCommerceStatus(null);
              setConfirmingCommerce(false);
            }}
          >
            {item === "health" ? "Health" : "Commerce"}
          </button>
        ))}
      </div>

      <div className="live-stage">
        <VoiceOrb
          mode={orbMode}
          level={level}
          size={showConversation ? "md" : "lg"}
          onClick={() => {
            if (recording) {
              stopRecording();
              return;
            }
            if (speaking) {
              stopPlayback();
              return;
            }
            if (!busy) void listenAndRespond();
          }}
          label="Speak"
          showLabel={false}
        />
      </div>

      <form
        className="live-text-composer"
        onSubmit={(event) => {
          event.preventDefault();
          void sendText();
        }}
      >
        <label className="sr-only" htmlFor="home-chat-input">Type a message</label>
        <input
          id="home-chat-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={lang === "en" ? "Type or speak…" : "Twerɛ anaa kasa…"}
          disabled={busy || recording}
        />
        <button type="submit" className="icon-action" disabled={!input.trim() || busy || recording} aria-label="Send message">
          <Send className="h-4 w-4" />
        </button>
      </form>

      {showConversation && (
        <div className="live-conversation" aria-live="polite">
          <div className="live-thread">
            {messages.map((message) => {
              const isCurrentUser = message.role === "USER" && message.id === userMessageId;
              if (message.role === "USER") {
                return (
                  <div key={message.id} className="live-heard-wrap">
                    {isCurrentUser && correctionOpen ? (
                      <form
                        className="transcript-correction"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void submitTranscriptCorrection();
                        }}
                      >
                        <input
                          value={correctionText}
                          onChange={(event) => setCorrectionText(event.target.value)}
                          aria-label="Correct transcript"
                          autoFocus
                        />
                        <button type="submit" className="mini-action" aria-label="Save correction">
                          <Check className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          className="mini-action"
                          aria-label="Cancel correction"
                          onClick={() => {
                            setCorrectionOpen(false);
                            setCorrectionText(heard ?? message.content);
                            setFeedbackStatus(null);
                            setShareAudioConsent(false);
                          }}
                        >
                          <X className="h-4 w-4" />
                        </button>
                        <label className="transcript-correction__consent">
                          <input
                            type="checkbox"
                            checked={shareAudioConsent}
                            onChange={(event) => setShareAudioConsent(event.target.checked)}
                          />
                          {lang === "en"
                            ? "Share this recording to improve the model"
                            : "Ma yɛmfa nne a wokae no nsiɛ model no"}
                        </label>
                      </form>
                    ) : (
                      <>
                        <p className="live-heard">{message.content}</p>
                        {isCurrentUser && !feedbackSent && (
                          <button
                            type="button"
                            className="transcript-correct-button"
                            onClick={() => {
                              setCorrectionText(message.content);
                              setCorrectionOpen(true);
                              setFeedbackStatus(null);
                            }}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                            Correct
                          </button>
                        )}
                        {isCurrentUser && feedbackStatus && (
                          <span className="transcript-feedback">{feedbackStatus}</span>
                        )}
                      </>
                    )}
                  </div>
                );
              }

              return (
                <div key={message.id} className="live-reply-wrap">
                  <div className="min-w-0">
                  <p className="live-reply">{message.content}</p>
                  <UnderstandingDetails data={message.understanding} />
                  </div>
                  {ttsB64 && message.id === latestAssistantId && (
                    <button
                      type="button"
                      className="icon-action"
                      aria-label={speaking ? "Stop reading" : "Read aloud"}
                      title={speaking ? "Stop reading" : "Read aloud"}
                      onClick={() => speaking ? stopPlayback() : playWav(ttsB64)}
                    >
                      {speaking ? <X className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                    </button>
                  )}
                </div>
              );
            })}
            {showListening && <ThinkingDots />}
            {showTranscribing && <ThinkingDots />}
            {showThinking && <ThinkingDots />}
            {reply && (
              <CommerceAction
                execution={commerceExecution}
                lang={lang}
                busy={confirmingCommerce}
                status={commerceStatus}
                onConfirm={async (productId, quantity) => {
                  setConfirmingCommerce(true);
                  setCommerceStatus(null);
                  try {
                    const res = await fetch("/api/commerce/confirm", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        confirm: true,
                        productId,
                        quantity,
                        source: "voice",
                      }),
                    });
                    if (!res.ok) throw new Error("Could not add to cart");
                    setCommerceStatus(lang === "en" ? "Added to cart" : "Wɔde aka cart no ho");
                  } catch (error) {
                    setCommerceStatus(friendlyVoiceError(error));
                  } finally {
                    setConfirmingCommerce(false);
                  }
                }}
              />
            )}
          </div>
          {status && !reply && <p className="live-error">{status}</p>}
          <div className="live-action-row">
            <label className="voice-model-picker" aria-label="Speech model">
              <span>ASR</span>
              <select value={asrModel} onChange={(event) => changeAsrModel(event.target.value as typeof asrModel)}>
                <option value="dondo">DONDO</option>
                <option value="v6">v6</option>
              </select>
            </label>
            <label className="voice-model-picker" aria-label="Understanding model mode">
              <span>Mind</span>
              <select
                value={understandingModelMode}
                onChange={(event) =>
                  changeUnderstandingModelMode(event.target.value as typeof understandingModelMode)
                }
              >
                <option value="shadow">Stable</option>
                <option value="assist">Research v0</option>
                <option value="assist_v1">Research v1</option>
              </select>
            </label>
            <button
              type="button"
              className={recording ? "icon-action chat-mic-action chat-mic-action--live" : "icon-action chat-mic-action"}
              disabled={busy || speaking}
              aria-label={recording ? "Stop recording" : "Record voice"}
              onClick={() => {
                if (recording) {
                  stopRecording();
                  return;
                }
                void listenAndRespond();
              }}
            >
              <Mic className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
