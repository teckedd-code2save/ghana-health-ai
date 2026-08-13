"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Pencil, ShoppingCart, Volume2, X } from "lucide-react";
import { useLang } from "@/components/lang-provider";
import { recordUntilSilence } from "@/lib/browser-audio";
import { VoiceOrb, modeLabel, type OrbMode } from "@/components/voice-orb";

type VoiceFocus = "health" | "commerce";

type VoiceTurnData = {
  conversationId?: string;
  asr?: { text?: string };
  understanding?: {
    reply?: string;
    commerceExecution?: CommerceExecution;
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

function friendlyVoiceError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error || "");
  if (
    message.includes("ECONNREFUSED") ||
    message.includes("Prisma") ||
    message.includes("TURBOPACK") ||
    message.includes("Can't reach database") ||
    message.includes("database is not reachable")
  ) {
    return "Service is starting. Try again in a moment.";
  }
  return message || "Try again";
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

function ThinkingDots({ label }: { label: string }) {
  return (
    <div className="live-thinking" aria-live="polite">
      <span className="live-thinking__label">{label}</span>
      <span className="typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

function stageLabel(stage: string | null) {
  switch (stage) {
    case "accepted":
    case "asr_started":
      return "Transcribing";
    case "asr_final":
    case "conversation":
    case "user_message":
    case "understanding":
      return "Thinking";
    case "assistant_message":
      return "Responding";
    case "tts":
      return "Speaking";
    default:
      return null;
  }
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
  const [vadState, setVadState] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [heard, setHeard] = useState<string | null>(null);
  const [reply, setReply] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [ttsB64, setTtsB64] = useState<string | null>(null);
  const [pipelineStage, setPipelineStage] = useState<string | null>(null);
  const [userMessageId, setUserMessageId] = useState<string | undefined>();
  const [correctionOpen, setCorrectionOpen] = useState(false);
  const [correctionText, setCorrectionText] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [commerceExecution, setCommerceExecution] = useState<CommerceExecution | null>(null);
  const [commerceStatus, setCommerceStatus] = useState<string | null>(null);
  const [confirmingCommerce, setConfirmingCommerce] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const recordAbortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const recordingRef = useRef(false);
  const speakingRef = useRef(false);
  const listenRef = useRef<() => void>(() => undefined);

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
    setPipelineStage(null);
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
      const { blob, durationMs, peakLevel } = await recordUntilSilence({
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

      if (blob.size < 400 || peakLevel < 0.01) throw new Error("Move closer and try again");
      if (durationMs < 400) throw new Error("Say a little more");

      setBusy(true);
      setPipelineStage("asr_started");

      const form = new FormData();
      form.append("audio", blob, "utterance.webm");
      form.append("language", lang);
      form.append("speak", "true");
      form.append("focus", focus);
      form.append("instruction", focusInstruction(focus));
      if (conversationId) form.append("conversationId", conversationId);

      const res = await fetch("/api/voice/converse/stream", { method: "POST", body: form });
      if (!res.ok) throw new Error("Couldn’t start voice turn");

      const data = await readVoiceStream(
        res,
        (text) => {
          setHeard(text);
          setPipelineStage("understanding");
        },
        (chunk) => {
          setPipelineStage("assistant_message");
          setReply((current) => `${current ?? ""}${chunk}`);
        },
        (stage) => setPipelineStage(stage.name),
      );

      const text = data.asr?.text?.trim();
      if (!text) throw new Error("I didn’t catch that");

      setConversationId(data.conversationId);
      setUserMessageId(data.userMessage?.id);
      setHeard(text);
      setCorrectionText(text);
      setReply(data.message?.content ?? data.understanding?.reply ?? "");
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
      setPipelineStage(null);
    }
  }

  useEffect(() => {
    listenRef.current = () => {
      void listenAndRespond();
    };
  });

  async function submitTranscriptCorrection() {
    const corrected = correctionText.trim();
    if (!conversationId || !heard || !corrected || corrected === heard.trim()) {
      setCorrectionOpen(false);
      return;
    }
    setFeedbackStatus(null);
    try {
      const res = await fetch("/api/voice/feedback", {
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
      if (!res.ok) throw new Error("Correction failed");
      setHeard(corrected);
      setCorrectionOpen(false);
      setFeedbackSent(true);
      setFeedbackStatus("Saved");
    } catch (error) {
      setFeedbackStatus(friendlyVoiceError(error));
    }
  }

  const showListening = recording && !heard && !reply && !status;
  const showTranscribing = busy && !heard && !reply && !status;
  const showThinking = busy && Boolean(heard) && !reply && !status;
  const showConversation =
    showListening || showTranscribing || showThinking || heard || reply || status;

  return (
    <section className="live-shell fade-up">
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
              setConversationId(undefined);
              setHeard(null);
              setReply(null);
              setStatus(null);
              setPipelineStage(null);
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
          size="lg"
          onClick={() => {
            if (recording) {
              stopRecording();
              return;
            }
            if (speaking) {
              stopPlayback();
              window.setTimeout(() => void listenAndRespond(), 120);
              return;
            }
            if (!busy) void listenAndRespond();
          }}
          label={status || stageLabel(pipelineStage) || modeLabel(orbMode, vadState)}
        />
      </div>

      {showConversation && (
        <div className="live-conversation" aria-live="polite">
          {heard && (
            <div className="live-heard-wrap">
              {correctionOpen ? (
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
                      setCorrectionText(heard);
                      setFeedbackStatus(null);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </button>
                </form>
              ) : (
                <>
                  <p className="live-heard">{heard}</p>
                  {!feedbackSent && (
                    <button
                      type="button"
                      className="transcript-correct-button"
                      onClick={() => {
                        setCorrectionText(heard);
                        setCorrectionOpen(true);
                        setFeedbackStatus(null);
                      }}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                      Correct
                    </button>
                  )}
                  {feedbackStatus && <span className="transcript-feedback">{feedbackStatus}</span>}
                </>
              )}
            </div>
          )}
          {showListening && <ThinkingDots label="Listening" />}
          {showTranscribing && <ThinkingDots label={stageLabel(pipelineStage) || "Transcribing"} />}
          {showThinking && <ThinkingDots label={stageLabel(pipelineStage) || "Thinking"} />}
          {reply && (
            <>
              <div className="live-reply-wrap">
                <p className="live-reply">{reply}</p>
                {ttsB64 && (
                  <button
                    type="button"
                    className="icon-action"
                    aria-label="Play again"
                    onClick={() => playWav(ttsB64)}
                  >
                    <Volume2 className="h-4 w-4" />
                  </button>
                )}
              </div>
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
            </>
          )}
          {status && !reply && <p className="live-error">{status}</p>}
        </div>
      )}
    </section>
  );
}
