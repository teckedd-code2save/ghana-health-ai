"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, ShieldCheck, Square, Volume2 } from "lucide-react";
import { useLang } from "@/components/lang-provider";

type ConverseResult = {
  asr?: { text: string; model: string; latencyMs: number };
  understanding?: {
    reply: string;
    intent: string;
    severity: string;
    escalate: boolean;
    engine: string;
  };
  tts?: { audioBase64: string; sampleRate?: number; model?: string; latencyMs?: number } | null;
  totalLatencyMs?: number;
  conversationId?: string;
  error?: string;
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
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [result, setResult] = useState<ConverseResult | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [speak, setSpeak] = useState(true);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function runConverse(blob: Blob) {
    setBusy(true);
    setStatus("ASR → understand → TTS…");
    setResult(null);
    try {
      const form = new FormData();
      form.append("audio", blob, "utterance.webm");
      form.append("language", lang);
      form.append("speak", speak ? "true" : "false");
      if (conversationId) form.append("conversationId", conversationId);

      const res = await fetch("/api/voice/converse", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Converse failed");

      setConversationId(data.conversationId);
      setResult(data);
      setStatus(
        `Done in ${data.totalLatencyMs}ms · ASR ${data.asr?.model} · brain ${data.understanding?.engine}`,
      );
      if (data.tts?.audioBase64) playWavBase64(data.tts.audioBase64);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function startRecording() {
    setStatus(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        void runConverse(blob);
      };
      mediaRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Mic permission denied");
    }
  }

  function stopRecording() {
    mediaRef.current?.stop();
    mediaRef.current = null;
    setRecording(false);
  }

  // Voice ID stubs kept minimal
  const [passphrase, setPassphrase] = useState("Me din de Ama Mensah twi");
  const [vidStatus, setVidStatus] = useState<string | null>(null);

  useEffect(() => {
    /* warm nothing — Modal cold-starts on first real call */
  }, []);

  async function enrollVoice() {
    setVidStatus(null);
    const res = await fetch("/api/voice/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passphrase, language: "tw" }),
    });
    const data = await res.json();
    setVidStatus(res.ok ? "Enrolled (embedding stub)" : data.error);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="glass rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <Mic className="h-5 w-5 text-[var(--teal)]" />
          <h2 className="font-[family-name:var(--font-display)] text-xl">Live conversation</h2>
        </div>
        <p className="mb-4 text-sm text-[var(--fg-muted)]">
          Real pipeline: <strong className="text-[var(--accent-soft)]">Twi ASR</strong> (Whisper
          Waxal fine-tune) → <strong className="text-[var(--accent-soft)]">LLM understand</strong> →{" "}
          <strong className="text-[var(--accent-soft)]">Akan TTS</strong> (MMS-TTS-aka).
        </p>

        <label className="mb-4 flex items-center gap-2 text-sm text-[var(--fg-muted)]">
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
          Speak reply (TTS)
        </label>

        <div className="flex flex-wrap gap-2">
          {!recording ? (
            <button
              onClick={() => void startRecording()}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-full bg-[var(--teal)] px-5 py-3 text-sm font-semibold text-[#062419] disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Mic className="h-4 w-4" />}
              {busy ? "Thinking…" : "Hold to talk — click start"}
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="mic-pulse inline-flex items-center gap-2 rounded-full bg-[var(--coral)] px-5 py-3 text-sm font-semibold text-white"
            >
              <Square className="h-4 w-4" /> Stop &amp; process
            </button>
          )}
        </div>

        {status && <p className="mt-4 text-xs text-[var(--accent-soft)]">{status}</p>}

        {result?.asr && (
          <div className="mt-4 space-y-3 rounded-2xl bg-black/20 p-4 text-sm">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">You said</p>
              <p className="mt-1">{result.asr.text}</p>
              <p className="mt-1 text-[10px] text-[var(--fg-muted)]">
                {result.asr.model} · {result.asr.latencyMs}ms
              </p>
            </div>
            {result.understanding && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
                  Understood · {result.understanding.intent} · {result.understanding.severity}
                  {result.understanding.escalate ? " · ESCALATE" : ""}
                </p>
                <p className="mt-1 whitespace-pre-wrap">{result.understanding.reply}</p>
                <p className="mt-1 text-[10px] text-[var(--fg-muted)]">
                  brain: {result.understanding.engine}
                </p>
              </div>
            )}
            {result.tts?.audioBase64 && (
              <button
                type="button"
                onClick={() => playWavBase64(result.tts!.audioBase64)}
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
          <h2 className="font-[family-name:var(--font-display)] text-xl">Voice ID (light)</h2>
        </div>
        <p className="mb-3 text-sm text-[var(--fg-muted)]">
          Enrollment still uses a local embedding stub. Intelligence work is on ASR/TTS/LLM.
        </p>
        <textarea
          value={passphrase}
          onChange={(e) => setPassphrase(e.target.value)}
          rows={3}
          className="mb-3 w-full rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
        />
        <button
          onClick={() => void enrollVoice()}
          className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[#1a1400]"
        >
          Enroll
        </button>
        {vidStatus && <p className="mt-2 text-sm text-[var(--accent-soft)]">{vidStatus}</p>}
      </section>
    </div>
  );
}
