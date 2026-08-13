/**
 * Client-side mic capture with end-of-speech (VAD) detection.
 */

export type LiveRecorder = {
  stop: () => Promise<Blob>;
  stream: MediaStream;
};

export type VadRecorderOptions = {
  /** RMS below this = silence (0–1 scale after Analyser normalisation) */
  silenceThreshold?: number;
  /** Consecutive quiet ms before auto-stop (after speech was heard) */
  silenceMs?: number;
  /** Max recording length */
  maxMs?: number;
  /** Require this much voiced audio before silence can end the turn */
  minSpeechMs?: number;
  onLevel?: (level: number) => void;
  onState?: (state: "listening" | "speech" | "silence" | "done") => void;
  signal?: AbortSignal;
  previewEveryMs?: number;
  previewMinMs?: number;
  onPreviewBlob?: (blob: Blob, elapsedMs: number) => void;
};

const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
    // Prefer 16 kHz if the browser honors it
    sampleRate: 16000,
  },
};

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "audio/webm";
  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
    return "audio/webm;codecs=opus";
  }
  if (MediaRecorder.isTypeSupported("audio/webm")) return "audio/webm";
  if (MediaRecorder.isTypeSupported("audio/mp4")) return "audio/mp4";
  return "";
}

export async function startLiveRecorder(): Promise<LiveRecorder> {
  const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  const mimeType = pickMimeType();
  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(
    stream,
    mimeType ? { mimeType, audioBitsPerSecond: 64_000 } : { audioBitsPerSecond: 64_000 },
  );
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  recorder.start(250);

  return {
    stream,
    stop: () =>
      new Promise((resolve, reject) => {
        recorder.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(chunks, { type: mimeType || "audio/webm" }));
        };
        recorder.onerror = () => {
          stream.getTracks().forEach((t) => t.stop());
          reject(new Error("Recording failed"));
        };
        if (recorder.state === "recording") recorder.stop();
        else {
          stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(chunks, { type: mimeType || "audio/webm" }));
        }
      }),
  };
}

/**
 * Record until the user stops speaking (silence after speech) or maxMs.
 * Returns webm/opus blob suitable for Modal ASR.
 */
export async function recordUntilSilence(
  opts: VadRecorderOptions = {},
): Promise<{ blob: Blob; durationMs: number; peakLevel: number; speechDetected: boolean }> {
  const silenceThreshold = opts.silenceThreshold ?? 0.018;
  const silenceMs = opts.silenceMs ?? 1100;
  const maxMs = opts.maxMs ?? 20_000;
  const minSpeechMs = opts.minSpeechMs ?? 400;
  const previewEveryMs = opts.previewEveryMs ?? 3500;
  const previewMinMs = opts.previewMinMs ?? 2500;

  const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  const mimeType = pickMimeType();
  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(
    stream,
    mimeType ? { mimeType, audioBitsPerSecond: 64_000 } : { audioBitsPerSecond: 64_000 },
  );
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.5;
  source.connect(analyser);
  const timeData = new Float32Array(analyser.fftSize);

  const startedAt = performance.now();
  let speechStartedAt: number | null = null;
  let lastLoudAt = startedAt;
  let lastPreviewAt = startedAt;
  let peakLevel = 0;
  let finished = false;

  opts.onState?.("listening");
  recorder.start(200);

  const rms = () => {
    analyser.getFloatTimeDomainData(timeData);
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) {
      const v = timeData[i]!;
      sum += v * v;
    }
    return Math.sqrt(sum / timeData.length);
  };

  return new Promise((resolve, reject) => {
    const finish = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(tick);
      opts.signal?.removeEventListener("abort", finish);
      opts.onState?.("done");
      const stopRec = () => {
        stream.getTracks().forEach((t) => t.stop());
        void audioCtx.close();
        const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
        resolve({
          blob,
          durationMs: performance.now() - startedAt,
          peakLevel,
          speechDetected: speechStartedAt != null,
        });
      };
      if (recorder.state === "recording") {
        recorder.onstop = stopRec;
        recorder.stop();
      } else {
        stopRec();
      }
    };

    recorder.onerror = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(tick);
      opts.signal?.removeEventListener("abort", finish);
      stream.getTracks().forEach((t) => t.stop());
      void audioCtx.close();
      reject(new Error("Recording failed"));
    };

    opts.signal?.addEventListener("abort", finish, { once: true });

    const tick = window.setInterval(() => {
      const level = rms();
      peakLevel = Math.max(peakLevel, level);
      opts.onLevel?.(level);

      const now = performance.now();
      if (level >= silenceThreshold) {
        if (speechStartedAt == null) {
          speechStartedAt = now;
          opts.onState?.("speech");
        }
        lastLoudAt = now;
      } else if (speechStartedAt != null) {
        opts.onState?.("silence");
        const spoken = lastLoudAt - speechStartedAt;
        const quiet = now - lastLoudAt;
        if (spoken >= minSpeechMs && quiet >= silenceMs) {
          finish();
          return;
        }
      }

      if (
        opts.onPreviewBlob &&
        speechStartedAt != null &&
        now - startedAt >= previewMinMs &&
        now - lastPreviewAt >= previewEveryMs &&
        chunks.length > 0
      ) {
        lastPreviewAt = now;
        opts.onPreviewBlob(
          new Blob([...chunks], { type: mimeType || "audio/webm" }),
          now - startedAt,
        );
      }

      if (now - startedAt >= maxMs) finish();
    }, 50);
  });
}

/** Decode any browser-supported audio blob → mono float32 @ 16 kHz + base64 PCM. */
export async function blobToPcmB64(
  blob: Blob,
): Promise<{ pcmB64: string; sampleRate: number; durationS: number }> {
  const ctx = new AudioContext();
  try {
    const raw = await blob.arrayBuffer();
    const decoded = await ctx.decodeAudioData(raw.slice(0));
    const channel = decoded.getChannelData(0);
    const targetSr = 16000;
    const ratio = decoded.sampleRate / targetSr;
    const outLen = Math.max(1, Math.floor(channel.length / ratio));
    const mono = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(channel.length - 1, i0 + 1);
      const t = srcIdx - i0;
      mono[i] = channel[i0]! * (1 - t) + channel[i1]! * t;
    }
    return {
      pcmB64: float32ToB64(mono),
      sampleRate: targetSr,
      durationS: outLen / targetSr,
    };
  } finally {
    await ctx.close();
  }
}

function float32ToB64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const slice = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode.apply(null, slice as unknown as number[]);
  }
  return btoa(binary);
}
