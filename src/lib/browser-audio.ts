/**
 * Client-side helpers: record mic audio + extract mono PCM for Voice ID.
 */

export type LiveRecorder = {
  stop: () => Promise<Blob>;
  stream: MediaStream;
};

export async function startLiveRecorder(): Promise<LiveRecorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";
  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(stream, { mimeType });
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  recorder.start(100);

  return {
    stream,
    stop: () =>
      new Promise((resolve, reject) => {
        recorder.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(chunks, { type: mimeType }));
        };
        recorder.onerror = () => {
          stream.getTracks().forEach((t) => t.stop());
          reject(new Error("Recording failed"));
        };
        if (recorder.state === "recording") recorder.stop();
        else {
          stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(chunks, { type: mimeType }));
        }
      }),
  };
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
