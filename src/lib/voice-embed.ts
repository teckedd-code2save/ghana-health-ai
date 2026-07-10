/**
 * Lightweight speaker embedding from PCM audio (not a passphrase hash).
 * Frame-level energy bands + ZCR + spectral tilt → L2-normalized vector.
 * Production upgrade path: ECAPA-TDNN / SpeechBrain on Modal.
 */

const EMBED_DIM = 64;
const TARGET_SR = 16000;

export function embeddingToB64(embedding: number[]): string {
  return Buffer.from(Float32Array.from(embedding).buffer).toString("base64");
}

export function embeddingFromB64(b64: string): number[] {
  const buf = Buffer.from(b64, "base64");
  const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return Array.from(arr);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

/** Decode raw base64 float32 little-endian mono PCM (client Web Audio path). */
export function pcmFromB64(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
}

/**
 * Build a fixed-size embedding from mono float samples in [-1, 1].
 * Resamples roughly if sampleRate differs from 16 kHz.
 */
export function embeddingFromPcm(
  samples: Float32Array | number[],
  sampleRate = TARGET_SR,
): number[] {
  const mono = resampleTo16k(samples, sampleRate);
  if (mono.length < 400) {
    throw new Error("Audio too short for voice enrollment (need ~0.5s+)");
  }

  const frameSize = 400; // 25ms @ 16k
  const hop = 160; // 10ms
  const nBands = 12;
  const bandAccum = new Array<number>(nBands).fill(0);
  let zcrSum = 0;
  let rmsSum = 0;
  let tiltSum = 0;
  let frames = 0;

  for (let start = 0; start + frameSize < mono.length; start += hop) {
    let energy = 0;
    let zcr = 0;
    let low = 0;
    let high = 0;
    for (let i = 0; i < frameSize; i++) {
      const x = mono[start + i]!;
      energy += x * x;
      if (i > 0) {
        const prev = mono[start + i - 1]!;
        if ((x >= 0 && prev < 0) || (x < 0 && prev >= 0)) zcr += 1;
      }
      if (i < frameSize / 2) low += x * x;
      else high += x * x;
    }
    const rms = Math.sqrt(energy / frameSize);
    // skip near-silence frames
    if (rms < 0.005) continue;

    // crude band energies via subframe windows
    const sub = Math.floor(frameSize / nBands);
    for (let b = 0; b < nBands; b++) {
      let e = 0;
      const off = start + b * sub;
      for (let i = 0; i < sub; i++) {
        const x = mono[off + i] ?? 0;
        e += x * x;
      }
      bandAccum[b]! += Math.log1p(e);
    }

    zcrSum += zcr / frameSize;
    rmsSum += rms;
    tiltSum += Math.log1p(high) - Math.log1p(low);
    frames += 1;
  }

  if (frames < 5) {
    throw new Error("Not enough voiced audio — speak clearly for 2–4 seconds");
  }

  const feats: number[] = [];
  for (let b = 0; b < nBands; b++) feats.push(bandAccum[b]! / frames);
  feats.push(zcrSum / frames, rmsSum / frames, tiltSum / frames);

  // delta-like: band ratios
  for (let b = 0; b < nBands - 1; b++) {
    feats.push(feats[b + 1]! - feats[b]!);
  }

  // pad / project to EMBED_DIM
  const out = new Array<number>(EMBED_DIM).fill(0);
  for (let i = 0; i < EMBED_DIM; i++) {
    out[i] = feats[i % feats.length]! * (1 + 0.07 * Math.sin(i * 1.7));
  }

  // L2 normalize
  const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0)) || 1;
  return out.map((v) => v / norm);
}

function resampleTo16k(samples: Float32Array | number[], sampleRate: number): Float32Array {
  const src =
    samples instanceof Float32Array ? samples : Float32Array.from(samples);
  if (sampleRate === TARGET_SR) return src;
  const ratio = sampleRate / TARGET_SR;
  const outLen = Math.max(1, Math.floor(src.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(src.length - 1, i0 + 1);
    const t = srcIdx - i0;
    out[i] = src[i0]! * (1 - t) + src[i1]! * t;
  }
  return out;
}
