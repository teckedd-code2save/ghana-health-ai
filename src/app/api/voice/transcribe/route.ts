import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { jsonOk, jsonError } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";

/**
 * Real voice transcription via Modal Twi Whisper.
 * No stub phrase rotation — audio is required.
 */
export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice:${ip}`, 20, 60);
    if (!rl.allowed) return jsonError("Too many voice requests", 429);

    if (!isModalAsrConfigured()) {
      return jsonError("MODAL_ASR_URL not configured — real ASR required", 503);
    }

    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    let audio: ArrayBuffer | null = null;
    let filename: string | undefined;
    let audioCt: string | undefined;
    let language: string | undefined;

    if (contentType.includes("multipart/form-data")) {
      const form = await req.formData();
      const file = form.get("audio");
      if (file instanceof Blob) {
        audio = await file.arrayBuffer();
        audioCt = file.type || undefined;
        if ("name" in file && typeof (file as File).name === "string") {
          filename = (file as File).name;
        }
      }
      const lang = form.get("language");
      if (typeof lang === "string") language = lang;
    } else if (
      contentType.includes("application/octet-stream") ||
      contentType.startsWith("audio/")
    ) {
      audio = await req.arrayBuffer();
      audioCt = contentType;
    }

    if (!audio || audio.byteLength < 100) {
      return jsonError("Audio required for transcription", 400);
    }

    const result = await modalTranscribe(audio, {
      contentType: audioCt,
      filename,
      language: language === "en" || user?.preferredLang === "en" ? "en" : undefined,
    });

    if (result.error && !result.text?.trim()) {
      return jsonError(result.error, 422, { model: result.model });
    }

    await prisma.transcriptSession.create({
      data: {
        userId: user?.id,
        language: result.language === "en" ? "en" : "tw",
        rawText: result.text,
        punctuated: result.text,
        speakers: [{ label: result.speaker ?? "Speaker 1", verified: result.verified ?? null }],
        audioDeleted: true,
        retained: false,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    await writeAudit({
      action: "voice.transcribe",
      actorId: user?.id,
      ip,
      meta: { mode: "modal", model: result.model, latencyMs: result.latency_ms },
    });

    return jsonOk({
      transcript: {
        text: result.text,
        speaker: result.speaker ?? "Speaker 1 (User)",
        language: result.language === "en" ? "en" : "tw",
        verified: result.verified ?? null,
        latencyMs: result.latency_ms,
        mode: "modal" as const,
        model: result.model,
        segments: result.segments,
      },
    });
  } catch (e) {
    console.error(e);
    return jsonError(e instanceof Error ? e.message : "Transcription failed", 500);
  }
}
