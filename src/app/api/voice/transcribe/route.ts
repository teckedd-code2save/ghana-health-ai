import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { stubTranscribeChunk } from "@/lib/voice-stub";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { jsonOk, jsonError } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";

/**
 * Voice transcription:
 * - VOICE_MODE=modal + MODAL_ASR_URL → Modal Twi Whisper GPU
 * - otherwise → local deterministic stub (dev / offline)
 */
export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice:${ip}`, 20, 60);
    if (!rl.allowed) return jsonError("Too many voice requests", 429);

    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    let audio: ArrayBuffer | null = null;
    let filename: string | undefined;
    let audioCt: string | undefined;

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
    } else if (
      contentType.includes("application/octet-stream") ||
      contentType.startsWith("audio/")
    ) {
      audio = await req.arrayBuffer();
      audioCt = contentType;
    }

    if (isModalAsrConfigured() && audio && audio.byteLength > 0) {
      try {
        const result = await modalTranscribe(audio, {
          contentType: audioCt,
          filename,
          language: user?.preferredLang === "en" ? "en" : undefined,
        });

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
      } catch (modalErr) {
        console.error("Modal ASR failed, falling back to stub:", modalErr);
        // fall through to stub so UX never hard-fails
      }
    }

    const result = await stubTranscribeChunk(audio, user?.id);

    await prisma.transcriptSession.create({
      data: {
        userId: user?.id,
        language: result.language,
        rawText: result.text,
        punctuated: result.text,
        speakers: [{ label: result.speaker, verified: result.verified }],
        audioDeleted: true,
        retained: false,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    return jsonOk({ transcript: result });
  } catch (e) {
    console.error(e);
    return jsonError("Transcription failed", 500);
  }
}
