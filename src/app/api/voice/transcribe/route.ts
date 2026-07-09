import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { stubTranscribeChunk } from "@/lib/voice-stub";
import { jsonOk, jsonError } from "@/lib/api";

/**
 * MVP stub endpoint — accepts optional audio blob, returns fake Twi/EN transcript.
 * Replace with Modal Parakeet streaming when VOICE_MODE=modal.
 */
export async function POST(req: Request) {
  try {
    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    let audio: ArrayBuffer | null = null;

    if (contentType.includes("multipart/form-data")) {
      const form = await req.formData();
      const file = form.get("audio");
      if (file instanceof Blob) audio = await file.arrayBuffer();
    } else if (contentType.includes("application/octet-stream")) {
      audio = await req.arrayBuffer();
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
