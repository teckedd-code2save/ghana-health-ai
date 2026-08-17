import { z } from "zod";
import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { writeAudit } from "@/lib/audit";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  conversationId: z.string().uuid(),
  messageId: z.string().uuid().optional(),
  originalTranscript: z.string().min(1).max(4000),
  correctedTranscript: z.string().trim().min(1).max(4000).optional(),
  rating: z.number().int().min(1).max(5).optional(),
  language: z.enum(["tw", "en", "ga"]).optional(),
  focus: z.enum(["health", "commerce"]).optional(),
  notes: z.string().trim().max(800).optional(),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice-feedback:${ip}`, 30, 60);
    if (!rl.allowed) return jsonError("Too many feedback updates — wait a minute", 429);

    // JSON (text-only) or multipart (with consented audio) — consent is
    // explicit per correction; audio is stored only when opted in.
    const contentType = req.headers.get("content-type") ?? "";
    let body: z.infer<typeof schema>;
    let audioBlob: Blob | null = null;
    let audioConsent = false;

    if (contentType.includes("multipart/form-data")) {
      const form = await req.formData();
      const file = form.get("audio");
      if (file instanceof Blob && file.size > 0) audioBlob = file;
      audioConsent = form.get("audioConsent") === "true";
      body = schema.parse({
        conversationId: stringValue(form.get("conversationId")),
        messageId: stringValue(form.get("messageId")),
        originalTranscript: stringValue(form.get("originalTranscript")),
        correctedTranscript: stringValue(form.get("correctedTranscript")),
        rating: form.get("rating") ? Number(form.get("rating")) : undefined,
        language: stringValue(form.get("language")),
        focus: stringValue(form.get("focus")),
        notes: stringValue(form.get("notes")),
      });
    } else {
      body = schema.parse(await req.json());
    }

    const user = await getSessionUser();

    const conversation = await prisma.conversation.findUnique({
      where: { id: body.conversationId },
      select: { id: true, userId: true },
    });
    if (!conversation) return jsonError("Conversation not found", 404);
    if (conversation.userId && user?.id !== conversation.userId) {
      return jsonError("Conversation not found", 404);
    }

    const message = body.messageId
      ? await prisma.message.findFirst({
          where: {
            id: body.messageId,
            conversationId: body.conversationId,
            role: "USER",
          },
          select: {
            id: true,
            language: true,
            metadata: true,
          },
        })
      : null;
    if (body.messageId && !message) return jsonError("Transcript message not found", 404);

    const metadata = isRecord(message?.metadata) ? message.metadata : {};
    const feedback = await prisma.asrFeedback.create({
      data: {
        userId: user?.id,
        conversationId: body.conversationId,
        messageId: message?.id,
        language: body.language ?? message?.language ?? user?.preferredLang ?? "tw",
        focus: body.focus,
        originalTranscript: body.originalTranscript,
        correctedTranscript: body.correctedTranscript,
        rating: body.rating,
        notes: body.notes,
        asrModel: stringValue(metadata.asrModel),
        asrRoute: stringValue(metadata.asrRoute),
        metadata: {
          source: "voice-ui",
          hasCorrection: Boolean(body.correctedTranscript),
          rating: body.rating ?? null,
        },
      },
    });

    let audioPath: string | undefined;
    if (audioConsent && audioBlob) {
      audioPath = (await storeConsentedAudio(feedback.id, audioBlob)) ?? undefined;
      if (audioPath) {
        await prisma.asrFeedback.update({
          where: { id: feedback.id },
          data: { audioConsent: true, audioPath },
        });
      }
    }

    await writeAudit({
      action: "voice.asr_feedback",
      actorId: user?.id,
      entityType: "asr_feedback",
      entityId: feedback.id,
      ip,
      meta: {
        conversationId: body.conversationId,
        messageId: message?.id,
        language: feedback.language,
        focus: body.focus,
        hasCorrection: Boolean(body.correctedTranscript),
        rating: body.rating ?? null,
        hasConsentedAudio: Boolean(audioPath),
      },
    });

    return jsonOk({ ok: true, feedbackId: feedback.id, hasConsentedAudio: Boolean(audioPath) });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return jsonError(error.issues[0]?.message ?? "Invalid feedback");
    }
    console.error("[voice/feedback]", error);
    return jsonError("Feedback failed", 500);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function stringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value : undefined;
}

const MAX_CONSENTED_AUDIO_BYTES = 10 * 1024 * 1024;

function audioExtension(blob: Blob): string {
  const type = (blob.type || "").toLowerCase();
  const name =
    typeof (blob as File).name === "string" ? (blob as File).name : "";
  if (type.includes("wav") || /\.wav$/i.test(name)) return ".wav";
  if (type.includes("mpeg") || type.includes("mp3") || /\.mp3$/i.test(name))
    return ".mp3";
  if (type.includes("ogg") || /\.ogg$/i.test(name)) return ".ogg";
  if (type.includes("mp4") || type.includes("m4a") || /\.m4a$/i.test(name))
    return ".m4a";
  return ".webm";
}

/**
 * Persist a consented correction clip to ASR_FEEDBACK_AUDIO_DIR
 * (default <cwd>/data/asr-feedback-audio) as <feedbackId>.<ext>.
 * Returns the stored filename, or null when rejected/skipped.
 */
async function storeConsentedAudio(
  feedbackId: string,
  blob: Blob,
): Promise<string | null> {
  const name =
    typeof (blob as File).name === "string" ? (blob as File).name : "";
  const type = (blob.type || "").toLowerCase();
  const looksAudio =
    type.startsWith("audio/") ||
    type.includes("webm") ||
    /\.(wav|mp3|ogg|m4a|webm)$/i.test(name);
  if (!looksAudio) return null;
  if (blob.size <= 0 || blob.size > MAX_CONSENTED_AUDIO_BYTES) return null;

  const { mkdir, writeFile } = await import("node:fs/promises");
  const path = await import("node:path");

  const dir =
    process.env.ASR_FEEDBACK_AUDIO_DIR ||
    path.join(process.cwd(), "data", "asr-feedback-audio");
  await mkdir(dir, { recursive: true });
  const filename = `${feedbackId}${audioExtension(blob)}`;
  const bytes = Buffer.from(await blob.arrayBuffer());
  await writeFile(path.join(dir, filename), bytes);
  return filename;
}
