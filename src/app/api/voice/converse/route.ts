import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { isModalTtsConfigured, modalSpeak } from "@/lib/modal-tts";
import { understandUtterance } from "@/lib/understand";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { writeAudit } from "@/lib/audit";
import type { LanguageCode } from "@prisma/client";

/**
 * Real voice loop:
 *   audio → Modal ASR (Twi Whisper) → LLM understand → Modal TTS (Akan MMS)
 */
export async function POST(req: Request) {
  const started = Date.now();
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`converse:${ip}`, 12, 60);
    if (!rl.allowed) return jsonError("Too many voice turns — wait a minute", 429);

    if (!isModalAsrConfigured()) {
      return jsonError("MODAL_ASR_URL not configured — real ASR required", 503);
    }

    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    let audio: ArrayBuffer | null = null;
    let language: LanguageCode = user?.preferredLang ?? "tw";
    let conversationId: string | undefined;
    let speak = true;

    if (contentType.includes("multipart/form-data")) {
      const form = await req.formData();
      const file = form.get("audio");
      if (file instanceof Blob) audio = await file.arrayBuffer();
      const lang = form.get("language");
      if (typeof lang === "string" && ["tw", "en", "ga", "ee", "dag"].includes(lang)) {
        language = lang as LanguageCode;
      }
      const cid = form.get("conversationId");
      if (typeof cid === "string" && cid) conversationId = cid;
      const sp = form.get("speak");
      if (sp === "false" || sp === "0") speak = false;
    } else {
      audio = await req.arrayBuffer();
    }

    if (!audio || audio.byteLength < 100) {
      return jsonError("Audio required for real conversation", 400);
    }

    // 1) ASR
    const asr = await modalTranscribe(audio, {
      language: language === "en" ? "en" : undefined,
      contentType: "audio/webm",
    });
    if (asr.error || !asr.text?.trim()) {
      return jsonError(asr.error || "Empty transcription", 422, { asr });
    }

    // 2) Conversation history for understanding
    let conversation = conversationId
      ? await prisma.conversation.findUnique({ where: { id: conversationId } })
      : null;

    if (!conversation) {
      conversation = await prisma.conversation.create({
        data: {
          userId: user?.id,
          language,
          channel: "VOICE",
          title: asr.text.slice(0, 60),
          intent: "UNKNOWN",
        },
      });
    }

    const prior = await prisma.message.findMany({
      where: { conversationId: conversation.id },
      orderBy: { createdAt: "asc" },
      take: 12,
    });

    const history = prior.map((m) => ({
      role: (m.role === "USER" ? "user" : "assistant") as "user" | "assistant",
      content: m.content,
    }));

    await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "USER",
        content: asr.text,
        language,
        speakerLabel: asr.speaker,
        latencyMs: asr.latency_ms,
        metadata: { model: asr.model, mode: "asr" },
      },
    });

    // 3) Understand
    const understanding = await understandUtterance({
      text: asr.text,
      language,
      history,
    });

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: { intent: understanding.intent, language },
    });

    const assistant = await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: "ASSISTANT",
        content: understanding.reply,
        language,
        intent: understanding.intent,
        disclaimer: understanding.intent === "HEALTH",
        latencyMs: Date.now() - started,
        metadata: {
          severity: understanding.severity,
          escalate: understanding.escalate,
          engine: understanding.engine,
          asr_model: asr.model,
        },
      },
    });

    // 4) TTS (optional)
    let tts: {
      audio_base64?: string;
      sample_rate?: number;
      format?: string;
      model?: string;
      latency_ms?: number;
    } | null = null;

    if (speak && isModalTtsConfigured()) {
      try {
        const ttsLang = language === "en" ? "en" : "tw";
        tts = await modalSpeak(understanding.reply, ttsLang);
      } catch (e) {
        console.error("[tts]", e);
      }
    }

    await prisma.transcriptSession.create({
      data: {
        userId: user?.id,
        language,
        rawText: asr.text,
        punctuated: asr.text,
        speakers: [{ label: asr.speaker, model: asr.model }],
        audioDeleted: true,
        retained: false,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    await writeAudit({
      action: "voice.converse",
      actorId: user?.id,
      entityType: "conversation",
      entityId: conversation.id,
      ip,
      meta: {
        asr_model: asr.model,
        understand: understanding.engine,
        severity: understanding.severity,
        tts: Boolean(tts && "audio_base64" in tts && tts.audio_base64),
      },
    });

    return jsonOk({
      conversationId: conversation.id,
      asr: {
        text: asr.text,
        model: asr.model,
        latencyMs: asr.latency_ms,
        language: asr.language,
      },
      understanding: {
        reply: understanding.reply,
        intent: understanding.intent,
        severity: understanding.severity,
        escalate: understanding.escalate,
        engine: understanding.engine,
      },
      message: {
        id: assistant.id,
        content: understanding.reply,
      },
      tts: tts?.audio_base64
        ? {
            audioBase64: tts.audio_base64,
            sampleRate: tts.sample_rate,
            format: tts.format || "wav",
            model: tts.model,
            latencyMs: tts.latency_ms,
          }
        : null,
      totalLatencyMs: Date.now() - started,
    });
  } catch (e) {
    console.error("[voice/converse]", e);
    return jsonError(e instanceof Error ? e.message : "Converse failed", 500);
  }
}
